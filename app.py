import argparse
import gc
import json
import os
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
from diffsynth import save_video
from diffsynth.utils.auxiliary import CameraTrajectory, load_video, homo_matrix_inverse
from diffsynth.utils.app import extract_point_cloud, build_scene_glb

parser = argparse.ArgumentParser()
parser.add_argument("--reconstructor_path", type=str,
                    default="models/NeoVerse/reconstructor.ckpt",
                    help="Path to reconstructor checkpoint")
args, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------
# Global model
# ---------------------------------------------------------------------------
OUTPUT_ROOT = "outputs/gradio"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
GLB_PATH = os.path.join(OUTPUT_ROOT, "scene.glb")
PREVIEW_PATH = os.path.join(OUTPUT_ROOT, "preview.mp4")
MASK_PATH = os.path.join(OUTPUT_ROOT, "mask.mp4")
OUTPUT_PATH = os.path.join(OUTPUT_ROOT, "output.mp4")
JSON_PATH = os.path.join(OUTPUT_ROOT, "trajectory.json")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading NeoVerse pipeline (reconstructor: {args.reconstructor_path})...")
pipe = WanVideoNeoVersePipeline.from_pretrained(
    local_model_path="models",
    reconstructor_path=args.reconstructor_path,
    lora_path="models/NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors",
    lora_alpha=1.0,
    torch_dtype=torch.bfloat16,
).to(device)
print("Pipeline loaded.")


def _export_scene(scene):
    """Export a trimesh.Scene to the fixed GLB path and return it."""
    scene.export(file_obj=GLB_PATH)
    return GLB_PATH


# ---------------------------------------------------------------------------
# 1. Upload handler
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _get_example_videos(config_path="examples/gallery.json"):
    """Scan directory for video/image files and return metadata list.

    If an ``examples.json`` exists in *directory*, it is used as the
    authoritative source (preserving order and per-example parameters).
    Files present on disk but absent from the JSON are appended with
    default parameters.
    """
    if not os.path.exists(config_path):
        return []
    _DEFAULTS = {
        "scene_type": "General scene",
        "camera_motion": "static",
        "angle": 0,
        "distance": 0,
        "orbit_radius": 0,
        "mode": "relative",
        "zoom_ratio": 1.0,
        "alpha_threshold": 1.0,
        "use_first_frame": True,
        "traj_file": None,
    }

    examples = []
    if os.path.exists(config_path):
        with open(config_path) as f:
            entries = json.load(f)
        for entry in entries:
            fpath = entry["file"]
            if not os.path.exists(fpath):
                continue
            ex = {**_DEFAULTS, **entry}
            examples.append(ex)
    return examples


def handle_upload(files, scene_type):
    """Load user media into a list of PIL images stored in gr.State."""
    if not files:
        return gr.update(), None, gr.update(interactive=False)
    static = scene_type == "Static scene"
    # Detect whether any file is a video
    video_path = None
    image_paths = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in VIDEO_EXTS:
            video_path = f
            break
        else:
            image_paths.append(f)
    if video_path:
        pil_images = load_video(video_path, 81, resolution=(560, 336),
                                resize_mode="center_crop", static_scene=static)
    elif image_paths:
        pil_images = load_video(image_paths, 81, resolution=(560, 336),
                                resize_mode="center_crop", static_scene=static)
    else:
        return gr.update(), None, gr.update(interactive=False)
    state = {"images": pil_images, "scene_type": scene_type}
    return state, pil_images, gr.update(interactive=True)


# ---------------------------------------------------------------------------
# 2. Reconstruction
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(state):
    """Run the reconstructor and return 3D scene."""
    if state is None or "images" not in state:
        raise gr.Error("Please upload a video or images first.")

    pil_images = state["images"]
    scene_type = state.get("scene_type", "General scene")
    static_flag = scene_type == "Static scene"
    S = len(pil_images)

    views = {
        "img": torch.stack([F.to_tensor(img)[None] for img in pil_images], dim=1).to(device),
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
    }
    if static_flag:
        views["is_static"] = torch.ones((1, S), dtype=torch.bool, device=device)
        views["timestamp"] = torch.zeros((1, S), dtype=torch.int64, device=device)
    else:
        views["is_static"] = torch.zeros((1, S), dtype=torch.bool, device=device)
        views["timestamp"] = torch.arange(0, S, dtype=torch.int64, device=device).unsqueeze(0)
    with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
        predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

    gaussians = predictions["splats"]
    input_intrs = predictions["rendered_intrinsics"][0]        # [S, 3, 3]
    input_cam2world = predictions["rendered_extrinsics"][0]     # [S, 4, 4]
    input_timestamps = predictions["rendered_timestamps"][0]    # [S]

    points, colors, frame_indices = extract_point_cloud(predictions)

    state["source_views"] = views
    state["gaussians"] = gaussians
    state["input_intrs"] = input_intrs
    state["input_cam2world"] = input_cam2world
    state["input_timestamps"] = input_timestamps
    state["points"] = points
    state["colors"] = colors
    state["frame_indices"] = frame_indices
    state["height"] = pil_images[0].size[1]
    state["width"] = pil_images[0].size[0]

    # Build GLB: 11-frame point cloud, all S cameras shown
    scene = build_scene_glb(points, colors, frame_indices, input_cam2world.cpu().numpy())
    glb_path = _export_scene(scene)

    return state, glb_path, gr.update(interactive=True)


# ---------------------------------------------------------------------------
# 3. Build trajectory
# ---------------------------------------------------------------------------
def build_trajectory(state, t_type, mode, angle, distance, orbit_radius, zoom_ratio, use_first_frame):
    """Build camera trajectory from UI rows, visualize, and export JSON."""
    if state is None or "gaussians" not in state:
        raise gr.Error("Run reconstruction first.")

    json_data = {
        "name": "gradio_traj",
        "mode": mode,
        "num_frames": 81,
        "zoom_ratio": zoom_ratio,
        "use_first_frame": use_first_frame,
        "keyframes": [
            {
                "0": [{"static": {}}]
            },
            {
                "80": [{t_type: {"angle": int(angle), "distance": float(distance), "orbit_radius": float(orbit_radius)}}]
            }
        ]
    }
    with open(JSON_PATH, "w") as f:
        json.dump(json_data, f, indent=2)
    cam_traj = CameraTrajectory.from_json(JSON_PATH)
    return cam_traj


def upload_trajectory(state, t_file):
    """Load trajectory JSON, build trajectory."""
    if state is None or "gaussians" not in state:
        raise gr.Error("Upload a trajectory JSON after reconstruction.")

    cam_traj = CameraTrajectory.from_json(t_file)
    return cam_traj


def handle_traj_upload(t_file):
    """Parse uploaded trajectory JSON and extract shared parameters."""
    if t_file is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    with open(t_file, "r") as f:
        data = json.load(f)
    mode = data.get("mode", "relative")
    zoom_ratio = data.get("zoom_ratio", 1.0)
    use_first_frame = data.get("use_first_frame", True)
    return mode, zoom_ratio, use_first_frame


# ---------------------------------------------------------------------------
# 4. Render preview
# ---------------------------------------------------------------------------
@torch.no_grad()
def preview(state, selected_tab, t_file, t_type, angle, distance, orbit_r,
            mode, zoom, use_ff, alpha_threshold):
    """Build trajectory then render preview.

    The active tab determines the trajectory source:
    *TAB_TRAJ_FILE* uses the uploaded JSON; *TAB_CAMERA_PARAMS* uses sliders.
    """
    if selected_tab == TAB_TRAJ_FILE:
        cam_traj = upload_trajectory(state, t_file)
        cam_traj.mode = mode
        cam_traj.zoom_ratio = zoom
        cam_traj.use_first_frame = use_ff
        with open(t_file, "r") as f:
            json_data = json.load(f)
        json_data["mode"] = mode
        json_data["zoom_ratio"] = zoom
        json_data["use_first_frame"] = use_ff
        with open(JSON_PATH, "w") as f:
            json.dump(json_data, f, indent=2)
    else:
        cam_traj = build_trajectory(
            state, t_type, mode, angle, distance, orbit_r, zoom, use_ff)
    static_flag = state.get("scene_type", "General scene") == "Static scene"
    input_cam2world = state["input_cam2world"]
    target_cam2world = cam_traj.c2w.to(device)
    if cam_traj.mode == "relative" and not static_flag:
        target_cam2world = input_cam2world @ target_cam2world

    scene = build_scene_glb(state["points"], state["colors"], state["frame_indices"],
                            target_cam2world.cpu().numpy())
    glb_path = _export_scene(scene)
    gaussians = state["gaussians"]
    input_intrs = state["input_intrs"]              # [S, 3, 3] tensor
    timestamps = state["input_timestamps"]           # [S] tensor
    H, W = state["height"], state["width"]

    if static_flag:
        K_81 = input_intrs[:1].repeat(81, 1, 1)
        ts_81 = timestamps[:1].repeat(81)
    else:
        K_81 = input_intrs
        ts_81 = timestamps

    # Apply zoom_ratio (matches inference.py)
    ratio = torch.linspace(1, cam_traj.zoom_ratio, K_81.shape[0], device=device)
    K_zoomed = K_81.clone()
    K_zoomed[:, 0, 0] *= ratio
    K_zoomed[:, 1, 1] *= ratio

    target_world2cam = homo_matrix_inverse(target_cam2world)
    target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
        gaussians,
        render_viewmats=[target_world2cam],
        render_Ks=[K_zoomed],
        render_timestamps=[ts_81],
        sh_degree=0, width=W, height=H,
    )

    target_mask = (target_alpha > alpha_threshold).float()
    if cam_traj.use_first_frame:
        pil_images = state["images"]
        first_frame_rgb = F.to_tensor(pil_images[0]).permute(1, 2, 0).to(device)
        target_rgb[0, 0] = first_frame_rgb
        target_mask[0, 0] = 1.0

    frames = []
    mask_frames = []
    for i in range(target_rgb.shape[1]):
        frame = (target_rgb[0, i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        frames.append(Image.fromarray(frame))
        mask_f = (target_mask[0, i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        if mask_f.ndim == 3 and mask_f.shape[2] == 1:
            mask_f = np.repeat(mask_f, 3, axis=2)
        elif mask_f.ndim == 2:
            mask_f = np.stack([mask_f] * 3, axis=-1)
        mask_frames.append(Image.fromarray(mask_f))

    state["target_rgb"] = target_rgb
    state["target_depth"] = target_depth
    state["target_mask"] = target_mask
    state["target_poses"] = target_cam2world.unsqueeze(0)
    state["target_intrs"] = K_zoomed.unsqueeze(0)

    save_video(frames, PREVIEW_PATH, fps=16)
    save_video(mask_frames, MASK_PATH, fps=16)
    return state, glb_path, PREVIEW_PATH, MASK_PATH, gr.update(interactive=True), JSON_PATH


# ---------------------------------------------------------------------------
# 5. Generate final video
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_final(state, prompt, negative_prompt, seed):
    """Run diffusion generation using rendered conditioning."""
    if state is None or "target_rgb" not in state:
        raise gr.Error("Run Render Preview first.")

    H, W = state["height"], state["width"]
    wrapped_data = {
        "source_views": state["source_views"],
        "target_rgb": state["target_rgb"],
        "target_depth": state["target_depth"],
        "target_mask": state["target_mask"],
        "target_poses": state["target_poses"],
        "target_intrs": state["target_intrs"],
    }

    seed = int(seed)
    generated_frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed, rand_device=device,
        height=H, width=W, num_frames=81,
        cfg_scale=1.0, num_inference_steps=4, tiled=False,
        **wrapped_data,
    )

    save_video(generated_frames, OUTPUT_PATH, fps=16)
    gc.collect()
    torch.cuda.empty_cache()
    return OUTPUT_PATH


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
theme = gr.themes.Base()

VALID_TYPES = sorted(CameraTrajectory.VALID_TRAJECTORY_TYPES)
TAB_CAMERA_PARAMS = "tab_camera_params"
TAB_TRAJ_FILE = "tab_traj_file"

with gr.Blocks(theme=theme, title="NeoVerse Interactive Demo") as demo:
    gr.HTML(
    """
    <div style="text-align: center;">
    <h1>
        <strong style="background: linear-gradient(to right, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">NeoVerse</strong>
        <span>: Enhancing 4D World Model with in-the-wild Monocular Videos</span>
    </h1>
    <p>
        📑 <a href="https://arxiv.org/abs/2601.00393">arXiv</a> &nbsp&nbsp | &nbsp&nbsp 🌐 <a href="https://neoverse-4d.github.io">Project</a> &nbsp&nbsp  | &nbsp&nbsp🖥️ <a href="https://github.com/IamCreateAI/NeoVerse">GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Yuppie1204/NeoVerse">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://www.modelscope.cn/models/Yuppie1204/NeoVerse">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 🎞️ <a href="https://www.bilibili.com/video/BV1ezvYBBEMi">BiliBili</a> &nbsp&nbsp | &nbsp&nbsp 🎥 <a href="https://youtu.be/1k8Ikf8zbZw">YouTube</a> &nbsp&nbsp
    </p>
    </div>
    <div style="font-size: 16px; line-height: 1.5;">
        <p>NeoVerse is a versatile 4D world model that turns monocular videos into free-viewpoint video generation.
        Given a single video or a set of images, NeoVerse reconstructs the underlying 4D scene and lets you
        render novel-trajectory videos along any custom camera path.</p>
    <ol>
        <li><strong>Upload</strong> &mdash; In the left column, upload a video or multiple images and select the scene type (General / Static).</li>
        <li><strong>Reconstruct</strong> &mdash; Click "Reconstruct" to perform 4D reconstruction. The middle column visualises the scene as a Gaussian-Splatting-centred point cloud so you can inspect the spatial layout and camera scale.</li>
        <li><strong>Design Camera Trajectory</strong> &mdash; Two input modes are available under the <em>Camera Parameters</em> and <em>Trajectory File</em> tabs:
            <ul>
                <li><em>Camera Parameters</em>: select a camera motion type (pan, tilt, orbit, push, etc.) and adjust angle, distance, and orbit radius with the sliders.</li>
                <li><em>Trajectory File</em>: upload a trajectory JSON file for full control over keyframes.</li>
            </ul>
            Click "Render" to preview RGB and mask renderings of the planned path.
        </li>
        <li><strong>Generate</strong> &mdash; In the right column, enter your prompt and click "Generate". NeoVerse synthesises the final video conditioned on the designed trajectory.</li>
    </ol>
    <h3>Key Parameters:</h3>
    <ul>
        <li><strong>Scene Type</strong> &mdash; <em>General</em>: for videos with camera or object motion; frames are sampled across the full time range. <em>Static</em>: for a single image or a stationary scene; all frames share the same timestamp.</li>
        <li><strong>Mode</strong> &mdash; <em>Relative</em>: the designed trajectory is composed with the reconstructed input camera, so movements are relative to the original viewpoint. <em>Global</em>: the trajectory matrices are used directly in world space.</li>
        <li><strong>Alpha Threshold</strong> &mdash; Controls the binary mask derived from the rendered alpha channel. Default 1.0 keeps all regions re-painted.</li>
    </ul>
    <p><strong>Note:</strong> Selecting an example from the gallery will automatically trigger reconstruction. Please wait a few seconds for it to complete before clicking "Render" to preview the target trajectory and renderings.</p>
    </div>
    """)
    app_state = gr.State(value=None)
    selected_tab_state = gr.State(value=TAB_CAMERA_PARAMS)

    with gr.Row():
        # ---- Left column: Upload ----
        with gr.Column(scale=1):
            scene_type = gr.Radio(["General scene", "Static scene"],
                                  value="General scene", label="Scene Type")
            file_upload = gr.File(file_count="multiple", label="Upload Video or Images",
                                  interactive=True, file_types=["image", "video"])
            image_gallery = gr.Gallery(label="Preview", columns=4, height=200,
                                       object_fit="contain")
            reconstruct_btn = gr.Button("Reconstruct", variant="primary",
                                        interactive=False)

            gr.Markdown("### Examples")
            _examples = _get_example_videos()
            if _examples:
                _gallery_items = [(ex["file"], ex["name"]) for ex in _examples]
                example_gallery = gr.Gallery(
                    value=_gallery_items,
                    label="Click to load",
                    columns=2, height=300,
                    object_fit="contain",
                    show_label=False,
                    interactive=True, preview=False, allow_preview=False,
                )

        # ---- Middle column: Visualization + Trajectory ----
        with gr.Column(scale=3):
            with gr.Row():
                model3d = gr.Model3D(label="Point Clouds Reference", height=350,
                                    zoom_speed=0.5, pan_speed=0.5, scale=1.0)
                with gr.Column(scale=1):
                    preview_video = gr.Video(label="RGB Rendering", height=170)
                    mask_video = gr.Video(label="Mask Rendering", height=170)

            with gr.Tabs() as traj_tabs:
                with gr.Tab("Camera Parameters", id=TAB_CAMERA_PARAMS):
                    with gr.Row():
                        traj_type = gr.Dropdown(choices=VALID_TYPES, value="static",
                                                label="Camera Motion")
                        traj_angle = gr.Slider(minimum=0, maximum=60, value=0,
                                               step=1, label="Angle")
                        traj_distance = gr.Slider(minimum=0, maximum=1, value=0,
                                                  step=0.01, label="Distance")
                        traj_orbit = gr.Slider(minimum=0, maximum=2, value=0,
                                               step=0.1, label="Orbit Radius")
                with gr.Tab("Trajectory File", id=TAB_TRAJ_FILE):
                    traj_upload = gr.File(
                        label="Upload Trajectory JSON",
                        file_types=[".json"],
                        file_count="single",
                        interactive=True,
                    )
            with gr.Row():
                traj_mode = gr.Radio(["relative", "global"], value="relative",
                                        label="Mode")
                zoom_ratio_input = gr.Slider(minimum=0.1, maximum=2, value=1.0,
                                                step=0.1, label="Zoom Ratio")
                alpha_threshold_input = gr.Slider(minimum=0, maximum=1, value=1.0,
                                                    step=0.01, label="Alpha Threshold")
                use_first_frame_input = gr.Checkbox(value=True,
                                                        label="Use First Frame")
            traj_download = gr.File(
                label="Download Trajectory JSON",
                interactive=False,
            )
            preview_btn = gr.Button("Render", variant="primary", interactive=False)

        # ---- Right column: Generation ----
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                value="A smooth video with complete scene content. "
                      "Inpaint any missing regions or margins naturally "
                      "to match the surrounding scene.",
            )
            neg_prompt = gr.Textbox(label="Negative Prompt", value="")
            seed = gr.Number(label="Seed", value=42, precision=0)
            output_video = gr.Video(label="Generated Video")
            generate_btn = gr.Button("Generate", variant="primary", interactive=False)

    # ================================================================
    # Wiring
    # ================================================================

    # Sync default params when camera motion type changes
    _DEFAULT_PARAMS = {
        "pan_left": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "pan_right": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "tilt_up": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "tilt_down": {"angle": 15, "distance": 0, "orbit_radius": 0},
        "move_left": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "move_right": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "push_in": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "pull_out": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "boom_up": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "boom_down": {"angle": 0, "distance": 0.1, "orbit_radius": 0},
        "orbit_left": {"angle": 15, "distance": 0, "orbit_radius": 1.0},
        "orbit_right": {"angle": 15, "distance": 0, "orbit_radius": 1.0},
        "static": {"angle": 0, "distance": 0, "orbit_radius": 0},
    }

    def _sync_traj_params(ttype):
        p = _DEFAULT_PARAMS.get(ttype, {})
        return p.get("angle", 0), p.get("distance", 0), p.get("orbit_radius", 0)

    traj_type.input(fn=_sync_traj_params,
                     inputs=[traj_type],
                     outputs=[traj_angle, traj_distance, traj_orbit])

    # Track selected tab via state
    def _on_tab_select(evt: gr.SelectData):
        return evt.value
    traj_tabs.select(fn=_on_tab_select, inputs=[], outputs=[selected_tab_state])

    # Upload
    file_upload.upload(fn=handle_upload,
                       inputs=[file_upload, scene_type],
                       outputs=[app_state, image_gallery, reconstruct_btn])
    scene_type.input(fn=handle_upload,
                      inputs=[file_upload, scene_type],
                      outputs=[app_state, image_gallery, reconstruct_btn])

    # Example gallery
    if _examples:
        def _load_example(evt: gr.SelectData):
            """Load an example and apply its preset parameters."""
            ex = _examples[evt.index]
            sc_type = ex.get("scene_type", "General scene")
            state, pil_images, btn_update = handle_upload([ex["file"]], sc_type)
            traj_file = ex.get("traj_file", None)
            if traj_file:
                tab_sel = gr.Tabs(selected=TAB_TRAJ_FILE)
                tab_id = TAB_TRAJ_FILE
            else:
                tab_sel = gr.Tabs(selected=TAB_CAMERA_PARAMS)
                tab_id = TAB_CAMERA_PARAMS
            return (state, pil_images, btn_update,
                    sc_type,
                    ex.get("camera_motion", "static"),
                    ex.get("angle", 0),
                    ex.get("distance", 0),
                    ex.get("orbit_radius", 0),
                    ex.get("mode", "relative"),
                    ex.get("zoom_ratio", 1.0),
                    ex.get("alpha_threshold", 1.0),
                    ex.get("use_first_frame", True),
                    traj_file,
                    tab_sel,
                    tab_id)

        example_gallery.select(
            fn=_load_example,
            inputs=[],
            outputs=[app_state, image_gallery, reconstruct_btn,
                     scene_type,
                     traj_type, traj_angle, traj_distance, traj_orbit,
                     traj_mode, zoom_ratio_input, alpha_threshold_input,
                     use_first_frame_input, traj_upload, traj_tabs,
                     selected_tab_state],
        ).then(
            fn=reconstruct,
            inputs=[app_state],
            outputs=[app_state, model3d, preview_btn],
        )

    # Reconstruct
    reconstruct_btn.click(
        fn=reconstruct,
        inputs=[app_state],
        outputs=[app_state, model3d, preview_btn],
    )
    # Preview (build trajectory + render + export JSON)
    # Active tab determines trajectory source
    preview_btn.click(
        fn=preview,
        inputs=[app_state, selected_tab_state, traj_upload, traj_type, traj_angle, traj_distance, traj_orbit,
                traj_mode, zoom_ratio_input, use_first_frame_input,
                alpha_threshold_input],
        outputs=[app_state, model3d,
                 preview_video, mask_video, generate_btn, traj_download],
    )

    # Sync shared params from uploaded trajectory JSON
    traj_upload.change(
        fn=handle_traj_upload,
        inputs=[traj_upload],
        outputs=[traj_mode, zoom_ratio_input, use_first_frame_input],
    )

    # Generate
    generate_btn.click(
        fn=generate_final,
        inputs=[app_state, prompt, neg_prompt, seed],
        outputs=[output_video],
    )


if __name__ == "__main__":
    demo.queue(max_size=5).launch(show_error=True, share=True)
