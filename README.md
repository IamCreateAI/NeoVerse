# NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos

[Yuxue Yang](https://yuxueyang1204.github.io)<sup>1, 2</sup>, [Lue Fan](https://lue.fan)<sup>1 ✉️ †</sup>, [Ziqi Shi](https://renshengji.github.io)<sup>1</sup>, [Junran Peng](https://jrpeng.github.io)<sup>1</sup>, [Feng Wang](https://happynear.wang)<sup>2</sup>, [Zhaoxiang Zhang](https://zhaoxiangzhang.net)<sup>1 ✉️</sup>

<sup>1</sup>NLPR & MAIS, CASIA&emsp; <sup>2</sup>CreateAI

<sup>✉️</sup>Corresponding Authors&emsp; <sup>†</sup>Project Lead

<a href='https://arxiv.org/abs/2601.00393'><img src='https://img.shields.io/badge/arXiv-2601.00393-b31b1b?logo=arxiv'></a> &nbsp;
<a href='https://neoverse-4d.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://huggingface.co/Yuppie1204/NeoVerse'><img src='https://img.shields.io/badge/Hugging Face-Model-gold?logo=huggingface'></a> &nbsp;
<a href='https://www.modelscope.cn/models/Yuppie1204/NeoVerse'><img src='https://img.shields.io/badge/ModelScope-Model-blueviolet?logo=modelscope'></a> &nbsp;
<a href='https://www.bilibili.com/video/BV1ezvYBBEMi'><img src='https://img.shields.io/badge/BiliBili-Video-479fd1?logo=bilibili'></a> &nbsp;
<a href='https://youtu.be/1k8Ikf8zbZw'><img src='https://img.shields.io/badge/YouTube-Video-orange?logo=youtube'></a>

NeoVerse is a versatile 4D world model that is capable of 4D reconstruction, novel-trajectory video generation, and rich downstream applications.

https://github.com/user-attachments/assets/4c957bd7-64e1-4a7e-9993-136740d911fe

**More videos are demonstrated on the [project website](https://neoverse-4d.github.io) for an enhanced view experience.**

## Updates

- **[2026-02-16]** Release inference scripts and model checkpoints in both [Hugging Face](https://huggingface.co/Yuppie1204/NeoVerse) and [ModelScope](https://www.modelscope.cn/models/Yuppie1204/NeoVerse).
- **[2026-01-01]** Release arXiv paper.


## TL;DR

- **Simple Inference Script** — Generate novel-trajectory videos with a single `python inference.py` command
- **Interactive Gradio Demo** — Step-by-step web UI for reconstruction, trajectory design, and generation
- **Multiple Reconstructors** — Supports different 3D reconstructors (e.g., [Depth Anything 3](https://depth-anything-3.github.io/)) via a plug-and-play interface
- **Fast Inference** — Inference pipeline completes in under 30 seconds with distilled LoRA acceleration on a single A800.

## Installation

### Step 1: Install Dependencies

We have tested NeoVerse on CUDA 12.1 with PyTorch 2.3.1 and CUDA 12.8 with PyTorch 2.7.1.

```bash
git clone https://github.com/IamCreateAI/NeoVerse.git
cd NeoVerse
conda create -n neoverse python=3.10 -y
conda activate neoverse

# For CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git

# For CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```

### Step 2: Download Model Checkpoints

```bash
hf download Yuppie1204/NeoVerse --local-dir models/NeoVerse
# Or using ModelScope
modelscope download --model Yuppie1204/NeoVerse --local_dir models/NeoVerse
```

Expected directory structure:
```
models/NeoVerse/
├── diffusion_pytorch_model-0000*-of-00006.safetensors
├── diffusion_pytorch_model.safetensors.index.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── reconstructor.ckpt
├── Wan2.1_VAE.pth
├── google/
│   └── ... (tokenizer files)
└── loras/
    └── Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors
```

## Usage

We provide two ways to try NeoVerse: a **command-line inference script** and an **interactive Gradio demo**.

### Inference Script

The inference script supports two trajectory input modes:

#### Predefined Trajectories with Adjustable Parameters

Use `--trajectory` to choose from 13 built-in camera motions, and fine-tune them with `--angle`, `--distance`, or `--orbit_radius`:

| Trajectory | Description |
|-----------|-------------|
| `pan_left` / `pan_right` | Rotate camera horizontally (yaw) |
| `tilt_up` / `tilt_down` | Rotate camera vertically (pitch) |
| `move_left` / `move_right` | Translate camera horizontally |
| `push_in` / `pull_out` | Translate camera forward / backward |
| `boom_up` / `boom_down` | Translate camera vertically |
| `orbit_left` / `orbit_right` | Arc around the scene center |
| `static` | Keep the original camera path |

```bash
# Tilt up
python inference.py \
    --input_path examples/videos/robot.mp4 \
    --trajectory tilt_up \
    --prompt "A two-arm robot assembles parts in front of a table." \
    --output_path outputs/tilt_up.mp4

# Move right by 0.2 units
python inference.py \
    --input_path examples/videos/tree_and_building.mp4 \
    --trajectory move_right \
    --distance 0.2 \
    --output_path outputs/move_right.mp4

# Zoom in 2x by adjusting the focal length
python inference.py \
    --input_path examples/videos/animal.mp4 \
    --trajectory static \
    --zoom_ratio 2.0 \
    --output_path outputs/zoom_in.mp4
```

#### Custom Trajectories from JSON

For full keyframe-level control, provide a trajectory JSON file via `--trajectory_file`:

```bash
# First orbit left, then pull out
python inference.py \
    --input_path examples/videos/movie.mp4 \
    --trajectory_file examples/trajectories/orbit_left_pull_out.json \
    --alpha_threshold 0.95 \
    --output_path outputs/orbit_left_pull_out.mp4

# Custom trajectory
python inference.py \
    --input_path examples/videos/driving.mp4 \
    --trajectory_file examples/trajectories/custom.json \
    --output_path outputs/custom_traj.mp4

# Custom trajectory on a static scene (single image input)
python inference.py \
    --input_path examples/videos/jungle.png \
    --static_scene \
    --trajectory_file examples/trajectories/custom2.json \
    --output_path outputs/custom_traj2.mp4

# Sparse keyframe poses with interpolation
python inference.py \
    --input_path examples/videos/driving2.mp4 \
    --trajectory_file examples/trajectories/sparse_matrices.json \
    --output_path outputs/keyframe_interpolation.mp4
```

See [docs/trajectory_format.md](docs/trajectory_format.md) for the JSON schema and [docs/coordinate_system.md](docs/coordinate_system.md) for the coordinate conventions. Ready-made examples are in `configs/trajectories/`.

You can validate a trajectory file without running inference:

```bash
python inference.py --trajectory_file my_trajectory.json --validate_only
```

#### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_path` | — | Input video or image path |
| `--trajectory` | — | Predefined trajectory type (see table above) |
| `--trajectory_file` | — | Path to a custom trajectory JSON (mutually exclusive with `--trajectory`) |
| `--output_path` | `outputs/inference.mp4` | Output video file path |
| `--prompt` | *(scene inpainting prompt)* | Text prompt for generation |
| `--static_scene` | off | Enable static scene mode (see below) |
| `--traj_mode` | `relative` | Trajectory coordinate mode (see below) |
| `--alpha_threshold` | `1.0` | Alpha mask threshold (see below) |
| `--reconstructor_path` | `models/NeoVerse/reconstructor.ckpt` | Path to reconstructor checkpoint |
| `--num_frames` | `81` | Number of output frames |
| `--height` / `--width` | `336` / `560` | Output resolution |
| `--disable_lora` | off | Use full 50-step inference instead of 4-step distilled LoRA |
| `--vis_rendering` | off | Save target-trajectory rendering visualizations alongside the output |
| `--seed` | `42` | Random seed |

**Scene Type** (`--static_scene`) — By default, NeoVerse treats the input as a *general scene*: frames are sampled across the full time range to capture camera and object motion. When `--static_scene` is set, all frames share the same timestamp, which is appropriate for a single image or a video with a completely stationary camera.

**Mode** (`--traj_mode`) — In `relative` mode (default), the designed trajectory is composed with the reconstructed input camera, so movements are relative to the original viewpoint. In `global` mode, the trajectory matrices are used directly in world space.

**Alpha Threshold** (`--alpha_threshold`) — After rendering the target viewpoint from the reconstructed 3D scene, pixels with alpha below this threshold are masked out and repainted by the diffusion model. Default `1.0` keeps all regions re-painted.

### Interactive Demo (Gradio)

Launch the web UI:

```bash
python app.py
```

The demo walks you through four steps:

1. **Upload** — Drop in a video or set of images and select the scene type (General / Static).
2. **Reconstruct** — Click `Reconstruct` to build a 4D Gaussian Splat scene. The 3D viewer shows Gaussian-Splatting-centred point cloud so you can inspect the spatial layout.
3. **Design Trajectory** — Pick a camera motion type and adjust sliders, or upload a trajectory JSON. Click `Render` to preview RGB and mask renderings.
4. **Generate** — Enter a prompt and click `Generate` to synthesize the final video.

### Alternative Reconstructors

NeoVerse also supports alternative reconstructors such as [Depth Anything 3](https://depth-anything-3.github.io/). Their predicted depth and camera parameters can be converted to pseudo Gaussian splats to plug into NeoVerse's pipeline.

Download the Depth Anything 3 checkpoint:

```bash
# Download model.safetensors from Hugging Face
wget https://huggingface.co/depth-anything/DA3-GIANT-1.1/resolve/main/model.safetensors -O models/da3_giant_1.1.safetensors
```

Then pass it via `--reconstructor_path`:

```bash
# CLI inference with Depth Anything 3
python inference.py \
    --input_path examples/videos/driving.mp4 \
    --trajectory_file examples/trajectories/custom.json \
    --reconstructor_path models/da3_giant_1.1.safetensors \
    --output_path outputs/custom_traj_da3.mp4

# Gradio demo with Depth Anything 3
python app.py --reconstructor_path models/da3_giant_1.1.safetensors
```

## Model Architecture

NeoVerse has two main components:

1. **Reconstructor** — Recovers 3D scene structure (Gaussian Splats + camera poses) from a monocular video. In the released version, we provide a [WorldMirror-based](https://3d-models.hunyuan.tencent.com/world/) reconstructor finetuned on 3D/4D datasets. What's more, NeoVerse is compatible with other reconstructors like [Depth Anything 3](https://depth-anything-3.github.io/) by converting their outputs to pseudo Gaussian splats.
2. **Video Diffusion Model** — Generates high-quality video frames conditioned on the reconstructed scene. Here we use a [WAN 2.1](https://github.com/Wan-Video/Wan2.1) backbone with [a 4-step distilled LoRA](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill-Lightx2v/tree/main/loras) for a fast inference speed.

For technical details, please refer to our [paper](https://arxiv.org/abs/2601.00393).

## Citation

If you find this work helpful, please help star the repository and consider citing it as follows. It would be greatly appreciated!

```bibtex
@article{yang2026neoverse,
  title={NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos},
  author={Yang, Yuxue and Fan, Lue and Shi, Ziqi and Peng, Junran and Wang, Feng and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2601.00393},
  year={2026}
}
```

## Acknowledgments

We sincerely thank the great work [VGGT](https://vgg-t.github.io/), [WorldMirror](https://3d-models.hunyuan.tencent.com/world/), [Depth Anything 3](https://depth-anything-3.github.io/), [Wan-Video](https://github.com/Wan-Video/Wan2.1), [TrajectoryCrafter](https://trajectorycrafter.github.io/), [ReCamMaster](https://jianhongbai.github.io/ReCamMaster/), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for their inspiring work and contributions to the 3D and video generation community.

## Contact Us

We believe NeoVerse has the potential to unlock a wide range of applications and we are excited to see how the community will use and build upon it. If you have any questions, suggestions, or want to share your results, please feel free to reach out to us via email [yangyuxue2023@ia.ac.cn](mailto:yangyuxue2023@ia.ac.cn) or WeChat ([Yuppie898988](Yuppie898988)). We also welcome you to open an issue on GitHub for any bug reports or feature requests.