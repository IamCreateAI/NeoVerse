import torch



class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype


    def get_name_dict(self, lora_state_dict):
        lora_name_dict = {}
        for key in lora_state_dict:
            if ".lora_B." not in key:
                continue
            keys = key.split(".")
            if len(keys) > keys.index("lora_B") + 2:
                keys.pop(keys.index("lora_B") + 1)
            keys.pop(keys.index("lora_B"))
            if keys[0] == "diffusion_model":
                keys.pop(0)
            keys.pop(-1)
            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(".lora_B.", ".lora_A."))
        return lora_name_dict


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_name_dict = self.get_name_dict(state_dict_lora)
        for name, module in model.named_modules():
            if name in lora_name_dict:
                weight_up = state_dict_lora[lora_name_dict[name][0]].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict_lora[lora_name_dict[name][1]].to(device=self.device, dtype=self.torch_dtype)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)
                state_dict = module.state_dict()
                state_dict["weight"] = state_dict["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
                module.load_state_dict(state_dict)
                updated_num += 1
        print(f"{updated_num} tensors are updated by LoRA.")


class LightX2VLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype


    def get_name_dict(self, lora_state_dict):
        lora_pairs = {}
        lora_diffs = {}

        def try_lora_pair(key, prefix, suffix_a, suffix_b, target_suffix):
            if key.endswith(suffix_a):
                base_name = key[len(prefix) :].replace(suffix_a, target_suffix)
                pair_key = key.replace(suffix_a, suffix_b)
                if pair_key in lora_state_dict:
                    lora_pairs[base_name] = (key, pair_key)

        def try_lora_diff(key, prefix, suffix, target_suffix):
            if key.endswith(suffix):
                base_name = key[len(prefix) :].replace(suffix, target_suffix)
                lora_diffs[base_name] = key

        prefixs = [
            "",  # empty prefix
            "diffusion_model.",
        ]
        for prefix in prefixs:
            for key in lora_state_dict.keys():
                if not key.startswith(prefix):
                    continue

                try_lora_pair(key, prefix, "lora_A.weight", "lora_B.weight", "weight")
                try_lora_pair(key, prefix, "lora_down.weight", "lora_up.weight", "weight")
                try_lora_diff(key, prefix, "diff", "weight")
                try_lora_diff(key, prefix, "diff_b", "bias")
                try_lora_diff(key, prefix, "diff_m", "modulation")

        return lora_pairs, lora_diffs


    def load(self, model: torch.nn.Module, state_dict_lora, alpha=1.0):
        updated_num = 0
        lora_pairs, lora_diffs = self.get_name_dict(state_dict_lora)
        update_weight_dict = model.state_dict()
        for name, param in update_weight_dict.items():
            if name in lora_pairs:
                name_lora_A, name_lora_B = lora_pairs[name]
                lora_A = state_dict_lora[name_lora_A].to(device=self.device, dtype=self.torch_dtype)
                lora_B = state_dict_lora[name_lora_B].to(device=self.device, dtype=self.torch_dtype)
                if param.shape == (lora_B.shape[0], lora_A.shape[1]):
                    weight_lora = torch.mm(lora_B, lora_A)
                    update_weight_dict[name] = param.to(device=self.device, dtype=self.torch_dtype) + alpha * weight_lora
                    updated_num += 1
            elif name in lora_diffs:
                name_diff = lora_diffs[name]
                lora_diff = state_dict_lora[name_diff].to(device=self.device, dtype=self.torch_dtype)
                if param.shape == lora_diff.shape:
                    update_weight_dict[name] = param.to(device=self.device, dtype=self.torch_dtype) + alpha * lora_diff
                    updated_num += 1
        model.load_state_dict(update_weight_dict)
        print(f"{updated_num} tensors are updated by LoRA.")
