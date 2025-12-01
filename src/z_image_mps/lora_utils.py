"""Manual LoRA weight merging for ZImagePipeline."""

import re
import safetensors
import torch
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LoRAMerger:
    """Manual LoRA weight merger for pipelines without built-in LoRA support."""

    def __init__(self, pipeline):
        """Initialize with a pipeline instance."""
        self.pipeline = pipeline
        self.original_weights = {}  # Store original weights for unloading

    def load_lora_weights(self, lora_path: str, lora_scale: float = 1.0) -> None:
        """
        Manually load and merge LoRA weights into the pipeline.

        Args:
            lora_path: Path to LoRA safetensors file
            lora_scale: Scale factor for LoRA weights (typically 0.0-2.0)
        """
        try:
            if self.original_weights:
                logger.info("Existing LoRA detected - restoring base weights before applying new LoRA")
                self.unload_lora_weights()

            # Load LoRA weights
            lora_state_dict = {}
            with safetensors.safe_open(lora_path, framework="pt") as f:
                for key in f.keys():
                    lora_state_dict[key] = f.get_tensor(key)

            # Store original weights before modification
            self._backup_original_weights(lora_state_dict, lora_scale)

            # Merge LoRA weights
            merged_count = self._merge_lora_weights(lora_state_dict, lora_scale)

            logger.info(f"Successfully merged {merged_count} LoRA weight pairs from {lora_path} with scale {lora_scale}")

        except Exception as e:
            logger.error(f"Failed to load LoRA from {lora_path}: {e}")
            raise

    def _backup_original_weights(self, lora_state_dict: Dict[str, torch.Tensor], lora_scale: float) -> None:
        """Store original weights before LoRA modification."""
        for lora_key in lora_state_dict.keys():
            # Get the corresponding module path from LoRA key
            module_path = self._get_module_path_from_lora_key(lora_key)
            if module_path:
                try:
                    module = self._get_module_from_path(module_path)
                    if module is not None and hasattr(module, 'weight'):
                        if module_path not in self.original_weights:
                            self.original_weights[module_path] = module.weight.data.clone()
                except Exception as e:
                    logger.debug(f"Could not backup weight for {module_path}: {e}")

    def _merge_lora_weights(self, lora_state_dict: Dict[str, torch.Tensor], lora_scale: float) -> int:
        """Merge LoRA weights into the pipeline modules."""
        merged_count = 0

        # Group LoRA weights by their corresponding modules
        lora_pairs = self._group_lora_weights(lora_state_dict)

        for module_path, lora_a_key, lora_b_key in lora_pairs:
            try:
                # Get LoRA matrices
                lora_a = lora_state_dict[lora_a_key]
                lora_b = lora_state_dict[lora_b_key]

                base_key = lora_a_key.replace('.lora_A.weight', '')
                alpha_key = f"{base_key}.alpha"
                alpha_tensor = lora_state_dict.get(alpha_key)

                rank = lora_b.shape[1] if lora_b.dim() >= 2 else lora_a.shape[0]
                alpha = alpha_tensor.item() if torch.is_tensor(alpha_tensor) else rank
                scale = lora_scale * (alpha / rank if rank else 1.0)

                # Calculate LoRA weight contribution: B @ A * scale
                lora_weight = (lora_b @ lora_a) * scale

                # Get target module
                module = self._get_module_from_path(module_path)
                if module is None or not hasattr(module, 'weight'):
                    logger.warning(f"Module {module_path} not found or has no weights")
                    continue

                # Check weight dimensions
                if module.weight.shape != lora_weight.shape:
                    logger.warning(f"Shape mismatch for {module_path}: module {module.weight.shape} vs LoRA {lora_weight.shape}")
                    continue

                # Merge LoRA weights
                with torch.no_grad():
                    module.weight.data += lora_weight.to(module.weight.device, module.weight.dtype)

                merged_count += 1

            except Exception as e:
                logger.error(f"Failed to merge LoRA weights for {module_path}: {e}")
                continue

        return merged_count

    def _group_lora_weights(self, lora_state_dict: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, str, str], ...]:
        """Group LoRA A and B weights by their corresponding modules."""
        lora_pairs = []

        for lora_key in lora_state_dict.keys():
            if '.lora_A.weight' in lora_key:
                # Find corresponding B key
                base_key = lora_key.replace('.lora_A.weight', '')
                b_key = f"{base_key}.lora_B.weight"

                if b_key in lora_state_dict:
                    module_path = self._get_module_path_from_lora_key(lora_key)
                    if module_path:
                        lora_pairs.append((module_path, lora_key, b_key))

        return tuple(lora_pairs)

    def _get_module_path_from_lora_key(self, lora_key: str) -> Optional[str]:
        """Convert LoRA key to module path in the transformer."""
        # Remove 'diffusion_model.' prefix if present
        key = lora_key.replace('diffusion_model.', '')

        # Handle different layer types
        attn_match = re.match(r'layers\.(\d+)\.attention\.(to_q|to_k|to_v|to_out\.0)', key)
        if attn_match:
            layer_idx, target = attn_match.groups()
            return f"transformer.layers.{layer_idx}.attention.{target}"

        ff_match = re.match(r'layers\.(\d+)\.feed_forward\.(w1|w2|w3)', key)
        if ff_match:
            layer_idx, ff_part = ff_match.groups()
            return f"transformer.layers.{layer_idx}.feed_forward.{ff_part}"

        adaln_match = re.match(r'layers\.(\d+)\.adaLN_modulation\.0', key)
        if adaln_match:
            layer_idx = adaln_match.group(1)
            return f"transformer.layers.{layer_idx}.adaLN_modulation.0"

        context_match = re.match(r'context_refiner\.(\d+)\.attention\.(to_q|to_k|to_v|to_out\.0)', key)
        if context_match:
            layer_idx, target = context_match.groups()
            return f"transformer.context_refiner.{layer_idx}.attention.{target}"

        noise_match = re.match(r'noise_refiner\.(\d+)\.attention\.(to_q|to_k|to_v|to_out\.0)', key)
        if noise_match:
            layer_idx, target = noise_match.groups()
            return f"transformer.noise_refiner.{layer_idx}.attention.{target}"

        # If no pattern matches, try direct conversion
        return f"transformer.{key}" if key.startswith('layers.') else None

    def _get_module_from_path(self, module_path: str):
        """Get module from the pipeline using dot notation path."""
        try:
            parts = module_path.split('.')
            current = self.pipeline

            for part in parts:
                if part.isdigit() and isinstance(current, (list, tuple, torch.nn.ModuleList)):
                    current = current[int(part)]
                else:
                    current = getattr(current, part)

            return current
        except AttributeError:
            return None

    def unload_lora_weights(self) -> None:
        """Restore original weights and remove all LoRA modifications."""
        if not self.original_weights:
            logger.warning("No LoRA weights to unload")
            return

        restored_count = 0
        for module_path, original_weight in self.original_weights.items():
            try:
                module = self._get_module_from_path(module_path)
                if module is not None and hasattr(module, 'weight'):
                    with torch.no_grad():
                        module.weight.data.copy_(original_weight.to(module.weight.device, module.weight.dtype))
                    restored_count += 1
            except Exception as e:
                logger.error(f"Failed to restore weights for {module_path}: {e}")

        self.original_weights.clear()
        logger.info(f"Restored original weights for {restored_count} modules")

    def set_lora_scale(self, new_scale: float) -> None:
        """Update LoRA scale by reloading with new scale."""
        if not self.original_weights:
            logger.warning("No LoRA weights loaded - cannot adjust scale")
            return

        # Store current loaded LoRA path (would need to be tracked separately)
        logger.warning("Scale adjustment requires reloading LoRA - please use load_lora_weights with new scale")


def load_lora_weights(pipeline, lora_path: str, lora_scale: float = 1.0) -> LoRAMerger:
    """
    Convenience function to load LoRA weights into a pipeline.

    Args:
        pipeline: The pipeline instance (e.g., ZImagePipeline)
        lora_path: Path to LoRA safetensors file
        lora_scale: Scale factor for LoRA weights

    Returns:
        LoRAMerger instance for managing the loaded LoRA
    """
    merger = LoRAMerger(pipeline)
    merger.load_lora_weights(lora_path, lora_scale)
    return merger
