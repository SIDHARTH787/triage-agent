"""
Resource-aware scheduler.

Formalizes and measures the GPU/CPU split arrived at empirically during
Phase 1/2 debugging. Reports VRAM usage via nvidia-smi and warns if the
configured budget is exceeded.
"""

import subprocess
import re
import logging

log = logging.getLogger("scheduler")


class ResourceScheduler:
    def __init__(self, config: dict):
        self.max_vram_gb = config.get("max_vram_gb", 8)
        self._nvidia_smi_available = self._check_nvidia_smi()

    @staticmethod
    def _check_nvidia_smi() -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True, timeout=5, check=True,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return False

    def get_vram_usage_mb(self) -> dict:
        if not self._nvidia_smi_available:
            return {"used_mb": None, "total_mb": None}

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5, check=True,
            )
            match = re.search(r"(\d+)\s*,\s*(\d+)", result.stdout)
            if match:
                return {"used_mb": int(match.group(1)), "total_mb": int(match.group(2))}
        except (subprocess.SubprocessError, ValueError, OSError) as e:
            log.debug("Could not parse nvidia-smi output: %s", e)

        return {"used_mb": None, "total_mb": None}

    def check_budget(self) -> dict:
        usage = self.get_vram_usage_mb()
        if usage["used_mb"] is None:
            return {**usage, "over_budget": False, "budget_mb": self.max_vram_gb * 1024}

        budget_mb = self.max_vram_gb * 1024
        over_budget = usage["used_mb"] > budget_mb

        if over_budget:
            log.warning(
                "VRAM usage (%d MB) exceeds configured budget (%d MB)",
                usage["used_mb"], budget_mb,
            )

        return {**usage, "over_budget": over_budget, "budget_mb": budget_mb}
