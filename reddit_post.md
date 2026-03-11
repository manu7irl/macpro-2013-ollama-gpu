# [Guide] Full GPU Acceleration for Ollama on Mac Pro 2013 (Dual FirePro D700) - Linux

Hey everyone! I finally managed to get full GPU acceleration working for **Ollama** on the legendary **Mac Pro 6.1 (2013 "Trashcan")** running Nobara Linux (and it should work on other distros too).

The problem with these machines is that they have dual **AMD FirePro D700s (Tahiti XT)**. By default, Linux uses the legacy `radeon` driver for these cards. While `radeon` works for display, it **does not support Vulkan or ROCm**, meaning Ollama defaults to the CPU, which is painfully slow.

### My Setup:
- **Model:** Mac Pro 6,1 (Late 2013)
- **CPU:** Xeon E5-1680 v2 (8C/16T @ 3.0 GHz)
- **RAM:** 32GB
- **GPU:** Dual AMD FirePro D700 (6GB each, 12GB total VRAM)
- **OS:** Nobara Linux (Fedora base, Kernel 6.18+)
- **Ollama:** 0.17.7 — **native install** (no Docker/Podman needed)

### The Solution:
Force the `amdgpu` driver for the Southern Islands (SI) architecture. Once `amdgpu` is active, Vulkan is enabled and Ollama picks up both GPUs automatically via `OLLAMA_VULKAN=1`.

### Actual Benchmarks (measured, not estimated):

**qwen3:8b** — the sweet spot for this hardware:
- Prompt eval: **~46 tok/sec**
- Generation: **~18 tok/sec**
- VRAM: ~5.9 GB split across both D700s
- 100% GPU offload

**qwen2.5-coder:14b** — for heavier code tasks:
- Prompt eval: **~43 tok/sec**
- Generation: **~11.5 tok/sec**
- VRAM: ~8.3 GB split across both D700s
- 100% GPU offload (49/49 layers)

On CPU alone these models run at <2 tok/sec. This fix makes the Trashcan a genuinely useful local LLM workstation in 2026.

**One gotcha:** qwen3.5:9b technically fits in VRAM but causes a GPU hang on Tahiti (`amdgpu: Fence fallback timer expired on ring gfx`) — the new qwen35 GGUF architecture uses tensor ops that GCN 1.0 can't handle via Vulkan. Stick with qwen3:8b or qwen2.5 models.

**What about image generation?** I tested Z-Image-Turbo (the #1 open-source text-to-image model right now) via two paths:
- **Ollama** (`ollama pull x/z-image-turbo`): Fails on Linux/AMD — the Ollama image runner requires `libcuda.so.1` (CUDA-only for now).
- **stable-diffusion.cpp with Vulkan:** Also crashes — same `vk::DeviceLostError` as qwen3.5 during diffusion sampling. GCN 1.0 doesn't have native fp16/bf16 matrix ops so the Vulkan shaders abort.
- **stable-diffusion.cpp CPU-only:** Works! Generates beautiful images. But ~43s/step on the Xeon = ~30 min per 512×512 at 20 steps. Fine for overnight batches, not for interactive use.

**Music generation (ACE-Step 1.5):** Text-to-music AI (prompt → full song). Installs and runs via Python 3.12 CPU-only venv. ROCm 6.x doesn't support Tahiti so no GPU acceleration. Functional, just slow.

So for LLM inference the Trashcan is genuinely great in 2026. For image/audio gen, Tahiti's GCN 1.0 Vulkan hits a wall — the diffusion ops need newer GPU features. LLMs are the sweet spot.

### How to do it:

**1. Update Kernel Parameters**

Add to `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="... radeon.si_support=0 amdgpu.si_support=1"
```

Or use the automated script in the repo:
```bash
sudo bash setup-gpu.sh
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
sudo reboot
```

**2. Verify the driver switched**
```bash
lspci -k | grep -A 3 -E "(VGA|3D)"
# Should show: Kernel driver in use: amdgpu
```

**3. Install Ollama natively**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**4. Add OLLAMA_VULKAN=1 to the systemd service**

Edit `/etc/systemd/system/ollama.service`:
```ini
Environment="OLLAMA_VULKAN=1"
```

```bash
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**5. Pull and run**
```bash
ollama pull qwen3:8b
ollama run qwen3:8b "hello"
ollama ps  # should show 100% GPU
```

Full repo with setup script, TUI, Podman Quadlet and Docker Compose configs:
https://github.com/manu7irl/macpro-2013-ollama-gpu

Hope this helps any fellow Trashcan owners out there!

#MacPro #Linux #Ollama #SelfHosted #AMD #FireProD700 #LocalLLM
