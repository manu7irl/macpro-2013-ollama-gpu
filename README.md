# Mac Pro 2013 (Trashcan) Ollama GPU Acceleration Fix

This repository provides the configuration and scripts to enable full GPU acceleration for **Ollama** on a **Mac Pro 6.1 (2013)** running **Nobara Linux** (or any Fedora/RHEL-based distro).

The Mac Pro 2013 features dual **AMD FirePro D700 (Tahiti XT)** GPUs. By default, Linux uses the `radeon` driver for these cards, which does not support Vulkan or ROCm, leaving Ollama to run on the CPU only.

This fix forces the `amdgpu` driver, which enables Vulkan support and allows Ollama to utilize both GPUs — whether you run Ollama **natively** or **in a container**.

## 💻 Hardware Profile (The "Trashcan")

- **CPU:** Intel Xeon E5-1680 v2 (8 Cores / 16 Threads @ 3.0 GHz)
- **RAM:** 32 GB DDR3 ECC
- **GPUs:** 2x AMD FirePro D700 (6GB GDDR5 VRAM each, total 12GB VRAM)
- **Architecture:** Tahiti XT (GCN 1.0 / Southern Islands)
- **OS:** Nobara Linux (Optimized Fedora-based distro)
- **Ollama Version:** 0.17.7 (native install)

---

## 📊 Benchmarks & Performance

All benchmarks run with Ollama installed natively, `OLLAMA_VULKAN=1`, dual FirePro D700s active via Vulkan.

### qwen3:8b — Recommended Sweet Spot (5.2 GB)

| Metric | Result |
|--------|--------|
| **GPU Offloading** | 100% GPU |
| **VRAM Usage** | ~5.9 GB (split across both D700s) |
| **Prompt eval** | **~46 tok/sec** |
| **Generation** | **~18 tok/sec** |
| **Load time (cold)** | ~6s |

### qwen2.5-coder:14b (9.0 GB)

| Metric | Result |
|--------|--------|
| **GPU Offloading** | 49/49 layers (100% GPU) |
| **VRAM Usage** | ~4.1 GB per GPU |
| **Prompt eval** | **~43 tok/sec** |
| **Generation** | **~11.5 tok/sec** |
| **Load time (cold)** | ~14s |

*On CPU alone these models run at <2 tok/sec. This fix makes the Trashcan a genuinely useful local LLM workstation in 2026.*

### Compatibility Notes

| Tool | GPU | Notes |
|------|-----|-------|
| **qwen3:8b** | ✅ 100% GPU | Best all-round model for this hardware |
| **qwen2.5-coder:14b** | ✅ 100% GPU | Great for code tasks |
| **qwen3.5:9b** | ❌ GPU hang | Fits in VRAM but `amdgpu: Fence fallback timer expired` on Tahiti — new `qwen35` GGUF tensor ops crash GCN 1.0. Runs CPU-only at ~2–3 tok/sec |
| **x/z-image-turbo** (Ollama) | ❌ Linux/AMD | Ollama's image gen uses an MLX runner requiring `libcuda.so.1` — CUDA-only on Linux. |
| **Z-Image-Turbo** (sd.cpp Vulkan) | ❌ GPU crash | `vk::DeviceLostError` during diffusion sampling — GCN 1.0 lacks fp16/bf16 matrix ops. |
| **Z-Image-Turbo** (sd.cpp CPU) | ✅ Works | ~43s/step on Xeon → ~30 min per 512×512 image. Beautiful output, just slow. |
| **ACE-Step 1.5** (music gen) | ❌ GPU | Tahiti not supported by ROCm 6.x. Installs and runs CPU-only via Python 3.12 venv. Slow but functional. |

**Vulkan is the only GPU compute path on this hardware.** ROCm, CUDA, and MLX all require newer architectures or NVIDIA/Apple Silicon.

### What is ACE-Step 1.5?

ACE-Step 1.5 is an open-source **music generation AI** (ACE Studio + Stepfun, 2025). Give it a text prompt and it generates a full song up to 4 minutes long. Features include voice cloning, lyric editing, remixing, and support for 50+ languages.

**Why not Ollama?** ACE-Step is a diffusion model pipeline (DiT + audio VAE + text encoder), not a text LLM. Ollama only runs GGUF text models — it has no concept of audio output or multi-step diffusion. They're fundamentally different tools.

**CPU install on this machine (Python 3.12 required):**
```bash
git clone https://github.com/ace-step/ACE-Step-1.5 ace-step-1.5
cd ace-step-1.5
python3.12 -m venv venv_cpu312 && source venv_cpu312/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy transformers diffusers soundfile scipy einops accelerate \
  loguru gradio fastapi uvicorn toml huggingface_hub vector-quantize-pytorch \
  safetensors numba matplotlib
pip install -e . --no-deps
# Patch for meta-tensor crash on CPU (transformers + vector-quantize-pytorch incompatibility):
python3.12 -c "
import site, pathlib
vqp = pathlib.Path(site.getsitepackages()[0]) / 'vector_quantize_pytorch'
# Fix 1: residual_fsq.py
rf = vqp / 'residual_fsq.py'; t = rf.read_text()
rf.write_text(t.replace('assert (levels_tensor > 1).all()', 'if not levels_tensor.is_meta:\n            assert (levels_tensor > 1).all()'))
# Fix 2: finite_scalar_quantization.py
fsq = vqp / 'finite_scalar_quantization.py'; t = fsq.read_text()
import re
t = re.sub(r'if return_indices:\n            self\.codebook_size = self\._levels\.prod\(\)\.item\(\)\n            implicit_codebook = self\._indices_to_codes\(torch\.arange\(self\.codebook_size\)\)\n            self\.register_buffer',
'if return_indices:\n            import math; self.codebook_size = math.prod(levels)\n            if not self._levels.is_meta:\n                implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))\n                self.register_buffer', t)
fsq.write_text(t)
print('Patches applied.')
"
# Start server (downloads ~10 GB of model weights on first run)
ACESTEP_LM_BACKEND=pt ACESTEP_INIT_LLM=false \
  python3.12 acestep/api_server.py --host 127.0.0.1 --port 8002
```

**Generate a song (API):**
```bash
# Submit
curl -X POST http://127.0.0.1:8002/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "upbeat electronic synthwave, driving beat, retro 80s vibes",
    "lyrics": "[verse]\nYour verse here\n[chorus]\nYour chorus here",
    "vocal_language": "en",
    "audio_duration": 20,
    "inference_steps": 8,
    "guidance_scale": 7.0,
    "seed": 42
  }'
# Poll (use task_id from above response)
curl -X POST http://127.0.0.1:8002/query_result \
  -H "Content-Type: application/json" \
  -d '{"task_id": "TASK_ID_HERE"}'
# Output MP3 saved to: .cache/acestep/tmp/api_audio/
```

**Performance:** ~4 min per 20-second song on the Xeon E5-1680 v2 at 8 steps. CPU-only — ROCm 6.x doesn't support Tahiti (GCN 1.0).

### What is Z-Image-Turbo?

Z-Image-Turbo is a **6B parameter text-to-image model** by Alibaba Tongyi MAI, ranked #1 open-source on the Arena image leaderboard (Feb 2026). It outperforms FLUX.1 [dev] in photorealism.

**Status on this hardware (tested):**
- **Ollama** (`ollama pull x/z-image-turbo`): ❌ Uses an MLX runner that requires `libcuda.so.1` on Linux — CUDA-only, AMD not supported.
- **stable-diffusion.cpp (Vulkan):** ❌ Crashes with `vk::DeviceLostError` during diffusion sampling — GCN 1.0 lacks the fp16/bf16 matrix ops the diffusion model needs.
- **stable-diffusion.cpp (CPU):** ✅ Works. Generates high-quality images. ~43s/step on the Xeon → ~30 min for a 512×512 image at 20 steps. Functional but slow.

**CPU install via stable-diffusion.cpp:**
```bash
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp && git submodule update --init --recursive
mkdir build-cpu && cd build-cpu
cmake .. -DSD_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)

# Download models (~6 GB total)
cd ../models
# Diffusion model (3.6 GB):
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('leejet/Z-Image-Turbo-GGUF', 'z_image_turbo-Q4_K.gguf', local_dir='.')"
# VAE (320 MB):
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('Comfy-Org/z_image_turbo', 'split_files/vae/ae.safetensors', local_dir='.')"
# LLM text encoder (2.4 GB):
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('unsloth/Qwen3-4B-Instruct-2507-GGUF', 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf', local_dir='.')"

# Generate an image
cd ..
./build-cpu/bin/sd-cli \
  --diffusion-model models/z_image_turbo-Q4_K.gguf \
  --vae models/split_files/vae/ae.safetensors \
  --llm models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  -p "a futuristic city at night, neon lights, cyberpunk" \
  --cfg-scale 1.0 --steps 20 -H 512 -W 512 -o output.png
```

---

## 🚀 The Fix at a Glance

1. **Kernel Parameters:** Disable `radeon` SI support and enable `amdgpu` SI support.
2. **GRUB Update:** Persist these changes in the bootloader.
3. **Ollama Config:** Set `OLLAMA_VULKAN=1` — either in the systemd service (native) or as a container environment variable.

---

## 🛠️ Step-by-Step: Native Install (Recommended)

### 1. Update Kernel Parameters

Add `radeon.si_support=0 amdgpu.si_support=1` to your kernel command line.

**Edit `/etc/default/grub`** — find `GRUB_CMDLINE_LINUX_DEFAULT` and append:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="... radeon.si_support=0 amdgpu.si_support=1"
```

Or run the automated script:
```bash
sudo bash setup-gpu.sh
```

**Regenerate GRUB:**
```bash
sudo grub2-mkconfig -o /boot/grub2/grub.cfg
```

### 2. Reboot
```bash
sudo reboot
```

### 3. Verify Driver
```bash
lspci -k | grep -A 3 -E "(VGA|3D)"
# Should show: Kernel driver in use: amdgpu
```

### 4. Install Ollama Natively

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 5. Configure the Systemd Service

Edit `/etc/systemd/system/ollama.service` to add `OLLAMA_VULKAN=1`:

```ini
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="OLLAMA_VULKAN=1"

[Install]
WantedBy=default.target
```

Apply and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ollama
```

### 6. Verify GPU is Active
```bash
ollama run qwen3:8b "hello"
# Check logs:
journalctl -u ollama -f
# Look for: Vulkan0, Vulkan1 in the output
```

---

## 🐳 Alternative: Container Deploy (Docker / Podman)

If you prefer a containerized setup, use the provided configs.

### Podman Quadlet

Copy `ollama.container` to `/etc/containers/systemd/`:
```bash
sudo cp ollama.container /etc/containers/systemd/
sudo systemctl daemon-reload
sudo systemctl start ollama
```

### Docker Compose
```bash
docker compose up -d
```

**Key requirements for containers:**
- Mount `/dev/dri` to the container.
- Set `OLLAMA_VULKAN=1` environment variable.
- Disable security labels (`SecurityLabel=disable` in Quadlet / `--security-opt label=disable` in Docker).

---

## 📂 Repository Structure

| File | Purpose |
|------|---------|
| `setup-gpu.sh` | Automated script to apply kernel parameter changes |
| `tui.sh` | Guided Terminal UI for the setup |
| `ollama.container` | Podman Quadlet config (containerized) |
| `docker-compose.yml` | Docker Compose config (containerized) |

---

## 🤝 Community

Inspired by the Mac Pro 2013 enthusiast community. If you found this helpful, share it on Reddit or GitHub!

---

*Tested on Nobara Linux 43 with Kernel 6.18+, Ollama 0.17.0.*
