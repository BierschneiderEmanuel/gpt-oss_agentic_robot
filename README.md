# gpt-oss_agentic_robot

Agentic robotic assistant that combines gpt-oss:20b, MediaPipe FaceMesh, YOLOv5 detection, a local image-LLM toolchain and a MyCobot controller. The main implementation lives in the repository root (see `gpt-oss_agentic_robot_BACKUP_v1_0_0.py`).

## Features
- gpt-oss:20b an agentic Large Language Model designed to function as a cognitive core.
- Real‑time face landmark detection and simple depth estimation (MediaPipe FaceMesh).
- Image cleanliness check via a local image LLM endpoint.
- YOLOv5 object detection (ONNX/pt support) and conversion to robot coordinate space.
- myCobot280 TCP interface for motion and gripper control.
- Simple voice I/O (pyttsx3 + speech_recognition) wired to a multi-tool LLM agent.

## Prerequisites
- Linux host.
- Python 3.9.23 (the repo expects this runtime for reproducible dependency installation).
- Camera device (V4L2) and microphone available to the process.
- MyCobot280 (or compatible) robot configured and reachable on the network (IP/port in code).
- Local image LLM endpoint available at `http://localhost:11434/api/generate` (used by DetectCleaniness).

## Install
1. Create and activate a Python 3.9.23 virtual environment:
```bash
python3.9 -m venv .venv
source .venv/bin/activate
# confirm Python is 3.9.23
python --version
```

2. Install requirements using the provided installer script (uses `requirements.txt`):
```bash
python3.9 install_script.py requirements.txt
```

Note: `install_script.py` prints commands and runs pip installs for each entry in `requirements.txt`.

3. Install VLM
```bash
ollama run bakllava
```

4. Install gpt-oss 20b
```bash
ollama run gpt-oss:20b
```

## MyCobot configuration
For MyCobot setup and the socket interface used by this project, refer to:
https://github.com/BierschneiderEmanuel/MyCobot280BSDSocketInterface

Follow that repository to configure your MyCobot firmware/network interface and confirm the robot server is reachable on the IP/port used in this project (default IP in code: `192.168.178.31`, port `9000`).

## Quickstart / Run
1. Configure your robots ip address (default IP in code: `192.168.178.31`, port `9000`)
2. Ensure the robot server is up and the camera + mic are attached.
3. Start the main script:
```bash
python3.9 gpt-oss_agentic_robot_BACKUP_v1_0_0.py
```
If you see ALSA/JACK noise on stderr during audio init and want to hide it, run:
```bash
python3.9 gpt-oss_agentic_robot_BACKUP_v1_0_0.py 2>/dev/null
```

## Basic testing instructions
- Position yourself in front of the camera so the face is visible in the preview window.
- When the script detects a person it speaks a greeting. Then it will listen for a voice prompt.
- Example voice prompts to test functionality:
  - "Who is Sam Altman?"
  - "How is the weather?" (this will exercise web-search tool if available)
  - "Who is Emanuel Bierschneider?"
- Expected behavior:
  - The agent will call local tools (Wikipedia/DuckDuckGo/DetectCleaniness) and speak back the answer using TTS.
  - If the DetectCleaniness tool returns "it is messy" (or similar), YOLO detection is triggered and the robot may start a cleanup sequence (if configured and safe).

## Troubleshooting
- No camera: adjust `cap_num` in the main file (V4L2 index).
- Microphone selection: the code attempts to auto-select a device with "Life" in its name — change the selection logic if needed.
- MyCobot connection fails: verify IP/port and consult the MyCobot280 socket interface repo above.
- Image LLM endpoint not responding: ensure your local model server is running at `http://localhost:11434/api/generate`.

## Notes & safety
- The code sends live robot commands. Test with the robot power off or in a safe configuration before enabling motion on a real robot.
- Use the provided `install_script.py` with Python 3.9.23 for installing dependencies to reduce compatibility issues.

## Files of interest
- `gpt-oss_agentic_robot_BACKUP_v1_0_0.py` — main runtime and agent logic
- `install_script.py` — installs packages listed in `requirements.txt`
- `requirements.txt` — Python dependencies
- `yolov5s.onnx` / `yolov5s.pt` — model files (if present)

License: MIT (see LICENSE file)