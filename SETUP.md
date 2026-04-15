# Setup Guide

## Requirements

- **Python 3.8+** (tested on 3.10/3.11)
- **Windows OS** (pydirectinput only works on Windows)
- **Webcam** (built-in or USB)

## Installation

1. Open a terminal in the project directory.

2. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Camera Setup

- Position your webcam so your **full upper body and legs** are visible.
- Stand about **1.5-2 meters** (5-7 feet) from the camera.
- Ensure good lighting — avoid strong backlighting (e.g., windows behind you).
- The camera feed is mirrored, so moving left on screen = moving left in-game.

## Verify Installation

Run a quick test:
```
python controller.py
```

You should see:
- A webcam window with two yellow vertical lines dividing the view into 3 sections.
- A 3-second calibration countdown.
- Your pose skeleton drawn in green/blue once calibration finishes.

Press `q` to quit.
