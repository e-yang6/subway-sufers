# Usage Guide

## Quick Start

1. Open Subway Surfers (browser or desktop).
2. Run the controller:
   ```
   python controller.py
   ```
3. Stand still during the 3-second calibration.
4. Start playing — use your body to control the game!

## Gesture Reference

| Gesture | Game Action | How to Perform |
|---------|------------|----------------|
| Move body left | Switch to left lane | Shift your hips into the left third of the camera view |
| Move body right | Switch to right lane | Shift your hips into the right third of the camera view |
| Return to center | Return to center lane | Move hips back to the middle third |
| Jump | Jump over obstacles | Jump so your ankles rise above the calibrated baseline |
| Squat / Duck | Roll under obstacles | Bend your knees so your torso lowers significantly |
| Clap hands | Activate hoverboard | Bring both wrists close together in front of your body |
| Stop moving | Pause game | Stand completely still for ~1.5 seconds |
| Resume moving | Unpause game | Start moving your legs again |

## Calibration

During the 3-second calibration at startup:
- **Stand upright** and face the camera.
- **Keep your arms at your sides.**
- **Stay still** — the system captures your baseline ankle and hip positions.

These baselines are used to detect jumps and squats. If detection feels off, restart the app to recalibrate.

## Tips for Best Results

- **Lighting**: Bright, even lighting works best. Avoid shadows on your body.
- **Camera angle**: Straight-on at roughly chest/waist height.
- **Distance**: Stand far enough that your full body (at least hips to head) is visible.
- **Clothing**: Avoid clothing that matches your background color.
- **Background**: A plain, uncluttered background improves pose detection accuracy.
- **Movement**: Make gestures deliberate — exaggerated movements are detected more reliably.

## On-Screen Display

- **Yellow lines**: Divide the view into left / center / right lanes.
- **Green highlight**: Shows which lane you're currently in.
- **Skeleton overlay**: Your detected pose drawn on the camera feed.
- **Status bar** (bottom): Shows current lane, running status, and last detected action.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Cannot open webcam" | Check that no other app is using the camera. Try unplugging and reconnecting a USB webcam. |
| Pose not detected | Ensure your body is visible in the frame. Improve lighting. |
| Jumps not registering | Jump higher, or restart to recalibrate. Make sure ankles are visible. |
| Squats not registering | Squat deeper. Ensure hips and shoulders are both visible. |
| Lane changes too sensitive | Move more slowly. Stay centered when not intending to switch. |
| Keys not working in game | Make sure the game window is focused/active. Run the script as administrator if needed. |
| Hoverboard triggers randomly | The clap threshold may need tuning — edit `CLAP_DISTANCE_THRESHOLD` in `controller.py`. |
| Game pauses unexpectedly | You may have been too still. Keep your legs moving slightly. Adjust `RUNNING_VARIANCE_THRESHOLD` if needed. |

## Tuning Thresholds

All thresholds are defined as constants at the top of `controller.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `JUMP_THRESHOLD` | 0.05 | How high ankles must rise above baseline to trigger a jump |
| `JUMP_COOLDOWN` | 1.0s | Minimum time between jump triggers |
| `SQUAT_RATIO_THRESHOLD` | 0.7 | Hip-shoulder distance ratio that triggers a squat |
| `SQUAT_COOLDOWN` | 1.0s | Minimum time between squat triggers |
| `CLAP_DISTANCE_THRESHOLD` | 0.05 | Max wrist distance to register a clap |
| `CLAP_COOLDOWN` | 2.0s | Minimum time between hoverboard triggers |
| `RUNNING_VARIANCE_THRESHOLD` | 0.0003 | Ankle variance below this = stopped |
