# Coordinate System Reference

This document describes the coordinate system conventions and transformation matrices used in NeoVerse.

## Overview

NeoVerse uses the **OpenCV coordinate convention** for all camera poses and trajectories. This reference covers axis definitions, transformation matrices for each camera operation, trajectory modes, and conversions from other coordinate systems.

## Coordinate System

### Axes

```
          +Z (Forward)
          /
         /
        /_________ +X (Right)
        |
        |
        |
        +Y (Down)
```

| Axis | Direction |
|------|-----------|
| **X** | Right |
| **Y** | Down |
| **Z** | Forward |

### World Coordinate System

The world coordinate system is defined as the camera coordinate system of the first frame.

## Camera Operations

All matrices below use the `angle` parameter (in degrees, converted to radians internally as `theta`). Translation operations use the `distance` parameter `d`. Orbit operations use both `angle` and `orbit_radius` (`r`).

### Pan (Rotation around Y-axis)

**pan_right** — Rotate camera viewing direction to the right (`theta` > 0):

```
[cos(theta)   0    sin(theta)   0]
[0            1    0            0]
[-sin(theta)  0    cos(theta)   0]
[0            0    0            1]
```

**pan_left** — Rotate camera viewing direction to the left (`theta` < 0):

```
[cos(theta)   0    sin(theta)   0]
[0            1    0            0]
[-sin(theta)  0    cos(theta)   0]
[0            0    0            1]
```

### Tilt (Rotation around X-axis)

**tilt_up** — Tilt camera viewing direction upward (`theta` > 0):

```
[1   0             0            0]
[0   cos(theta)   -sin(theta)   0]
[0   sin(theta)    cos(theta)   0]
[0   0             0            1]
```

**tilt_down** — Tilt camera viewing direction downward (`theta` < 0):

```
[1   0             0            0]
[0   cos(theta)   -sin(theta)   0]
[0   sin(theta)    cos(theta)   0]
[0   0             0            1]
```

### Translation

**move_left** — Translate along -X:

```
[1   0   0   -d]
[0   1   0    0]
[0   0   1    0]
[0   0   0    1]
```

**move_right** — Translate along +X:

```
[1   0   0   d]
[0   1   0   0]
[0   0   1   0]
[0   0   0   1]
```

**boom_up** — Translate along -Y (upward):

```
[1   0   0    0]
[0   1   0   -d]
[0   0   1    0]
[0   0   0    1]
```

**boom_down** — Translate along +Y (downward):

```
[1   0   0   0]
[0   1   0   d]
[0   0   1   0]
[0   0   0   1]
```

**push_in** — Translate along +Z (forward):

```
[1   0   0   0]
[0   1   0   0]
[0   0   1   d]
[0   0   0   1]
```

**pull_out** — Translate along -Z (backward):

```
[1   0   0    0]
[0   1   0    0]
[0   0   1   -d]
[0   0   0    1]
```

### Orbit (Rotation + Arc Translation)

Orbit operations combine a Y-axis rotation with an arc translation so the camera moves along a circular path while keeping the subject in view.

**orbit_right** — Arc clockwise in XZ plane (angle `theta` > 0, radius `r`):

```
[cos(theta)    0   -sin(theta)    r*sin(theta)     ]
[0             1    0              0               ]
[sin(theta)    0    cos(theta)     r*(1-cos(theta))]
[0             0    0              1               ]
```

**orbit_left** — Arc clockwise in XZ plane (angle `theta` < 0, radius `r`):

```
[cos(theta)    0   -sin(theta)    r*sin(theta)     ]
[0             1    0              0               ]
[sin(theta)    0    cos(theta)     r*(1-cos(theta))]
[0             0    0              1               ]
```

The rotation component faces the camera toward the orbit center, while the translation component places the camera on the arc.

## Trajectory Modes

### Relative Mode (Default)

Trajectory transformations are applied relative to the original camera motion extracted from the input video:

```
T_target = T_original * T_trajectory
```

The original camera motion is preserved and the trajectory is layered on top. A `pan_left` in relative mode will pan left while maintaining any existing forward motion.

### Global Mode

Trajectory transformations are used directly as camera-to-world poses:

```
T_target = T_trajectory
```

The original camera motion is ignored. Use this when you need absolute positioning or want to define a completely new camera path.

## Composing Transformations

When multiple operations appear in a single keyframe, they are composed via matrix multiplication in list order:

```
T_final = T_op1 * T_op2 * T_op3 * ...
```

Matrix multiplication is **not commutative** — order matters.

**Example:**

```json
{"1.0": [
  {"pan_left": {"angle": 30}},
  {"push_in": {"distance": 0.1}}
]}
```

This first rotates left, **then** translates forward in the rotated frame (curved path). Reversing the order would translate forward first, then rotate (straight path followed by rotation in place).

Additionally, operations at each keyframe accumulate onto the running pose from previous keyframes. The accumulated poses are then interpolated across the full frame range.

---

For more information, see:
- [trajectory_format.md](trajectory_format.md) — Trajectory file format specification
- [examples/trajectories/](../examples/trajectories/) — Example trajectory files
