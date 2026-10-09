# Sixthsense

**Sixthsense enables robots to sense external forces and torques across their entire body and across tasks using proprioception alone.** This anonymous review package includes an ONNX model, recorded robot data, a browser replay viewer, and MuJoCo simulation for standing, walking, lying, and TWIST2 motion tracking.

## 1. Installation

Install Python 3.13, then run from the project directory:

```sh
python -m pip install -r requirements.txt
```

## 2. Recorded Robot Replay

This supplementary package includes a compact dataset subset with 43 recordings across standing, walking, and lying, including HDF5 signals and paired camera videos. Personal source paths are removed from metadata; numerical signals and video files are preserved. The complete dataset is distributed separately during review; author-linked external URLs are withheld.

### 2.1 Standing

```sh
python sixthsense.py real stand
```

![Standing robot replay](assets/demos/sixthsense_real_stand.gif)


### 2.2 Walking

```sh
python sixthsense.py real walk
```

![Walking robot replay](assets/demos/sixthsense_real_walk.gif)


### 2.3 Lying

```sh
python sixthsense.py real lie
```

![Lying robot replay](assets/demos/sixthsense_real_lie.gif)


## 3. Simulation

The GIFs show selected simulation rollouts using the native MuJoCo UI. Red curves show true force and torque; green curves show estimates aligned to the same target timestamps.

### 3.1 Standing

```sh
python sixthsense.py sim stand
```

![Standing online simulation with true and estimated wrenches](assets/demos/sixthsense_sim_stand.gif)

### 3.2 Walking

```sh
python sixthsense.py sim walk
```

![Walking online simulation with true and estimated wrenches](assets/demos/sixthsense_sim_walk.gif)

### 3.3 Lying

```sh
python sixthsense.py sim lie
```

![Lying online simulation with true and estimated wrenches](assets/demos/sixthsense_sim_lie.gif)

### 3.4 TWIST2 Motion Tracking

```sh
python sixthsense.py sim tracking --motion 1 --single-motion
```

![TWIST2 online motion tracking with true and estimated wrenches](assets/demos/sixthsense_sim_tracking.gif)


## 4. Network Architecture

The diagram summarizes the main generation and readout path. The model jointly generates contact gate scores and a **6D wrench** `[Fx, Fy, Fz, Tx, Ty, Tz]` for each of 30 body links at every frame.

```mermaid
%%{init: {"flowchart": {"curve": "stepBefore"}}}%%
flowchart TD
    X["Sensor Observations X"] -->|"X [B, 50, 96]"| C["Sensor Feature Conditioning"]

    Z["Gaussian Noise Z"] -->|"Z_joint [B, 50, 210]"| J["Joint Contact–6D Wrench Generation"]
    C -->|"Conditioning features"| J

    J -->|"g [B, 50, 30]"| G["Link Gate Scores g"]
    J -->|"W [B, 50, 6, 30]"| W["6D Wrenches: Force + Torque"]
    G -->|"g"| R["Per-Link Contact Readout and Wrench Gating"]
    W -->|"W"| R

    R -->|"A [B, 50, 30]"| A["Per-Link Contact Mask"]
    R -->|"W_out [B, 50, 6, 30]"| O["Gated 6D Wrenches for All 50 Frames"]
```

`B` is the batch size; `50` is the number of frames; `30` is the number of target links; `6` denotes three force and three torque components.

| Variable | Meaning |
|---|---|
| `X` | Robot signals: 9 IMU features and position, velocity and motor torque for each of 29 joints. |
| `Z_joint` | Random noise: 7 channels per link, 210 in total. |
| `C` | Features extracted from the robot signals. |
| `g`, `W` | Contact gate scores and 6D wrenches for each link. |
| `A`, `W_out` | Contact mask and final wrenches: `W_out = A * W`. Inactive links have zero force and torque. |

The ONNX outputs `raw_wrench` (`W`) and `wrench` (`W_out`) both have shape `[B, 50, 6, 30]`: 50 frames, 6 components and 30 links per sample. Components are `[Fx, Fy, Fz, Tx, Ty, Tz]`, with force in N and torque in N·m. Both use pelvis axes; torque is measured about each link's body origin.
