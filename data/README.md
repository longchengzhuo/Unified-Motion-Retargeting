# data/

Source motion. The bundled clip is `walk_slow.bvh`: Xsens mocap, 23 joints,
17099 frames at 240 Hz, centimetres, Y-up — about 71 s of slow walking.

## Using a different BVH

The parser is [`umr/bodies/bvh.py`](../umr/bodies/bvh.py) (numpy + scipy only). It assumes:

1. **Channel layout** — root has 6 channels (3 translation + 3 rotation), every other joint has 3.
   The Euler order is read from the `CHANNELS` line and interpreted as intrinsic
   (`Yrotation Xrotation Zrotation` → $R_y R_x R_z$).
2. **Units and orientation** are configurable. `source.length_scale` handles centimetres (`0.01`);
   the default basis (X=left, Y=up, Z=forward) is converted to MuJoCo's (X=forward, Y=left, Z=up).
   `source.auto_face_x: true` estimates and corrects the yaw from the frame-0 ankle→toe vector.
3. **Joint names** index the segment geometry table `SEGMENT_SPECS` in
   [`umr/bodies/human_mjcf.py`](../umr/bodies/human_mjcf.py), which currently covers the Xsens
   23-joint naming (`Hips`, `Chest`..`Chest4`, `Neck`, `Head`, `LeftCollar`, `LeftShoulder`,
   `LeftElbow`, `LeftWrist`, `LeftHip`, `LeftKnee`, `LeftAnkle`, `LeftToe`, and the right-side
   equivalents). A different skeleton means editing that table plus `SEGMENT_ADJACENCY` and the two
   weight tables.

Then point `source.bvh` in your config at the new file.

Xsens exports usually start with a calibration frame where every channel is zero.
`first_motion_frame()` skips it during retargeting but keeps it as the canonical T-pose for Stage I.

Sanity check after swapping the file:

```bash
python -c "
from umr.bodies import bvh
d = bvh.load_bvh('data/<your-file>.bvh')
print(d.num_frames, 'frames @', round(d.fps, 2), 'FPS,', d.num_joints, 'joints')
print('actor height %.3f m' % bvh.actor_height(d))
print('first motion frame:', bvh.first_motion_frame(d))
"
```

The estimated height should land in 1.5–2.0 m; if it does not, `length_scale` or the axis
convention is wrong.
