# Experiment: elbow torque from a phone video

**Question.** Can a phone video of a throw measure the mechanics that drive elbow torque well enough to improve
on velocity + body size alone?

**Answer: not yet.** Even in a best case, video mechanics barely help.

| Inputs (397 lab pitches, validated on unseen pitchers) | R² | typical error |
|---|---|---|
| Velocity + size (no video) | 0.37 | 12.1 Nm |
| + mechanics, perfect lab tracking | 0.49 | 11.0 Nm |
| + mechanics, 240 fps phone video (simulated) | 0.40 | 12.0 Nm |
| + mechanics, 60 fps phone video (simulated) | 0.38 | 12.0 Nm |
| 240 fps video, no radar gun | 0.28 | 13.2 Nm |

Velocity from video alone: R² 0.14 (±3.4 mph). The key mechanic, layback, is the forearm's rotation: small, fast and
hard for single-camera pose tracking. On a real 240 fps iPhone clip (Driveline's, `media/sample_throw.mp4`), the in-browser
tracker followed the arm well in 2D, but automatic release detection picked the follow-through.

## Files

| | |
|---|---|
| `video_sim.py` | turns each lab pitch into what a phone + pose tracker would see (jitter, depth error, dropped frames) |
| `video_features.js` | runs the browser feature extractor (`features.js`) on those simulated clips |
| `video_model.py` | the accuracy table above; writes `video_model.json` |
| `video.js` | runs MediaPipe Pose on a video in the browser, every frame at its true timestamp |
| `features.js` | events (foot plant, max layback, release) and mechanics from a 3D skeleton |
| `media/sample_throw.mp4` | Driveline OpenBiomechanics iPhone clip, CC BY-NC-SA 4.0 |

`vendor/mediapipe/` holds the files from the npm package `@mediapipe/pose@0.5.1675469404` (Apache-2.0); it isn't in the
zip because of its size: `npm pack @mediapipe/pose@0.5.1675469404` and copy the package's files there.

Run (after `pipeline/run_all.sh`): `python3 video_sim.py && node video_features.js && python3 video_model.py`
