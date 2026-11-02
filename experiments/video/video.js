/*
 * video.js — run MediaPipe Pose on a video, entirely in the browser.
 *
 * The clip is played very slowly and each displayed frame is grabbed with requestVideoFrameCallback,
 * which reports the frame's true timestamp. That matters: phone slow-mo drops frames, and speeds and
 * timings are only right if every frame keeps its real time. Nothing is uploaded anywhere.
 */
(function (root) {
  "use strict";
  // MediaPipe landmark indices we keep: nose, shoulders, elbows, wrists, hips, knees, ankles
  const KEEP = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

  async function track(video, opts) {
    opts = Object.assign({ base: "vendor/mediapipe/", rate: 0.0625, onProgress: () => {} }, opts || {});
    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) throw new Error("This browser can't read video frames one by one. Try Chrome, Edge or Safari.");
    const pose = new root.Pose({ locateFile: (f) => opts.base + f });
    pose.setOptions({ modelComplexity: 1, smoothLandmarks: false, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    let result = null;
    pose.onResults((r) => { result = r; });
    await pose.initialize();

    const canvas = document.createElement("canvas");
    const w = video.videoWidth, h = video.videoHeight, s = Math.min(1, 960 / Math.max(w, h));
    canvas.width = Math.round(w * s); canvas.height = Math.round(h * s);
    const ctx = canvas.getContext("2d");
    const frames = [];
    let lastT = -1, stopped = false;

    video.muted = true; video.loop = false;
    if (video.currentTime > 0) { video.currentTime = 0; }
    if (video.seeking) await new Promise((r) => video.addEventListener("seeked", r, { once: true }));
    if (video.readyState < 2) await new Promise((r) => video.addEventListener("canplay", r, { once: true }));
    video.playbackRate = opts.rate;

    // Pause on every new frame, track it, then resume: every frame gets processed however slow the device is.
    const done = new Promise((resolve, reject) => {
      const finish = () => { if (!stopped) { stopped = true; resolve(); } };
      const onFrame = async (now, meta) => {
        if (stopped) return;
        const t = meta.mediaTime;
        if (t > lastT + 1e-4) {
          video.pause(); lastT = t;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          try {
            result = null;
            await pose.send({ image: canvas });
            if (result && result.poseWorldLandmarks && result.poseLandmarks) {
              frames.push({
                t,
                wl: KEEP.map((i) => { const p = result.poseWorldLandmarks[i]; return [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4), +(p.visibility || 0).toFixed(2)]; }),
                lm: KEEP.map((i) => { const p = result.poseLandmarks[i]; return [+p.x.toFixed(4), +p.y.toFixed(4)]; }),
              });
            }
          } catch (e) { stopped = true; reject(e); return; }
          opts.onProgress(Math.min(1, t / video.duration), frames.length);
          if (video.ended || t >= video.duration - 0.002) return finish();
          video.requestVideoFrameCallback(onFrame);
          video.play().catch(reject);
          return;
        }
        if (!video.ended) video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
      video.addEventListener("ended", () => setTimeout(finish, 300), { once: true });
      video.play().catch(reject);
    });
    await done;
    video.pause(); video.playbackRate = 1;
    try { pose.close(); } catch (e) {}
    return { frames, width: w, height: h, keep: KEEP };
  }

  /** Convert stored frames (KEEP order) into the full-index layout features.js expects. */
  function toMediaPipeFrames(rec) {
    return rec.frames.map((f) => {
      const wl = new Array(33).fill(null);
      rec.keep.forEach((idx, k) => { wl[idx] = f.wl[k]; });
      return { t: f.t, wl };
    });
  }

  root.VIDEO = { track, toMediaPipeFrames, KEEP };
})(window);
