"""
physics.py — rigid-body mechanics of the throwing arm, from joint-center trajectories.

Everything here is computed from scratch using only 3D joint-center positions
(shoulder, elbow, wrist, hand) sampled at 360 Hz, body mass, and textbook
anthropometrics. No lab-computed forces or moments are used as inputs; those
are only used afterwards to validate the result.

Main pieces
-----------
1. Segment inertial parameters (de Leva 1996, adjusted Zatsiorsky–Seluyanov values).
2. Segment reference frames built from joint centers.
3. Angular velocity from rotating frames:  w = 1/2 * sum_i (e_i x de_i/dt).
4. Newton–Euler inverse dynamics of the distal arm (forearm + hand + ball) about
   the elbow joint center:
       F_elbow = sum m_i (a_i - g)
       M_elbow = sum (r_i x m_i (a_i - g)) + I*alpha + w x (I*w)
   then projected onto the elbow's varus/valgus axis. The internal varus moment
   is the torque the elbow structures (UCL + flexor-pronator muscles + bone)
   must supply to stop the forearm being flung into valgus during late cocking.
5. Segment kinetic energy and elbow joint power.

Coordinate system (lab): +x toward home plate, +y toward first base, +z up.
Left-handers are mirrored (y -> -y) so every pitcher is analysed as a righty.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

G = np.array([0.0, 0.0, -9.81])  # gravity, m/s^2
BALL_MASS = 0.145                 # kg (MLB regulation 5–5.25 oz)
FS = 360.0                        # Hz, OpenBiomechanics capture rate

# de Leva (1996) male segment parameters.
#   mass: fraction of body mass
#   com : COM location as fraction of segment length from the proximal joint
#   rg  : radii of gyration as fraction of segment length (sagittal, transverse, longitudinal)
DE_LEVA = {
    "upper_arm": dict(mass=0.0271, com=0.5772, rg=(0.285, 0.269, 0.158)),
    "forearm":   dict(mass=0.0162, com=0.4574, rg=(0.276, 0.265, 0.121)),
    "hand":      dict(mass=0.0061, com=0.7900, rg=(0.628, 0.513, 0.401)),
}


def lowpass(x: np.ndarray, cutoff: float, fs: float = FS, order: int = 4) -> np.ndarray:
    """Zero-lag Butterworth low-pass along axis 0 (filtfilt => effective order 8)."""
    b, a = butter(order // 2, cutoff / (fs / 2.0))
    padlen = min(3 * max(len(a), len(b)), x.shape[0] - 1)
    return filtfilt(b, a, x, axis=0, padlen=padlen)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1e-12, n)


def deriv(x: np.ndarray, dt: float) -> np.ndarray:
    """Central-difference time derivative along axis 0."""
    return np.gradient(x, dt, axis=0)


def frame_angular_velocity(e1, e2, e3, dt):
    """Angular velocity of an orthonormal frame: w = 1/2 * sum e_i x de_i/dt."""
    return 0.5 * (np.cross(e1, deriv(e1, dt)) + np.cross(e2, deriv(e2, dt)) + np.cross(e3, deriv(e3, dt)))


def inertia_global(R: np.ndarray, I_local_diag: np.ndarray) -> np.ndarray:
    """R: (T,3,3) with columns = local axes in global coords. Returns (T,3,3) global inertia tensors."""
    return np.einsum("tij,j,tkj->tik", R, I_local_diag, R)


def arm_inverse_dynamics(
    t: np.ndarray,
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    hand: np.ndarray,
    body_mass: float,
    br_time: float,
    cutoff: float | None = None,
    ball_offset: float = 0.5,
) -> dict:
    """
    Newton–Euler inverse dynamics for the distal arm about the elbow joint center.

    Parameters
    ----------
    t : (T,) time in s
    shoulder, elbow, wrist, hand : (T,3) joint-center positions in m (lefties already mirrored)
    body_mass : kg
    br_time : ball-release time in s (ball mass is carried until this instant)
    cutoff : optional extra low-pass cutoff (Hz) applied to positions before differentiation
    ball_offset : ball centre placed at hand + ball_offset * (hand - wrist), i.e. in the fingers

    Returns a dict of time series (all length T).
    """
    dt = float(np.median(np.diff(t)))
    if cutoff is not None:
        shoulder, elbow, wrist, hand = (lowpass(p, cutoff, 1.0 / dt) for p in (shoulder, elbow, wrist, hand))

    # --- segment frames -------------------------------------------------------
    upper = shoulder - elbow                       # elbow -> shoulder
    fore = wrist - elbow                           # elbow -> wrist
    z_f = unit(elbow - wrist)                      # forearm long axis (points proximally)
    z_u = unit(shoulder - elbow)                   # upper-arm long axis (points proximally)
    n_flex = unit(np.cross(upper, fore))           # elbow flexion/extension axis (normal to arm plane)
    v_axis = unit(np.cross(z_f, n_flex))           # varus/valgus axis: perpendicular to forearm in arm plane
    v_u = unit(np.cross(z_u, n_flex))

    R_f = np.stack([n_flex, v_axis, z_f], axis=2)  # columns are axes
    w_f = frame_angular_velocity(n_flex, v_axis, z_f, dt)
    w_u = frame_angular_velocity(n_flex, v_u, z_u, dt)
    a_f = deriv(w_f, dt)                           # angular acceleration of forearm

    # --- segment lengths, masses, COMs ----------------------------------------
    L_f = np.median(np.linalg.norm(fore, axis=1))
    L_u = np.median(np.linalg.norm(upper, axis=1))
    hand_vec = hand - wrist
    L_h = 2.0 * np.median(np.linalg.norm(hand_vec, axis=1))   # hand_jc sits mid-hand

    p = DE_LEVA
    m_f, m_h, m_u = (p[s]["mass"] * body_mass for s in ("forearm", "hand", "upper_arm"))
    c_f = elbow + p["forearm"]["com"] * fore
    c_h = hand                                      # hand_jc used as hand COM
    c_b = hand + ball_offset * hand_vec             # ball in the fingers
    c_u = shoulder + p["upper_arm"]["com"] * (elbow - shoulder)

    has_ball = (t <= br_time).astype(float)[:, None]
    m_b = BALL_MASS * has_ball

    acc = lambda x: deriv(deriv(x, dt), dt)
    a_cf, a_ch, a_cb = acc(c_f), acc(c_h), acc(c_b)

    # forearm inertia about its COM in local frame (n, v, z) -> (transverse, sagittal, longitudinal)
    rg = p["forearm"]["rg"]
    I_f_local = m_f * (np.array([rg[1], rg[0], rg[2]]) * L_f) ** 2
    I_f = inertia_global(R_f, I_f_local)
    rgh = p["hand"]["rg"]
    I_h = inertia_global(R_f, m_h * (np.array([rgh[1], rgh[0], rgh[2]]) * L_h) ** 2)

    # --- Newton: net force the upper arm applies to the forearm at the elbow -----
    F = m_f * (a_cf - G) + m_h * (a_ch - G) + m_b * (a_cb - G)

    # --- Euler: net internal moment at the elbow (about the elbow joint centre) ---
    def lin_term(c, m, a):
        return np.cross(c - elbow, m * (a - G))

    Iw_f = np.einsum("tij,tj->ti", I_f, w_f)
    Iw_h = np.einsum("tij,tj->ti", I_h, w_f)
    rot = (np.einsum("tij,tj->ti", I_f, a_f) + np.cross(w_f, Iw_f)
           + np.einsum("tij,tj->ti", I_h, a_f) + np.cross(w_f, Iw_h))
    M = lin_term(c_f, m_f, a_cf) + lin_term(c_h, m_h, a_ch) + lin_term(c_b, m_b, a_cb) + rot

    # Varus sign convention: + = internal varus moment (resisting valgus), fixed below.
    varus = -np.einsum("ti,ti->t", M, v_axis)
    flexion = np.einsum("ti,ti->t", M, n_flex)
    pronation = np.einsum("ti,ti->t", M, z_f)

    # contribution split: inertial "flinging" of forearm/hand/ball vs rotational inertia
    varus_rot_part = -np.einsum("ti,ti->t", rot, v_axis)

    # --- energy ---------------------------------------------------------------
    v_cf, v_ch, v_cb, v_cu = (deriv(x, dt) for x in (c_f, c_h, c_b, c_u))
    rgu = p["upper_arm"]["rg"]
    R_u = np.stack([n_flex, v_u, z_u], axis=2)
    I_u = inertia_global(R_u, m_u * (np.array([rgu[1], rgu[0], rgu[2]]) * L_u) ** 2)
    ke_rot = lambda I, w: 0.5 * np.einsum("ti,tij,tj->t", w, I, w)
    ke = (0.5 * m_u * (v_cu ** 2).sum(1) + ke_rot(I_u, w_u)
          + 0.5 * m_f * (v_cf ** 2).sum(1) + ke_rot(I_f, w_f)
          + 0.5 * m_h * (v_ch ** 2).sum(1) + ke_rot(I_h, w_f))
    ke_ball = 0.5 * m_b[:, 0] * (v_cb ** 2).sum(1)

    # elbow joint power: internal moment on forearm dotted with relative angular velocity
    elbow_power = np.einsum("ti,ti->t", M, w_f - w_u)

    # elbow flexion angle (deg) from geometry
    cosang = np.einsum("ti,ti->t", unit(upper), unit(fore))
    elbow_flex = 180.0 - np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    return dict(
        varus=varus, flexion_moment=flexion, pronation_moment=pronation,
        varus_rot_part=varus_rot_part, elbow_force=F, elbow_force_mag=np.linalg.norm(F, axis=1),
        arm_ke=ke, ball_ke=ke_ball, elbow_power=elbow_power, elbow_flex=elbow_flex,
        hand_speed=np.linalg.norm(v_ch, axis=1), ball_speed_est=np.linalg.norm(v_cb, axis=1),
        w_forearm=np.linalg.norm(w_f, axis=1),
        seg=dict(L_f=L_f, L_u=L_u, L_h=L_h, m_f=m_f, m_h=m_h, m_u=m_u),
    )
