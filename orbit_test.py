import numpy as np
import math
from sgp4.api import Satrec, jday
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# DEMO MODE
# ============================================================

DEMO_MODE = True
DEMO_POSITION_OFFSET = np.array([6, -4, 3])
DEMO_VELOCITY_OFFSET = np.array([0, 0.05, 0])

# ============================================================
# SETTINGS
# ============================================================

MAX_SATS = 1000
EARTH_RADIUS = 6371
TIME_STEP = 20
MANEUVER_WINDOW = 86400
PC_THRESHOLD = 1e-4

POS_SIGMA = 0.5
VEL_SIGMA = 0.0005

# ============================================================
# LOAD CATALOG WITH COVARIANCE
# ============================================================

def load_catalog(filename, limit):
    sats = []
    with open(filename, "r") as f:
        lines = f.readlines()

    count = 0
    for i in range(0, len(lines), 3):
        if count >= limit:
            break
        try:
            name = lines[i].strip()
            l1 = lines[i+1].strip()
            l2 = lines[i+2].strip()
            sat = Satrec.twoline2rv(l1, l2)

            P0 = np.diag([
                POS_SIGMA**2, POS_SIGMA**2, POS_SIGMA**2,
                VEL_SIGMA**2, VEL_SIGMA**2, VEL_SIGMA**2
            ])

            sats.append({
                "name": name,
                "sat": sat,
                "P": P0
            })

            count += 1
        except:
            continue

    return sats

objects = load_catalog("active.txt", MAX_SATS)

# ============================================================
# FIND ISS
# ============================================================

iss_obj = None
for obj in objects:
    if "ISS" in obj["name"].upper():
        iss_obj = obj
        break

# ============================================================
# SYNTHETIC THREAT
# ============================================================

class SyntheticSat:
    def __init__(self, base_sat):
        self.base_sat = base_sat

    def sgp4(self, jd, fr):
        e, r, v = self.base_sat.sgp4(jd, fr)
        r = np.array(r)
        v = np.array(v)

        if DEMO_MODE:
            r += DEMO_POSITION_OFFSET
            v += DEMO_VELOCITY_OFFSET

        return 0, r.tolist(), v.tolist()

synthetic_P = np.diag([
    POS_SIGMA**2, POS_SIGMA**2, POS_SIGMA**2,
    VEL_SIGMA**2, VEL_SIGMA**2, VEL_SIGMA**2
])

objects.append({
    "name": "TEST_OBJECT",
    "sat": SyntheticSat(iss_obj["sat"]),
    "P": synthetic_P
})

# ============================================================
# UTILITIES
# ============================================================

def analytical_tca(r1, v1, r2, v2):
    r_rel = r1 - r2
    v_rel = v1 - v2
    denom = np.dot(v_rel, v_rel)
    if denom == 0:
        return np.linalg.norm(r_rel), 0
    tca = -np.dot(r_rel, v_rel) / denom
    r1_tca = r1 + v1 * tca
    r2_tca = r2 + v2 * tca
    return np.linalg.norm(r1_tca - r2_tca), tca

def propagate_covariance(P, dt):
    I3 = np.eye(3)
    Z3 = np.zeros((3,3))
    F = np.block([
        [I3, dt * I3],
        [Z3, I3]
    ])
    return F @ P @ F.T

def compute_pc_covariance(r1, P1, r2, P2):
    r_rel = r1 - r2
    P_rel = P1 + P2
    P_pos = P_rel[0:3, 0:3]

    try:
        P_inv = np.linalg.inv(P_pos)
    except:
        return 0.0

    D2 = r_rel.T @ P_inv @ r_rel
    return math.exp(-0.5 * D2)

# ============================================================
# 3D COVARIANCE ELLIPSOID
# ============================================================

ellipsoid_artists = []

def plot_covariance_ellipsoid(ax, center, P, color="yellow", scale=3.0):
    P_pos = P[0:3, 0:3]

    try:
        eigvals, eigvecs = np.linalg.eigh(P_pos)
    except:
        return

    eigvals = np.maximum(eigvals, 1e-12)
    radii = scale * np.sqrt(eigvals)

    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack((x, y, z), axis=0)

    for i in range(3):
        sphere[i] *= radii[i]

    ellipsoid = np.tensordot(eigvecs, sphere.reshape(3, -1), axes=1)
    ellipsoid = ellipsoid.reshape(3, len(u), len(v))

    artist = ax.plot_wireframe(
        ellipsoid[0] + center[0],
        ellipsoid[1] + center[1],
        ellipsoid[2] + center[2],
        rstride=2,
        cstride=2,
        color=color,
        alpha=0.3
    )

    ellipsoid_artists.append(artist)

# ============================================================
# PLOT SETUP
# ============================================================

plt.style.use("dark_background")
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, np.pi, 40)
x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
z = EARTH_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color="blue", alpha=0.3)

ax.set_xlim([-20000, 20000])
ax.set_ylim([-20000, 20000])
ax.set_zlim([-20000, 20000])

leo_scatter = ax.scatter([], [], [], s=6, color="#00FFFF")
meo_scatter = ax.scatter([], [], [], s=8, color="#FF00FF")
heo_scatter = ax.scatter([], [], [], s=10, color="#FFA500")

status_text = fig.text(0.5, 0.95, "", ha="center", fontsize=16)

sim_time = datetime.now(timezone.utc)

# ============================================================
# UPDATE LOOP
# ============================================================

def update(frame):
    global sim_time, ellipsoid_artists

    sim_time += timedelta(seconds=TIME_STEP)

    jd, fr = jday(sim_time.year, sim_time.month, sim_time.day,
                  sim_time.hour, sim_time.minute, sim_time.second)

    # Remove old ellipsoids safely
    for artist in ellipsoid_artists:
        try:
            artist.remove()
        except:
            pass
    ellipsoid_artists = []

    leo_positions, meo_positions, heo_positions = [], [], []
    threats = []

    iss_r = None
    iss_v = None
    iss_P = None

    for obj in objects:
        e, r, v = obj["sat"].sgp4(jd, fr)
        if e != 0:
            continue

        r = np.array(r)
        v = np.array(v)

        obj["P"] = propagate_covariance(obj["P"], TIME_STEP)

        altitude = np.linalg.norm(r) - EARTH_RADIUS
        if altitude < 2000:
            leo_positions.append(r)
        elif altitude < 35786:
            meo_positions.append(r)
        else:
            heo_positions.append(r)

        if obj == iss_obj:
            iss_r = r
            iss_v = v
            iss_P = obj["P"]

    if iss_r is not None:
        plot_covariance_ellipsoid(ax, iss_r, iss_P, color="cyan")

    for obj in objects:
        if obj == iss_obj:
            continue

        e, r2, v2 = obj["sat"].sgp4(jd, fr)
        if e != 0:
            continue

        r2 = np.array(r2)
        v2 = np.array(v2)

        miss, tca = analytical_tca(iss_r, iss_v, r2, v2)
        pc = compute_pc_covariance(iss_r, iss_P, r2, obj["P"])

        if 0 < tca < MANEUVER_WINDOW and pc > PC_THRESHOLD:
            threats.append((obj["name"], pc, miss))
            plot_covariance_ellipsoid(ax, r2, obj["P"], color="red")

    if leo_positions:
        leo_scatter._offsets3d = tuple(np.array(leo_positions).T)
    if meo_positions:
        meo_scatter._offsets3d = tuple(np.array(meo_positions).T)
    if heo_positions:
        heo_scatter._offsets3d = tuple(np.array(heo_positions).T)

    if threats:
        status_text.set_text("🚨 COLLISION RISK (Covariance + Ellipsoid)")
        status_text.set_color("red")
    else:
        status_text.set_text("🟢 SAFE")
        status_text.set_color("green")

    ax.set_title(f"Simulation Time: {sim_time.strftime('%H:%M:%S UTC')}")
    return leo_scatter

ani = FuncAnimation(fig, update, interval=80)
plt.show()