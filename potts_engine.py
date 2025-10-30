# potts_engine.py
import numpy as np
from skimage.measure import label, regionprops
import math
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

# ---------- Material database (basic values you can refine) ----------
MATERIALS = {
    "Inconel-718": {
        "yield_strength": 650e6,   # Pa (example typical)
        "fatigue_limit": 300e6,    # Pa (approx)
        "Q": 240e3,                # activation energy J/mol (placeholder)
        "rho_crit": 1.0            # critical dislocation density (arbitrary units)
    },
    "SS321": {
        "yield_strength": 350e6,
        "fatigue_limit": 200e6,
        "Q": 180e3,
        "rho_crit": 1.0
    }
}

# ---------- utility ----------
def neighborhood_indices(x, y, Lx, Ly, moore=True):
    neigh = []
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            if dx==0 and dy==0:
                continue
            if not moore and abs(dx)+abs(dy)==2:
                continue
            nx = (x+dx) % Lx
            ny = (y+dy) % Ly
            neigh.append((nx,ny))
    return neigh

# ---------- energy calc helpers ----------
def delta_energy(grid, x, y, new_label, J=1.0, moore=True):
    Lx, Ly = grid.shape
    E_before = 0.0
    E_after = 0.0
    for nx, ny in neighborhood_indices(x, y, Lx, Ly, moore=moore):
        E_before += J * (1 if grid[nx,ny] != grid[x,y] else 0)
        E_after  += J * (1 if grid[nx,ny] != new_label else 0)
    return E_after - E_before

# ---------- core Potts step (numba and numpy versions) ----------
if NUMBA_AVAILABLE:
    from numba import njit, prange

    @njit
    def potts_montecarlo_step_numba(grid, rng_states, J, kT_local, moore):
        # rng_states is a 2D array or flattened; for simplicity we'll use numpy random in main loop
        Lx, Ly = grid.shape
        for ix in range(Lx):
            for iy in range(Ly):
                # pick random neighbor orientation
                dx = np.int64(np.random.randint(-1, 2))
                dy = np.int64(np.random.randint(-1, 2))
                nx = (ix + dx) % Lx
                ny = (iy + dy) % Ly
                if dx == 0 and dy == 0:
                    continue
                new_label = grid[nx, ny]
                dE = 0.0
                # compute dE
                for dx2 in range(-1, 2):
                    for dy2 in range(-1, 2):
                        if dx2 == 0 and dy2 == 0:
                            continue
                        if not moore and abs(dx2)+abs(dy2)==2:
                            continue
                        xx = (ix + dx2) % Lx
                        yy = (iy + dy2) % Ly
                        E_before = 1.0 if grid[xx,yy] != grid[ix,iy] else 0.0
                        E_after  = 1.0 if grid[xx,yy] != new_label else 0.0
                        dE += (E_after - E_before)
                if dE <= 0.0 or np.random.rand() < np.exp(-dE / (kT_local[ix,iy] + 1e-12)):
                    grid[ix,iy] = new_label
        return grid

# fallback pure-numpy version (slower but stable)
def potts_montecarlo_step(grid, J, kT_local, moore=True, trials=None):
    Lx, Ly = grid.shape
    if trials is None:
        trials = Lx * Ly
    for _ in range(trials):
        x = np.random.randint(0, Lx)
        y = np.random.randint(0, Ly)
        # choose neighbor
        neis = neighborhood_indices(x, y, Lx, Ly, moore=moore)
        nx, ny = neis[np.random.randint(len(neis))]
        new_label = grid[nx, ny]
        dE = delta_energy(grid, x, y, new_label, J=J, moore=moore)
        kT = kT_local[x,y]
        if dE <= 0 or np.random.rand() < math.exp(-dE / (kT + 1e-12)):
            grid[x,y] = new_label
    return grid

# ---------- dislocation and nucleation ----------
def initialize_grid(Lx, Ly, n_init):
    grid = np.random.randint(0, n_init, size=(Lx, Ly), dtype=np.int32)
    return grid

def compute_grain_stats(grid):
    labeled = label(grid, connectivity=1)
    props = regionprops(labeled)
    areas = [p.area for p in props]
    if len(areas)==0:
        return {"n_grains":0, "mean_area":0, "areas":np.array([])}
    return {"n_grains":len(areas), "mean_area":np.mean(areas), "areas":np.array(areas)}

def update_dislocation(rho, strain_increment, T_field, k_accum=0.1, k_recovery=0.01):
    # simplified evolution: rho += k_accum * strain_increment - k_recovery * f(T) * rho
    # f(T) increases with temperature (recovery faster)
    fT = 1.0 + 0.01 * (T_field - 300.0)  # simple linear mapping
    rho += k_accum * strain_increment
    rho -= k_recovery * fT * rho
    rho[rho < 0] = 0.0
    return rho

def nucleation_step(grid, rho, T_field, strain_rate_field, material_props, nuc_prefactor=0.001):
    Lx, Ly = grid.shape
    new_nuclei = []
    for x in range(Lx):
        for y in range(Ly):
            # only nucleate at grain boundaries (simple test: any neighbor with different label)
            is_boundary = False
            for nx, ny in neighborhood_indices(x, y, Lx, Ly, moore=True):
                if grid[nx,ny] != grid[x,y]:
                    is_boundary = True
                    break
            if not is_boundary:
                continue
            # simple probability
            Q = material_props.get("Q", 200e3)
            T = T_field[x,y]
            rate_factor = math.exp(-Q / (8.314 * T + 1e-12))
            P = nuc_prefactor * strain_rate_field[x,y] * rate_factor * (rho[x,y] / (material_props.get("rho_crit",1.0)+1e-12))
            if np.random.rand() < P:
                # create a new grain id (use max+1)
                new_id = grid.max() + 1
                grid[x,y] = new_id
                new_nuclei.append((x,y))
    return grid, new_nuclei

# ---------- fatigue check ----------
def fatigue_check(strain_field, material_props):
    # Compare local equivalent strain to an allowable strain based on yield (very simplified)
    # allowable_strain = yield_strength / (E) -> but we lack E -> use empirical ratio
    yield_strength = material_props.get("yield_strength", 300e6)
    fatigue_limit = material_props.get("fatigue_limit", 200e6)
    # Define allowable engineering strain fraction (very simplified)
    # Use a heuristic: allowable_strain = fatigue_limit / yield_strength (dimensionless) * 0.02
    allowable = (fatigue_limit / (yield_strength + 1e-12)) * 0.05
    fatigued_mask = strain_field > allowable
    return fatigued_mask, allowable

# ---------- top-level runner ----------
def run_simulation(Lx=200, Ly=200, n_init=100, J=1.0, mcs_steps=500,
                   T_global=900.0, strain=0.02, strain_rate=1.0,
                   material_name="Inconel-718", use_numba=False, snapshot_every=50):
    mat = MATERIALS.get(material_name, MATERIALS["Inconel-718"])
    # Create grids
    grid = initialize_grid(Lx, Ly, n_init)
    rho = np.zeros((Lx, Ly), dtype=np.float32)       # dislocation density field
    T_field = np.ones((Lx, Ly), dtype=float) * T_global
    strain_field = np.ones((Lx, Ly), dtype=float) * strain
    strain_rate_field = np.ones((Lx, Ly), dtype=float) * strain_rate

    # fatigue check
    fatigued_mask, allowable_strain = fatigue_check(strain_field, mat)
    if fatigued_mask.any():
        return {
            "status": "fatigue",
            "fatigued_mask": fatigued_mask,
            "allowable_strain": allowable_strain
        }

    snapshots = []
    stats = []
    for mcs in range(1, mcs_steps+1):
        kT_local = np.ones_like(T_field) * (0.1 + 0.001*(T_field/1000.0))  # mapping T->kT (heuristic)
        if NUMBA_AVAILABLE and use_numba:
            # Use the numba accelerated loop (note: uses numpy rand inside numba)
            try:
                grid = potts_montecarlo_step_numba(grid, None, J, kT_local, True)
            except Exception:
                grid = potts_montecarlo_step(grid, J, kT_local, moore=True)
        else:
            grid = potts_montecarlo_step(grid, J, kT_local, moore=True)

        # dislocation update (simple)
        rho = update_dislocation(rho, strain / mcs_steps, T_field)

        # nucleation
        grid, newn = nucleation_step(grid, rho, T_field, strain_rate_field, mat)

        if mcs % snapshot_every == 0 or mcs == 1 or mcs == mcs_steps:
            s = compute_grain_stats(grid)
            stats.append({"mcs": mcs, "n_grains": s["n_grains"], "mean_area": s["mean_area"]})
            snapshots.append((mcs, grid.copy(), rho.copy()))
    return {"status":"ok", "snapshots": snapshots, "stats": stats, "final_grid": grid, "rho": rho}
