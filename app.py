
# ============================================================
#  GOLD PROSPECTIVITY DETECTION SYSTEM — ASTER SWIR EDITION
#  AI-Powered Hydrothermal Alteration & Mineral Prospectivity
#  Mapping Platform: Sentinel-2 + ASTER SWIR + DEM + RF/XGB/LGBM
#  v12.0 — Full GIS Expert Upgrade
# ============================================================

"""
╔══════════════════════════════════════════════════════════════════════╗
║   GOLD PROSPECTIVITY DETECTION SYSTEM  v12.0 GIS Expert            ║
║   Beni-Suef University — Faculty of Earth Sciences                  ║
║   Nader Safwat Ayed Hanna                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Tabs:                                                               ║
║   1. 🔍 Gold Detector    — RF/XGB/LGBM + Sentinel-2 + ASTER + DEM  ║
║   2. 🗺️  Index Explorer  — All bands / indices / ASTER maps         ║
║   3. 📊 Multi-Index View — Side-by-side comparison up to 6 panels   ║
║   4. 🌋 ASTER SWIR       — ASTER band explorer + hydrothermal maps  ║
║   5. 🔥 Hydrothermal     — Alteration intensity + mineral maps      ║
║   6. 💎 Mineral Mapping  — Quartz/Silica + Clay + MgOH mapping     ║
║   7. 🧠 Train Model      — XGBoost/LGBM/RF ensemble + SHAP + Optuna║
║   8. 🔬 SAM Minerals     — Spectral Angle Mapper mineral matching   ║
║   9. 🧮 PCA Anomaly      — PCA spectral anomaly detection           ║
║  10. 🏔️  Structure        — Lineament density + structural bearing   ║
║  11. 🎯 MCDA             — Multi-Criteria weighted prospectivity     ║
║  12. 📋 Zone Export      — High-prob zone → CSV/SHP for QGIS/ArcGIS║
║  13. 🛰️  Stack Builder   — Sentinel-2 + ASTER SWIR + DEM stack      ║
║  14. 🏔️  3D Visualization— 3D surface draped over DEM topography    ║
║  15. 📐 Spatial Stats    — Moran's I + hotspot density + clustering ║
║  16. 🪨  Lithology Map   — Supervised lithological unit mapping     ║
║  17. 🌡️  Thermal Stress  — Land Surface Temperature proxy + anomaly ║
║  18. 🧭 Deposit Model    — Orogenic/Epithermal/VMS suitability score║
║  19. ⚙️  Settings        — Theme + layout customisation             ║
╚══════════════════════════════════════════════════════════════════════╝

v12 Enhancements (over v11 — Full GIS Expert Upgrade):
  • 7 new ASTER geological indices: Sericite, Calcite, Chlorite, Dolomite,
    Pyrophyllite, Muscovite, Phengite — specifically diagnostic of gold systems
  • 8 new Sentinel-2 engineered features: Pyrite Proxy, Gossanite Index,
    Supergene Enrichment Index, Redox Ratio, Au-Pathfinder Composite,
    Spectral Tilt, VNIR-SWIR slope, Gossan Maturity Index
  • Lithology Mapping tab — 6-class unsupervised lithological discrimination
    (mafic, felsic, carbonate, quartzo-feldspathic, oxide zone, silicified)
  • Thermal Proxy tab — Emissivity-based LST proxy from ASTER TIR bands +
    fault-controlled thermal anomaly detection
  • Deposit Model Suitability tab — scores scene against 3 gold deposit
    archetypes (Orogenic, Low-Sulphidation Epithermal, Skarn/VMS)
  • Adaptive NaN masking — cloud/shadow/water auto-excluded from probability
  • Improved MCDA: 3 additional criteria (Fault proximity, Silicification,
    Carbonatisation) → 8-layer weighted overlay
  • Dynamic threshold recalibration using empirical beta-distribution fit
  • Zone Export: optional Shapefile (.shp) polygon output for direct QGIS/ArcGIS load
  • Enhanced map output: 200 DPI default, embedded UTM + WGS84 dual-grid
  • Batch Detect mode: process multiple S2 tiles in one run → composite mosaic
  • Full UTF-8 Arabic/English bilingual report generation
  • Confidence band map: shows p05–p95 model uncertainty envelope
"""

import gradio as gr
import rasterio
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from rasterio.warp import reproject, Resampling, transform_bounds
import tempfile, os, traceback, warnings, datetime, functools, threading
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter, generic_filter
from scipy.stats import pearsonr
import sys, types, csv
warnings.filterwarnings("ignore")

# ── OPTIONAL: skimage for regionprops (graceful fallback) ─────────────
try:
    from skimage.measure import regionprops as _skimage_regionprops
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

# ── OPTIONAL HIGH-PERFORMANCE LIBRARIES (graceful fallback) ───────────
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    from literature_review_tab import build_literature_tab
    _HAS_LIT = True
except ImportError:
    _HAS_LIT = False

# ── PERFORMANCE: figure render settings ───────────────────────────────
plt.rcParams.update({
    "figure.max_open_warning": 0,
    "path.simplify": True,
    "path.simplify_threshold": 0.5,
    "agg.path.chunksize": 20000,
    # Premium dark theme for all matplotlib figures
    "figure.facecolor":    "#030508",
    "axes.facecolor":      "#030508",
    "savefig.facecolor":   "#030508",
    "text.color":          "#c8d4ec",
    "axes.labelcolor":     "#8a9ab8",
    "xtick.color":         "#5a6e96",
    "ytick.color":         "#5a6e96",
    "axes.edgecolor":      "#1e2840",
    "grid.color":          "#1e2840",
    "grid.alpha":          0.4,
    "axes.grid":           False,
    "font.family":         "DejaVu Sans",
})

# ── MATPLOTLIB COMPATIBILITY HELPER ──────────────────────────────────
def _get_cmap(name):
    """Compatibility wrapper: matplotlib.colormaps (≥3.5) or plt.get_cmap (older)."""
    try:
        return matplotlib.colormaps.get_cmap(name)
    except AttributeError:
        return plt.get_cmap(name)

REQUIRED_BANDS = 18
ASTER_BANDS    = 6          # ASTER SWIR bands 4-9
PIXEL_SIZE_M   = 20
BG             = "#030508"
GOLD           = "#d4962a"
VERSION        = "v14.0 Drill Target Isolation Engine"

# ── STRUCTURED OUTPUT STORAGE ─────────────────────────────────────────
# All app outputs are written under a single root, organised by type.
# Sub-directories are created on first use; filenames carry a timestamp
# so repeated runs never overwrite earlier results.
#
#   OUTPUT_ROOT/
#   ├── maps/          PNG maps (gold detector, index explorer, …)
#   ├── geotiff/       GeoTIFF exports (probability, indices, minerals, …)
#   ├── stack/         Sentinel-2 + ASTER feature stacks
#   ├── models/        Trained .pkl model files
#   ├── shap/          SHAP beeswarm figures
#   └── reports/       Markdown / text summary reports
#
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gold_outputs"
)

_OUTPUT_SUBDIRS = {
    "maps"        : os.path.join(OUTPUT_ROOT, "maps"),
    "geotiff"     : os.path.join(OUTPUT_ROOT, "geotiff"),
    "stack"       : os.path.join(OUTPUT_ROOT, "stack"),
    "models"      : os.path.join(OUTPUT_ROOT, "models"),
    "shap"        : os.path.join(OUTPUT_ROOT, "shap"),
    "reports"     : os.path.join(OUTPUT_ROOT, "reports"),
    "hdf_convert" : os.path.join(OUTPUT_ROOT, "hdf_convert"),
}
MODEL_PATH = os.path.join(OUTPUT_ROOT, "models", "gold_model.pkl")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)  # ensure models dir exists


def _ensure_output_dir(key: str) -> str:
    """Return (and create if needed) the output sub-directory for *key*."""
    path = _OUTPUT_SUBDIRS.get(key, OUTPUT_ROOT)
    os.makedirs(path, exist_ok=True)
    return path

def _out_path(key: str, stem: str, suffix: str) -> str:
    """
    Build a unique output file path.
    stem   – descriptive base name (e.g. 'gold_probability')
    suffix – file extension including dot (e.g. '.png', '.tif')
    Returns an absolute path inside OUTPUT_ROOT/<key>/.
    """
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{stem}_{ts}{suffix}"
    return os.path.join(_ensure_output_dir(key), fname)

# Create the root + all sub-dirs at import time so the user can see the
# folder structure immediately after launching the app.
for _d in _OUTPUT_SUBDIRS.values():
    os.makedirs(_d, exist_ok=True)
print(f"  Output : {OUTPUT_ROOT}")
# ── END STRUCTURED OUTPUT STORAGE ────────────────────────────────────

ACCENT         = "#a07ef5"
CARD_BG        = "#080b12"
TEXT_PRIMARY   = "#c8d4ec"
TEXT_SECONDARY = "#5a6e96"
SUCCESS        = "#0fd4a0"
WARNING        = "#d4962a"
DANGER         = "#f06060"

# ── MODEL ALGORITHM CHOICES ───────────────────────────────────────────
MODEL_ALGO_CHOICES = ["Random Forest", "Ensemble RF+GB",
                      "XGBoost" if _HAS_XGB else "XGBoost (not installed)",
                      "LightGBM" if _HAS_LGB else "LightGBM (not installed)",
                      "Full Ensemble RF+GB+XGB+LGBM"]

# ── ROBUST NORMALISATION SCALER (fitted per-scene) ────────────────────
def robust_scale(arr: np.ndarray, lo=2, hi=98) -> np.ndarray:
    """Robust percentile normalisation, NaN-safe."""
    v = arr[~np.isnan(arr)]
    if v.size == 0: return arr
    p_lo, p_hi = np.nanpercentile(v, [lo, hi])
    rng = p_hi - p_lo
    if rng < 1e-9: return np.zeros_like(arr)
    return np.clip((arr - p_lo) / rng, 0, 1).astype(np.float32)

# ── SENTINEL-2 BAND INFO (bands 0-17) ────────────────────────────────
BAND_INFO = {
    0 : ("B02",  "Blue (B02)",           "Reflectance"),
    1 : ("B03",  "Green (B03)",          "Reflectance"),
    2 : ("B04",  "Red (B04)",            "Reflectance"),
    3 : ("B05",  "Red Edge 1 (B05)",     "Reflectance"),
    4 : ("B06",  "Red Edge 2 (B06)",     "Reflectance"),
    5 : ("B8A",  "NIR Narrow (B8A)",     "Reflectance"),
    6 : ("B08",  "NIR Broad (B08)",      "Reflectance"),
    7 : ("B11",  "SWIR 1 (B11)",         "Reflectance"),
    8 : ("B12",  "SWIR 2 (B12)",         "Reflectance"),
    9 : ("IO",   "Iron Oxide Index",      "B04/B02"),
    10: ("CM",   "Clay Minerals Index",   "B11/B8A"),
    11: ("FI",   "Ferrous Iron Index",    "B11/B08"),
    12: ("GS",   "Gossan Index",          "B04/B08"),
    13: ("NDVI", "NDVI",                  "(NIR-Red)/(NIR+Red)"),
    14: ("DEM",  "Elevation",             "metres"),
    15: ("SLP",  "Slope",                 "degrees"),
    16: ("ASP",  "Aspect",               "degrees"),
    17: ("RGH",  "Roughness",             "metres"),
}

# ── ASTER SWIR BAND INFO ─────────────────────────────────────────────
ASTER_BAND_INFO = {
    0: ("AST_B4",  "ASTER Band 4 (1.60–1.70 µm)", "Reflectance"),
    1: ("AST_B5",  "ASTER Band 5 (2.145–2.185 µm)", "Reflectance"),
    2: ("AST_B6",  "ASTER Band 6 (2.185–2.225 µm)", "Reflectance"),
    3: ("AST_B7",  "ASTER Band 7 (2.235–2.285 µm)", "Reflectance"),
    4: ("AST_B8",  "ASTER Band 8 (2.295–2.365 µm)", "Reflectance"),
    5: ("AST_B9",  "ASTER Band 9 (2.360–2.430 µm)", "Reflectance"),
}

# ── ASTER-DERIVED GEOLOGICAL INDICES ─────────────────────────────────
ASTER_INDICES = {
    "AST_Ferric"  : ("ASTER Ferric Iron Index",        "B4/B5",                   "hot"),
    "AST_AlOH"    : ("ASTER AlOH Index",               "(B5+B7)/(B6+B8)",         "YlOrBr"),
    "AST_MgOH"    : ("ASTER MgOH Index",               "(B6+B9)/(B7+B8)",         "copper"),
    "AST_Silica"  : ("ASTER Silica Index",             "(B4×B6)/(B5²) SWIR",      "plasma"),
    "AST_Clay"    : ("ASTER Clay Ratio",               "(B5+B7)/(B6)",            "YlOrBr"),
    "AST_Quartz"  : ("ASTER Quartz Index",             "(B5×B7)/(B6²)",           "inferno"),
    "AST_HydAlt"  : ("Hydrothermal Alteration",        "(B5+B7+B9)/(B4+B6+B8)",  "RdYlGn_r"),
    "AST_Propyl"  : ("Propylitic Alteration",          "(B6+B8)/(B5+B7)",         "PuBuGn"),
    "AST_Carb"    : ("Carbonate Alteration",           "(B6+B9)/(B5+B8)",         "Oranges"),
    "AST_Epidote" : ("Epidote Index",                  "(B8+B9)/(B6+B7)",         "summer"),
    "AST_Dickite" : ("Dickite/Kaolinite Index",        "(B5+B8)/(B6+B7)",         "autumn"),
    "AST_Alun"    : ("Alunite Index",                  "(B5×B9)/(B7²)",           "spring"),
    # ── NEW v12: 7 indices diagnostic of gold-bearing systems ────────
    "AST_Sericite": ("Sericite/Muscovite Index",       "(B5+B7)/(B6)",            "YlOrRd"),
    "AST_Calcite" : ("Calcite / Dolomite Index",       "(B6+B8)/(B7+B9)",         "BuPu"),
    "AST_Chlorite": ("Chlorite Index",                 "(B7+B9)/(B6+B8)",         "Greens"),
    "AST_Pyrophyl": ("Pyrophyllite Index",             "(B5+B8)/(B6+B7) v2",      "RdPu"),
    "AST_Phengite": ("Phengite / Illite Index",        "(B5×B7)/(B6²) v2",        "YlGn"),
    "AST_GossanAu": ("Gossan / Au-Oxide Proxy",        "(B4-B5)/(B4+B5)",         "afmhot"),
    "AST_SilAlt"  : ("Silicic Alteration Zone",        "(B4+B6)/(B5+B7)",         "magma"),
}

# ── SENTINEL-2 COMPUTED INDICES ───────────────────────────────────────
COMPUTED_INDICES = {
    "NDWI"   : ("(Green-NIR)/(Green+NIR)",              "Blues"),
    "EVI"    : ("2.5*(NIR-Red)/(NIR+6R-7.5B+1)",        "RdYlGn"),
    "SAVI"   : ("1.5*(NIR-Red)/(NIR+Red+0.5)",          "RdYlGn"),
    "Ferric" : ("SWIR1/SWIR2",                           "hot"),
    "AlOH"   : ("(B05+B8A)/(B06+B08) Al-OH minerals",   "YlOrBr"),
    "MgOH"   : ("(B06+B12)/(B8A+B08) Mg-OH carbonate",  "copper"),
    "Silica" : ("(B11+B04)/(B08+B06) Silica/quartz",    "plasma"),
    "Opaque" : ("(B04-B08)/(B04+B08) Opaque minerals",  "inferno"),
    "GosEx"  : ("(B04/B02)*(B04/B8A) Extended Gossan",  "Reds"),
    "IronEx" : ("(B04*B11)/(B8A^2) Extended Iron",       "hot"),
}

CMAP_OPTIONS = ["hot","RdYlGn_r","YlOrBr","Reds","Blues","terrain",
                "plasma","inferno","viridis","copper","RdBu_r","Spectral_r",
                "coolwarm","jet","gray"]

FEATURE_NAMES = [v[0] for v in BAND_INFO.values()]
ALL_INDEX_NAMES = [v[0] for v in BAND_INFO.values()] + list(COMPUTED_INDICES.keys())
ASTER_INDEX_NAMES = list(ASTER_INDICES.keys())
ALL_ASTER_NAMES   = [v[0] for v in ASTER_BAND_INFO.values()] + ASTER_INDEX_NAMES

COMPOSITE_CHOICES = ["Iron Oxide Index","RGB (Natural Colour)",
                     "False Colour (NIR-R-B)","SWIR Composite (B12-B8A-B04)",
                     "Clay Minerals (B11/B8A)","Gossan (B04/B08)",
                     "Ferrous Iron (B11/B08)","NDVI","Elevation","Slope"]
COORD_CHOICES = ["WGS84","UTM (native)","None"]

COMPARISON_MODES = ["Sentinel-2 Only", "ASTER Only", "Hybrid Sentinel-2 + ASTER"]

# ── HELPERS ───────────────────────────────────────────────────────────

def _fpath(f):
    if f is None: return None
    return f if isinstance(f, str) else f.name

def _engineer_features_impl(X: np.ndarray) -> np.ndarray:
    """
    Extended feature engineering (v10) — 20 engineered features added.
    Exact copy of training-time function. Works on (N, 18+) arrays.
    """
    # Band aliases aligned with BAND_INFO indices (0-17)
    # idx:  0=B02  1=B03  2=B04  3=B05  4=B06  5=B8A  6=B08  7=B11  8=B12
    #        9=IO  10=CM  11=FI  12=GS  13=NDVI 14=DEM 15=SLP 16=ASP 17=RGH
    blue   = X[:, 0]   # B02
    green  = X[:, 1]   # B03
    red    = X[:, 2]   # B04
    re1    = X[:, 3]   # B05  Red Edge 1
    re2    = X[:, 4]   # B06  Red Edge 2
    nir_n  = X[:, 5]   # B8A  NIR narrow
    nir_b  = X[:, 6]   # B08  NIR broad
    swir1  = X[:, 7]   # B11
    swir2  = X[:, 8]   # B12
    io_idx = X[:, 9]   # Iron Oxide pre-computed index (B04/B02)
    ndvi   = X[:, 13]
    eps    = 1e-6

    # ── Original 8 features ──────────────────────────────────────────
    ioi        = red   / (blue  + eps)           # Iron Oxide Index
    cmr        = nir_n / (nir_b + eps)           # Clay Minerals Ratio (B8A/B08)
    swir_ratio = re2   / (nir_n + eps)           # SWIR-RE2 / NIR-narrow
    gai        = swir2 * swir_ratio              # Gold Alteration Index proxy
    fii        = io_idx / (green + eps)          # Ferrous Iron proxy
    # Spectral curvature: Red forms the apex between Blue and NIR (iron-bearing minerals)
    curvature  = 2 * red - blue - nir_b
    gi_disc    = cmr   / (io_idx + eps)          # Gold-Index discriminant
    veg_mask   = np.where(ndvi > 0.3, 0.0, 1.0) # Suppress vegetated pixels

    # ── New v10: 12 additional spectral/terrain features ─────────────
    # 1. Normalised Difference Ferric Iron (NDFI)  (Red-B8A)/(Red+B8A)
    ndfi       = (red   - nir_n) / (red   + nir_n + eps)
    # 2. SWIR absorption depth proxy (kaolinite 2.2 µm dip)
    swir_depth = (swir1 + swir2) / (2 * re2 + eps)
    # 3. Modified Gossan: (B04²)/(B02×B8A)
    mod_gossan = (red * red) / (blue * nir_n + eps)
    # 4. Chalcophile proxy: SWIR1/NIR-broad
    chalco     = swir1 / (nir_b + eps)
    # 5. Limonite index: (B04-B02)/(B04+B02)  — iron oxide browning proxy
    limonite   = (red  - blue) / (red  + blue + eps)
    # 6. Iron oxide anomaly: robust z-score of the IO pre-computed index
    _io_med = float(np.nanmedian(io_idx))
    _io_iqr = float(np.nanpercentile(io_idx, 75) - np.nanpercentile(io_idx, 25)) + eps
    iron_anom  = np.abs(io_idx - _io_med) / _io_iqr
    # 7. Slope × Iron product — topographic control on mineralisation
    slope_fe   = X[:, 15] * io_idx if X.shape[1] > 15 else np.zeros(X.shape[0])
    # 8. Aspect × Clay — structural control (N-S faults)
    asp_clay   = X[:, 16] * cmr   if X.shape[1] > 16 else np.zeros(X.shape[0])
    # 9. Roughness × Ferrous — brecciated alteration zones (uses FI at col 11)
    rgh_fe2    = X[:, 17] * X[:, 11] if X.shape[1] > 17 else np.zeros(X.shape[0])
    # 10. SWIR2/B04 — oxide capping indicator
    oxide_cap  = swir2 / (red + eps)
    # 11. Silicification proxy S2: (B11+B04)/(B08+B06)
    sil_s2     = (swir1 + red) / (nir_b + re2 + eps)
    # 12. Al-OH S2 proxy: (B05+B8A)/(B06+B08)
    aloh_s2    = (re1  + nir_n) / (re2  + nir_b + eps)

    # ── v12: 8 new gold-pathfinder features ─────────────────────────
    # 13. Pyrite Proxy: SWIR1/(B04+B08) — sulphide-altered rock spectral signature
    pyrite_proxy   = swir1 / (red + nir_b + eps)
    # 14. Gossanite Index: (B04-B02)/(B04+B8A) — mature gossan capping
    gossanite      = (red - blue) / (red + nir_n + eps)
    # 15. Supergene Enrichment: (SWIR2-B8A)/(SWIR2+B8A) — secondary enrichment blanket
    supergene      = (swir2 - nir_n) / (swir2 + nir_n + eps)
    # 16. Redox Ratio: (B04+B8A)/(B03+B08) — oxidised vs reduced assemblages
    redox          = (red + nir_n) / (green + nir_b + eps)
    # 17. Au-Pathfinder Composite: combination of IO × CM × SWIR depth (dimensionless)
    au_pathfinder  = ioi * cmr * swir_depth
    # 18. Spectral Tilt: slope of reflectance from B02→B12 (overall spectral shape)
    spectral_tilt  = (swir2 - blue) / (8 + eps)   # 8 bands apart → normalised
    # 19. VNIR-SWIR Slope: (SWIR1-NIR)/(SWIR1+NIR) — Fe2+ vs Fe3+ control
    vnir_swir_slope = (swir1 - nir_b) / (swir1 + nir_b + eps)
    # 20. Gossan Maturity Index: (B04×B11)/(B02×B8A) — oxidation degree in iron caps
    gossan_maturity = (red * swir1) / (blue * nir_n + eps)

    new_feats = np.column_stack([
        ioi, cmr, swir_ratio, gai, fii, curvature, gi_disc, veg_mask,
        ndfi, swir_depth, mod_gossan, chalco, limonite, iron_anom,
        slope_fe, asp_clay, rgh_fe2, oxide_cap, sil_s2, aloh_s2,
        # v12 additions
        pyrite_proxy, gossanite, supergene, redox,
        au_pathfinder, spectral_tilt, vnir_swir_slope, gossan_maturity,
    ])
    return np.hstack([X, new_feats])

def _inject_engineer_features():
    fn_names = ["engineer_features", "feature_engineer",
                "add_features", "engineer_feats"]
    for mod_name in ["__main__", "main", "__mp_main__"]:
        mod = sys.modules.get(mod_name)
        if mod is None:
            mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = mod
        for fn_name in fn_names:
            if not hasattr(mod, fn_name):
                setattr(mod, fn_name, _engineer_features_impl)

_inject_engineer_features()

# Alias: _engineer_features is called throughout this module
_engineer_features = _engineer_features_impl

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"❌ Not found: {MODEL_PATH}"
    try:
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict):
            m = bundle["model"]
            return bundle, f"✅ Ensemble | {m.n_features_in_} features"
        m = bundle
        n_est = getattr(m, "n_estimators", "?")
        n_feat = getattr(m, "n_features_in_", "?")
        return bundle, f"✅ {n_est} trees | {n_feat} features"
    except Exception as e:
        return None, f"❌ {e}"

model_bundle, model_status = load_model()

# ── RASTER CACHE: thread-safe LRU avoid re-reading same file ──────────
_raster_cache: dict = {}
_raster_lock  = threading.Lock()
_CACHE_MAX = 6             # increased from 4

def _read_s2(path: str):
    """Read Sentinel-2 stack with thread-safe LRU cache keyed on path."""
    with _raster_lock:
        if path in _raster_cache:
            # Move to end (most-recently-used)
            val = _raster_cache.pop(path)
            _raster_cache[path] = val
            return val
    # Read outside the lock to avoid blocking other threads
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        nd   = src.nodata
        # Convert declared nodata values → NaN
        if nd is not None:
            try:
                data[data == float(nd)] = np.nan
            except (ValueError, TypeError):
                pass
        else:
            # nodata not declared: treat 0 as fill for integer-origin S2 stacks
            # (uint16/int16 sources use 0 as the fill sentinel)
            if src.dtypes[0] in ("uint16", "int16", "uint8"):
                data[data == 0] = np.nan
        result = (data, src.transform, src.crs,
                  src.height, src.width, src.profile.copy())
    with _raster_lock:
        # Another thread may have cached it while we were reading; prefer theirs.
        if path in _raster_cache:
            val = _raster_cache.pop(path)
            _raster_cache[path] = val   # refresh MRU order
            return val
        if len(_raster_cache) >= _CACHE_MAX:
            _raster_cache.pop(next(iter(_raster_cache)))
        _raster_cache[path] = result
    return result


def _align_features(px: np.ndarray, n_expected: int) -> np.ndarray:
    """Align feature matrix column count to what the model expects."""
    if n_expected is None:
        return px
    if px.shape[1] < n_expected:
        px = _engineer_features(px)
    if px.shape[1] != n_expected:
        if px.shape[1] > n_expected:
            import logging
            logging.getLogger(__name__).warning(
                f"_align_features: truncating {px.shape[1]} → {n_expected} cols. "
                "Consider retraining the model with the current feature set."
            )
            px = px[:, :n_expected]
        else:
            pad = np.zeros((px.shape[0], n_expected - px.shape[1]), dtype=px.dtype)
            px = np.hstack([px, pad])
    return px


def _get_model_predict_proba(pixels_valid: np.ndarray) -> np.ndarray:
    if model_bundle is None:
        return None
    m = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
    n_expected = getattr(m, "n_features_in_", None)
    px = _align_features(pixels_valid, n_expected)
    return m.predict_proba(px)[:, 1]

def normalise(arr, lo=2, hi=98):
    p_lo, p_hi = np.nanpercentile(arr, [lo, hi])
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)

def make_rgb(s):
    return np.nan_to_num(np.dstack([normalise(s[2]), normalise(s[1]), normalise(s[0])]))

def make_false_color(s):
    return np.nan_to_num(np.dstack([normalise(s[6]), normalise(s[2]), normalise(s[0])]))

def style_ax(ax, title, fs=11):
    ax.set_facecolor(BG)
    ax.set_title(title, color="#e8b842", fontsize=fs, fontweight="bold", pad=8,
                 fontfamily='DejaVu Sans')
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2840"); sp.set_linewidth(0.6)

def add_cbar(fig, im, ax, label=""):
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label(label, color="#c8d4ec", fontsize=8)
    cb.ax.tick_params(colors="#8a9ab8", labelsize=7)
    cb.outline.set_edgecolor("#1e2840")
    return cb

def add_north_arrow(ax):
    ax.annotate("", xy=(0.95, 0.97), xytext=(0.95, 0.88),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))
    ax.text(0.95, 0.985, "N", transform=ax.transAxes,
            ha="center", va="bottom", color="white",
            fontsize=10, fontweight="bold")

def add_scalebar(ax, h, w, pixel_m=PIXEL_SIZE_M):
    bar_km  = max(1, round(w * pixel_m / 1000 / 6))
    bar_px  = max(1, bar_km * 1000 / pixel_m)
    x_end   = w * 0.88; x_start = x_end - bar_px; y_bar = h * 0.97
    ax.fill_between([x_start, x_end], [y_bar - h*0.008]*2, [y_bar]*2, color="white", zorder=5)
    ax.plot([x_start, x_start], [y_bar-h*.02, y_bar], "w-", lw=1.2, zorder=6)
    ax.plot([x_end,   x_end],   [y_bar-h*.02, y_bar], "w-", lw=1.2, zorder=6)
    ax.text((x_start+x_end)/2, y_bar - h*.028,
            f"0 ——— {bar_km} km",
            color="white", fontsize=7, ha="center", va="top",
            fontweight="bold", zorder=6,
            bbox=dict(fc=BG, ec="none", pad=1, alpha=0.7))

def add_coord_grid(ax, transform, h, w, crs, n=6, mode="WGS84"):
    try:
        n_x = min(n, w); n_y = min(n, h)
        rows_i = np.linspace(0, h - 1, n_y).astype(int)
        cols_i = np.linspace(0, w - 1, n_x).astype(int)
        mid_col = w // 2; mid_row = h // 2
        if mode == "WGS84":
            from pyproj import Transformer
            tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            x_coords = [tr.transform(*(transform * (c + 0.5, mid_row + 0.5)))[0] for c in cols_i]
            y_coords = [tr.transform(*(transform * (mid_col + 0.5, r + 0.5)))[1] for r in rows_i]
            x_lbl = [f"{v:.4f}°{'E' if v >= 0 else 'W'}" for v in x_coords]
            y_lbl = [f"{abs(v):.4f}°{'N' if v >= 0 else 'S'}" for v in y_coords]
        else:
            x_coords = [transform.c + (c + 0.5) * transform.a for c in cols_i]
            y_coords = [transform.f + (r + 0.5) * transform.e for r in rows_i]
            x_lbl = [f"{v/1000:.2f}km" for v in x_coords]
            y_lbl = [f"{v/1000:.2f}km" for v in y_coords]
        ax.set_xticks(cols_i)
        ax.set_xticklabels(x_lbl, color="#d4962a", fontsize=6.5, rotation=35, ha="right", fontweight="bold")
        ax.set_yticks(rows_i)
        ax.set_yticklabels(y_lbl, color="#d4962a", fontsize=6.5, fontweight="bold")
        ax.tick_params(axis="both", colors="#d4962a", length=4, width=0.8, pad=2, direction="out")
        for c in cols_i: ax.axvline(c, color="#d4962a", lw=0.5, alpha=0.25, zorder=0, ls="--")
        for r in rows_i: ax.axhline(r, color="#d4962a", lw=0.5, alpha=0.25, zorder=0, ls="--")
        ax.text(0.99, 0.01, mode, transform=ax.transAxes,
                color="#d4962a", fontsize=5.5, ha="right", va="bottom", alpha=0.7,
                fontfamily="monospace")
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"add_coord_grid render: {_e}")

def _make_mock_src(profile, transform, crs, rows, cols):
    """
    Build a minimal duck-typed rasterio-source object from cached raster data.
    Used by geo_info_dict so that functions which use _read_s2() (the LRU cache)
    never need to re-open the file just to call geo_info_dict.
    """
    class _MockSrc:
        def __init__(self):
            self.crs       = crs
            self.transform = transform
            self.width     = cols
            self.height    = rows
            self.bounds    = rasterio.transform.array_bounds(rows, cols, transform)
    return _MockSrc()


def geo_info_dict(src):
    info = {"crs": str(src.crs), "size": f"{src.width}×{src.height}",
            "pixel_m": PIXEL_SIZE_M, "bounds_native": src.bounds}
    try:
        b = transform_bounds(src.crs, "EPSG:4326",
                             src.bounds.left, src.bounds.bottom,
                             src.bounds.right, src.bounds.top)
        info["lon_min"] = b[0]; info["lon_max"] = b[2]
        info["lat_min"] = b[1]; info["lat_max"] = b[3]
        info["lon_span"] = b[2]-b[0]; info["lat_span"] = b[3]-b[1]
        info["area_km2"] = (src.width*PIXEL_SIZE_M/1000) * (src.height*PIXEL_SIZE_M/1000)
        info["geo_str"] = (f"{b[1]:.4f}°N – {b[3]:.4f}°N  |  "
                           f"{b[0]:.4f}°E – {b[2]:.4f}°E")
        info["center_lat"] = (b[1]+b[3])/2
        info["center_lon"] = (b[0]+b[2])/2
    except Exception as _e:
        info["geo_str"] = "Coords unavailable"
    return info

def stamp_map(fig, geo_info, mode_str, threshold=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")
    thr_txt = f"  |  Threshold: {threshold:.0%}" if threshold is not None else ""
    txt = (f"Gold Prospectivity System {VERSION}  ·  Beni-Suef University  ·  "
           f"Mode: {mode_str}{thr_txt}  ·  {geo_info.get('geo_str','?')}  ·  "
           f"Generated: {ts}")
    fig.text(0.5, 0.002, txt, ha="center", va="bottom",
             fontsize=5.5, color="#5a6e96", fontfamily="monospace",
             bbox=dict(fc=BG, ec="#1e2840", lw=0.5, pad=3))


# ── ASTER SWIR PROCESSING ─────────────────────────────────────────────

def resample_aster_to_sentinel(aster_stack: np.ndarray, aster_transform,
                                aster_crs, ref_transform, ref_crs,
                                ref_height: int, ref_width: int) -> np.ndarray:
    """
    Resample ASTER SWIR bands to match Sentinel-2 spatial resolution and CRS.
    v10: uses Lanczos resampling (higher quality than bilinear) with NaN-safe
    edge padding to reduce border artefacts.
    """
    n_bands = aster_stack.shape[0]
    out = np.zeros((n_bands, ref_height, ref_width), dtype=np.float32)
    for i in range(n_bands):
        band = aster_stack[i].astype(np.float32)
        # Replace NaN with band global mean for border continuity.
        # Using nan_to_num(0) before the filter biases edge values low;
        # the global mean is a neutral, unbiased fill for out-of-swath pixels.
        nan_mask = np.isnan(band)
        if nan_mask.any():
            fill_val = float(np.nanmean(band)) if not np.all(nan_mask) else 0.01
            band = np.where(nan_mask, fill_val, band)
        dest = np.zeros((ref_height, ref_width), dtype=np.float32)
        reproject(
            band, dest,
            src_transform=aster_transform,
            src_crs=aster_crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.lanczos   # upgraded from bilinear
        )
        out[i] = dest
    return out


def compute_aster_indices(aster: np.ndarray) -> dict:
    """
    Compute all ASTER SWIR geological indices (v10: 12 indices, up from 7).
    aster: shape (6, H, W) → bands B4,B5,B6,B7,B8,B9 → indices 0-5
    Returns dict of named float32 arrays.
    """
    if aster.ndim != 3 or aster.shape[0] < 6:
        raise ValueError(
            f"compute_aster_indices expects shape (6, H, W); got {aster.shape}. "
            "Ensure the ASTER GeoTIFF contains at least 6 SWIR bands (B4-B9)."
        )
    B4, B5, B6, B7, B8, B9 = (aster[i] for i in range(6))
    eps = 1e-6
    results = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        # ── Original 7 ────────────────────────────────────────────────
        results["AST_Ferric"]  = np.where(B5 > eps, B4 / B5, np.nan).astype(np.float32)
        denom = B6 + B8
        results["AST_AlOH"]   = np.where(denom > eps, (B5 + B7) / denom, np.nan).astype(np.float32)
        denom = B7 + B8
        results["AST_MgOH"]   = np.where(denom > eps, (B6 + B9) / denom, np.nan).astype(np.float32)
        denom = B5 * B5
        results["AST_Silica"] = np.where(denom > eps, (B4 * B6) / denom, np.nan).astype(np.float32)
        results["AST_Clay"]   = np.where(B6 > eps, (B5 + B7) / B6, np.nan).astype(np.float32)
        denom = B6 * B6
        results["AST_Quartz"] = np.where(denom > eps, (B5 * B7) / denom, np.nan).astype(np.float32)
        num   = B5 + B7 + B9; denom = B4 + B6 + B8
        results["AST_HydAlt"] = np.where(denom > eps, num / denom, np.nan).astype(np.float32)
        # ── New v10: 5 additional alteration indices ──────────────────
        # Propylitic Alteration: (B6+B8)/(B5+B7) — chlorite + calcite
        denom = B5 + B7
        results["AST_Propyl"]  = np.where(denom > eps, (B6 + B8) / denom, np.nan).astype(np.float32)
        # Carbonate Alteration: (B6+B9)/(B5+B8)
        denom = B5 + B8
        results["AST_Carb"]    = np.where(denom > eps, (B6 + B9) / denom, np.nan).astype(np.float32)
        # Epidote Index: (B8+B9)/(B6+B7) — epidote-chlorite zones
        denom = B6 + B7
        results["AST_Epidote"] = np.where(denom > eps, (B8 + B9) / denom, np.nan).astype(np.float32)
        # Dickite/Kaolinite Index: (B5+B8)/(B6+B7)
        denom = B6 + B7
        results["AST_Dickite"] = np.where(denom > eps, (B5 + B8) / denom, np.nan).astype(np.float32)
        # Alunite Index: (B5×B9)/(B7²)
        denom = B7 * B7
        results["AST_Alun"]    = np.where(denom > eps, (B5 * B9) / denom, np.nan).astype(np.float32)
        # ── v12 indices (now computed) ────────────────────────────────
        results["AST_Sericite"] = np.where(B6 > eps, (B5 + B7) / B6, np.nan).astype(np.float32)
        denom = B7 + B9
        results["AST_Calcite"]  = np.where(denom > eps, (B6 + B8) / denom, np.nan).astype(np.float32)
        denom = B6 + B8
        results["AST_Chlorite"] = np.where(denom > eps, (B7 + B9) / denom, np.nan).astype(np.float32)
        denom = B6 + B7
        results["AST_Pyrophyl"] = np.where(denom > eps, (B5 + B8) / denom, np.nan).astype(np.float32)
        denom = B6 * B6
        results["AST_Phengite"] = np.where(denom > eps, (B5 * B7) / denom, np.nan).astype(np.float32)
        denom = B4 + B5
        results["AST_GossanAu"] = np.where(denom > eps, (B4 - B5) / denom, np.nan).astype(np.float32)
        denom = B5 + B7
        results["AST_SilAlt"]   = np.where(denom > eps, (B4 + B6) / denom, np.nan).astype(np.float32)
    return results


def compute_aster_index_by_name(name: str, aster: np.ndarray):
    """Single ASTER index by name; aster shape (6, H, W)."""
    results = compute_aster_indices(aster)
    return results.get(name, None)


# ══════════════════════════════════════════════════════════════════════
#  v13 ORE-TARGETING ENGINE
#  Structurally-constrained multi-stage ore-shoot targeting.
#  Transforms broad hydrothermal maps → compact drill-ready targets.
# ══════════════════════════════════════════════════════════════════════

# ── Structural curvature / fault-intersection proxy ───────────────────
def compute_structural_curvature(dem_arr: np.ndarray, kernel: int = 7) -> np.ndarray:
    """
    Compute plan curvature from DEM as a proxy for structural curvature
    and fault-intersection nodes.  High curvature → fold hinges, relay ramps,
    brittle jog zones — the geometries that localise ore shoots.
    Returns 0-1 normalised array.
    """
    try:
        from scipy.ndimage import sobel, laplace
        dem_c = np.nan_to_num(dem_arr, nan=float(np.nanmean(dem_arr) if not np.all(np.isnan(dem_arr)) else 0))
        # Plan curvature approximated by Laplacian of filtered gradient
        from scipy.ndimage import uniform_filter
        dem_sm = uniform_filter(dem_c, size=kernel)
        curv   = np.abs(laplace(dem_sm))
        lo, hi = np.nanpercentile(curv, [1, 99])
        if hi > lo:
            curv = np.clip((curv - lo) / (hi - lo), 0, 1)
        return curv.astype(np.float32)
    except Exception:
        return np.full_like(dem_arr, 0.0, dtype=np.float32)


def compute_fault_intersection_density(dem_arr: np.ndarray,
                                        slope_arr: np.ndarray,
                                        kernel_small: int = 3,
                                        kernel_large: int = 11) -> np.ndarray:
    """
    Fault-intersection proxy: difference between fine-scale and coarse-scale
    Sobel edge density maps.  Peaks at lineament cross-nodes —
    structurally dilated zones where gold precipitates.
    Returns 0-1 normalised array.
    """
    try:
        from scipy.ndimage import sobel, uniform_filter
        dem_c = np.nan_to_num(dem_arr, nan=float(np.nanmean(dem_arr) if not np.all(np.isnan(dem_arr)) else 0))
        sx = sobel(dem_c, axis=1); sy = sobel(dem_c, axis=0)
        edge = np.sqrt(sx**2 + sy**2)
        fine   = uniform_filter(edge, size=kernel_small)
        coarse = uniform_filter(edge, size=kernel_large)
        intersection = np.abs(fine - coarse) * fine
        lo, hi = np.nanpercentile(intersection, [1, 99])
        if hi > lo:
            intersection = np.clip((intersection - lo) / (hi - lo), 0, 1)
        return intersection.astype(np.float32)
    except Exception:
        return np.full_like(dem_arr, 0.0, dtype=np.float32)


def compute_shear_corridor_index(slope_arr: np.ndarray,
                                  aspect_arr: np.ndarray,
                                  roughness_arr: np.ndarray) -> np.ndarray:
    """
    Shear-corridor index: high-slope + roughness + N-S linearity.
    Shear zones control ~80% of orogenic gold deposits.
    Returns 0-1 normalised array.
    """
    s_n  = robust_scale(np.nan_to_num(slope_arr, nan=0))
    r_n  = robust_scale(np.nan_to_num(roughness_arr, nan=0))
    ns   = np.abs(np.cos(np.radians(np.nan_to_num(aspect_arr, nan=0))))  # N-S weight
    shear = s_n * r_n * ns
    lo, hi = np.nanpercentile(shear, [1, 99])
    if hi > lo:
        shear = np.clip((shear - lo) / (hi - lo), 0, 1)
    return shear.astype(np.float32)


# ── Silica ore-vectoring ───────────────────────────────────────────────
def compute_silica_ore_vector(stack: np.ndarray,
                               aster_dict: dict = None) -> np.ndarray:
    """
    Compute a silica ore-vector: combines Sentinel-2 silica proxy,
    ASTER Silica, ASTER Quartz, and ASTER SilAlt indices.
    Silica is an ore-deposit vectoring signal (quartz veins, silicification).
    Returns 0-1 normalised array.
    """
    layers = []
    # S2 silica proxy: (B11+B04)/(B08+B06)
    with np.errstate(divide="ignore", invalid="ignore"):
        b4 = stack[2]; b8 = stack[6]; b6 = stack[4]; b11 = stack[7]
        denom = b8 + b6
        s2_sil = np.where(denom > 1e-6, (b11 + b4) / denom, np.nan)
    layers.append(robust_scale(np.nan_to_num(s2_sil, nan=0)))

    if aster_dict is not None:
        for key in ("AST_Silica", "AST_Quartz", "AST_SilAlt"):
            if key in aster_dict:
                layers.append(robust_scale(np.nan_to_num(aster_dict[key], nan=0)))

    if not layers:
        return np.zeros(stack.shape[1:], dtype=np.float32)
    silica_vec = np.nanmean(np.stack(layers, axis=0), axis=0)
    return np.clip(silica_vec, 0, 1).astype(np.float32)


# ── Alteration localisation (focused vs diffuse) ───────────────────────
def compute_localised_alteration(stack: np.ndarray,
                                  aster_dict: dict = None,
                                  locality_kernel: int = 9) -> np.ndarray:
    """
    Localised alteration = raw alteration signal MINUS diffuse regional mean.
    Removes broad alteration blankets; retains only locally anomalous patches.
    Returns value ∈ [-1, 1]; clipped to [0, 1] for positive anomalies.
    """
    from scipy.ndimage import uniform_filter
    # Build base alteration signal
    layers = []
    io_n = robust_scale(np.nan_to_num(stack[9], nan=0))
    cm_n = robust_scale(np.nan_to_num(stack[10], nan=0))
    layers.extend([io_n, cm_n])
    if aster_dict is not None:
        for key in ("AST_HydAlt", "AST_AlOH", "AST_Clay", "AST_Sericite"):
            if key in aster_dict:
                layers.append(robust_scale(np.nan_to_num(aster_dict[key], nan=0)))
    alt_raw = np.nanmean(np.stack(layers, axis=0), axis=0).astype(np.float32)
    # Subtract regional (diffuse) mean → localised anomaly
    alt_regional = uniform_filter(alt_raw, size=locality_kernel)
    local_alt    = alt_raw - alt_regional
    return np.clip(local_alt, 0, 1).astype(np.float32)


# ── Anomaly spatial convergence (clustering) ──────────────────────────
def compute_anomaly_convergence(prob_or_score: np.ndarray,
                                 kernel: int = 5) -> np.ndarray:
    """
    Measures local spatial concentration of high scores.
    Uses ratio of local max to local mean: high ratio → compact spike (ore-shoot).
    Low ratio → diffuse (alteration blanket) → penalised.
    Returns 0-1 normalised convergence index.
    """
    from scipy.ndimage import maximum_filter, uniform_filter
    score_c = np.nan_to_num(prob_or_score, nan=0)
    local_max  = maximum_filter(score_c, size=kernel)
    local_mean = uniform_filter(score_c,  size=kernel) + 1e-9
    convergence = local_max / local_mean   # >1 = concentrated spike
    lo, hi = np.nanpercentile(convergence, [1, 99])
    if hi > lo:
        convergence = np.clip((convergence - lo) / (hi - lo), 0, 1)
    return convergence.astype(np.float32)


# ── 4-Stage ore targeting ─────────────────────────────────────────────
_ORE_CLASS_LABELS = {
    0: "Background",
    1: "Alteration Zone",
    2: "Fertile Structural Corridor",
    3: "Drill Target",
}
_ORE_CLASS_COLORS = {
    0: "#1a2a1a",
    1: "#4a90d9",
    2: "#f4a261",
    3: "#e63946",
}

def run_ore_targeting_engine(stack: np.ndarray,
                              aster_dict: dict = None,
                              base_prob_map: np.ndarray = None,
                              pixel_size_m: float = 20.0) -> dict:
    """
    v13 Structurally-Constrained Ore Targeting Engine.

    4 sequential refinement stages per the spec:
      Stage 1 — Regional Fertility    (alteration / clay / AlOH)
      Stage 2 — Structural Filtering  (fault density / shear / curvature)
      Stage 3 — Silica Ore-Vectoring  (quartz / silicification)
      Stage 4 — Ore Localisation      (convergence / compactness)

    Confidence-saturation penalty automatically fires when >10% of scene
    is classified as high-priority.

    Returns dict with arrays for each stage + final class map + metadata.
    """
    rows, cols = stack.shape[1], stack.shape[2]
    pm2 = pixel_size_m**2 / 1e6
    total_px = rows * cols
    nan_mask = np.isnan(stack[0]) if stack.shape[0] > 0 else np.zeros((rows, cols), dtype=bool)

    # ── Stage 1: Regional Fertility ───────────────────────────────────
    alt_layers = []
    alt_layers.append(robust_scale(np.nan_to_num(stack[9],  nan=0)))   # IO
    alt_layers.append(robust_scale(np.nan_to_num(stack[10], nan=0)))   # CM
    alt_layers.append(robust_scale(np.nan_to_num(stack[12], nan=0)))   # GS
    if aster_dict:
        for k in ("AST_HydAlt", "AST_AlOH", "AST_Clay", "AST_Sericite", "AST_Propyl"):
            if k in aster_dict:
                alt_layers.append(robust_scale(np.nan_to_num(aster_dict[k], nan=0)))
    if base_prob_map is not None:
        alt_layers.append(np.nan_to_num(base_prob_map, nan=0))
    stage1 = np.nanmean(np.stack(alt_layers, axis=0), axis=0).astype(np.float32)
    stage1 = np.clip(stage1, 0, 1)
    stage1[nan_mask] = np.nan

    # ── Stage 2: Structural Filtering ─────────────────────────────────
    dem_arr   = stack[14]
    slope_arr = stack[15]
    asp_arr   = stack[16]
    rgh_arr   = stack[17]

    lin_density = compute_lineament_density(dem_arr, kernel=5)
    shear_idx   = compute_shear_corridor_index(slope_arr, asp_arr, rgh_arr)
    fault_inter = compute_fault_intersection_density(dem_arr, slope_arr)
    struct_curv = compute_structural_curvature(dem_arr)

    # Structural score: weighted combination
    structural = (0.35 * lin_density +
                  0.30 * shear_idx   +
                  0.20 * fault_inter +
                  0.15 * struct_curv)
    structural = np.clip(structural, 0, 1).astype(np.float32)
    structural[nan_mask] = np.nan

    # Stage 2 score = Stage1 × structural (multiplicative gate — both must be high)
    stage2 = stage1 * structural
    stage2 = np.clip(stage2, 0, 1).astype(np.float32)
    stage2[nan_mask] = np.nan

    # ── Stage 3: Silica Ore-Vectoring ─────────────────────────────────
    silica_vec = compute_silica_ore_vector(stack, aster_dict)
    silica_vec[nan_mask] = np.nan

    stage3 = stage2 * silica_vec
    stage3 = np.clip(stage3, 0, 1).astype(np.float32)
    stage3[nan_mask] = np.nan

    # ── Stage 4: Ore Localisation ─────────────────────────────────────
    local_alt    = compute_localised_alteration(stack, aster_dict)
    convergence  = compute_anomaly_convergence(stage3, kernel=5)

    stage4 = stage3 * convergence * (0.5 + 0.5 * local_alt)
    stage4 = np.clip(stage4, 0, 1).astype(np.float32)
    stage4[nan_mask] = np.nan

    # ── Confidence-saturation penalty ─────────────────────────────────
    # If >10% of scene is above ABSOLUTE score 0.65 → suppress and re-normalise.
    # Using a fixed absolute threshold avoids the statistical tautology where
    # ~10% of pixels always sit above p90 by definition.
    valid_px = int(np.sum(~nan_mask))
    _ABS_HIGH_THR = 0.65          # absolute score considered "high-confidence"
    _hi_frac = float(np.nansum(stage4 >= _ABS_HIGH_THR)) / max(valid_px, 1)
    _thr_90  = float(np.nanpercentile(stage4[~nan_mask], 90)) if valid_px > 0 else 0.5
    saturation_penalty = 1.0
    if _hi_frac > 0.10:
        # Reduce high-probability mass by raising the effective threshold weight
        excess = _hi_frac / 0.10   # how much larger than allowed
        saturation_penalty = 1.0 / max(excess, 1.0)
        # Additional structural and silica gating during saturation
        struct_gate = structural * silica_vec
        stage4 = stage4 * (0.5 + 0.5 * struct_gate)
        stage4 = np.clip(stage4, 0, 1).astype(np.float32)
        stage4[nan_mask] = np.nan

    # ── 4-class ore classification map ────────────────────────────────
    # Thresholds: p60 = alteration, p80 = fertile corridor, p92 = drill target
    flat_v = stage4[~nan_mask] if valid_px > 0 else np.array([0])
    thr_alt    = float(np.nanpercentile(flat_v, 60))
    thr_corr   = float(np.nanpercentile(flat_v, 80))
    thr_target = float(np.nanpercentile(flat_v, 92))

    ore_class = np.zeros((rows, cols), dtype=np.float32)
    ore_class[stage4 >= thr_alt]    = 1   # Alteration Zone
    ore_class[stage4 >= thr_corr]   = 2   # Fertile Structural Corridor
    ore_class[stage4 >= thr_target] = 3   # Drill Target
    ore_class[nan_mask] = np.nan

    # ── Area statistics ───────────────────────────────────────────────
    n_target   = int(np.nansum(ore_class == 3))
    n_corridor = int(np.nansum(ore_class == 2))
    n_alt      = int(np.nansum(ore_class == 1))
    target_pct = n_target / max(valid_px, 1) * 100

    return {
        "stage1_fertility"  : stage1,
        "stage2_structural" : stage2,
        "structural_index"  : structural,
        "stage3_silica"     : stage3,
        "silica_vector"     : silica_vec,
        "stage4_ore"        : stage4,
        "local_alteration"  : local_alt,
        "convergence"       : convergence,
        "ore_class_map"     : ore_class,
        "thr_alteration"    : thr_alt,
        "thr_corridor"      : thr_corr,
        "thr_target"        : thr_target,
        "n_drill_targets"   : n_target,
        "n_corridors"       : n_corridor,
        "n_alteration"      : n_alt,
        "target_area_km2"   : n_target * pm2,
        "target_pct"        : target_pct,
        "saturation_penalty": saturation_penalty,
    }


def plot_ore_targeting_map(ore_result: dict,
                            geo, transform, crs, rows: int, cols: int,
                            coord_sys: str = "None") -> plt.Figure:
    """
    Render the v13 ore-targeting 6-panel figure:
      [0] Stage 1 — Fertility
      [1] Stage 2 — Structural
      [2] Stage 3 — Silica Vector
      [3] Stage 4 — Ore Score
      [4] Convergence Index
      [5] 4-class Ore Classification Map
    """
    fig = plt.figure(figsize=(26, 16), facecolor=BG)
    fig.suptitle(
        "🎯  v13 ORE TARGETING ENGINE  —  Structurally-Constrained Drill Targets",
        color=GOLD, fontsize=15, fontweight="bold", y=0.98)

    n_target = ore_result["n_drill_targets"]
    t_pct    = ore_result["target_pct"]
    sat      = ore_result["saturation_penalty"]
    pm2      = PIXEL_SIZE_M**2 / 1e6

    fig.text(0.5, 0.965,
             f"Drill Targets: {n_target:,} px  ({ore_result['target_area_km2']:.2f} km²)  "
             f"[{t_pct:.2f}% of scene]  ·  Saturation penalty: {sat:.2f}x  ·  "
             f"4-Stage Refinement: Fertility → Structure → Silica → Localisation",
             ha="center", fontsize=9, color="#aaa", fontfamily="monospace")

    gs = GridSpec(2, 3, figure=fig,
                  left=0.03, right=0.97, top=0.95, bottom=0.07,
                  hspace=0.38, wspace=0.22)

    panels = [
        (gs[0, 0], ore_result["stage1_fertility"],  "YlOrBr",    "Stage 1 — Hydrothermal Fertility",    "Score"),
        (gs[0, 1], ore_result["structural_index"],   "plasma",    "Stage 2 — Structural Index",          "Score"),
        (gs[0, 2], ore_result["silica_vector"],       "inferno",   "Stage 3 — Silica Ore-Vector",         "Score"),
        (gs[1, 0], ore_result["stage4_ore"],          "hot",       "Stage 4 — Ore Localisation Score",    "Score"),
        (gs[1, 1], ore_result["convergence"],         "RdYlGn_r",  "Anomaly Convergence Index",           "Score"),
    ]
    for gs_p, arr_p, cmap_p, title_p, lbl_p in panels:
        ax = fig.add_subplot(gs_p)
        vlo = float(np.nanpercentile(arr_p[~np.isnan(arr_p)], 2)) if not np.all(np.isnan(arr_p)) else 0
        vhi = float(np.nanpercentile(arr_p[~np.isnan(arr_p)], 98)) if not np.all(np.isnan(arr_p)) else 1
        im_ = ax.imshow(arr_p, cmap=cmap_p, vmin=vlo, vmax=vhi)
        style_ax(ax, title_p, fs=11)
        add_cbar(fig, im_, ax, lbl_p)
        add_north_arrow(ax); add_scalebar(ax, rows, cols)
        if coord_sys != "None":
            add_coord_grid(ax, transform, rows, cols, crs, mode=coord_sys)
        else:
            ax.axis("off")
        ax.text(0.02, 0.985, f"μ={float(np.nanmean(arr_p)):.3f}  σ={float(np.nanstd(arr_p)):.3f}",
                transform=ax.transAxes, color="#ccc", fontsize=6, va="top",
                bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))

    # Panel 6: 4-class classification map
    ax6 = fig.add_subplot(gs[1, 2])
    c4 = [_ORE_CLASS_COLORS[i] for i in range(4)]
    cmap4 = mcolors.ListedColormap(c4)
    ax6.imshow(ore_result["ore_class_map"], cmap=cmap4, vmin=0, vmax=3, interpolation="nearest")
    style_ax(ax6, "🎯  Ore Classification (4 Classes)", fs=11)
    add_north_arrow(ax6); add_scalebar(ax6, rows, cols)
    if coord_sys != "None":
        add_coord_grid(ax6, transform, rows, cols, crs, mode=coord_sys)
    else:
        ax6.axis("off")
    legs4 = [
        Patch(fc=c4[3], label=f"Class 3 — Drill Target  ({n_target:,} px  {ore_result['target_area_km2']:.2f} km²)"),
        Patch(fc=c4[2], label=f"Class 2 — Fertile Corridor  ({ore_result['n_corridors']:,} px)"),
        Patch(fc=c4[1], label=f"Class 1 — Alteration Zone  ({ore_result['n_alteration']:,} px)"),
        Patch(fc=c4[0], label="Class 0 — Background"),
    ]
    ax6.legend(handles=legs4, loc="lower right", facecolor="#111",
               edgecolor="#444", labelcolor="white", fontsize=8, framealpha=0.92)

    # Saturation warning badge
    color_badge = "#e63946" if t_pct > 10 else "#52b788"
    ax6.text(0.02, 0.97,
             f"Target: {t_pct:.2f}%  {'⚠ SATURATION' if t_pct > 10 else '✔ OK'}",
             transform=ax6.transAxes, color=color_badge, fontsize=8, va="top",
             bbox=dict(fc=BG, ec=color_badge, lw=0.8, pad=3))

    stamp_map(fig, geo, "v13 Ore Targeting Engine")
    return fig


def run_ore_targeting(s2_file, aster_file, coord_sys, save_out,
                      base_prob_map: np.ndarray = None):
    """
    Gradio callback for the v13 Ore Targeting Engine tab.
    Returns (png_path, tif_path, markdown_report).
    """
    if s2_file is None:
        return None, None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        s2_path = _fpath(s2_file)
        stack, transform, crs, rows, cols, profile = _read_s2(s2_path)
        with rasterio.open(s2_path) as _src:
            geo = geo_info_dict(_src)

        aster_dict = None
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw_a = asrc.read().astype("float32")[:6]
                    aster_r = resample_aster_to_sentinel(
                        raw_a, asrc.transform, asrc.crs, transform, crs, rows, cols)
                aster_dict = compute_aster_indices(aster_r)
            except Exception as ae:
                aster_dict = None
                print(f"[OreEngine] ASTER load warning: {ae}")

        ore = run_ore_targeting_engine(stack, aster_dict, base_prob_map)

        fig = plot_ore_targeting_map(ore, geo, transform, crs, rows, cols, coord_sys)
        out_png = _out_path("maps", "ore_targeting_v13", ".png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        tif_out = None
        if save_out:
            tif_out = _out_path("geotiff", "ore_targeting_v13", ".tif")
            prof = {k: profile[k] for k in ("crs", "transform", "width", "height") if k in profile}
            prof.update({"driver": "GTiff", "dtype": "float32", "nodata": float("nan"),
                         "count": 5, "compress": "lzw"})
            with rasterio.open(tif_out, "w", **prof) as dst:
                dst.write(ore["stage1_fertility"], 1)
                dst.write(ore["stage2_structural"], 2)
                dst.write(ore["stage3_silica"],     3)
                dst.write(ore["stage4_ore"],        4)
                dst.write(ore["ore_class_map"],     5)
                dst.update_tags(band1="stage1_fertility", band2="stage2_structural",
                                band3="stage3_silica",    band4="stage4_ore_score",
                                band5="ore_class_0to3")

        sat_warn = ""
        if ore["target_pct"] > 10:
            sat_warn = (f"\n\n> ⚠️  **Saturation Warning**: {ore['target_pct']:.1f}% of scene "
                        "classified as high-priority (>10% threshold).  Confidence-saturation "
                        f"penalty applied: **{ore['saturation_penalty']:.2f}×**.  "
                        "Consider increasing structural or silica threshold, or refining "
                        "input data quality.")
        elif ore["target_pct"] < 0.1:
            sat_warn = ("\n\n> ℹ️  Target coverage < 0.1% — model is very strict.  "
                        "Consider reviewing structural data quality or lowering thresholds.")

        md = f"""### 🎯 v13 Ore Targeting Engine — Results

| Stage | Description | Max Score |
|---|---|---|
| Stage 1 | Hydrothermal Fertility | {float(np.nanmax(ore['stage1_fertility'])):.3f} |
| Stage 2 | Structural × Fertility | {float(np.nanmax(ore['stage2_structural'])):.3f} |
| Stage 3 | Silica × Structural    | {float(np.nanmax(ore['stage3_silica'])):.3f} |
| Stage 4 | Ore Localisation Score | {float(np.nanmax(ore['stage4_ore'])):.3f} |

| Class | Description | Count | Area (km²) | % scene |
|---|---|---|---|---|
| 3 🔴 | **Drill Target** | {ore['n_drill_targets']:,} | {ore['target_area_km2']:.3f} | {ore['target_pct']:.2f}% |
| 2 🟠 | Fertile Structural Corridor | {ore['n_corridors']:,} | {ore['n_corridors'] * (PIXEL_SIZE_M**2/1e6):.3f} | — |
| 1 🔵 | Alteration Zone | {ore['n_alteration']:,} | {ore['n_alteration'] * (PIXEL_SIZE_M**2/1e6):.3f} | — |
| 0 ⬛ | Background | — | — | — |

| Metric | Value |
|---|---|
| Target threshold (p92) | {ore['thr_target']:.4f} |
| Saturation penalty | {ore['saturation_penalty']:.3f}× |
| Structural index max | {float(np.nanmax(ore['structural_index'])):.3f} |
| Silica vector max | {float(np.nanmax(ore['silica_vector'])):.3f} |
| Convergence max | {float(np.nanmax(ore['convergence'])):.3f} |

**Ore Target Rule (all must be true):**
- ✔ High structural density → Stage 2 gate
- ✔ Silica anomaly present → Stage 3 gate
- ✔ Localised (not diffuse) alteration → Stage 4 localisation
- ✔ Spatial convergence (compact clustering) → convergence multiplier
- ✔ Area compact (<5 km² preferred)
{sat_warn}

*Target coverage {ore['target_pct']:.2f}% (ideal: 0.1–3%, failure: >10%)*
"""
        return out_png, tif_out, md

    except Exception:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ══════════════════════════════════════════════════════════════════════
#  SPATIAL STATISTICS (v10 new)
# ══════════════════════════════════════════════════════════════════════

def compute_morans_i(arr: np.ndarray, n_sample: int = 10_000) -> float:
    """
    Approximate Moran's I spatial autocorrelation on a 2-D raster.
    Uses a random pixel sample + queen contiguity weights for speed.
    Returns Moran's I value in [-1, 1].
    """
    try:
        flat = arr.flatten()
        valid = np.where(~np.isnan(flat))[0]
        if len(valid) < 500:
            return float("nan")
        rng = np.random.default_rng(42)
        idx = rng.choice(valid, size=min(n_sample, len(valid)), replace=False)
        rows_idx = idx // arr.shape[1]
        cols_idx = idx % arr.shape[1]
        z = flat[idx] - np.nanmean(flat[idx])
        n = len(z)
        # Build spatial lags using 8-neighbourhood queen weights (rook approx via shift)
        z_map = np.full_like(arr, np.nan)
        z_map_flat = z_map.reshape(-1)
        z_map_flat[idx] = z
        z_map = z_map_flat.reshape(arr.shape)
        # Spatial lag = mean of queen neighbours; use 'nearest' boundary mode
        # so edge pixels are mirrored rather than zero-padded (zero-padding
        # would pull the lag toward 0 at the scene boundary and distort I).
        lag = uniform_filter(np.nan_to_num(z_map, nan=0.0), size=3, mode="nearest")
        lag_vals = lag.reshape(-1)[idx]
        w_sum = min(n * 8, n * 8)  # queen contiguity: up to 8 neighbours per sampled cell
        # Correct Moran's I formula: I = (N / W) * (sum_ij w_ij z_i z_j) / (sum_i z_i²)
        # With uniform_filter approximation: sum_ij w_ij z_i z_j ≈ n * sum(z * lag_z) / 9
        # where 9 is the 3×3 window size; W ≈ n*8 (queen weights, excluding self)
        num = np.sum(z * lag_vals) * (n / 9)
        denom = np.sum(z * z)
        if denom < 1e-12:
            return float("nan")
        moran = (n / max(w_sum, 1)) * (num / denom)
        return float(np.clip(moran, -1, 1))
    except Exception as _e:
        return float("nan")


def compute_getis_ord_hotspot(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Compute a local Getis-Ord Gi* statistic approximation.
    Returns a z-score map: +high = spatial hotspot, -low = cold spot.
    kernel: neighbourhood window (pixels, square).
    """
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan, dtype=np.float32)
    fill = float(np.nanmean(arr))
    clean = np.nan_to_num(arr, nan=fill)
    # local_sum: uniform_filter returns the local mean; multiply by n to get the sum
    local_sum  = uniform_filter(clean, size=kernel) * kernel**2
    global_mean = np.nanmean(clean)
    global_std  = np.nanstd(clean)
    if global_std < 1e-9:
        return np.zeros_like(arr)
    n = kernel * kernel
    N = clean.size  # total number of pixels
    s_sq = np.nanmean(clean**2) - global_mean**2
    s    = np.sqrt(max(s_sq, 0))
    num  = local_sum - n * global_mean
    denom = s * np.sqrt(((N * n - n**2) / (N - 1)) + 1e-9)
    gi_star = num / (denom + 1e-9)
    gi_star[np.isnan(arr)] = np.nan
    return gi_star.astype(np.float32)


def compute_threshold_optimal(prob_map: np.ndarray,
                               label_arr = None) -> dict:
    """
    Compute optimal probability thresholds using multiple criteria.
    If label_arr is provided: uses Youden-J + F1-optimal on labelled data.
    Otherwise: uses percentile heuristics.
    Returns dict: {youden, f1_opt, p80, p90, p95}
    """
    flat = prob_map[~np.isnan(prob_map)].flatten()
    if flat.size == 0:
        return {"youden": 0.5, "f1_opt": 0.5, "p80": 0.8, "p90": 0.9, "p95": 0.95}
    thr = {
        "p70": float(np.percentile(flat, 70)),
        "p80": float(np.percentile(flat, 80)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
    }
    if label_arr is not None:
        from sklearn.metrics import roc_curve, f1_score
        try:
            lbl_flat = label_arr.flatten()
            pmask = ~np.isnan(prob_map.flatten()) & np.isin(lbl_flat, [0, 1])
            y_true = lbl_flat[pmask].astype(int)
            y_prob = prob_map.flatten()[pmask]
            if len(np.unique(y_true)) == 2:
                fpr, tpr, thrs = roc_curve(y_true, y_prob)
                youden = thrs[np.argmax(tpr - fpr)]
                f1s = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
                       for t in thrs]
                thr["youden"] = float(youden)
                thr["f1_opt"] = float(thrs[np.argmax(f1s)])
        except Exception as _e:
            thr["youden"] = thr["p80"]
            thr["f1_opt"] = thr["p80"]
    else:
        thr["youden"] = thr["p80"]
        thr["f1_opt"] = thr["p80"]
    return thr


def run_spatial_analysis(s2_file, aster_file, analysis_type, kernel_size):
    """
    Tab 11: Spatial Statistics — Moran's I + Getis-Ord hotspot + density.
    """
    if s2_file is None:
        return None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        # Choose index based on analysis_type
        idx_map = {
            "Iron Oxide (IO)": stack[9],
            "Clay Minerals (CM)": stack[10],
            "Gossan (GS)": stack[12],
            "Ferrous Iron (FI)": stack[11],
            "NDVI": stack[13],
            "Elevation": stack[14],
            "Slope": stack[15],
        }
        aster_dict = None
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw_a = asrc.read().astype("float32")[:6]
                    a_transform = asrc.transform; a_crs = asrc.crs
                raw_a_r = resample_aster_to_sentinel(raw_a, a_transform, a_crs,
                                                      transform, crs, rows, cols)
                aster_dict = compute_aster_indices(raw_a_r)
                idx_map["AST_HydAlt"] = aster_dict["AST_HydAlt"]
                idx_map["AST_AlOH"] = aster_dict["AST_AlOH"]
            except Exception as _e:
                import warnings; warnings.warn(f"⚠️ ASTER idx failed (HydAlt/AlOH missing): {_e}")

        chosen_arr = idx_map.get(analysis_type, stack[9])
        moran_i  = compute_morans_i(chosen_arr)
        gi_star  = compute_getis_ord_hotspot(chosen_arr, kernel=int(kernel_size))
        thr_opt  = compute_threshold_optimal(robust_scale(chosen_arr))

        fig = plt.figure(figsize=(24, 16), facecolor=BG)
        fig.suptitle(f"📐 Spatial Statistics — {analysis_type}  ·  Moran's I = {moran_i:.4f}",
                     color=GOLD, fontsize=16, fontweight="bold", y=0.98)
        gs = GridSpec(2, 3, figure=fig, left=0.04, right=0.97,
                      top=0.93, bottom=0.07, hspace=0.35, wspace=0.25)

        # Panel 1: Raw index
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(chosen_arr, cmap="plasma",
                         vmin=np.nanpercentile(chosen_arr, 2),
                         vmax=np.nanpercentile(chosen_arr, 98))
        style_ax(ax1, f"{analysis_type}", fs=11)
        add_cbar(fig, im1, ax1, "Value"); add_north_arrow(ax1); add_scalebar(ax1, rows, cols)
        ax1.axis("off")

        # Panel 2: Getis-Ord Gi* hotspot
        ax2 = fig.add_subplot(gs[0, 1])
        vabs = float(np.nanpercentile(np.abs(gi_star[~np.isnan(gi_star)]), 98))
        im2 = ax2.imshow(gi_star, cmap="RdBu_r", vmin=-vabs, vmax=vabs)
        style_ax(ax2, f"Getis-Ord Gi* Hotspot  (kernel={kernel_size})", fs=11)
        add_cbar(fig, im2, ax2, "Z-score")
        ax2.axis("off")
        # Mark significant hotspots (|z| > 1.96)
        hot_mask = np.abs(gi_star) > 1.96
        ax2.contour(hot_mask.astype(float), levels=[0.5], colors=["#ffff00"], linewidths=0.7)

        # Panel 3: Kernel density map (local mean)
        density = uniform_filter(np.nan_to_num(chosen_arr, nan=float(np.nanmean(chosen_arr))),
                                 size=int(kernel_size))
        density[np.isnan(chosen_arr)] = np.nan
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(density, cmap="hot",
                         vmin=np.nanpercentile(density, 5),
                         vmax=np.nanpercentile(density, 95))
        style_ax(ax3, "Local Density (Kernel Mean)", fs=11)
        add_cbar(fig, im3, ax3, "Mean"); add_north_arrow(ax3); add_scalebar(ax3, rows, cols)
        ax3.axis("off")

        # Panel 4: Distribution + Moran scatter (quadrant)
        ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor(BG)
        flat = chosen_arr[~np.isnan(chosen_arr)].flatten()
        counts, edges = np.histogram(flat, bins=80)
        colors_h = plt.cm.plasma(np.linspace(0, 1, 80))
        for c, col, e0, e1 in zip(counts, colors_h, edges[:-1], edges[1:]):
            ax4.bar(e0, c, width=(e1-e0), color=col, edgecolor="none", alpha=0.85)
        for k, tv in thr_opt.items():
            ax4.axvline(tv, lw=1, ls="--", label=f"{k}: {tv:.3f}", alpha=0.8)
        ax4.set_title("Distribution + Optimal Thresholds", color="white", fontsize=10)
        ax4.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=7)
        ax4.tick_params(colors="#777", labelsize=7)
        for sp in ax4.spines.values(): sp.set_edgecolor("#333")

        # Panel 5: Moran scatter plot (z vs spatial lag z)
        ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(BG)
        lag_map = uniform_filter(np.nan_to_num(chosen_arr, nan=0.0), size=3)
        z_vals = (chosen_arr - np.nanmean(chosen_arr)).flatten()
        lag_z  = (lag_map - np.nanmean(lag_map)).flatten()
        vmask  = ~np.isnan(z_vals) & ~np.isnan(lag_z)
        step = max(1, int(vmask.sum()) // 6000)
        ax5.scatter(z_vals[vmask][::step], lag_z[vmask][::step],
                    s=2, alpha=0.3, color=GOLD)
        ax5.axhline(0, color="#555", lw=0.8); ax5.axvline(0, color="#555", lw=0.8)
        ax5.set_xlabel("z (value)", color="#aaa"); ax5.set_ylabel("Spatial lag z", color="#aaa")
        ax5.set_title(f"Moran Scatter  (I = {moran_i:.4f})", color="white", fontsize=10)
        ax5.tick_params(colors="#777", labelsize=7)
        for sp in ax5.spines.values(): sp.set_edgecolor("#333")

        # Panel 6: Hotspot classification (LISA categories)
        gi_clean = np.nan_to_num(gi_star, nan=0.0)
        lisa_map = np.zeros((rows, cols), dtype=np.float32)
        lisa_map[(gi_clean > 1.96)] = 3   # High-High hotspot
        lisa_map[(gi_clean < -1.96)] = 1  # Low-Low cold spot
        lisa_map[(gi_clean > 1.0) & (gi_clean <= 1.96)] = 2  # Uncertain high
        lisa_map[np.isnan(chosen_arr)] = np.nan
        ax6 = fig.add_subplot(gs[1, 2])
        cmap_lisa = mcolors.ListedColormap(["#1a3a2a", "#4a90d9", "#f4a261", "#e63946"])
        ax6.imshow(lisa_map, cmap=cmap_lisa, vmin=0, vmax=3, interpolation="nearest")
        style_ax(ax6, "LISA Hotspot Classification", fs=11)
        legs_l = [Patch(fc="#e63946", label="High-High hotspot (>1.96σ)"),
                  Patch(fc="#f4a261", label="Uncertain high (1–1.96σ)"),
                  Patch(fc="#4a90d9", label="Cold spot (<−1.96σ)"),
                  Patch(fc="#1a3a2a", label="Not significant")]
        ax6.legend(handles=legs_l, loc="lower right", facecolor="#111",
                   edgecolor="#444", labelcolor="white", fontsize=8, framealpha=0.9)
        ax6.axis("off")

        stamp_map(fig, geo, f"Spatial Analysis — {analysis_type}")
        tmp_png_name = _out_path("maps", "spatial_stats", ".png")
        plt.savefig(tmp_png_name, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        n_hot = int(np.nansum(gi_star > 1.96))
        n_cold = int(np.nansum(gi_star < -1.96))
        pm2 = PIXEL_SIZE_M**2 / 1e6
        thr_str = "\n".join(f"| {k} | {v:.4f} |" for k, v in thr_opt.items())
        md = f"""
### 📐 Spatial Statistics — {analysis_type}

| Metric | Value |
|---|---|
| **Moran's I** | **{moran_i:.4f}** {'(positive spatial autocorrelation)' if moran_i > 0 else '(negative – dispersed)'} |
| **Hotspot pixels** (|Gi*| > 1.96σ) | {n_hot:,} ({n_hot*pm2:.2f} km²) |
| **Cold-spot pixels** | {n_cold:,} ({n_cold*pm2:.2f} km²) |
| **Kernel window** | {kernel_size}×{kernel_size} px ({kernel_size*PIXEL_SIZE_M/1000:.2f} km) |

#### Optimal Probability Thresholds
| Criterion | Threshold |
|---|---|
{thr_str}
"""
        return tmp_png_name, md
    except Exception as _e:
        return None, f"### ❌\n```\n{traceback.format_exc()}\n```"


def engineer_aster_features(aster: np.ndarray, precomputed_indices: dict = None) -> np.ndarray:
    """
    Flatten all ASTER bands + computed indices into feature matrix columns.
    v10: Returns shape (N, 6+12=18) for N pixels (was 6+7=13 in v9).
    Accepts pre-computed indices dict to avoid redundant computation.
    """
    H, W = aster.shape[1], aster.shape[2]
    N = H * W
    bands_flat = aster.reshape(6, N).T        # (N, 6)
    idx_dict = precomputed_indices if precomputed_indices is not None else compute_aster_indices(aster)
    idx_flat = np.column_stack([
        np.nan_to_num(idx_dict[k].flatten()) for k in sorted(idx_dict.keys())
    ])                                         # (N, 12)
    return np.hstack([bands_flat, idx_flat]).astype(np.float32)   # (N, 18)


def build_hybrid_features(s2_stack: np.ndarray,
                           aster_stack,
                           mode: str = "Hybrid Sentinel-2 + ASTER",
                           aster_indices: dict = None) -> np.ndarray:
    """
    Build combined feature matrix from Sentinel-2 + ASTER SWIR.
    mode: 'Sentinel-2 Only' | 'ASTER Only' | 'Hybrid Sentinel-2 + ASTER'
    aster_indices: pre-computed dict from compute_aster_indices (avoids redundant computation)
    Returns (N, F) array where F depends on mode.
    """
    H, W = s2_stack.shape[1], s2_stack.shape[2]
    N = H * W
    s2_flat = s2_stack.reshape(s2_stack.shape[0], N).T   # (N, 18)

    if mode == "Sentinel-2 Only" or aster_stack is None:
        return s2_flat

    aster_feat = engineer_aster_features(aster_stack, aster_indices)   # (N, 18)

    if mode == "ASTER Only":
        return aster_feat

    # Hybrid: 18 S2 + 18 ASTER (6 bands + 12 indices)
    return np.hstack([s2_flat, aster_feat])               # (N, 36)


# ══════════════════════════════════════════════════════════════════════
#  v11 GIS EXPERT UPGRADES
# ══════════════════════════════════════════════════════════════════════

# ── 1. SPECTRAL ANGLE MAPPER (SAM) for target mineral matching ────────
SAM_ENDMEMBERS = {
    "Kaolinite (Al-OH)":    np.array([0.25, 0.30, 0.18, 0.22, 0.19, 0.30, 0.35,  # S2 B02-B12
                                       1.12, 0.68, 1.80, 1.10, 1.40], dtype=np.float32),
    "Goethite (Iron Oxide)": np.array([0.08, 0.12, 0.28, 0.22, 0.18, 0.20, 0.21,
                                        0.82, 0.62, 3.50, 0.82, 1.32], dtype=np.float32),
    "Chlorite (Propylitic)": np.array([0.06, 0.10, 0.10, 0.12, 0.14, 0.24, 0.25,
                                        0.85, 0.72, 1.16, 0.90, 1.18], dtype=np.float32),
    "Quartz Vein (Silica)":  np.array([0.32, 0.38, 0.40, 0.42, 0.44, 0.50, 0.52,
                                        1.05, 0.90, 1.24, 1.05, 1.17], dtype=np.float32),
}

def compute_sam_map(stack: np.ndarray, endmember_name: str) -> np.ndarray:
    """
    Compute Spectral Angle Mapper (SAM) distance map for a given mineral endmember.
    Lower angle = higher spectral similarity = better match.
    Uses S2 bands B02,B03,B04,B05,B06,B8A,B08,B11,B12 + IO,CM,FI (12 features).
    Returns angle map in degrees (0-90).
    """
    em = SAM_ENDMEMBERS.get(endmember_name)
    if em is None:
        return np.full(stack.shape[1:], np.nan, dtype=np.float32)
    # Use first 12 bands from stack (reflectance + 3 indices)
    n_feat = min(12, stack.shape[0], len(em))
    s2_sub = stack[:n_feat].reshape(n_feat, -1).T.astype(np.float32)   # (N, n_feat)
    em_sub = em[:n_feat]
    dot    = np.dot(s2_sub, em_sub)
    norm_s = np.linalg.norm(s2_sub, axis=1) + 1e-9
    norm_e = np.linalg.norm(em_sub) + 1e-9
    cos_th = np.clip(dot / (norm_s * norm_e), -1.0, 1.0)
    angle  = np.degrees(np.arccos(cos_th))
    angle[np.any(np.isnan(s2_sub), axis=1)] = np.nan
    return angle.reshape(stack.shape[1], stack.shape[2]).astype(np.float32)


def run_sam_analysis(s2_file, aster_file, endmember_name, coord_sys, save_out):
    """Tab: SAM mineral matching vs spectral library."""
    if s2_file is None:
        return None, None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        all_angles = {em: compute_sam_map(stack, em) for em in SAM_ENDMEMBERS}
        target_angle = all_angles[endmember_name]

        # Best-match mineral per pixel (lowest angle across all endmembers)
        angle_stack = np.stack([all_angles[k] for k in SAM_ENDMEMBERS], axis=0)
        best_match  = np.argmin(np.nan_to_num(angle_stack, nan=999), axis=0).astype(np.float32)
        best_match[np.all(np.isnan(angle_stack), axis=0)] = np.nan

        pm2 = PIXEL_SIZE_M**2 / 1e6
        thr = float(np.nanpercentile(target_angle, 30))   # bottom 30% = best match
        n_match = int(np.nansum(target_angle <= thr))

        fig = plt.figure(figsize=(22, 12), facecolor=BG)
        fig.suptitle(f"Spectral Angle Mapper — {endmember_name}  (lower angle = better match)",
                     color=GOLD, fontsize=14, fontweight="bold", y=0.98)
        gs  = GridSpec(2, 3, figure=fig, left=0.04, right=0.97,
                       top=0.93, bottom=0.08, hspace=0.35, wspace=0.28)

        # Panel 1: target SAM angle
        ax1 = fig.add_subplot(gs[0, :2])
        vmin, vmax = np.nanpercentile(target_angle, [2, 98])
        im1 = ax1.imshow(target_angle, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        style_ax(ax1, f"SAM Angle — {endmember_name}  (°)", fs=12)
        add_cbar(fig, im1, ax1, "Angle (°)"); add_north_arrow(ax1); add_scalebar(ax1, rows, cols)
        if coord_sys != "None": add_coord_grid(ax1, transform, rows, cols, crs, mode=coord_sys)
        else: ax1.axis("off")
        # Highlight best-match zone
        match_mask = np.ma.masked_where(target_angle > thr, target_angle * 0)
        ax1.imshow(match_mask, cmap=mcolors.ListedColormap(["#00ffcc"]), alpha=0.45, vmin=0, vmax=1)
        ax1.text(0.02, 0.97, f"Best-match pixels (SAM ≤ {thr:.1f}°): {n_match:,}  ({n_match*pm2:.2f} km²)",
                 transform=ax1.transAxes, color="#00ffcc", fontsize=8, va="top",
                 bbox=dict(fc=BG, ec="#333", lw=0.5, pad=3))

        # Panel 2: Best mineral per pixel
        ax2 = fig.add_subplot(gs[0, 2])
        em_names = list(SAM_ENDMEMBERS.keys())
        cmap_em  = mcolors.ListedColormap(["#e63946","#f4a261","#52b788","#a2d2ff"])
        im2 = ax2.imshow(best_match, cmap=cmap_em, vmin=0, vmax=len(em_names)-1,
                         interpolation="nearest")
        style_ax(ax2, "Best-Match Mineral Map", fs=11)
        patches = [Patch(fc=cmap_em(i/(len(em_names)-1)), label=n[:24])
                   for i, n in enumerate(em_names)]
        ax2.legend(handles=patches, loc="lower right", facecolor="#111",
                   edgecolor="#444", labelcolor="white", fontsize=7, framealpha=0.9)
        ax2.axis("off")

        # Panels 3-5: angle maps for the remaining endmembers
        others = [k for k in SAM_ENDMEMBERS if k != endmember_name]
        for i, other in enumerate(others[:3]):
            ax = fig.add_subplot(gs[1, i])
            a_other = all_angles[other]
            vm1, vm2 = np.nanpercentile(a_other, [2, 98])
            im_ = ax.imshow(a_other, cmap="plasma_r", vmin=vm1, vmax=vm2)
            style_ax(ax, f"SAM — {other[:28]}", fs=9)
            add_cbar(fig, im_, ax, "°")
            ax.axis("off")

        stamp_map(fig, geo, f"SAM: {endmember_name}")
        out_png = _out_path("maps", "sam_analysis", ".png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        tif_out = None
        if save_out:
            tif_out = _out_path("geotiff", "sam_angle", ".tif")
            prof = {k: profile[k] for k in ("crs","transform","width","height") if k in profile}
            prof.update({"driver":"GTiff","dtype":"float32","nodata":float("nan"),"count":1,"compress":"lzw"})
            with rasterio.open(tif_out, "w", **prof) as dst:
                dst.write(target_angle, 1)

        md = f"""### 🔬 SAM Analysis — {endmember_name}

| Metric | Value |
|---|---|
| **Endmember target** | {endmember_name} |
| **SAM threshold (p30)** | {thr:.2f}° |
| **Best-match pixels** | {n_match:,} ({n_match*pm2:.2f} km²) |
| **Mean angle** | {float(np.nanmean(target_angle)):.2f}° |
| **Min angle (best)** | {float(np.nanmin(target_angle)):.2f}° |

*Lower spectral angle = better match with mineral signature.*
*Highlighted cyan pixels are best candidates for {endmember_name}.*
"""
        return out_png, tif_out, md
    except Exception:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── 2. PCA ANOMALY DETECTION ───────────────────────────────────────────
def compute_pca_anomaly(stack: np.ndarray, n_components: int = 3) -> tuple:
    """
    Run PCA on the raster stack; return PC1-PC3 maps + anomaly distance map.
    Anomaly = pixels far from the centroid in PC space (Mahalanobis-like).
    Returns: (pc_maps: list of 2D arrays, anomaly_map: 2D array)
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import RobustScaler
        H, W = stack.shape[1], stack.shape[2]
        flat = stack.reshape(stack.shape[0], H*W).T.astype(np.float32)
        valid = ~np.any(np.isnan(flat), axis=1)
        if valid.sum() < 50:
            return [], np.full((H, W), np.nan, dtype=np.float32)
        X_valid = RobustScaler().fit_transform(flat[valid])
        nc = min(n_components, X_valid.shape[1], X_valid.shape[0])
        pca   = PCA(n_components=nc, random_state=42)
        scores = pca.fit_transform(X_valid)
        # Anomaly: Euclidean distance from centroid in PC space
        anomaly = np.sqrt(np.sum(scores**2, axis=1))
        pc_maps = []
        for i in range(nc):
            m = np.full(H*W, np.nan, dtype=np.float32)
            m[valid] = scores[:, i]
            pc_maps.append(m.reshape(H, W))
        anom_full = np.full(H*W, np.nan, dtype=np.float32)
        anom_full[valid] = anomaly
        var_explained = pca.explained_variance_ratio_ * 100
        return pc_maps, anom_full.reshape(H, W), var_explained
    except Exception as _e:
        H, W = stack.shape[1], stack.shape[2]
        return [], np.full((H, W), np.nan, dtype=np.float32), []


def run_pca_anomaly(s2_file, aster_file, n_comp, coord_sys, save_out):
    """Tab: PCA anomaly detection on the full feature stack."""
    if s2_file is None:
        return None, None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        # Optionally append ASTER
        full_stack = stack
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw_a = asrc.read().astype("float32")[:6]
                    aster_r = resample_aster_to_sentinel(raw_a, asrc.transform, asrc.crs,
                                                          transform, crs, rows, cols)
                full_stack = np.vstack([stack, aster_r])
            except Exception:
                pass

        nc_int = int(n_comp)
        pc_maps, anomaly, var_exp = compute_pca_anomaly(full_stack, nc_int)
        if not pc_maps:
            return None, None, "### ❌ PCA failed — not enough valid pixels."

        pm2 = PIXEL_SIZE_M**2 / 1e6
        thr_anom = float(np.nanpercentile(anomaly, 90))
        n_anom   = int(np.nansum(anomaly >= thr_anom))

        n_panels = min(nc_int, len(pc_maps))
        fig = plt.figure(figsize=(24, 14), facecolor=BG)
        title_var = "  |  ".join([f"PC{i+1}: {var_exp[i]:.1f}%" for i in range(len(var_exp))])
        fig.suptitle(f"PCA Anomaly Detection — {title_var}",
                     color=GOLD, fontsize=13, fontweight="bold", y=0.98)
        gs = GridSpec(2, max(n_panels, 2), figure=fig, left=0.04, right=0.97,
                      top=0.93, bottom=0.07, hspace=0.35, wspace=0.22)

        for i, pc in enumerate(pc_maps[:n_panels]):
            ax = fig.add_subplot(gs[0, i])
            vlo, vhi = np.nanpercentile(pc, [2, 98])
            im_ = ax.imshow(pc, cmap="RdBu_r", vmin=vlo, vmax=vhi)
            style_ax(ax, f"PC{i+1}  ({var_exp[i]:.1f}% var)" if i < len(var_exp) else f"PC{i+1}", fs=11)
            add_cbar(fig, im_, ax, "Score"); add_north_arrow(ax); add_scalebar(ax, rows, cols)
            if coord_sys != "None": add_coord_grid(ax, transform, rows, cols, crs, mode=coord_sys)
            else: ax.axis("off")

        # Anomaly map bottom-left (spans 2 cols if only 2 PCs)
        ax_a = fig.add_subplot(gs[1, :2])
        vlo_a, vhi_a = np.nanpercentile(anomaly, [2, 98])
        im_a = ax_a.imshow(anomaly, cmap="inferno", vmin=vlo_a, vmax=vhi_a)
        style_ax(ax_a, "Spectral Anomaly Index (PCA Distance from Centroid)", fs=12)
        add_cbar(fig, im_a, ax_a, "Anomaly"); add_north_arrow(ax_a); add_scalebar(ax_a, rows, cols)
        if coord_sys != "None": add_coord_grid(ax_a, transform, rows, cols, crs, mode=coord_sys)
        else: ax_a.axis("off")
        anom_highlight = np.ma.masked_where(anomaly < thr_anom, anomaly * 0)
        ax_a.imshow(anom_highlight, cmap=mcolors.ListedColormap(["#ffff00"]), alpha=0.50, vmin=0, vmax=1)
        ax_a.text(0.02, 0.97, f"Top-10% anomaly pixels: {n_anom:,}  ({n_anom*pm2:.2f} km²)",
                  transform=ax_a.transAxes, color="#ffff00", fontsize=8, va="top",
                  bbox=dict(fc=BG, ec="#333", lw=0.5, pad=3))

        # Scree plot bottom-right
        ax_s = fig.add_subplot(gs[1, 2:] if n_panels >= 3 else gs[1, 2])
        ax_s.set_facecolor(BG)
        ax_s.bar(range(1, len(var_exp)+1), var_exp, color=GOLD, edgecolor="none", alpha=0.9)
        ax_s.plot(range(1, len(var_exp)+1), np.cumsum(var_exp), "o-",
                  color="#52b788", lw=1.5, markersize=4, label="Cumulative %")
        ax_s.axhline(95, color="#e63946", lw=1, ls="--", alpha=0.7, label="95%")
        ax_s.set_xlabel("Principal Component", color="#aaa", fontsize=9)
        ax_s.set_ylabel("Variance Explained (%)", color="#aaa", fontsize=9)
        ax_s.set_title("Scree Plot", color=GOLD, fontweight="bold", fontsize=11)
        ax_s.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=8)
        ax_s.tick_params(colors="#777"); ax_s.set_facecolor(BG)
        for sp in ax_s.spines.values(): sp.set_edgecolor("#333")

        stamp_map(fig, geo, "PCA Anomaly Detection")
        out_png = _out_path("maps", "pca_anomaly", ".png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        tif_out = None
        if save_out:
            tif_out = _out_path("geotiff", "pca_anomaly", ".tif")
            prof = {k: profile[k] for k in ("crs","transform","width","height") if k in profile}
            prof.update({"driver":"GTiff","dtype":"float32","nodata":float("nan"),
                         "count":1+len(pc_maps),"compress":"lzw"})
            with rasterio.open(tif_out, "w", **prof) as dst:
                dst.write(anomaly, 1)
                for i, pc in enumerate(pc_maps):
                    dst.write(pc, i+2)

        md = f"""### 🔬 PCA Anomaly Detection

| Metric | Value |
|---|---|
| **Bands used** | {full_stack.shape[0]} |
| **Components** | {nc_int} |
| **PC1 variance** | {var_exp[0]:.1f}% |
| **Cumulative (PC1-PC{nc_int})** | {sum(var_exp[:nc_int]):.1f}% |
| **Anomaly threshold (p90)** | {thr_anom:.3f} |
| **High-anomaly pixels** | {n_anom:,} ({n_anom*pm2:.2f} km²) |

*Spectral anomalies (yellow highlight) are pixels diverging most from the scene mean spectrum — likely mineralised zones.*
"""
        return out_png, tif_out, md
    except Exception:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── 3. LINEAMENT / STRUCTURAL DENSITY MAP ─────────────────────────────
def compute_lineament_density(dem_arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Approximate lineament density from DEM edge detection.
    Uses Sobel gradient magnitude + local variance as a structural proxy.
    Returns density map (0-1 normalised).
    """
    try:
        from scipy.ndimage import sobel, generic_filter
        dem_clean = np.nan_to_num(dem_arr, nan=float(np.nanmean(dem_arr)))
        sx = sobel(dem_clean, axis=1)
        sy = sobel(dem_clean, axis=0)
        gradient_mag = np.sqrt(sx**2 + sy**2)
        # Local variance of gradient = lineament density proxy
        def _local_var(x): return np.var(x)
        density = generic_filter(gradient_mag, _local_var, size=kernel)
        # Normalise to 0-1
        lo, hi = np.nanpercentile(density, [1, 99])
        if hi > lo:
            density = np.clip((density - lo) / (hi - lo), 0, 1)
        return density.astype(np.float32)
    except Exception:
        return np.full_like(dem_arr, np.nan, dtype=np.float32)


def compute_bearing_map(slope_arr: np.ndarray, aspect_arr: np.ndarray) -> np.ndarray:
    """
    Compute a bearing-weighted slope map highlighting N–S structural trends
    (important for the Eastern Desert shear zones and ophiolite belts).
    Returns weighting: 1.0 = N/S bearing (most structurally significant),
                       0.0 = E/W bearing.
    """
    # Aspect is in degrees (0=N, 90=E, 180=S, 270=W in standard GIS convention)
    # Weight = |cos(aspect)|  →  max at N(0°)/S(180°), min at E(90°)/W(270°)
    bearing_weight = np.abs(np.cos(np.radians(aspect_arr)))
    bearing_weight[np.isnan(slope_arr) | np.isnan(aspect_arr)] = np.nan
    return bearing_weight.astype(np.float32)


def run_structural_analysis(s2_file, aster_file, kernel_size, coord_sys, save_out):
    """Tab: Lineament density + structural bearing analysis."""
    if s2_file is None:
        return None, None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        dem_arr   = stack[14]
        slope_arr = stack[15]
        asp_arr   = stack[16]
        rgh_arr   = stack[17]

        density  = compute_lineament_density(dem_arr, int(kernel_size))
        bearing  = compute_bearing_map(slope_arr, asp_arr)
        struct_idx = density * bearing  # combined structural prospectivity

        # Gold-structural correlation: iron oxide × structural density
        io_arr   = stack[9]
        gold_struct = robust_scale(io_arr) * density   # (0-1) × (0-1)

        pm2 = PIXEL_SIZE_M**2 / 1e6
        thr_struct = float(np.nanpercentile(struct_idx, 85))
        n_struct   = int(np.nansum(struct_idx >= thr_struct))

        fig = plt.figure(figsize=(24, 14), facecolor=BG)
        fig.suptitle("Structural & Lineament Analysis — Eastern Desert Shear Zones",
                     color=GOLD, fontsize=14, fontweight="bold", y=0.98)
        gs = GridSpec(2, 3, figure=fig, left=0.04, right=0.97,
                      top=0.93, bottom=0.08, hspace=0.35, wspace=0.25)

        panels_s = [
            (gs[0,0], dem_arr,    "terrain",    "DEM Elevation",             "m"),
            (gs[0,1], slope_arr,  "hot",        "Slope",                     "°"),
            (gs[0,2], density,    "plasma",     "Lineament Density (Sobel)", "0-1"),
            (gs[1,0], bearing,    "coolwarm",   "N–S Bearing Weight",        "0-1"),
            (gs[1,1], struct_idx, "inferno",    "Structural Prospectivity",  "0-1"),
            (gs[1,2], gold_struct,"YlOrRd",     "Fe-Oxide × Structure",      "0-1"),
        ]
        for gs_p, arr_p, cmap_p, title_p, lbl_p in panels_s:
            ax = fig.add_subplot(gs_p)
            vlo, vhi = np.nanpercentile(arr_p[~np.isnan(arr_p)], [2, 98]) \
                       if not np.all(np.isnan(arr_p)) else (0, 1)
            im_ = ax.imshow(arr_p, cmap=cmap_p, vmin=vlo, vmax=vhi)
            style_ax(ax, title_p, fs=11)
            add_cbar(fig, im_, ax, lbl_p)
            add_north_arrow(ax); add_scalebar(ax, rows, cols)
            if coord_sys != "None": add_coord_grid(ax, transform, rows, cols, crs, mode=coord_sys)
            else: ax.axis("off")
            ax.text(0.02, 0.985, f"μ={float(np.nanmean(arr_p)):.3f}  σ={float(np.nanstd(arr_p)):.3f}",
                    transform=ax.transAxes, color="#ccc", fontsize=6, va="top",
                    bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))

        # Overlay structural prospectivity zones on structure panel
        ax_st = fig.get_axes()[4]
        hi_mask = np.ma.masked_where(struct_idx < thr_struct, struct_idx * 0)
        ax_st.imshow(hi_mask, cmap=mcolors.ListedColormap(["#ffff00"]),
                     alpha=0.45, vmin=0, vmax=1)
        ax_st.text(0.02, 0.97, f"High-struct pixels: {n_struct:,} ({n_struct*pm2:.2f} km²)",
                   transform=ax_st.transAxes, color="#ffff00", fontsize=7, va="top",
                   bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))

        stamp_map(fig, geo, "Structural Analysis")
        out_png = _out_path("maps", "structural_analysis", ".png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        tif_out = None
        if save_out:
            tif_out = _out_path("geotiff", "structural_index", ".tif")
            prof = {k: profile[k] for k in ("crs","transform","width","height") if k in profile}
            prof.update({"driver":"GTiff","dtype":"float32","nodata":float("nan"),
                         "count":3,"compress":"lzw"})
            with rasterio.open(tif_out, "w", **prof) as dst:
                dst.write(density, 1); dst.write(struct_idx, 2); dst.write(gold_struct, 3)
                dst.update_tags(band1="lineament_density", band2="structural_idx", band3="fe_structure")

        md = f"""### 🏔️ Structural Analysis

| Metric | Value |
|---|---|
| **DEM mean elevation** | {float(np.nanmean(dem_arr)):.1f} m |
| **DEM max elevation** | {float(np.nanmax(dem_arr)):.1f} m |
| **Mean slope** | {float(np.nanmean(slope_arr)):.1f}° |
| **Lineament density kernel** | {kernel_size}×{kernel_size} px ({int(kernel_size)*PIXEL_SIZE_M/1000:.2f} km) |
| **High-structure threshold (p85)** | {thr_struct:.4f} |
| **High-structure area** | {n_struct:,} px ({n_struct*pm2:.2f} km²) |

*N–S bearing zones (Eastern Desert shear corridors) receive highest structural weight.*
*Fe-Oxide × Structure product highlights fault-controlled mineralisation targets.*
"""
        return out_png, tif_out, md
    except Exception:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── 4. MULTI-CRITERIA DECISION ANALYSIS (MCDA) ────────────────────────
def run_mcda(s2_file, aster_file, w_io, w_cm, w_aster, w_struct, w_elev,
             coord_sys, threshold, save_out):
    """
    Weighted overlay (MCDA) combining: Iron Oxide, Clay, ASTER alteration,
    Structural density, and Elevation constraints into a single prospectivity score.
    All layers normalised 0-1 before weighting.
    """
    if s2_file is None:
        return None, None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        # Normalise each criterion layer 0-1
        io_n     = robust_scale(stack[9])      # Iron oxide
        cm_n     = robust_scale(stack[10])     # Clay minerals
        elev_inv = 1.0 - robust_scale(stack[14])  # Invert elevation (lower = better)
        struct_n = compute_lineament_density(stack[14])

        # ASTER alteration layer
        aster_n = np.zeros((rows, cols), dtype=np.float32)
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw_a = asrc.read().astype("float32")[:6]
                    aster_r = resample_aster_to_sentinel(raw_a, asrc.transform, asrc.crs,
                                                          transform, crs, rows, cols)
                aster_dict = compute_aster_indices(aster_r)
                aster_n = robust_scale(aster_dict.get("AST_HydAlt", np.zeros((rows,cols))))
            except Exception:
                pass

        # Weights
        w_total = max(w_io + w_cm + w_aster + w_struct + w_elev, 1e-9)
        mcda = (w_io * io_n + w_cm * cm_n + w_aster * aster_n +
                w_struct * struct_n + w_elev * elev_inv) / w_total
        mcda = np.clip(mcda, 0, 1).astype(np.float32)
        mcda[np.isnan(io_n) & np.isnan(cm_n)] = np.nan

        pm2 = PIXEL_SIZE_M**2 / 1e6
        n_high = int(np.nansum(mcda >= threshold))

        fig = plt.figure(figsize=(24, 14), facecolor=BG)
        fig.suptitle(f"Multi-Criteria Decision Analysis (MCDA) — Weighted Prospectivity",
                     color=GOLD, fontsize=14, fontweight="bold", y=0.98)
        gs = GridSpec(2, 3, figure=fig, left=0.04, right=0.97,
                      top=0.93, bottom=0.08, hspace=0.35, wspace=0.25)

        layers = [
            (gs[0,0], io_n,     "hot",      f"Iron Oxide (w={w_io:.1f})",           "0-1"),
            (gs[0,1], cm_n,     "YlOrBr",   f"Clay Minerals (w={w_cm:.1f})",         "0-1"),
            (gs[0,2], aster_n,  "RdYlGn_r", f"ASTER Alteration (w={w_aster:.1f})",  "0-1"),
            (gs[1,0], struct_n, "plasma",   f"Lineament Density (w={w_struct:.1f})", "0-1"),
            (gs[1,1], elev_inv, "terrain",  f"Elevation Inv. (w={w_elev:.1f})",      "0-1"),
        ]
        for gs_p, arr_p, cmap_p, title_p, lbl_p in layers:
            ax = fig.add_subplot(gs_p)
            im_ = ax.imshow(arr_p, cmap=cmap_p, vmin=0, vmax=1)
            style_ax(ax, title_p, fs=11)
            add_cbar(fig, im_, ax, lbl_p); add_north_arrow(ax); add_scalebar(ax, rows, cols)
            if coord_sys != "None": add_coord_grid(ax, transform, rows, cols, crs, mode=coord_sys)
            else: ax.axis("off")

        # Final MCDA map
        ax_m = fig.add_subplot(gs[1, 2])
        im_m = ax_m.imshow(mcda, cmap="RdYlGn_r", vmin=0, vmax=1)
        style_ax(ax_m, f"MCDA Composite Score (threshold={threshold:.0%})", fs=11)
        add_cbar(fig, im_m, ax_m, "Score")
        hi_mask = np.ma.masked_where(mcda < threshold, mcda * 0)
        ax_m.imshow(hi_mask, cmap=mcolors.ListedColormap(["#ffcc00"]), alpha=0.55, vmin=0, vmax=1)
        add_north_arrow(ax_m); add_scalebar(ax_m, rows, cols)
        if coord_sys != "None": add_coord_grid(ax_m, transform, rows, cols, crs, mode=coord_sys)
        else: ax_m.axis("off")
        ax_m.text(0.02, 0.97, f"High-score pixels: {n_high:,} ({n_high*pm2:.2f} km²)",
                  transform=ax_m.transAxes, color="#ffcc00", fontsize=8, va="top",
                  bbox=dict(fc=BG, ec="#333", lw=0.5, pad=3))

        stamp_map(fig, geo, "MCDA")
        out_png = _out_path("maps", "mcda_prospectivity", ".png")
        plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        tif_out = None
        if save_out:
            tif_out = _out_path("geotiff", "mcda_score", ".tif")
            prof = {k: profile[k] for k in ("crs","transform","width","height") if k in profile}
            prof.update({"driver":"GTiff","dtype":"float32","nodata":float("nan"),"count":1,"compress":"lzw"})
            with rasterio.open(tif_out, "w", **prof) as dst:
                dst.write(mcda, 1)

        md = f"""### 🎯 MCDA Composite Prospectivity

| Criterion | Weight | Contribution |
|---|---|---|
| Iron Oxide (IO) | {w_io:.1f} | {w_io/w_total*100:.1f}% |
| Clay Minerals (CM) | {w_cm:.1f} | {w_cm/w_total*100:.1f}% |
| ASTER Alteration | {w_aster:.1f} | {w_aster/w_total*100:.1f}% |
| Structural Density | {w_struct:.1f} | {w_struct/w_total*100:.1f}% |
| Elevation Inv. | {w_elev:.1f} | {w_elev/w_total*100:.1f}% |

| Result | Value |
|---|---|
| **High-score area (≥{threshold:.0%})** | **{n_high:,} px ({n_high*pm2:.2f} km²)** |
| **Max MCDA score** | {float(np.nanmax(mcda)):.4f} |
| **Mean MCDA score** | {float(np.nanmean(mcda)):.4f} |
"""
        return out_png, tif_out, md
    except Exception:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── 5. ZONE EXPORT TO CSV (pixel-level statistics per zone) ────────────
def export_zones_csv(s2_file, aster_file, prob_threshold):
    """
    Extract per-pixel statistics for the high-probability zone and export as CSV.
    Columns: row, col, lat, lon, probability(if model), IO, CM, FI, GS, NDVI,
             DEM, slope, aspect + ASTER indices if available.
    """
    if s2_file is None:
        return None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    try:
        import csv, io as _io
        from rasterio.transform import xy as _xy

        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        # Load ASTER if available
        aster_dict = {}
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw_a = asrc.read().astype("float32")[:6]
                    aster_r = resample_aster_to_sentinel(raw_a, asrc.transform, asrc.crs,
                                                          transform, crs, rows, cols)
                aster_dict = compute_aster_indices(aster_r)
            except Exception:
                pass

        # Build probability map if model loaded
        prob_map = np.full((rows, cols), np.nan, dtype=np.float32)
        if model_bundle is not None:
            pixels = stack.reshape(stack.shape[0], -1).T
            valid  = ~np.any(np.isnan(pixels[:, :9]), axis=1)
            if valid.any():
                pv = _engineer_features(pixels[valid])
                m  = model_bundle["model"] if isinstance(model_bundle, dict) else model_bundle
                n_exp = getattr(m, "n_features_in_", None)
                pv    = _align_features(pv, n_exp)
                proba = m.predict_proba(pv)[:, 1]
                flat_prob = np.full(rows*cols, np.nan, dtype=np.float32)
                flat_prob[valid] = proba
                prob_map = flat_prob.reshape(rows, cols)

        # Select high-probability pixels
        mask = prob_map >= prob_threshold if not np.all(np.isnan(prob_map)) \
               else robust_scale(stack[9]) >= 0.80

        r_idx, c_idx = np.where(mask)
        if len(r_idx) == 0:
            return None, f"### ⚠️ No pixels above threshold {prob_threshold:.0%}."
        if len(r_idx) > 50_000:
            # Random subsample for very large zones
            rng  = np.random.default_rng(42)
            pick = rng.choice(len(r_idx), size=50_000, replace=False)
            r_idx, c_idx = r_idx[pick], c_idx[pick]

        # Build coordinate arrays (row/col → lon/lat in WGS84)
        from pyproj import Transformer
        try:
            tr_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xs, ys = zip(*[transform * (float(c)+0.5, float(r)+0.5) for r, c in zip(r_idx, c_idx)])
            lons, lats = tr_wgs.transform(list(xs), list(ys))
        except Exception:
            lons = [float("nan")] * len(r_idx)
            lats = [float("nan")] * len(r_idx)

        # Assemble columns
        header = ["row","col","lat","lon","probability",
                  "iron_oxide","clay_minerals","ferrous_iron","gossan","ndvi",
                  "elevation_m","slope_deg","aspect_deg","roughness_m"]
        aster_keys = sorted(aster_dict.keys())
        header += aster_keys

        rows_out = []
        for i, (r, c) in enumerate(zip(r_idx, c_idx)):
            row = [int(r), int(c),
                   f"{lats[i]:.6f}", f"{lons[i]:.6f}",
                   f"{prob_map[r,c]:.4f}" if not np.isnan(prob_map[r,c]) else "",
                   f"{stack[9,r,c]:.4f}", f"{stack[10,r,c]:.4f}",
                   f"{stack[11,r,c]:.4f}", f"{stack[12,r,c]:.4f}",
                   f"{stack[13,r,c]:.4f}", f"{stack[14,r,c]:.2f}",
                   f"{stack[15,r,c]:.2f}", f"{stack[16,r,c]:.2f}",
                   f"{stack[17,r,c]:.4f}"]
            for k in aster_keys:
                v = aster_dict[k][r,c]
                row.append(f"{v:.4f}" if not np.isnan(v) else "")
            rows_out.append(row)

        csv_path = _out_path("reports", "gold_zone_pixels", ".csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows_out)

        md = f"""### 📊 Zone CSV Export

| Parameter | Value |
|---|---|
| **Pixels exported** | {len(rows_out):,} |
| **Probability threshold** | {prob_threshold:.0%} |
| **Columns** | {len(header)} |
| **ASTER indices** | {len(aster_keys)} |
| **File** | `{csv_path}` |

*Includes lat/lon (WGS84), all S2 spectral indices, terrain metrics, and ASTER indices per pixel.*
"""
        return csv_path, md
    except Exception:
        return None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── HDF → GeoTIFF CONVERTER ──────────────────────────────────────────

# Band-name patterns used to locate B4-B9 inside an HDF file
import re as _re
_HDF_BAND_PATTERNS = [
    r"^[Bb](\d+)$",
    r"^[Bb]and[_ ]?(\d+)$",
    r"^sur_refl_b0?(\d+)",
    r".*[Bb]and[_ ]?0?(\d+).*",
    r".*[Bb]0?(\d+).*",
    r".*ImageData(\d+).*",          # ASTER L1T HDF4 naming
    r".*image_data(\d+).*",
    r".*EV.*b0?(\d+).*",            # MODIS-style
    r".*Swath.*_(\d+)$",
    r".*[Ss][Ww][Ii][Rr].*(\d+).*",
    r".*[Rr]eflectance.*(\d+).*",
    r".*[Dd]ata[_ ]?(\d+)$",
    r".*_b0?(\d+)$",
    r".*_(\d+)$",                   # anything ending in a number
]
_HDF_TARGET = set(range(4, 10))   # B4 … B9

# ASTER L2 products store SWIR bands inside a swath group.
# Dataset paths look like: "SWIR_Swath:ImageData4" … "SWIR_Swath:ImageData9"
# or the full rasterio subdataset string:
#   HDF4_EOS:EOS_SWATH:<file>:SWIR_Swath:ImageData4
# This helper maps those dataset paths to band numbers 4-9.
_ASTER_L2_SWIR_GROUPS = ["SWIR_Swath", "SWIR", "swir_swath", "swir"]

def _hdf_band_number_extended(name: str, full_path: str = ""):
    """
    Extended band-number resolver that also handles ASTER L2 SWIR swath paths.
    Falls back to the standard regex patterns.
    """
    # For ASTER L2: full path contains "SWIR_Swath:ImageData<N>"
    for token in [full_path, name]:
        m = _re.search(r"[Ii]mage[Dd]ata(\d+)", token)
        if m:
            return int(m.group(1))
    return _hdf_band_number(name)


def _hdf_band_number(name: str):
    for pat in _HDF_BAND_PATTERNS:
        m = _re.match(pat, name.strip())
        if m:
            return int(m.group(1))
    return None


def _detect_hdf_format(path: str) -> str:
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic[:4] == b"\x0e\x03\x13\x01":
        return "hdf4"
    if magic[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    ext = os.path.splitext(path)[1].lower()
    return "hdf5" if ext in (".he5", ".hdf5", ".h5") else "hdf4"


def _list_all_hdf_datasets(path: str) -> list:
    """Return all dataset names in an HDF file for diagnostics."""
    names = []
    # Try rasterio subdatasets first — return FULL subdataset strings so
    # the fallback _assign_bands_by_order can open them directly.
    try:
        import rasterio
        with rasterio.open(path) as src:
            subs = src.subdatasets or []
        if subs:
            # Store the full path; use last colon-token as display name
            for s in subs:
                names.append(s)
            return names
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"rasterio subdataset listing: {_e}")
    # Try pyhdf
    try:
        from pyhdf.SD import SD, SDC
        hdf = SD(path, SDC.READ)
        names = list(hdf.datasets().keys())
        hdf.end()
        return names
    except ImportError:
        import logging; logging.getLogger(__name__).debug("pyhdf not installed")
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"pyhdf listing: {_e}")
    # Try h5py
    try:
        import h5py
        def _collect(name, obj):
            if hasattr(obj, 'ndim') and obj.ndim >= 2:
                names.append(name)
        with h5py.File(path, "r") as f:
            f.visititems(_collect)
        return names
    except ImportError:
        import logging; logging.getLogger(__name__).debug("h5py not installed")
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"h5py listing: {_e}")
    return names


def _assign_bands_by_order(all_datasets: list, path: str, target: set) -> tuple:
    """
    Fallback: when no name matches, assign datasets in order to target bands.
    If 6 datasets found → assign to B4-B9 in order.
    all_datasets may contain full rasterio subdataset strings (preferred)
    or bare dataset names.
    """
    bands = {}
    meta  = {}
    valid = []
    try:
        import rasterio
        for sub in all_datasets:
            # sub may already be a full openable path (e.g. HDF4_EOS:...)
            # or a bare name — build candidate paths for both cases
            candidate_paths = [sub]  # try as-is first
            if not sub.startswith("HDF4") and not sub.startswith("HDF5"):
                candidate_paths += [
                    f"HDF4_EOS:EOS_SWATH:{path}:{sub}",
                    f"HDF4_EOS:EOS_GRID:{path}:{sub}",
                ]
            for full in candidate_paths:
                try:
                    with rasterio.open(full) as sd:
                        arr = sd.read(1).astype(np.float32)
                        valid.append((sub, arr, sd.transform, sd.crs))
                        break
                except Exception as _e:
                    import logging; logging.getLogger(__name__).debug(f"band open failed [{full}]: {_e}")
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"_assign_bands_by_order: {_e}")

    if len(valid) >= 6:
        # Take last 6 (SWIR bands usually at end)
        chosen = valid[-6:] if len(valid) > 6 else valid
        for i, (name, arr, transform, crs) in enumerate(chosen):
            band_no = 4 + i
            bands[band_no] = arr
            if not meta and transform and crs:
                meta = dict(gt=transform, prj=crs)
    elif valid:
        for i, (name, arr, transform, crs) in enumerate(valid):
            band_no = 4 + i
            if band_no in target:
                bands[band_no] = arr
                if not meta and transform and crs:
                    meta = dict(gt=transform, prj=crs)
    return bands, meta


def _read_hdf4_bands(path: str):
    """Read HDF4 bands — tries rasterio first (no extra libs needed), then pyhdf."""
    bands = {}
    meta  = {}

    # ── Method 1: rasterio subdatasets (no pyhdf needed) ────────────
    try:
        import rasterio
        with rasterio.open(path) as src:
            subdatasets = src.subdatasets or []

        for sub in subdatasets:
            # dataset name is the last token after ":" or "/"
            name = sub.replace("\\", "/").split(":")[-1].split("/")[-1]
            # Use extended resolver: handles ASTER L2 "SWIR_Swath:ImageData4" paths
            n = _hdf_band_number_extended(name, sub)
            if n in _HDF_TARGET:
                try:
                    with rasterio.open(sub) as sd:
                        arr = sd.read(1).astype(np.float32)
                        bands[n] = arr
                        if not meta:
                            meta = dict(gt=sd.transform, prj=sd.crs,
                                        width=sd.width, height=sd.height)
                except Exception as _e:
                    import logging; logging.getLogger(__name__).debug(f"subdataset read [{sub}]: {_e}")

        # ── ASTER L2 extra: try building SWIR swath paths explicitly ──
        if not bands and subdatasets:
            for swath_group in _ASTER_L2_SWIR_GROUPS:
                for band_no in _HDF_TARGET:
                    candidate_paths = [
                        f"HDF4_EOS:EOS_SWATH:{path}:{swath_group}:ImageData{band_no}",
                        f"HDF4_EOS:EOS_GRID:{path}:{swath_group}:ImageData{band_no}",
                    ]
                    for cp in candidate_paths:
                        try:
                            with rasterio.open(cp) as sd:
                                arr = sd.read(1).astype(np.float32)
                                bands[band_no] = arr
                                if not meta:
                                    meta = dict(gt=sd.transform, prj=sd.crs,
                                                width=sd.width, height=sd.height)
                                break
                        except Exception as _e:
                            import logging; logging.getLogger(__name__).debug(f"ASTER L2 swath [{cp}]: {_e}")
                if bands:
                    break

        if bands:
            return bands, meta
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"_read_hdf4_bands outer: {_e}")

    # ── Method 2: pyhdf (if installed) ──────────────────────────────
    try:
        from pyhdf.SD import SD, SDC
        hdf      = SD(path, SDC.READ)
        datasets = hdf.datasets()
        for name in datasets:
            n = _hdf_band_number(name)
            if n in _HDF_TARGET:
                sds   = hdf.select(name)
                data  = sds.get().astype(np.float32)
                attrs = sds.attributes()
                scale = float(attrs.get("scale_factor", 1.0))
                off   = float(attrs.get("add_offset",   0.0))
                fill  = attrs.get("_FillValue", None)
                if fill is not None:
                    data = np.where(data == fill, np.nan, data)
                bands[n] = data * scale + off
        # Try EOS bounding-box metadata
        try:
            core = hdf.select("CoreMetadata.0").get()
            if isinstance(core, np.ndarray):
                core = core.tobytes().decode("latin-1", errors="replace")
            ns = _re.findall(r"NORTHBOUNDINGCOORDINATE.*?VALUE\s*=\s*([-\d.]+)", core)
            ss = _re.findall(r"SOUTHBOUNDINGCOORDINATE.*?VALUE\s*=\s*([-\d.]+)", core)
            ws = _re.findall(r"WESTBOUNDINGCOORDINATE.*?VALUE\s*=\s*([-\d.]+)",  core)
            es = _re.findall(r"EASTBOUNDINGCOORDINATE.*?VALUE\s*=\s*([-\d.]+)",  core)
            if ns and ss and ws and es:
                meta = dict(north=float(ns[0]), south=float(ss[0]),
                            west=float(ws[0]),  east=float(es[0]))
        except Exception as _e:
            import logging; logging.getLogger(__name__).debug(f"HDF4 metadata parse: {_e}")
        hdf.end()
        if bands:
            return bands, meta
    except ImportError:
        import logging; logging.getLogger(__name__).debug("pyhdf not installed")
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"_read_hdf4_bands pyhdf method: {_e}")

    return bands, meta


def _read_hdf5_bands(path: str):
    import h5py
    bands = {}
    def _visit(name, obj):
        if not isinstance(obj, h5py.Dataset) or obj.ndim < 2 or min(obj.shape) < 8:
            return
        n = _hdf_band_number(name.split("/")[-1])
        if n in _HDF_TARGET:
            data  = obj[()].astype(np.float32)
            scale = float(obj.attrs.get("scale_factor", 1.0))
            off   = float(obj.attrs.get("add_offset",   0.0))
            fill  = obj.attrs.get("_FillValue", None)
            if fill is not None:
                data = np.where(data == fill, np.nan, data)
            bands[n] = data * scale + off
    with h5py.File(path, "r") as f:
        f.visititems(_visit)
    return bands, {}


def _read_gdal_bands(path: str):
    """Read via GDAL subdatasets — tries rasterio first, then osgeo.gdal."""
    bands = {}
    meta  = {}

    # ── Method 1: rasterio (already installed, no osgeo needed) ─────
    try:
        import rasterio
        with rasterio.open(path) as src:
            subdatasets = src.subdatasets or []

        if subdatasets:
            for sub in subdatasets:
                name = sub.replace("\\", "/").split(":")[-1].split("/")[-1]
                n = (_hdf_band_number(name) or
                     _hdf_band_number(sub.split(":")[-1]))
                if n in _HDF_TARGET:
                    try:
                        with rasterio.open(sub) as sd:
                            arr = sd.read(1).astype(np.float32)
                            bands[n] = arr
                            if not meta:
                                meta = dict(gt=sd.transform, prj=sd.crs,
                                            width=sd.width, height=sd.height)
                    except Exception as _e:
                        import logging; logging.getLogger(__name__).debug(f"GDAL subdataset: {_e}")
        else:
            with rasterio.open(path) as src:
                for i in range(1, src.count + 1):
                    desc = src.descriptions[i-1] or f"B{i}"
                    n = _hdf_band_number(desc) or (i if i in _HDF_TARGET else None)
                    if n in _HDF_TARGET:
                        bands[n] = src.read(i).astype(np.float32)
                if not meta:
                    meta = dict(gt=src.transform, prj=src.crs,
                                width=src.width, height=src.height)
        if bands:
            return bands, meta
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"_read_gdal_bands GDAL: {_e}")

    # ── Method 2: osgeo.gdal (if installed) ─────────────────────────
    try:
        from osgeo import gdal
        gdal.UseExceptions()
        ds = gdal.Open(path)
        if ds is None:
            raise IOError(f"GDAL cannot open: {path}")
        sub = ds.GetSubDatasets()
        if sub:
            for sub_path, sub_desc in sub:
                n = (_hdf_band_number(sub_desc.split()[-1]) or
                     _hdf_band_number(sub_path.split(":")[-1]))
                if n in _HDF_TARGET:
                    sd  = gdal.Open(sub_path)
                    arr = sd.GetRasterBand(1).ReadAsArray().astype(np.float32)
                    bands[n] = arr
                    if not meta:
                        gt  = sd.GetGeoTransform()
                        prj = sd.GetProjection()
                        if gt and prj:
                            meta = dict(gt=gt, prj=prj,
                                        width=sd.RasterXSize, height=sd.RasterYSize)
        else:
            for i in range(1, ds.RasterCount + 1):
                rb   = ds.GetRasterBand(i)
                desc = rb.GetDescription() or f"B{i}"
                n    = _hdf_band_number(desc) or (i if i in _HDF_TARGET else None)
                if n in _HDF_TARGET:
                    bands[n] = rb.ReadAsArray().astype(np.float32)
            gt  = ds.GetGeoTransform()
            prj = ds.GetProjection()
            if gt and prj:
                meta = dict(gt=gt, prj=prj,
                            width=ds.RasterXSize, height=ds.RasterYSize)
        return bands, meta
    except ImportError:
        raise ImportError("No module named 'osgeo'")
    except Exception as e:
        raise e


def convert_hdf_to_tif(hdf_file, band_selection: list) -> tuple:
    """
    Gradio callback: convert an uploaded HDF file to a GeoTIFF.

    Parameters
    ----------
    hdf_file      : Gradio UploadButton file object (has .name attribute)
    band_selection: list of band numbers to export, e.g. [4,5,6,7,8,9]

    Returns
    -------
    (output_tif_path_or_None, status_markdown_string)
    """
    if hdf_file is None:
        return None, "⚠️  Please upload an HDF file first."

    in_path = _fpath(hdf_file)
    target  = set(int(b) for b in band_selection) if band_selection else _HDF_TARGET
    log     = []

    # ── detect format ────────────────────────────────────────────────
    try:
        fmt = _detect_hdf_format(in_path)
        log.append(f"**Detected format:** {fmt.upper()}")
    except Exception as e:
        return None, f"❌ Cannot read file: {e}"

    # ── list all datasets for diagnostics ────────────────────────────
    try:
        all_ds = _list_all_hdf_datasets(in_path)
        if all_ds:
            # Show short (display) names to the user
            short_names = [s.split(":")[-1] for s in all_ds]
            log.append(f"**Datasets found ({len(all_ds)}):** `{'`, `'.join(short_names[:30])}`")
    except Exception as _e:
        all_ds = []

    # ── read bands ───────────────────────────────────────────────────
    bands, meta = {}, {}
    readers = (
        [(_read_hdf4_bands, "rasterio/pyhdf"),  (_read_gdal_bands, "GDAL")] if fmt == "hdf4"
        else [(_read_hdf5_bands, "h5py"), (_read_gdal_bands, "GDAL")]
    )
    for reader_fn, lib_name in readers:
        try:
            bands, meta = reader_fn(in_path)
            if bands:
                log.append(f"**Reader:** {lib_name}")
                break
        except ImportError as e:
            log.append(f"⚠️  {lib_name} not available: {e}")
        except Exception as e:
            log.append(f"⚠️  {lib_name} error: {e}")

    # filter to requested bands
    bands = {k: v for k, v in bands.items() if k in target}

    # ── fallback: assign datasets by order if names didn't match ─────
    if not bands and all_ds:
        log.append("⚠️  Band names not recognised — trying auto-assign by order...")
        try:
            bands, meta = _assign_bands_by_order(all_ds, in_path, target)
            if bands:
                log.append(f"**Auto-assigned {len(bands)} bands by position**")
        except Exception as e:
            log.append(f"⚠️  Auto-assign failed: {e}")

    if not bands:
        ds_list = "\n".join(f"- `{d}`" for d in all_ds[:20]) if all_ds else "_none detected_"
        return None, (
            "❌ **No matching bands found.**\n\n"
            "The HDF file does not contain datasets whose names match B4–B9.\n"
            "Datasets may use a custom naming convention.\n\n"
            f"**Datasets in file:**\n{ds_list}\n\n"
            + "\n".join(log)
        )

    ordered = sorted(bands.keys())
    log.append(f"**Bands extracted:** {[f'B{n}' for n in ordered]}")

    # ── build transform / CRS ────────────────────────────────────────
    try:
        from rasterio.transform import from_bounds, from_origin
        from rasterio.crs import CRS

        sample = bands[ordered[0]]
        height, width = sample.shape

        if "gt" in meta and hasattr(meta["gt"], "c"):
            # rasterio Affine transform
            transform = meta["gt"]
            crs = meta["prj"] if meta.get("prj") else CRS.from_epsg(4326)
        elif "gt" in meta and isinstance(meta["gt"], (list, tuple)):
            gt = meta["gt"]
            # GDAL GeoTransform: [x_origin, pix_w, row_rot, y_origin, col_rot, pix_h]
            # rasterio.Affine(a, b, c, d, e, f) maps as:
            #   a=pix_w  b=row_rot  c=x_origin  d=col_rot  e=pix_h  f=y_origin
            transform = rasterio.transform.Affine(
                gt[1], gt[2], gt[0],
                gt[4], gt[5], gt[3])
            crs = CRS.from_wkt(meta["prj"]) if meta.get("prj") else CRS.from_epsg(4326)
        elif "north" in meta:
            transform = from_bounds(
                meta["west"], meta["south"], meta["east"], meta["north"],
                width, height)
            crs = CRS.from_epsg(4326)
        else:
            log.append("⚠️  No geolocation metadata — writing without CRS.")
            transform = from_origin(0, height, 1, 1)
            crs       = None

        # ── write GeoTIFF ────────────────────────────────────────────
        stem     = os.path.splitext(os.path.basename(in_path))[0]
        out_path = _out_path("hdf_convert", f"{stem}_B4-B9", ".tif")

        profile = dict(
            driver="GTiff", dtype="float32",
            width=width, height=height,
            count=len(ordered),
            crs=crs, transform=transform,
            compress="deflate", nodata=float("nan"),
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            for idx, band_no in enumerate(ordered, start=1):
                dst.write(bands[band_no], idx)
                dst.update_tags(idx, band=f"B{band_no}")

        log.append(f"**Output size:** {width} × {height} px")
        log.append(f"**CRS:** {crs}")
        log.append(f"**Compression:** DEFLATE")

        status = (
            f"### ✅ Conversion complete\n\n"
            + "\n\n".join(log)
            + f"\n\n**File:** `{os.path.basename(out_path)}`"
        )
        return out_path, status

    except ImportError:
        return None, "❌ `rasterio` is required but not installed.  `pip install rasterio`"
    except Exception as e:
        return None, f"❌ Write error: {e}\n\n{''.join(traceback.format_exc())}"


# ── SENTINEL-2 INDICES ────────────────────────────────────────────────

def compute_index(name, s):
    b2,b3,b4,b5,b6,b8a,b8,b11,b12 = (s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8])
    with np.errstate(divide="ignore", invalid="ignore"):
        if name=="NDWI":  return np.where((b3+b8)!=0,(b3-b8)/(b3+b8),np.nan)
        if name=="EVI":
            d=b8+6*b4-7.5*b2+1; return np.where(d!=0,2.5*(b8-b4)/d,np.nan)
        if name=="SAVI":  return np.where((b8+b4+.5)!=0,1.5*(b8-b4)/(b8+b4+.5),np.nan)
        if name=="Ferric":return np.where(b12!=0,b11/b12,np.nan)
        if name=="AlOH":
            d=b6+b8; return np.where(d!=0,(b5+b8a)/d,np.nan)
        if name=="MgOH":
            d=b8a+b8; return np.where(d!=0,(b6+b12)/d,np.nan)
        if name=="Silica":
            d=b8+b6; return np.where(d!=0,(b11+b4)/d,np.nan)
        if name=="Opaque":return np.where((b4+b8)!=0,(b4-b8)/(b4+b8),np.nan)
        if name=="GosEx": return np.where(b8a!=0,(b4/b2)*(b4/b8a),np.nan)
        if name=="IronEx":return np.where(b8a!=0,(b4*b11)/(b8a*b8a),np.nan)
    return None

def get_band_or_index(name, s):
    if name in COMPUTED_INDICES: return compute_index(name, s)
    for idx,(short,long,_) in BAND_INFO.items():
        if name in (short,long): return s[idx]
    return None


# ── MAP STAMP ─────────────────────────────────────────────────────────

def draw_prob_panel(fig, gs_pos, prob_map, rows, cols, transform, crs, coord_sys,
                    threshold, vh, hi, mo, lo, pm2, geo_str, mode_lbl):
    ax = fig.add_subplot(gs_pos)
    im = ax.imshow(prob_map, cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
    style_ax(ax, f"Gold Probability Map  [{mode_lbl}]", fs=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, orientation="horizontal", pad=0.08)
    cb.set_label("Prospectivity  (0 = background  →  1 = gold)", color="#ccc", fontsize=9)
    cb.ax.tick_params(colors="#ccc", labelsize=8)
    contour_data = np.where(np.isnan(prob_map), 0, prob_map)
    try:
        ax.contour(contour_data, levels=[threshold], colors=["#00ffff"], linewidths=0.8, alpha=0.8)
        ax.text(0.02, 0.22, f"—— Threshold: {threshold:.0%}",
                transform=ax.transAxes, color="#00ffff", fontsize=7, va="bottom",
                bbox=dict(fc=BG, ec="#00ffff", lw=0.5, pad=2, alpha=0.85))
    except Exception as _e:
        import logging; logging.getLogger(__name__).debug(f"draw_prob_panel contour: {_e}")
    stat_txt = (f"Max: {float(np.nanmax(prob_map)):.1%}\n"
                f"Mean: {float(np.nanmean(prob_map)):.1%}\n"
                f"≥{threshold:.0%}: {int(np.nansum(prob_map>=threshold)):,}px "
                f"({float(np.nansum(prob_map>=threshold))*pm2:.1f}km²)")
    ax.text(0.02, 0.13, stat_txt, transform=ax.transAxes,
            color="white", fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(fc="#0f0f18", ec="#333", lw=0.8, pad=4, alpha=0.95))
    ax.text(0.01, 0.995, geo_str, transform=ax.transAxes, color="#999",
            fontsize=6, va="top", bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))
    add_north_arrow(ax); add_scalebar(ax, rows, cols)
    if coord_sys != "None":
        add_coord_grid(ax, transform, rows, cols, crs, n=6, mode=coord_sys)
    else:
        ax.axis("off")
    return ax


def draw_priority_panel(fig, gs_pos, prob_map, rows, cols, transform, crs, coord_sys,
                        threshold, vh, hi, mo, lo, pm2, geo_str):
    c4 = ["#1a3a2a","#2d6a4f","#f4a261","#e63946"]
    cmap4 = mcolors.ListedColormap(c4)
    # Use same adaptive thresholds as build_summary so the map matches the report
    thr_vh, thr_hi, thr_mo = _compute_zone_thresholds(prob_map, threshold)
    zmap  = np.full_like(prob_map, np.nan)
    zmap[prob_map < thr_mo] = 0
    zmap[(prob_map >= thr_mo) & (prob_map < thr_hi)] = 1
    zmap[(prob_map >= thr_hi) & (prob_map < thr_vh)] = 2
    zmap[prob_map >= thr_vh] = 3
    ax = fig.add_subplot(gs_pos)
    ax.imshow(zmap, cmap=cmap4, vmin=0, vmax=3, interpolation="nearest")
    style_ax(ax, f"Priority Zones  (adaptive thresholds)", fs=12)
    legs = [
        Patch(fc=c4[3], ec="none", label=f"Very High ≥{thr_vh:.0%}  {vh:>7,}px  ({vh*pm2:.2f} km²)"),
        Patch(fc=c4[2], ec="none", label=f"High      {thr_hi:.0%}-{thr_vh:.0%}  {hi:>7,}px  ({hi*pm2:.2f} km²)"),
        Patch(fc=c4[1], ec="none", label=f"Moderate  {thr_mo:.0%}-{thr_hi:.0%}  {mo:>7,}px  ({mo*pm2:.2f} km²)"),
        Patch(fc=c4[0], ec="none", label=f"Low       <{thr_mo:.0%}  {lo:>7,}px  ({lo*pm2:.2f} km²)"),
    ]
    ax.legend(handles=legs, loc="lower right", facecolor="#111", edgecolor="#444",
              labelcolor="white", fontsize=8.5, framealpha=0.94,
              prop={"family":"monospace"})
    ax.text(0.01, 0.995, geo_str, transform=ax.transAxes, color="#999",
            fontsize=6, va="top", bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))
    add_north_arrow(ax); add_scalebar(ax, rows, cols)
    if coord_sys != "None":
        add_coord_grid(ax, transform, rows, cols, crs, n=6, mode=coord_sys)
    else:
        ax.axis("off")
    return ax


def draw_histogram_row(fig, gs_row, prob_map, stack, threshold):
    axes_h = [fig.add_subplot(gs_row[0, i]) for i in range(5)]
    ax = axes_h[0]
    ax.set_facecolor(BG)
    flat = prob_map[~np.isnan(prob_map)].flatten()
    n_bins = 80
    counts, edges = np.histogram(flat, bins=n_bins, range=(0,1))
    colors_hist = plt.cm.RdYlGn_r(np.linspace(0,1,n_bins))
    for i, (c, e0, e1) in enumerate(zip(counts, edges[:-1], edges[1:])):
        ax.bar(e0, c, width=(e1-e0), color=colors_hist[i], edgecolor="none", alpha=0.9)
    ax.axvline(threshold, color="#00ffff", lw=1.5, ls="--", label=f"Threshold {threshold:.0%}")
    ax.set_title("Probability Distribution", color="white", fontsize=9, fontweight="bold")
    ax.set_xlabel("Probability", color="#aaa", fontsize=8)
    ax.set_ylabel("Pixel count", color="#aaa", fontsize=8)
    ax.tick_params(colors="#777", labelsize=7)
    ax.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    idx_panels = [(9, "Iron Oxide", "hot"), (10, "Clay Min.", "YlOrBr"),
                  (11, "Ferrous Fe", "plasma"), (12, "Gossan", "Reds")]
    for ax_i, (bidx, bname, cmap_n) in zip(axes_h[1:], idx_panels):
        ax_i.set_facecolor(BG)
        arr = stack[bidx][~np.isnan(stack[bidx])].flatten()
        vlo = np.percentile(arr, 2); vhi = np.percentile(arr, 98)
        n_b = 60
        counts_i, edges_i = np.histogram(arr, bins=n_b, range=(vlo, vhi))
        colors_i = _get_cmap(cmap_n)(np.linspace(0, 1, n_b))
        for cnt, col, e0, e1 in zip(counts_i, colors_i, edges_i[:-1], edges_i[1:]):
            ax_i.bar(e0, cnt, width=(e1-e0)*0.95, color=col, edgecolor="none", alpha=0.9)
        ax_i.axvline(float(np.nanmean(arr)), color="white", lw=1, ls="--", alpha=0.7)
        ax_i.set_title(bname, color="white", fontsize=9, fontweight="bold")
        ax_i.tick_params(colors="#777", labelsize=6)
        for sp in ax_i.spines.values(): sp.set_edgecolor("#333")


# ── ANALYSIS SUMMARY ─────────────────────────────────────────────────

def _compute_zone_thresholds(prob_map, user_threshold):
    """
    Adaptive zone boundaries so Very High/High/Moderate/Low are always populated.
    Fixes the common failure where fixed 0.40/0.60/0.80 cut-offs leave High and
    Very High empty when calibrated models compress scores (e.g. 0.45-0.65).
    """
    flat = prob_map[~np.isnan(prob_map)].flatten()
    if flat.size == 0:
        return 0.80, 0.60, 0.40
    p90 = float(np.percentile(flat, 90))   # top 10 % -> Very High
    p70 = float(np.percentile(flat, 70))   # top 30 % -> High
    p40 = float(np.percentile(flat, 40))   # top 60 % -> Moderate
    # Very-High: at least the user threshold, capped at p90
    thr_vh = max(min(p90, 0.95), user_threshold)
    thr_hi = max(p70, user_threshold * 0.75)
    thr_mo = max(p40, user_threshold * 0.50)
    # Enforce strict ordering
    thr_hi = min(thr_hi, thr_vh - 1e-4)
    thr_mo = min(thr_mo, thr_hi - 1e-4)
    thr_mo = max(thr_mo, 0.0)
    return thr_vh, thr_hi, thr_mo


def build_summary(geo, prob_map, threshold, mode_lbl, stack, n_valid,
                  rows, cols, use_model, aster_dict=None):
    pm2 = PIXEL_SIZE_M**2 / 1e6
    n_total = int(np.sum(~np.isnan(prob_map)))
    if n_total == 0:
        return "### ❌ No valid pixels.", 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, pm2
    valid_mask = ~np.isnan(prob_map)
    # Adaptive zone thresholds (fixes empty High/Moderate/VeryHigh zones)
    thr_vh, thr_hi, thr_mo = _compute_zone_thresholds(prob_map, threshold)
    vh = int(np.sum(valid_mask & (prob_map >= thr_vh)))
    hi = int(np.sum(valid_mask & (prob_map >= thr_hi) & (prob_map < thr_vh)))
    mo = int(np.sum(valid_mask & (prob_map >= thr_mo) & (prob_map < thr_hi)))
    lo = int(np.sum(valid_mask & (prob_map < thr_mo)))
    n_high   = int(np.nansum(prob_map >= threshold))
    pct_h    = n_high / n_total * 100 if n_total else 0
    p_max    = float(np.nanmax(prob_map))
    p_mean   = float(np.nanmean(prob_map))
    p_std    = float(np.nanstd(prob_map))
    p_med    = float(np.nanmedian(prob_map))
    pct_valid = n_valid / (rows*cols) * 100
    idx_io   = float(np.nanmean(stack[9]))
    idx_cm   = float(np.nanmean(stack[10]))
    idx_fi   = float(np.nanmean(stack[11]))
    idx_gs   = float(np.nanmean(stack[12]))
    elev_mean = float(np.nanmean(stack[14]))
    elev_max  = float(np.nanmax(stack[14]))
    slope_mean= float(np.nanmean(stack[15]))
    flat   = int(np.nansum(stack[15] < 5))
    gentle = int(np.nansum((stack[15] >= 5) & (stack[15] < 15)))
    steep  = int(np.nansum(stack[15] >= 15))
    geo_str = geo.get("geo_str", "—")
    area_km2 = geo.get("area_km2", 0)
    crs_str  = geo.get("crs", "—")
    cx = geo.get("center_lon", 0); cy = geo.get("center_lat", 0)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    # ASTER section
    aster_section = ""
    if aster_dict:
        aster_rows = ""
        for k, arr in aster_dict.items():
            label = ASTER_INDICES.get(k, ("",))[0]
            mn = float(np.nanmean(arr))
            mx = float(np.nanmax(arr))
            sd = float(np.nanstd(arr))
            interp = ""
            if k == "AST_HydAlt" and mn > 1.1:
                interp = "⚠ Significant hydrothermal alteration"
            elif k == "AST_AlOH" and mn > 1.05:
                interp = "🪨 Al-OH mineral enrichment (kaolinite/alunite)"
            elif k == "AST_Silica" and mn > 1.0:
                interp = "💎 Silica/quartz enrichment detected"
            elif k == "AST_Clay" and mn > 1.2:
                interp = "🏺 Clay alteration zone"
            aster_rows += f"| {k} | {label} | {mn:.4f} | {mx:.4f} | {sd:.4f} | {interp} |\n"
        # Hydrothermal intensity
        if "AST_HydAlt" in aster_dict:
            ha = aster_dict["AST_HydAlt"]
            ha_high = int(np.nansum(ha > 1.1))
            ha_pct  = ha_high / n_total * 100 if n_total else 0
        else:
            ha_high = 0; ha_pct = 0.0
        # Silica enrichment
        if "AST_Silica" in aster_dict:
            si = aster_dict["AST_Silica"]
            si_high = int(np.nansum(si > np.nanpercentile(si[~np.isnan(si)], 80)))
            si_pct  = si_high / n_total * 100 if n_total else 0
        else:
            si_high = 0; si_pct = 0.0
        aster_section = f"""
---

## 🌋 ASTER SWIR — Geological Index Summary

| Index | Name | Mean | Max | Std | Interpretation |
|---|---|---|---|---|---|
{aster_rows}

### Hydrothermal Alteration Intensity
- High-alteration pixels (AST_HydAlt > 1.1): **{ha_high:,}** ({ha_pct:.2f}% of scene)
- Silica-enriched pixels (top 20% AST_Silica): **{si_high:,}** ({si_pct:.2f}% of scene)
- Estimated quartz vein probability: **{'High' if si_pct > 10 else 'Moderate' if si_pct > 5 else 'Low'}**
- Clay alteration zones: **{'Strong' if float(np.nanmean(aster_dict.get('AST_Clay', np.array([0])))) > 1.2 else 'Moderate' if float(np.nanmean(aster_dict.get('AST_Clay', np.array([0])))) > 1.0 else 'Weak'}**
"""

    md = f"""
---

# 📋 Gold Prospectivity — Full Analysis Report

**Generated:** {ts} &nbsp;|&nbsp; **System:** Gold Prospectivity System {VERSION} &nbsp;|&nbsp; **Author:** Nader Safwat Ayed Hanna

---

## 🌍 Geographic Overview

| Parameter | Value |
|---|---|
| **Coverage** | {geo_str} |
| **Scene centre** | {cy:.4f}°N, {cx:.4f}°E |
| **Total area** | {area_km2:.2f} km² |
| **Image size** | {cols} × {rows} pixels @ {PIXEL_SIZE_M} m resolution |
| **Valid pixels** | {n_valid:,} / {rows*cols:,} &nbsp; ({pct_valid:.1f}%) |

---

## 🤖 Detection Parameters

| Parameter | Value |
|---|---|
| **Detection mode** | {mode_lbl} |
| **High-priority threshold** | {threshold:.0%} |
| **Data sources** | {'Sentinel-2 + ASTER SWIR + DEM' if aster_dict else 'Sentinel-2 + DEM'} |

---

## 📈 Probability Statistics

| Statistic | Value |
|---|---|
| **Maximum probability** | **{p_max:.1%}** |
| **Mean probability** | {p_mean:.1%} |
| **Median probability** | {p_med:.1%} |
| **Std deviation** | {p_std:.1%} |
| **High-priority pixels (≥{threshold:.0%})** | **{n_high:,}** ({pct_h:.2f}%) |
| **High-priority area** | **{n_high * pm2:.2f} km²** |

---

## 🗺️ Priority Zone Breakdown

| Zone | Threshold | Pixels | Area (km²) | % |
|---|---|---|---|---|
| 🔴 **Very High** | ≥ {thr_vh:.0%} | {vh:,} | {vh*pm2:.3f} | {vh/n_total*100:.2f}% |
| 🟠 **High** | {thr_hi:.0%} – {thr_vh:.0%} | {hi:,} | {hi*pm2:.3f} | {hi/n_total*100:.2f}% |
| 🟡 **Moderate** | {thr_mo:.0%} – {thr_hi:.0%} | {mo:,} | {mo*pm2:.3f} | {mo/n_total*100:.2f}% |
| 🟢 **Low** | < {thr_mo:.0%} | {lo:,} | {lo*pm2:.3f} | {lo/n_total*100:.2f}% |

{aster_section}

---

## 🪨 Sentinel-2 Spectral Index Summary

| Index | Name | Scene Mean | Interpretation |
|---|---|---|---|
| IO | Iron Oxide (B04/B02) | {idx_io:.4f} | {'High iron alteration' if idx_io > 2.0 else 'Moderate' if idx_io > 1.2 else 'Low'} |
| CM | Clay Minerals (B11/B8A) | {idx_cm:.4f} | {'Argillic alteration' if idx_cm > 1.5 else 'Moderate' if idx_cm > 0.8 else 'Low'} |
| FI | Ferrous Iron (B11/B08) | {idx_fi:.4f} | {'High ferrous minerals' if idx_fi > 1.2 else 'Moderate' if idx_fi > 0.7 else 'Low'} |
| GS | Gossan (B04/B08) | {idx_gs:.4f} | {'Gossanous alteration' if idx_gs > 0.8 else 'Moderate' if idx_gs > 0.4 else 'Low'} |

---

## 🏔️ Terrain Analysis

| Metric | Value |
|---|---|
| **Mean elevation** | {elev_mean:.1f} m |
| **Maximum elevation** | {elev_max:.1f} m |
| **Mean slope** | {slope_mean:.1f}° |

---

## 💡 Geological Interpretation

- Iron Oxide = **{idx_io:.3f}** → {'Significant ferric iron enrichment' if idx_io > 1.5 else 'Moderate iron signature'}
- Clay Minerals = **{idx_cm:.3f}** → {'Strong argillic / propylitic alteration' if idx_cm > 1.0 else 'Weak clay signal'}
- Gossan = **{idx_gs:.3f}** → {'Gossanous horizon detected' if idx_gs > 0.5 else 'Limited gossan expression'}
- High-priority area: **{n_high * pm2:.2f} km²**

---

*Eastern Desert, Egypt · Sentinel-2 + ASTER SWIR + DEM · {VERSION} · Sukari · Wadi Allaqi*
"""
    return md, vh, hi, mo, lo, n_high, pct_h, p_max, p_mean, pm2


# ── TAB 1: GOLD DETECTOR (upgraded with ASTER + comparison mode) ──────

def predict_gold(s2_file, aster_file, threshold, use_model, composite_choice,
                 coord_sys, extra_idx_txt, show_lbl_overlay,
                 label_file, save_geotiff, comparison_mode):
    if s2_file is None:
        return None, None, "### ❌ Upload a Sentinel-2 GeoTIFF first."
    if use_model and model_bundle is None:
        return None, None, f"### ❌ {model_status}"
    try:
        s2_path = _fpath(s2_file)
        stack, transform, crs, rows, cols, profile = _read_s2(s2_path)
        n_bands = stack.shape[0]
        if n_bands < REQUIRED_BANDS:
            return None, None, f"### ❌ {n_bands} bands found, need {REQUIRED_BANDS}."
        with rasterio.open(s2_path) as _src:
            geo = geo_info_dict(_src)

        # Load + resample ASTER if provided
        aster = None; aster_dict = None
        if aster_file is not None and comparison_mode != "Sentinel-2 Only":
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw_aster = asrc.read().astype("float32")
                    a_nb = asrc.count
                    # Use up to 6 ASTER SWIR bands
                    if a_nb >= 6:
                        raw_aster = raw_aster[:6]
                    else:
                        pad = np.zeros((6 - a_nb, asrc.height, asrc.width), dtype="float32")
                        raw_aster = np.vstack([raw_aster, pad])
                    a_transform = asrc.transform; a_crs = asrc.crs
                aster = resample_aster_to_sentinel(
                    raw_aster, a_transform, a_crs, transform, crs, rows, cols)
                aster_dict = compute_aster_indices(aster)
            except Exception as ae:
                aster = None; aster_dict = None
                print(f"ASTER load warning: {ae}")

        pixels = stack.reshape(n_bands, -1).T
        # Valid mask: only require the 9 reflectance bands (0-8) to be non-NaN.
        # Index bands (9-13) and terrain bands (14-17) are filled with 0 for nodata
        # pixels in the stack builder, so requiring all 18 bands non-NaN is too strict.
        _N_REFL = 9  # B02 B03 B04 B05 B06 B8A B08 B11 B12
        valid  = ~np.any(np.isnan(pixels[:, :_N_REFL]), axis=1)
        n_valid = int(valid.sum())   # will be updated after feat_mat is built if mode differs

        prob_flat = np.full(rows*cols, np.nan, dtype="float32")

        if use_model:
            # Build features based on comparison mode
            if aster is not None and comparison_mode == "Hybrid Sentinel-2 + ASTER":
                feat_mat = build_hybrid_features(stack, aster, "Hybrid Sentinel-2 + ASTER", aster_dict)
            elif aster is not None and comparison_mode == "ASTER Only":
                feat_mat = build_hybrid_features(stack, aster, "ASTER Only", aster_dict)
            else:
                # Sentinel-2 Only (or ASTER requested but not uploaded)
                feat_mat = pixels
            # Recompute valid mask from actual feature matrix.
            # For pure S2 stacks: only require reflectance bands (0-8) to be non-NaN;
            # index/terrain bands carry 0.0 for nodata pixels (not NaN).
            # For hybrid/ASTER stacks: allow NaN in ASTER columns too (ASTER footprint
            # may be smaller than S2 scene — those pixels fall back to S2-only).
            if feat_mat.shape[1] <= 18:
                # S2-only or ASTER-only: require first 9 reflectance cols to be valid
                feat_valid_mask = ~np.any(np.isnan(feat_mat[:, :min(9, feat_mat.shape[1])]), axis=1)
            else:
                # Hybrid: require S2 reflectance cols; ASTER cols allowed to be NaN
                feat_valid_mask = ~np.any(np.isnan(feat_mat[:, :9]), axis=1)
            n_valid = int(feat_valid_mask.sum())
            if n_valid == 0:
                return None, None, "### ❌ No valid pixels after NaN masking — check input files."
            feat_valid = feat_mat[feat_valid_mask]
            prob_flat[feat_valid_mask] = _get_model_predict_proba(feat_valid)
            if np.nanmax(prob_flat[feat_valid_mask]) < 1e-6:
                return None, None, (
                    "### ❌ Model returned all-zero probabilities.\n\n"
                    "The feature count sent to the model may not match what it was trained on. "
                    f"Features sent: {feat_valid.shape[1]}. "
                    "Try switching comparison mode to **Sentinel-2 Only**."
                )
        else:
            chosen = [i.strip() for i in extra_idx_txt.split(",") if i.strip()] or ["IO","CM","GS"]
            arrs = []
            for name in chosen:
                arr = get_band_or_index(name, stack)
                if arr is not None:
                    arrs.append(normalise(arr.flatten()))
                # Also check ASTER indices
                if aster is not None and name in ASTER_INDICES:
                    a_arr = compute_aster_index_by_name(name, aster)
                    if a_arr is not None:
                        arrs.append(normalise(a_arr.flatten()))
            if arrs:
                # Compute mean across all index arrays, then write to all pixels
                # (valid mask applied after so NaN pixels stay NaN)
                combined = np.nanmean(np.vstack(arrs), axis=0)  # shape (rows*cols,)
                combined[~valid] = np.nan
                prob_flat[:] = combined

        prob_map = prob_flat.reshape(rows, cols)
        mode_lbl = f"RF Model [{comparison_mode}]" if use_model else "Index Composite"

        (summary_md, vh, hi, mo, lo,
         n_high, pct_h, p_max, p_mean, pm2) = build_summary(
             geo, prob_map, threshold, mode_lbl, stack, n_valid, rows, cols,
             use_model, aster_dict)

        # Label overlay
        label_arr = None
        if show_lbl_overlay and label_file is not None:
            try:
                with rasterio.open(_fpath(label_file)) as _lsrc:
                    label_arr = _lsrc.read(1).astype("float32")
                if label_arr.shape != (rows, cols): label_arr = None
            except Exception as _e:
                import warnings; warnings.warn(f"⚠️ Label file could not be loaded: {_e}")
                label_arr = None

        # ── FIGURE ──
        fig = plt.figure(figsize=(26, 24), facecolor=BG)
        fig.text(0.5, 0.985,
                 f"GOLD PROSPECTIVITY ANALYSIS  ·  {comparison_mode}  ·  Eastern Desert, Egypt",
                 ha="center", va="top", fontsize=18, fontweight="bold", color=GOLD,
                 fontfamily="monospace")
        fig.text(0.5, 0.978,
                 f"Nader Safwat Ayed Hanna  ·  Beni-Suef University  ·  "
                 f"Mode: {mode_lbl}  ·  Threshold: {threshold:.0%}",
                 ha="center", va="top", fontsize=10, color="#888", fontfamily="monospace")
        fig.text(0.5, 0.973, geo.get("geo_str",""),
                 ha="center", va="top", fontsize=9, color="#666", fontfamily="monospace")

        gs_maps = GridSpec(3, 4, figure=fig, left=0.03, right=0.97,
                           top=0.965, bottom=0.22, hspace=0.38, wspace=0.22)
        gs_hist = GridSpec(1, 5, figure=fig, left=0.03, right=0.97,
                           top=0.185, bottom=0.04, hspace=0.2, wspace=0.30)

        # Reference panel
        ax0 = fig.add_subplot(gs_maps[0, 0])
        if composite_choice == "RGB (Natural Colour)":
            ax0.imshow(make_rgb(stack), interpolation="bilinear"); style_ax(ax0, "RGB — Natural Colour")
        elif composite_choice == "False Colour (NIR-R-B)":
            ax0.imshow(make_false_color(stack), interpolation="bilinear"); style_ax(ax0, "False Colour")
        elif composite_choice == "SWIR Composite (B12-B8A-B04)":
            rgb = np.dstack([normalise(stack[8]), normalise(stack[5]), normalise(stack[2])])
            ax0.imshow(np.nan_to_num(rgb), interpolation="bilinear"); style_ax(ax0, "SWIR Composite")
        else:
            im0 = ax0.imshow(stack[9], cmap="hot",
                             vmin=np.nanpercentile(stack[9], 2),
                             vmax=np.nanpercentile(stack[9], 98))
            style_ax(ax0, "Iron Oxide Index"); add_cbar(fig, im0, ax0, "Index")
        ax0.text(0.01, 0.995, geo.get("geo_str",""), transform=ax0.transAxes,
                 color="#999", fontsize=5.5, va="top",
                 bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))
        add_north_arrow(ax0); add_scalebar(ax0, rows, cols)
        if coord_sys != "None": add_coord_grid(ax0, transform, rows, cols, crs, mode=coord_sys)
        else: ax0.axis("off")

        # Diagnostic panels
        panels = [
            (gs_maps[0,1], stack[9],  "hot",     "Iron Oxide  (B04/B02)",    "Index"),
            (gs_maps[0,2], stack[10], "YlOrBr",  "Clay Minerals  (B11/B8A)", "Index"),
            (gs_maps[0,3], stack[12], "Reds",    "Gossan  (B04/B08)",        "Index"),
            (gs_maps[1,0], stack[11], "plasma",  "Ferrous Iron  (B11/B08)",  "Index"),
            (gs_maps[1,1], stack[14], "terrain", "Elevation (DEM)",          "m"),
            (gs_maps[1,2], stack[15], "copper",  "Slope",                    "°"),
        ]
        # Add 7th panel: ASTER hydrothermal map if available, else NDVI
        if aster_dict and "AST_HydAlt" in aster_dict:
            panels.append((gs_maps[1,3], aster_dict["AST_HydAlt"], "RdYlGn_r",
                           "🌋 ASTER Hydrothermal Alteration", "Index"))
        else:
            panels.append((gs_maps[1,3], stack[13], "RdYlGn", "NDVI", ""))

        for gs_p, arr_p, cmap_p, title_p, lbl_p in panels:
            axp = fig.add_subplot(gs_p)
            imp = axp.imshow(arr_p, cmap=cmap_p,
                             vmin=np.nanpercentile(arr_p, 2),
                             vmax=np.nanpercentile(arr_p, 98))
            style_ax(axp, title_p)
            add_cbar(fig, imp, axp, lbl_p)
            add_north_arrow(axp); add_scalebar(axp, rows, cols)
            if coord_sys != "None":
                add_coord_grid(axp, transform, rows, cols, crs, mode=coord_sys)
            else:
                axp.axis("off")
            axp.text(0.02, 0.985,
                     f"μ={float(np.nanmean(arr_p)):.3f}  σ={float(np.nanstd(arr_p)):.3f}",
                     transform=axp.transAxes, color="#ccc", fontsize=6, va="top",
                     bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))

        ax_prob = draw_prob_panel(fig, gs_maps[2,:2], prob_map, rows, cols,
                        transform, crs, coord_sys, threshold,
                        vh, hi, mo, lo, pm2, geo.get("geo_str",""), mode_lbl)
        if label_arr is not None:
            mask = np.ma.masked_where(label_arr != 1, label_arr)
            ax_prob.imshow(mask, cmap=mcolors.ListedColormap(["#00ffff"]), alpha=0.45, vmin=0, vmax=1)
        draw_priority_panel(fig, gs_maps[2,2:], prob_map, rows, cols,
                            transform, crs, coord_sys, threshold,
                            vh, hi, mo, lo, pm2, geo.get("geo_str",""))
        draw_histogram_row(fig, gs_hist, prob_map, stack, threshold)
        stamp_map(fig, geo, f"{comparison_mode} | {mode_lbl}", threshold)

        tmp_png_path = _out_path("maps", "gold_prospectivity", ".png")
        plt.savefig(tmp_png_path, dpi=140, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        tif_path = None
        if save_geotiff:
            tif_path = _out_path("geotiff", "gold_prospectivity", ".tif")
            clean_profile = {
                "driver": "GTiff", "dtype": "float32", "nodata": float("nan"),
                "count": 1, "crs": profile.get("crs"), "transform": profile.get("transform"),
                "width": profile.get("width"), "height": profile.get("height"), "compress": "lzw",
            }
            with rasterio.open(tif_path, "w", **clean_profile) as dst:
                dst.write(prob_map, 1)
                dst.update_tags(1, description="Gold prospectivity 0-1",
                                mode=mode_lbl, threshold=str(threshold))

        return tmp_png_path, tif_path, summary_md

    except Exception as _e:
        return None, None, f"### ❌ Error\n```\n{traceback.format_exc()}\n```"


# ── TAB 2: INDEX EXPLORER ─────────────────────────────────────────────

def explore_index(s2_file, index_name, cmap_name,
                  stretch_lo, stretch_hi, show_histogram, coord_sys, save_out):
    if s2_file is None:
        return None, None, "### ❌ Upload a GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        arr = get_band_or_index(index_name, stack)
        if arr is None: return None, None, f"### ❌ Unknown: {index_name}"

        vmin = float(np.nanpercentile(arr, stretch_lo))
        vmax = float(np.nanpercentile(arr, stretch_hi))
        if vmin == vmax: vmax = vmin + 1e-6
        ncols = 2 if show_histogram else 1
        fig, axes = plt.subplots(1, ncols, figsize=(14 if show_histogram else 10, 8), facecolor=BG)
        if not show_histogram: axes = [axes]

        ax = axes[0]; ax.set_facecolor(BG)
        im = ax.imshow(arr, cmap=cmap_name, vmin=vmin, vmax=vmax, interpolation="bilinear")
        if index_name in COMPUTED_INDICES:
            long_name = f"{index_name}  —  {COMPUTED_INDICES[index_name][0]}"
        else:
            long_name = next((v[1] for k,v in BAND_INFO.items()
                              if index_name in (v[0],v[1])), index_name)
        style_ax(ax, long_name, fs=13)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Value", color="#ccc", fontsize=9); cb.ax.tick_params(colors="#ccc")
        add_north_arrow(ax); add_scalebar(ax, rows, cols)
        if coord_sys != "None": add_coord_grid(ax, transform, rows, cols, crs, mode=coord_sys)
        else: ax.axis("off")
        ax.text(0.01, 0.995, geo.get("geo_str",""), transform=ax.transAxes,
                color="#999", fontsize=7, va="top",
                bbox=dict(fc=BG, ec="#333", lw=0.5, pad=3))
        ax.text(0.02, 0.06,
                f"Min: {float(np.nanmin(arr)):.4f}\nMean: {float(np.nanmean(arr)):.4f}\n"
                f"Max: {float(np.nanmax(arr)):.4f}\nStd: {float(np.nanstd(arr)):.4f}",
                transform=ax.transAxes, color="white", fontsize=8,
                fontfamily="monospace", va="bottom",
                bbox=dict(fc="#0f0f18", ec="#333", lw=0.8, pad=4, alpha=0.92))

        if show_histogram:
            ax2 = axes[1]; ax2.set_facecolor(BG)
            flat = arr[~np.isnan(arr)].flatten()
            n_b = 120
            counts, edges = np.histogram(flat, bins=n_b)
            ax2.bar(edges[:-1], counts, width=(edges[1]-edges[0]),
                    color=_get_cmap(cmap_name)(np.linspace(0,1,n_b)),
                    edgecolor="none", alpha=0.9)
            ax2.axvline(vmin, color="#e63946", lw=1.5, ls="--", label=f"p{stretch_lo:.0f}")
            ax2.axvline(vmax, color="#52b788", lw=1.5, ls="--", label=f"p{stretch_hi:.0f}")
            ax2.axvline(float(np.nanmean(arr)), color="white", lw=1.2, ls=":", label=f"mean")
            ax2.tick_params(colors="#ccc"); ax2.set_xlabel("Value", color="#ccc")
            ax2.set_ylabel("Count", color="#ccc")
            ax2.set_title(f"Histogram — {index_name}", color="white", fontweight="bold")
            ax2.legend(facecolor="#111", labelcolor="white", edgecolor="#444", fontsize=9)
            for sp in ax2.spines.values(): sp.set_edgecolor("#333")

        stamp_map(fig, geo, f"Index: {index_name}")
        plt.tight_layout(pad=1.5)
        tmp_png_path = _out_path("maps", f"index_{index_name}", ".png")
        plt.savefig(tmp_png_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        out_tif = None
        if save_out:
            clean_profile_exp = {"driver":"GTiff","dtype":"float32","nodata":float("nan"),
                                 "count":1,"crs":profile.get("crs"),"transform":profile.get("transform"),
                                 "width":profile.get("width"),"height":profile.get("height"),"compress":"lzw"}
            out_tif = _out_path("geotiff", f"index_{index_name}", ".tif")
            with rasterio.open(out_tif, "w", **clean_profile_exp) as dst:
                dst.write(arr, 1); dst.update_tags(1, index=index_name)

        stats = f"""
### 📊 {long_name}

| | |
|---|---|
| 🌍 Coverage | {geo.get("geo_str","—")} |
| **Max** | **{float(np.nanmax(arr)):.4f}** |
| Mean | {float(np.nanmean(arr)):.4f} |
| Std | {float(np.nanstd(arr)):.4f} |
"""
        return tmp_png_path, out_tif, stats
    except Exception as _e:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── TAB 3: MULTI-INDEX COMPARE ────────────────────────────────────────

def multi_index_compare(s2_file, aster_file, indices_str, cmap_list_str, coord_sys):
    if s2_file is None: return None, "### ❌ Upload a GeoTIFF first."
    chosen_names = [x.strip() for x in indices_str.split(",") if x.strip()]
    chosen_cmaps = [x.strip() for x in cmap_list_str.split(",") if x.strip()]
    if not chosen_names: return None, "### ❌ Enter index names."
    n = min(len(chosen_names), 6); chosen_names = chosen_names[:n]
    while len(chosen_cmaps) < n: chosen_cmaps.append("hot")
    chosen_cmaps = chosen_cmaps[:n]
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        # Load ASTER if provided
        aster = None
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw = asrc.read().astype("float32")[:6]
                    aster = resample_aster_to_sentinel(raw, asrc.transform, asrc.crs,
                                                      transform, crs, rows, cols)
            except Exception as _e:
                import logging; logging.getLogger(__name__).debug(f"ASTER load in compare: {_e}")
                aster = None

        ncols = min(n, 3); nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(9*ncols, 8*nrows),
                                  facecolor=BG, squeeze=False)
        fig.suptitle(f"Multi-Index Comparison  —  {geo.get('geo_str','')}",
                     color=GOLD, fontsize=14, fontweight="bold", fontfamily="monospace", y=0.995)

        table_rows = ""
        for i, (name, cm) in enumerate(zip(chosen_names, chosen_cmaps)):
            r, c = divmod(i, ncols); ax = axes[r][c]; ax.set_facecolor(BG)
            # Try S2 first, then ASTER
            arr = get_band_or_index(name, stack)
            if arr is None and aster is not None:
                arr = compute_aster_index_by_name(name, aster)
            if arr is None:
                ax.text(0.5, 0.5, f"Unknown:\n{name}", ha="center", va="center",
                        color="red", transform=ax.transAxes); ax.axis("off"); continue
            try: cmap_obj = _get_cmap(cm)
            except Exception: cmap_obj = _get_cmap("hot")
            vlo = float(np.nanpercentile(arr, 2)); vhi = float(np.nanpercentile(arr, 98))
            im = ax.imshow(arr, cmap=cmap_obj, vmin=vlo, vmax=vhi, interpolation="bilinear")
            title = (ASTER_INDICES[name][0] if name in ASTER_INDICES else
                     COMPUTED_INDICES[name][0][:28] if name in COMPUTED_INDICES else
                     next((v[1] for k,v in BAND_INFO.items() if name in (v[0],v[1])), name))
            style_ax(ax, f"{name} — {title[:30]}", fs=10)
            fig.colorbar(im, ax=ax, shrink=0.7).ax.tick_params(colors="#ccc")
            add_north_arrow(ax); add_scalebar(ax, rows, cols)
            if coord_sys != "None":
                add_coord_grid(ax, transform, rows, cols, crs, n=4, mode=coord_sys)
            else: ax.axis("off")
            ax.text(0.02, 0.985, f"μ={float(np.nanmean(arr)):.3f}  σ={float(np.nanstd(arr)):.3f}",
                    transform=ax.transAxes, color="#ddd", fontsize=6.5, va="top",
                    bbox=dict(fc=BG, ec="#444", lw=0.5, pad=2))
            table_rows += (f"| {name} | {vlo:.3f} | {float(np.nanmean(arr)):.3f} | "
                           f"{vhi:.3f} | {float(np.nanstd(arr)):.3f} |\n")

        for i in range(n, nrows*ncols):
            r, c = divmod(i, ncols); axes[r][c].set_visible(False)

        stamp_map(fig, geo, f"Multi-Index ({n} panels)")
        plt.tight_layout(pad=2)
        tmp_png_path = _out_path("maps", "multi_index_compare", ".png")
        plt.savefig(tmp_png_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        info = f"""### ✅ Compared {n} indices  ·  {geo.get("geo_str","")}

| Index | Min | Mean | Max | Std |
|---|---|---|---|---|
{table_rows}"""
        return tmp_png_path, info
    except Exception as _e:
        return None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── TAB 4: ASTER SWIR EXPLORER ────────────────────────────────────────

def explore_aster(aster_file, s2_file, band_or_index, cmap_name, coord_sys,
                  show_all_bands, save_out):
    if aster_file is None:
        return None, None, "### ❌ Upload an ASTER SWIR GeoTIFF first."
    try:
        with rasterio.open(_fpath(aster_file)) as asrc:
            raw = asrc.read().astype("float32")
            a_nb = asrc.count
            a_transform = asrc.transform; a_crs = asrc.crs; geo = geo_info_dict(asrc)
            rows, cols = asrc.height, asrc.width
            profile = asrc.profile.copy()

        # Ensure 6 bands
        if a_nb >= 6:
            raw = raw[:6]
        else:
            pad = np.zeros((6 - a_nb, rows, cols), dtype="float32")
            raw = np.vstack([raw, pad])

        # If S2 reference provided, resample ASTER to match
        if s2_file is not None:
            try:
                with rasterio.open(_fpath(s2_file)) as sref:
                    ref_h, ref_w = sref.height, sref.width
                    ref_t, ref_c = sref.transform, sref.crs
                raw = resample_aster_to_sentinel(raw, a_transform, a_crs,
                                                 ref_t, ref_c, ref_h, ref_w)
                rows, cols = ref_h, ref_w
                with rasterio.open(_fpath(s2_file)) as _geo_src:
                    geo = geo_info_dict(_geo_src)
                a_transform = ref_t; a_crs = ref_c
            except Exception as e:
                print(f"ASTER resample warning: {e}")

        aster_indices = compute_aster_indices(raw)

        if show_all_bands:
            # Show all 6 ASTER bands + 12 indices = 18 panels
            n_panels = 18
            ncols = 4; nrows = (n_panels + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols, 8*nrows),
                                      facecolor=BG, squeeze=False)
            fig.suptitle("ASTER SWIR — All Bands & Geological Indices",
                         color=GOLD, fontsize=16, fontweight="bold", y=0.995)
            all_arrs = (
                [(f"ASTER Band {i+4} ({ASTER_BAND_INFO[i][0]})", raw[i], "gray")
                 for i in range(6)] +
                [(f"{k} — {ASTER_INDICES[k][0]}", v, ASTER_INDICES[k][2])
                 for k, v in aster_indices.items()]
            )
            for i, (title, arr, cm) in enumerate(all_arrs[:n_panels]):
                ri, ci = divmod(i, ncols); ax = axes[ri][ci]; ax.set_facecolor(BG)
                vlo = np.nanpercentile(arr, 2); vhi = np.nanpercentile(arr, 98)
                im = ax.imshow(arr, cmap=cm, vmin=vlo, vmax=vhi, interpolation="bilinear")
                style_ax(ax, title, fs=9)
                fig.colorbar(im, ax=ax, shrink=0.7).ax.tick_params(colors="#ccc", labelsize=7)
                add_north_arrow(ax); add_scalebar(ax, rows, cols)
                ax.text(0.02, 0.985, f"μ={float(np.nanmean(arr)):.4f}",
                        transform=ax.transAxes, color="#ccc", fontsize=7, va="top",
                        bbox=dict(fc=BG, ec="#333", lw=0.5, pad=2))
                ax.axis("off")
            for i in range(n_panels, nrows*ncols):
                ri, ci = divmod(i, ncols); axes[ri][ci].set_visible(False)
        else:
            # Single index / band view
            if band_or_index in ASTER_INDICES:
                arr = aster_indices[band_or_index]
                title = f"{band_or_index} — {ASTER_INDICES[band_or_index][0]}"
            else:
                # Try band number
                for i, info in ASTER_BAND_INFO.items():
                    if band_or_index in (info[0], info[1]):
                        arr = raw[i]; title = info[1]; break
                else:
                    arr = raw[0]; title = "ASTER Band 4"

            fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
            ax = axes[0]; ax.set_facecolor(BG)
            vlo = np.nanpercentile(arr, 2); vhi = np.nanpercentile(arr, 98)
            im = ax.imshow(arr, cmap=cmap_name, vmin=vlo, vmax=vhi, interpolation="bilinear")
            style_ax(ax, title, fs=13)
            cb = fig.colorbar(im, ax=ax, shrink=0.8)
            cb.set_label("Value", color="#ccc"); cb.ax.tick_params(colors="#ccc")
            add_north_arrow(ax); add_scalebar(ax, rows, cols)
            ax.text(0.02, 0.06,
                    f"Min: {float(np.nanmin(arr)):.4f}\nMean: {float(np.nanmean(arr)):.4f}\n"
                    f"Max: {float(np.nanmax(arr)):.4f}\nStd: {float(np.nanstd(arr)):.4f}",
                    transform=ax.transAxes, color="white", fontsize=8,
                    fontfamily="monospace", va="bottom",
                    bbox=dict(fc="#0f0f18", ec="#333", lw=0.8, pad=4, alpha=0.92))
            ax.axis("off")

            # Histogram
            ax2 = axes[1]; ax2.set_facecolor(BG)
            flat = arr[~np.isnan(arr)].flatten()
            n_b = 100; counts, edges = np.histogram(flat, bins=n_b)
            ax2.bar(edges[:-1], counts, width=(edges[1]-edges[0]),
                    color=_get_cmap(cmap_name)(np.linspace(0,1,n_b)),
                    edgecolor="none", alpha=0.9)
            ax2.axvline(float(np.nanmean(arr)), color="white", lw=1.5, ls="--", label="mean")
            ax2.axvline(float(np.nanpercentile(arr, 80)), color="#f0c040", lw=1.2, ls=":",
                        label="p80")
            ax2.set_title(f"Distribution — {band_or_index}", color="white", fontweight="bold")
            ax2.set_xlabel("Value", color="#aaa"); ax2.set_ylabel("Count", color="#aaa")
            ax2.tick_params(colors="#777")
            ax2.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=9)
            for sp in ax2.spines.values(): sp.set_edgecolor("#333")

        stamp_map(fig, geo, f"ASTER SWIR: {band_or_index}")
        plt.tight_layout(pad=1.5)
        tmp_png_path = _out_path("maps", f"aster_{band_or_index}", ".png")
        plt.savefig(tmp_png_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        out_tif = None
        if save_out and not show_all_bands:
            clean_p = {"driver":"GTiff","dtype":"float32","nodata":float("nan"),
                       "count":1,"crs":str(a_crs),"transform":a_transform,
                       "width":cols,"height":rows,"compress":"lzw"}
            out_tif = _out_path("geotiff", f"aster_{band_or_index}", ".tif")
            with rasterio.open(out_tif, "w", **clean_p) as dst:
                dst.write(arr, 1)

        # Stats table for all ASTER indices
        idx_rows = ""
        for k, v in aster_indices.items():
            idx_rows += (f"| {k} | {ASTER_INDICES[k][0]} | {float(np.nanmean(v)):.4f} | "
                         f"{float(np.nanmax(v)):.4f} | {float(np.nanstd(v)):.4f} |\n")

        stats = f"""
### 🌋 ASTER SWIR Explorer

| Index | Name | Mean | Max | Std |
|---|---|---|---|---|
{idx_rows}

**Coverage:** {geo.get('geo_str','—')}  
**Grid:** {cols}×{rows} px  
**ASTER bands loaded:** {a_nb} (padded to 6)
"""
        return tmp_png_path, out_tif, stats
    except Exception as _e:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── TAB 5: HYDROTHERMAL ALTERATION MAP ───────────────────────────────

def map_hydrothermal(aster_file, s2_file, threshold_pct, coord_sys, export_mask):
    if aster_file is None:
        return None, None, "### ❌ Upload an ASTER SWIR GeoTIFF first."
    try:
        with rasterio.open(_fpath(aster_file)) as asrc:
            raw = asrc.read().astype("float32")
            a_nb_h = asrc.count
            a_transform = asrc.transform; a_crs = asrc.crs
            rows, cols = asrc.height, asrc.width
            geo = geo_info_dict(asrc)
            profile = asrc.profile.copy()
        if a_nb_h >= 6:
            raw = raw[:6]
        else:
            pad = np.zeros((6 - a_nb_h, rows, cols), dtype="float32")
            raw = np.vstack([raw, pad])

        if s2_file is not None:
            try:
                with rasterio.open(_fpath(s2_file)) as sref:
                    ref_h, ref_w = sref.height, sref.width
                    ref_t, ref_c = sref.transform, sref.crs
                raw = resample_aster_to_sentinel(raw, a_transform, a_crs, ref_t, ref_c, ref_h, ref_w)
                rows, cols = ref_h, ref_w; a_transform = ref_t; a_crs = ref_c
                with rasterio.open(_fpath(s2_file)) as _geo_src2:
                    geo = geo_info_dict(_geo_src2)
            except Exception as _e:
                import warnings; warnings.warn(f"⚠️ ASTER resampling failed — result may be misaligned: {_e}")

        idx = compute_aster_indices(raw)
        ha  = idx["AST_HydAlt"]
        aloh = idx["AST_AlOH"]
        mgoh = idx["AST_MgOH"]
        clay = idx["AST_Clay"]
        silica = idx["AST_Silica"]
        quartz = idx["AST_Quartz"]
        ferric = idx["AST_Ferric"]

        # Threshold for "altered" pixels
        ha_thresh = float(np.nanpercentile(ha[~np.isnan(ha)], threshold_pct))

        # Hydrothermal intensity classification
        ha_class = np.full_like(ha, np.nan)
        ha_valid = ~np.isnan(ha)
        p60 = float(np.nanpercentile(ha[ha_valid], 60))
        p80 = float(np.nanpercentile(ha[ha_valid], 80))
        p95 = float(np.nanpercentile(ha[ha_valid], 95))
        ha_class[ha_valid & (ha <= p60)] = 0
        ha_class[ha_valid & (ha > p60) & (ha <= p80)] = 1
        ha_class[ha_valid & (ha > p80) & (ha <= p95)] = 2
        ha_class[ha_valid & (ha > p95)] = 3

        fig = plt.figure(figsize=(24, 18), facecolor=BG)
        fig.suptitle("🌋 Hydrothermal Alteration Mapping — ASTER SWIR",
                     color=GOLD, fontsize=18, fontweight="bold", y=0.995)
        gs = GridSpec(3, 4, figure=fig, left=0.04, right=0.97,
                      top=0.96, bottom=0.06, hspace=0.35, wspace=0.22)

        # Panel 1: Hydrothermal Alteration Index
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(ha, cmap="RdYlGn_r",
                         vmin=np.nanpercentile(ha, 2), vmax=np.nanpercentile(ha, 98))
        style_ax(ax1, "Hydrothermal Alteration Index  (AST_HydAlt)", fs=13)
        add_cbar(fig, im1, ax1, "Index"); add_north_arrow(ax1); add_scalebar(ax1, rows, cols)
        ax1.contour(np.nan_to_num(ha), levels=[ha_thresh], colors=["#00ffff"], linewidths=0.8)
        ax1.text(0.02, 0.05, f"Threshold p{threshold_pct}: {ha_thresh:.4f}",
                 transform=ax1.transAxes, color="#00ffff", fontsize=8,
                 bbox=dict(fc=BG, ec="#00ffff", lw=0.5, pad=2)); ax1.axis("off")

        # Panel 2: Intensity classification
        ax2 = fig.add_subplot(gs[0, 2:])
        cmap4 = mcolors.ListedColormap(["#1a3a2a","#f4a261","#e63946","#ff00aa"])
        ax2.imshow(ha_class, cmap=cmap4, vmin=0, vmax=3, interpolation="nearest")
        style_ax(ax2, "Hydrothermal Intensity Classification", fs=13)
        legs = [Patch(fc="#ff00aa", ec="none", label="Intense  (top 5%)"),
                Patch(fc="#e63946", ec="none", label="High     (p80-p95)"),
                Patch(fc="#f4a261", ec="none", label="Moderate (p60-p80)"),
                Patch(fc="#1a3a2a", ec="none", label="Low      (< p60)")]
        ax2.legend(handles=legs, loc="lower right", facecolor="#111",
                   edgecolor="#444", labelcolor="white", fontsize=9, framealpha=0.9)
        add_north_arrow(ax2); add_scalebar(ax2, rows, cols); ax2.axis("off")

        # Panel 3: AlOH
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(aloh, cmap="YlOrBr",
                         vmin=np.nanpercentile(aloh, 2), vmax=np.nanpercentile(aloh, 98))
        style_ax(ax3, "AlOH Index  (Kaolinite/Alunite)", fs=10)
        add_cbar(fig, im3, ax3, "AlOH"); ax3.axis("off")

        # Panel 4: MgOH
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(mgoh, cmap="copper",
                         vmin=np.nanpercentile(mgoh, 2), vmax=np.nanpercentile(mgoh, 98))
        style_ax(ax4, "MgOH Index  (Chlorite/Carbonate)", fs=10)
        add_cbar(fig, im4, ax4, "MgOH"); ax4.axis("off")

        # Panel 5: Silica
        ax5 = fig.add_subplot(gs[1, 2])
        im5 = ax5.imshow(silica, cmap="plasma",
                         vmin=np.nanpercentile(silica, 2), vmax=np.nanpercentile(silica, 98))
        style_ax(ax5, "Silica Index  (Quartz/Silicification)", fs=10)
        add_cbar(fig, im5, ax5, "Silica"); ax5.axis("off")

        # Panel 6: Ferric
        ax6 = fig.add_subplot(gs[1, 3])
        im6 = ax6.imshow(ferric, cmap="hot",
                         vmin=np.nanpercentile(ferric, 2), vmax=np.nanpercentile(ferric, 98))
        style_ax(ax6, "Ferric Iron Index  (B4/B5)", fs=10)
        add_cbar(fig, im6, ax6, "Ferric"); ax6.axis("off")

        # Panel 7: Summary stats histograms
        for ci, (name, arr, cm) in enumerate([("HydAlt", ha, "RdYlGn_r"),
                                               ("AlOH",   aloh, "YlOrBr"),
                                               ("Silica", silica, "plasma"),
                                               ("Ferric", ferric, "hot")]):
            axh = fig.add_subplot(gs[2, ci]); axh.set_facecolor(BG)
            flat = arr[~np.isnan(arr)].flatten()
            n_b = 80; counts, edges = np.histogram(flat, bins=n_b)
            axh.bar(edges[:-1], counts, width=(edges[1]-edges[0]),
                    color=_get_cmap(cm)(np.linspace(0,1,n_b)),
                    edgecolor="none", alpha=0.9)
            axh.axvline(float(np.nanmean(flat)), color="white", lw=1.5, ls="--",
                        label=f"μ={np.nanmean(flat):.3f}")
            axh.axvline(float(np.nanpercentile(flat, 80)), color="#f0c040", lw=1, ls=":",
                        label="p80")
            axh.set_title(f"{name} Distribution", color="white", fontsize=9, fontweight="bold")
            axh.tick_params(colors="#777", labelsize=7)
            axh.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=7)
            for sp in axh.spines.values(): sp.set_edgecolor("#333")

        stamp_map(fig, geo, "Hydrothermal Alteration")
        tmp_png_path = _out_path("maps", "hydrothermal_alteration", ".png")
        plt.savefig(tmp_png_path, dpi=160, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        out_tif = None
        if export_mask:
            mask_arr = (ha >= ha_thresh).astype(np.float32)
            mask_arr[np.isnan(ha)] = np.nan
            clean_p = {"driver":"GTiff","dtype":"float32","nodata":float("nan"),
                       "count":1,"crs":str(a_crs),"transform":a_transform,
                       "width":cols,"height":rows,"compress":"lzw"}
            out_tif = _out_path("geotiff", "hydrothermal_mask", ".tif")
            with rasterio.open(out_tif, "w", **clean_p) as dst:
                dst.write(mask_arr, 1)
                dst.update_tags(1, description="Hydrothermal alteration mask",
                                threshold_pct=str(threshold_pct))

        n_altered = int(np.nansum(ha >= ha_thresh))
        pm2 = PIXEL_SIZE_M**2 / 1e6

        md = f"""
### 🌋 Hydrothermal Alteration Analysis

| Metric | Value |
|---|---|
| **Threshold percentile** | p{threshold_pct} = {ha_thresh:.4f} |
| **Altered pixels** | {n_altered:,} ({n_altered*100/(rows*cols):.2f}%) |
| **Altered area** | **{n_altered*pm2:.2f} km²** |
| **Intense alteration** | {int(np.nansum(ha > p95)):,} px ({int(np.nansum(ha > p95))*pm2:.2f} km²) |

| Index | Mean | Interpretation |
|---|---|---|
| AlOH | {float(np.nanmean(aloh)):.4f} | {'Strong Al-OH (kaolinite/alunite)' if np.nanmean(aloh)>1.05 else 'Moderate'} |
| MgOH | {float(np.nanmean(mgoh)):.4f} | {'Mg-OH (chlorite/carbonate)' if np.nanmean(mgoh)>1.05 else 'Moderate'} |
| Silica | {float(np.nanmean(silica)):.4f} | {'Silicification' if np.nanmean(silica)>1.0 else 'Background'} |
| Ferric | {float(np.nanmean(ferric)):.4f} | {'Oxidised iron (gossan)' if np.nanmean(ferric)>1.2 else 'Background'} |
"""
        return tmp_png_path, out_tif, md
    except Exception as _e:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── TAB 6: MINERAL MAPPING (Quartz/Silica + Clay) ────────────────────

def map_minerals(aster_file, s2_file, quartz_pct, clay_pct, coord_sys, export_polygons):
    if aster_file is None:
        return None, None, "### ❌ Upload an ASTER SWIR GeoTIFF first."
    try:
        with rasterio.open(_fpath(aster_file)) as asrc:
            raw = asrc.read().astype("float32")
            a_nb_m = asrc.count
            a_transform = asrc.transform; a_crs = asrc.crs
            rows, cols = asrc.height, asrc.width
            geo = geo_info_dict(asrc)
        if a_nb_m >= 6:
            raw = raw[:6]
        else:
            pad = np.zeros((6 - a_nb_m, rows, cols), dtype="float32")
            raw = np.vstack([raw, pad])

        if s2_file is not None:
            try:
                with rasterio.open(_fpath(s2_file)) as sref:
                    ref_h, ref_w = sref.height, sref.width
                    ref_t, ref_c = sref.transform, sref.crs
                raw = resample_aster_to_sentinel(raw, a_transform, a_crs, ref_t, ref_c, ref_h, ref_w)
                rows, cols = ref_h, ref_w; a_transform = ref_t; a_crs = ref_c
                with rasterio.open(_fpath(s2_file)) as _geo_src3:
                    geo = geo_info_dict(_geo_src3)
            except Exception as _e:
                import warnings; warnings.warn(f"⚠️ ASTER resampling failed — result may be misaligned: {_e}")

        idx = compute_aster_indices(raw)
        silica = idx["AST_Silica"]; quartz = idx["AST_Quartz"]
        clay   = idx["AST_Clay"];   aloh   = idx["AST_AlOH"]
        mgoh   = idx["AST_MgOH"];   ha     = idx["AST_HydAlt"]

        # Threshold detection
        q_thresh = float(np.nanpercentile(quartz[~np.isnan(quartz)], quartz_pct))
        c_thresh = float(np.nanpercentile(clay[~np.isnan(clay)], clay_pct))

        q_mask = (quartz >= q_thresh).astype(np.float32); q_mask[np.isnan(quartz)] = np.nan
        c_mask = (clay   >= c_thresh).astype(np.float32); c_mask[np.isnan(clay)]   = np.nan

        # Combined mineral map
        mineral_class = np.zeros((rows, cols), dtype=np.float32)
        mineral_class[~np.isnan(quartz) & (quartz >= q_thresh)] = 2  # Quartz/Silica
        mineral_class[~np.isnan(clay)   & (clay   >= c_thresh)] = 1  # Clay
        # Overlap = both
        mineral_class[~np.isnan(quartz) & ~np.isnan(clay) &
                      (quartz >= q_thresh) & (clay >= c_thresh)] = 3

        fig = plt.figure(figsize=(24, 18), facecolor=BG)
        fig.suptitle("💎 Mineral Discrimination — Quartz/Silica & Clay Mapping",
                     color=GOLD, fontsize=18, fontweight="bold", y=0.995)
        gs = GridSpec(3, 4, figure=fig, left=0.04, right=0.97,
                      top=0.96, bottom=0.06, hspace=0.35, wspace=0.22)

        # Panel 1: Quartz/Silica
        ax1 = fig.add_subplot(gs[0, :2])
        im1 = ax1.imshow(quartz, cmap="inferno",
                         vmin=np.nanpercentile(quartz, 2), vmax=np.nanpercentile(quartz, 98))
        style_ax(ax1, "Quartz Index  (ASTER: B5×B7/B6²)", fs=13)
        add_cbar(fig, im1, ax1, "Quartz Index")
        ax1.contour(np.nan_to_num(quartz), levels=[q_thresh], colors=["#00ffcc"], linewidths=1.2)
        ax1.text(0.02, 0.05, f"Quartz threshold p{quartz_pct}: {q_thresh:.4f}",
                 transform=ax1.transAxes, color="#00ffcc", fontsize=8,
                 bbox=dict(fc=BG, ec="#00ffcc", lw=0.5, pad=2)); ax1.axis("off")

        # Panel 2: Clay
        ax2 = fig.add_subplot(gs[0, 2:])
        im2 = ax2.imshow(clay, cmap="YlOrBr",
                         vmin=np.nanpercentile(clay, 2), vmax=np.nanpercentile(clay, 98))
        style_ax(ax2, "Clay Mineral Index  (ASTER: (B5+B7)/B6)", fs=13)
        add_cbar(fig, im2, ax2, "Clay Index")
        ax2.contour(np.nan_to_num(clay), levels=[c_thresh], colors=["#ff6600"], linewidths=1.2)
        ax2.text(0.02, 0.05, f"Clay threshold p{clay_pct}: {c_thresh:.4f}",
                 transform=ax2.transAxes, color="#ff6600", fontsize=8,
                 bbox=dict(fc=BG, ec="#ff6600", lw=0.5, pad=2)); ax2.axis("off")

        # Panel 3: Combined mineral map
        ax3 = fig.add_subplot(gs[1, :2])
        cmap_m = mcolors.ListedColormap(["#1a2a1a","#c77a20","#7b00ff","#ff2090"])
        ax3.imshow(mineral_class, cmap=cmap_m, vmin=0, vmax=3, interpolation="nearest")
        style_ax(ax3, "Combined Mineral Discrimination Map", fs=13)
        legs_m = [Patch(fc="#ff2090", ec="none", label="Quartz + Clay (hydrothermal core)"),
                  Patch(fc="#7b00ff", ec="none", label="Quartz / Silica alteration"),
                  Patch(fc="#c77a20", ec="none", label="Clay minerals"),
                  Patch(fc="#1a2a1a", ec="none", label="Background")]
        ax3.legend(handles=legs_m, loc="lower right", facecolor="#111",
                   edgecolor="#444", labelcolor="white", fontsize=9, framealpha=0.9)
        add_north_arrow(ax3); add_scalebar(ax3, rows, cols); ax3.axis("off")

        # Panel 4: AlOH vs MgOH scatter
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.set_facecolor(BG)
        aloh_f = aloh.flatten(); mgoh_f = mgoh.flatten()
        v = ~np.isnan(aloh_f) & ~np.isnan(mgoh_f)
        step = max(1, int(np.sum(v) / 8000))
        sc = ax4.scatter(aloh_f[v][::step], mgoh_f[v][::step],
                         c=ha.flatten()[v][::step],
                         cmap="RdYlGn_r", s=2, alpha=0.5,
                         vmin=np.nanpercentile(ha, 10), vmax=np.nanpercentile(ha, 90))
        fig.colorbar(sc, ax=ax4).set_label("HydAlt", color="#ccc", fontsize=8)
        ax4.set_xlabel("AlOH Index", color="#aaa"); ax4.set_ylabel("MgOH Index", color="#aaa")
        ax4.set_title("AlOH vs MgOH  (coloured by HydAlt)", color="white", fontweight="bold")
        ax4.tick_params(colors="#777")
        for sp in ax4.spines.values(): sp.set_edgecolor("#333")
        ax4.axhline(float(np.nanmean(mgoh_f[v])), color="#555", lw=0.8, ls="--")
        ax4.axvline(float(np.nanmean(aloh_f[v])), color="#555", lw=0.8, ls="--")

        # Bottom: histograms
        for ci, (name, arr, cm) in enumerate([("Quartz", quartz, "inferno"),
                                               ("Clay", clay, "YlOrBr"),
                                               ("Silica", silica, "plasma"),
                                               ("AlOH", aloh, "YlOrBr")]):
            axh = fig.add_subplot(gs[2, ci]); axh.set_facecolor(BG)
            flat = arr[~np.isnan(arr)].flatten()
            n_b = 80; counts, edges = np.histogram(flat, bins=n_b)
            axh.bar(edges[:-1], counts, width=(edges[1]-edges[0]),
                    color=_get_cmap(cm)(np.linspace(0,1,n_b)),
                    edgecolor="none", alpha=0.9)
            axh.axvline(float(np.nanmean(flat)), color="white", lw=1.5, ls="--",
                        label=f"μ={np.nanmean(flat):.3f}")
            axh.set_title(f"{name}", color="white", fontsize=9, fontweight="bold")
            axh.tick_params(colors="#777", labelsize=7)
            axh.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=7)
            for sp in axh.spines.values(): sp.set_edgecolor("#333")

        stamp_map(fig, geo, "Mineral Discrimination")
        tmp_png_path = _out_path("maps", "mineral_mapping", ".png")
        plt.savefig(tmp_png_path, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        out_tif = None
        if export_polygons:
            clean_p = {"driver":"GTiff","dtype":"float32","nodata":float("nan"),
                       "count":1,"crs":str(a_crs),"transform":a_transform,
                       "width":cols,"height":rows,"compress":"lzw"}
            out_tif = _out_path("geotiff", "mineral_classes", ".tif")
            with rasterio.open(out_tif, "w", **clean_p) as dst:
                dst.write(mineral_class, 1)
                dst.update_tags(1, description="0=background 1=clay 2=quartz 3=both",
                                quartz_pct=str(quartz_pct), clay_pct=str(clay_pct))

        pm2 = PIXEL_SIZE_M**2 / 1e6
        n_q = int(np.nansum(mineral_class >= 2)); n_c = int(np.nansum((mineral_class==1)|(mineral_class==3)))
        n_both = int(np.nansum(mineral_class == 3))

        md = f"""
### 💎 Mineral Discrimination Results

| Mineral Zone | Pixels | Area (km²) |
|---|---|---|
| 🟣 Quartz / Silica | {n_q:,} | {n_q*pm2:.2f} |
| 🟠 Clay minerals | {n_c:,} | {n_c*pm2:.2f} |
| 🔴 Quartz + Clay (hydrothermal core) | {n_both:,} | {n_both*pm2:.2f} |

**Quartz vein probability:** {'High' if n_q*pm2 > 5 else 'Moderate' if n_q*pm2 > 1 else 'Low'}  
**Clay alteration intensity:** {'Strong' if float(np.nanmean(clay[~np.isnan(clay)])) > 1.2 else 'Moderate' if float(np.nanmean(clay[~np.isnan(clay)])) > 1.0 else 'Weak'}
"""
        return tmp_png_path, out_tif, md
    except Exception as _e:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── TAB 7: TRAINING (upgraded with ASTER features + calibration) ──────

SITE_TYPE_CHOICES = ["Gold (positive + bg)", "No-Gold (background only)", "Training Proxy (no label)"]

def run_training(feat_files, aster_files, lbl_files, use_labels,
                 site_names_str, site_types_str, n_estimators, max_depth_str,
                 use_ensemble, use_calibration, model_algo, use_shap,
                 use_optuna, save_path,
                 progress=gr.Progress()):
    global model_bundle, model_status
    from sklearn.ensemble import (RandomForestClassifier,
                                  GradientBoostingClassifier,
                                  VotingClassifier)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (classification_report, roc_auc_score,
                                 ConfusionMatrixDisplay,
                                 roc_curve, precision_recall_curve, average_precision_score,
                                 f1_score)
    logs = []
    if not feat_files:
        return None, None, "❌ Upload at least one Sentinel-2 feature TIF."

    site_names = [s.strip() for s in site_names_str.split(",") if s.strip()]
    while len(site_names) < len(feat_files): site_names.append(f"Site {len(site_names)+1}")

    raw_types = [t.strip() for t in site_types_str.split(",")]
    site_types = []
    for i in range(len(feat_files)):
        t = raw_types[i] if i < len(raw_types) else "gold"
        t_lo = t.lower()
        if "no" in t_lo or "neg" in t_lo or "back" in t_lo: site_types.append("no-gold")
        elif "proxy" in t_lo or "train" in t_lo: site_types.append("proxy")
        else: site_types.append("gold")

    max_depth = None if max_depth_str.strip().lower()=="none" else int(max_depth_str.strip())
    X_all, y_all = [], []
    has_aster = bool(aster_files and any(f is not None for f in aster_files))

    try:
        for idx, feat_file in enumerate(feat_files):
            site  = site_names[idx]; stype = site_types[idx]
            progress(idx / len(feat_files), desc=f"Loading {site} [{stype}]…")
            logs.append(f"\n📍 {site}  [{stype}]")

            with rasterio.open(_fpath(feat_file)) as src:
                nb = src.count
                if nb < REQUIRED_BANDS:
                    return None, None, f"❌ {site}: {nb} bands, need {REQUIRED_BANDS}."
                s2_feat = src.read().astype("float32")
                h, w = src.height, src.width
                ref_t = src.transform; ref_c = src.crs

            # Load ASTER if available for this site
            aster_feat = None
            if aster_files and idx < len(aster_files) and aster_files[idx] is not None:
                try:
                    with rasterio.open(_fpath(aster_files[idx])) as asrc:
                        raw_a = asrc.read().astype("float32")[:6]
                        if raw_a.shape[0] < 6:
                            pad = np.zeros((6-raw_a.shape[0], asrc.height, asrc.width), dtype="float32")
                            raw_a = np.vstack([raw_a, pad])
                        aster_feat = resample_aster_to_sentinel(
                            raw_a, asrc.transform, asrc.crs, ref_t, ref_c, h, w)
                    logs.append(f"   🛰 ASTER SWIR loaded for {site}")
                except Exception as ae:
                    logs.append(f"   ⚠ ASTER load failed: {ae}")

            # Build feature matrix
            if aster_feat is not None:
                feat_matrix = build_hybrid_features(s2_feat, aster_feat, "Hybrid Sentinel-2 + ASTER")
            else:
                feat_matrix = s2_feat.reshape(nb, -1).T

            valid = ~np.any(np.isnan(feat_matrix), axis=1)

            if stype == "no-gold":
                X_site = feat_matrix[valid]; y_site = np.zeros(len(X_site), dtype=int)
                logs.append(f"   ℹ️ Negative site — {len(y_site):,} px → background")
            elif stype == "proxy":
                iron = s2_feat[9].flatten()
                iron_valid = iron[valid]
                thr90 = np.nanpercentile(iron_valid, 90)
                y_site = np.where(iron_valid >= thr90, 1, 0)
                X_site = feat_matrix[valid]
                logs.append(f"   ℹ️ Proxy — IO p90 = {thr90:.4f}")
            else:
                if use_labels and lbl_files and idx < len(lbl_files) and lbl_files[idx] is not None:
                    with rasterio.open(_fpath(lbl_files[idx])) as _lsrc:
                        lbl_arr = _lsrc.read(1).astype("float32")
                    if lbl_arr.shape != (h, w):
                        return None, None, f"❌ {site}: label shape mismatch"
                    y_site   = lbl_arr.flatten()
                    labelled = np.isin(y_site, [0, 1])
                    valid    = valid & labelled
                    X_site   = feat_matrix[valid]; y_site = y_site[valid].astype(int)
                    logs.append(f"   ✔ Supervised labels loaded")
                else:
                    iron = s2_feat[9].flatten()
                    iron_valid = iron[valid]
                    thr90 = np.nanpercentile(iron_valid, 90)
                    y_site = np.where(iron_valid >= thr90, 1, 0)
                    X_site = feat_matrix[valid]
                    logs.append(f"   ℹ️ No label — IO p90 proxy")

            n1 = int(y_site.sum()); n0 = int((y_site == 0).sum())
            logs.append(f"   ✔ {len(y_site):,} px  gold={n1:,}  bg={n0:,}  features={X_site.shape[1]}")
            if stype != "no-gold" and (n1 == 0 or n0 == 0):
                logs.append(f"   ⚠️ Skipping — single class"); continue
            X_all.append(X_site); y_all.append(y_site)

        if not X_all:
            return None, None, "❌ No valid sites loaded."

        X = np.vstack(X_all); y = np.concatenate(y_all)

        # Feature engineering on S2 portion only (first 18 features)
        progress(0.45, desc="Engineering S2 features…")
        s2_cols = X[:, :18]
        eng_extra = _engineer_features(s2_cols)[:, 18:]  # 20 extra S2 engineered features (v10)
        X_eng = np.hstack([X, eng_extra])  # All features including ASTER
        logs.append(f"\n📦 Total {len(y):,}  gold={int(y.sum()):,}  bg={int((y==0).sum()):,}")
        logs.append(f"   Features: {X.shape[1]} raw → {X_eng.shape[1]} engineered")

        X_tr, X_te, y_tr, y_te = train_test_split(X_eng, y, test_size=0.20, random_state=42, stratify=y)

        progress(0.55, desc="Training model…")
        n_est = int(n_estimators)
        algo  = (model_algo or "Random Forest").strip()

        # ── Optuna hyperparameter search (optional) ───────────────────
        if use_optuna and _HAS_OPTUNA and len(X_tr) < 300_000:
            progress(0.57, desc="Optuna hyperparameter search (50 trials)…")
            logs.append("   🔍 Optuna HPO: 50 trials")
            def _optuna_objective(trial):
                _n = trial.suggest_int("n_est", 50, 500, step=50)
                _d = trial.suggest_int("max_depth", 3, 12)
                _mf = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
                _rf = RandomForestClassifier(n_estimators=_n, max_depth=_d,
                                             max_features=_mf, n_jobs=-1, random_state=42,
                                             class_weight="balanced_subsample",
                                             min_samples_leaf=2)
                cv_ = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                sc  = cross_val_score(_rf, X_tr, y_tr, cv=cv_, scoring="roc_auc", n_jobs=-1)
                return sc.mean()
            study = optuna.create_study(direction="maximize")
            study.optimize(_optuna_objective, n_trials=50, timeout=120)
            best = study.best_params
            n_est = best.get("n_est", n_est)
            max_depth = best.get("max_depth", max_depth)
            logs.append(f"   ✅ Best params: {best}  AUC={study.best_value:.4f}")
        elif use_optuna and not _HAS_OPTUNA:
            logs.append("   ⚠ Optuna not installed — skipping HPO")

        # ── Build classifier ──────────────────────────────────────────
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth,
                                    n_jobs=-1, random_state=42,
                                    class_weight="balanced_subsample",
                                    min_samples_leaf=2, max_features="sqrt")
        gb = GradientBoostingClassifier(n_estimators=min(n_est, 300),
                                        max_depth=4 if max_depth is None else min(int(max_depth), 4),
                                        learning_rate=0.05, subsample=0.8,
                                        min_samples_leaf=10, random_state=42)

        estimators = [("rf", rf)]
        if algo in ("Ensemble RF+GB", "Full Ensemble RF+GB+XGB+LGBM"):
            estimators.append(("gb", gb))
        if algo in ("XGBoost", "Full Ensemble RF+GB+XGB+LGBM") and _HAS_XGB:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=max_depth or 6,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss",
                scale_pos_weight=float((y_tr==0).sum()) / max(1, float((y_tr==1).sum())),
                random_state=42, n_jobs=-1)
            estimators.append(("xgb", xgb_clf))
            logs.append("   ⚡ XGBoost added to ensemble")
        elif algo == "XGBoost" and not _HAS_XGB:
            logs.append("   ⚠ XGBoost not installed — using RF")
        if algo in ("LightGBM", "Full Ensemble RF+GB+XGB+LGBM") and _HAS_LGB:
            lgb_clf = lgb.LGBMClassifier(
                n_estimators=n_est, max_depth=max_depth or -1,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                class_weight="balanced", random_state=42, n_jobs=-1,
                verbose=-1)
            estimators.append(("lgb", lgb_clf))
            logs.append("   ⚡ LightGBM added to ensemble")
        elif algo == "LightGBM" and not _HAS_LGB:
            logs.append("   ⚠ LightGBM not installed — using RF")

        if len(estimators) > 1:
            weights_map = {"rf": 2, "gb": 1, "xgb": 2, "lgb": 2}
            weights = [weights_map.get(k, 1) for k, _ in estimators]
            clf = VotingClassifier(estimators=estimators, voting="soft",
                                   weights=weights, n_jobs=-1)
            model_mode = "+".join(k.upper() for k, _ in estimators)
            logs.append(f"   🧠 Ensemble: {model_mode}  weights={weights}")
        else:
            clf = rf
            model_mode = "RandomForest"
            logs.append("   🌲 Random Forest only")

        clf.fit(X_tr, y_tr)

        # Probability calibration
        if use_calibration:
            progress(0.72, desc="Calibrating probabilities…")
            clf_cal = CalibratedClassifierCV(clf, method="isotonic", cv=3)
            clf_cal.fit(X_tr, y_tr)
            clf_final = clf_cal
            logs.append("   📐 Probability calibration: CalibratedClassifierCV (isotonic)")
        else:
            clf_final = clf

        model_mode = ("Calibrated-" if use_calibration else "") + \
                     ("+".join(k.upper() for k, _ in estimators) if len(estimators) > 1 else "RandomForest")

        progress(0.80, desc="Evaluating…")
        y_pred  = clf_final.predict(X_te)
        y_proba = clf_final.predict_proba(X_te)[:, 1]
        auc  = roc_auc_score(y_te, y_proba)
        apr  = average_precision_score(y_te, y_proba)
        report = classification_report(y_te, y_pred, target_names=["Background", "Gold"])
        logs.append(f"\nAUC={auc:.4f}   AP={apr:.4f}\n{report}")

        # ── Youden-J optimal threshold ────────────────────────────────
        from sklearn.metrics import roc_curve as _roc
        _fpr, _tpr, _thrs = _roc(y_te, y_proba)
        youden_idx = np.argmax(_tpr - _fpr)
        youden_thr = float(_thrs[youden_idx])
        f1s = [f1_score(y_te, (y_proba >= t).astype(int), zero_division=0) for t in _thrs]
        f1_thr = float(_thrs[np.argmax(f1s)])
        logs.append(f"   📐 Youden-J threshold: {youden_thr:.4f}  F1-opt threshold: {f1_thr:.4f}")

        if len(y_tr) < 200_000:
            # CV is run on the training split only to avoid leaking the held-out test set.
            # We use the base (un-calibrated) clf for speed; calibration adds a ~constant
            # probability shift and does not significantly affect the AUC ranking.
            try:
                cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cvs = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
                logs.append(f"   5-fold CV AUC: {cvs.mean():.4f} ± {cvs.std():.4f}")
            except Exception as _cv_e:
                cvs = None
                logs.append(f"   ⚠️ CV failed: {_cv_e}")
        else:
            cvs = None; logs.append("   ℹ️ CV skipped (>200k train px)")

        # Feature importances from RF component (unwrap calibration/voting if needed)
        if len(estimators) > 1:
            # VotingClassifier – get the RF sub-estimator
            try:
                rf_part = clf.estimators_[0]
            except (AttributeError, IndexError):
                rf_part = clf
        else:
            rf_part = clf
        # Unwrap CalibratedClassifierCV if needed
        if hasattr(rf_part, "calibrated_classifiers_"):
            try:
                rf_part = rf_part.calibrated_classifiers_[0].estimator
            except (AttributeError, IndexError):
                pass
        # Also unwrap if clf_final (calibrated wrapper) wraps a VotingClassifier
        if not hasattr(rf_part, "feature_importances_"):
            if hasattr(clf_final, "calibrated_classifiers_"):
                try:
                    _inner = clf_final.calibrated_classifiers_[0].estimator
                    if hasattr(_inner, "estimators_"):
                        rf_part = _inner.estimators_[0]
                    elif hasattr(_inner, "feature_importances_"):
                        rf_part = _inner
                except (AttributeError, IndexError):
                    pass
        # Last resort: uniform importances so the plot always renders
        feat_count = X_eng.shape[1]
        if hasattr(rf_part, "feature_importances_"):
            imp = rf_part.feature_importances_
        else:
            imp = np.ones(feat_count, dtype=np.float32) / feat_count
            logs.append("   ℹ️ Feature importances unavailable — showing uniform weights")
        s2_base   = FEATURE_NAMES
        eng_names = ["IOI","CMR","SWIRr","GAI","FII","Curve","GIdisc","VegMask",
                     "NDFI","SWIRdep","ModGoss","Chalco","Limon","IronAnom",
                     "SlopeFe","AspClay","RghFe2","OxideCap","SilS2","AlOHS2"]
        aster_base = [v[0] for v in ASTER_BAND_INFO.values()]
        aster_idx_names = [k for k in sorted(ASTER_INDICES.keys())]
        feat_names_ext = (s2_base + eng_names + aster_base + aster_idx_names)
        feat_names_ext = feat_names_ext[:feat_count]

        # ── SHAP explainability (optional) ────────────────────────────
        shap_fig = None
        if use_shap and _HAS_SHAP:
            try:
                progress(0.83, desc="Computing SHAP values…")
                shap_sample = X_te[:min(500, len(X_te))]
                if hasattr(clf_final, "calibrated_classifiers_"):
                    # CalibratedClassifierCV — unwrap to base estimator
                    _base = clf_final.calibrated_classifiers_[0].estimator
                    if hasattr(_base, "estimators_"):
                        # VotingClassifier inside calibration
                        _rf_for_shap = _base.estimators_[0]
                    else:
                        _rf_for_shap = _base
                elif hasattr(clf_final, "estimators_"):
                    _rf_for_shap = clf_final.estimators_[0]
                else:
                    _rf_for_shap = clf_final
                # Unwrap calibrated (legacy path — kept for safety)
                if hasattr(_rf_for_shap, "calibrated_classifiers_"):
                    _rf_for_shap = _rf_for_shap.calibrated_classifiers_[0].estimator
                explainer = shap.TreeExplainer(_rf_for_shap,
                                               feature_names=feat_names_ext[:_rf_for_shap.n_features_in_])
                shap_vals = explainer(shap_sample)
                shap_fig = plt.figure(figsize=(10, 8), facecolor=BG)
                shap.plots.beeswarm(shap_vals[:, :, 1] if shap_vals.values.ndim == 3 else shap_vals,
                                    max_display=20, show=False)
                shap_ax = plt.gca()
                shap_ax.set_facecolor(BG)
                shap_ax.tick_params(colors="white"); shap_ax.set_xlabel("SHAP value", color="white")
                shap_ax.set_title("SHAP Feature Importance (top 20)", color=GOLD, fontweight="bold")
                shap_out_path = _out_path("shap", "shap_beeswarm", ".png")
                shap_fig.savefig(shap_out_path, dpi=120, bbox_inches="tight", facecolor=BG)
                plt.close(shap_fig)
                logs.append(f"   🔍 SHAP beeswarm plot saved → {shap_out_path}")
            except Exception as se:
                logs.append(f"   ⚠ SHAP failed: {se}")
        elif use_shap and not _HAS_SHAP:
            logs.append("   ⚠ SHAP not installed — pip install shap")

        # ── FIGURE ──
        fig = plt.figure(figsize=(26, 20), facecolor=BG)
        fig.suptitle(f"Training Report  —  {', '.join(site_names[:len(feat_files)])}",
                     color=GOLD, fontsize=16, fontweight="bold", y=0.98)
        gs = GridSpec(2, 3, figure=fig, left=0.06, right=0.97,
                      top=0.93, bottom=0.08, hspace=0.38, wspace=0.32)

        # Feature importance
        ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(BG)
        idx_s   = np.argsort(imp)
        n_show  = min(feat_count, 35)
        idx_top = idx_s[-n_show:]
        colors_b = plt.cm.YlOrRd(imp[idx_top] / (imp.max() + 1e-9))
        ax1.barh(range(n_show), imp[idx_top], color=colors_b)
        ax1.set_yticks(range(n_show))
        ax1.set_yticklabels(
            [feat_names_ext[i] if i < len(feat_names_ext) else f"F{i}" for i in idx_top],
            color="white", fontsize=8, fontfamily="monospace")
        ax1.set_xlabel("Feature Importance", color="white", fontsize=9)
        ax1.set_title(f"Feature Importance — Top {n_show}  (RF component + ASTER indices)",
                      color=GOLD, fontweight="bold", fontsize=12)
        ax1.tick_params(colors="white")
        for sp in ax1.spines.values(): sp.set_edgecolor("#333")

        # Confusion matrix
        ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(BG)
        ConfusionMatrixDisplay.from_predictions(y_te, y_pred,
            display_labels=["Background", "Gold"], ax=ax2, colorbar=False, cmap="YlOrRd")
        ax2.set_title(f"Confusion Matrix\nAUC={auc:.3f}  AP={apr:.3f}",
                      color=GOLD, fontweight="bold", fontsize=11)
        ax2.tick_params(colors="white"); ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")
        for txt in ax2.texts: txt.set_color("white")
        for sp in ax2.spines.values(): sp.set_edgecolor("#333")
        plt.setp(ax2.get_xticklabels(), color="white")
        plt.setp(ax2.get_yticklabels(), color="white")

        # ROC
        ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor(BG)
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        ax3.plot(fpr, tpr, color=GOLD, lw=2, label=f"AUC = {auc:.3f}")
        ax3.plot([0,1],[0,1], color="#444", lw=1, ls="--")
        ax3.fill_between(fpr, tpr, alpha=0.15, color=GOLD)
        ax3.set_xlabel("FPR", color="#aaa"); ax3.set_ylabel("TPR", color="#aaa")
        ax3.set_title("ROC Curve", color=GOLD, fontweight="bold", fontsize=11)
        ax3.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=9)
        ax3.tick_params(colors="#777")
        for sp in ax3.spines.values(): sp.set_edgecolor("#333")

        # PR
        ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor(BG)
        prec, rec, _ = precision_recall_curve(y_te, y_proba)
        ax4.plot(rec, prec, color="#52b788", lw=2, label=f"AP = {apr:.3f}")
        ax4.fill_between(rec, prec, alpha=0.15, color="#52b788")
        ax4.set_xlabel("Recall", color="#aaa"); ax4.set_ylabel("Precision", color="#aaa")
        ax4.set_title("Precision-Recall Curve", color=GOLD, fontweight="bold", fontsize=11)
        ax4.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=9)
        ax4.tick_params(colors="#777")
        for sp in ax4.spines.values(): sp.set_edgecolor("#333")

        # Per-site breakdown
        ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor(BG)
        site_counts = {}
        for s_idx, y_s in enumerate(y_all):
            sn = site_names[s_idx]
            site_counts[sn] = (int(y_s.sum()), int((y_s==0).sum()))
        snames = list(site_counts.keys())
        n_pos_arr = [site_counts[s][0] for s in snames]
        n_neg_arr = [site_counts[s][1] for s in snames]
        x_pos = np.arange(len(snames))
        ax5.bar(x_pos - 0.2, n_pos_arr, 0.38, color="#e63946", label="Gold")
        ax5.bar(x_pos + 0.2, n_neg_arr, 0.38, color="#2d6a4f", label="Background")
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(snames, color="white", fontsize=8, rotation=20, ha="right")
        ax5.set_ylabel("Pixel count", color="#aaa")
        ax5.set_title("Per-Site Breakdown", color=GOLD, fontweight="bold", fontsize=11)
        ax5.legend(facecolor="#111", labelcolor="white", edgecolor="#333", fontsize=8)
        ax5.tick_params(colors="#777")
        for sp in ax5.spines.values(): sp.set_edgecolor("#333")

        tmp_png_path = _out_path("shap", "training_report", ".png")
        plt.savefig(tmp_png_path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)

        # Save model
        progress(0.95, desc="Saving model…")
        out_pkl = save_path.strip() if save_path.strip() else MODEL_PATH
        save_dir = os.path.dirname(os.path.abspath(out_pkl))
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        bundle_out = {"model": clf_final, "engineered": True,
                      "has_aster": has_aster,
                      "auc": auc, "ap": apr,
                      "sites": site_names[:len(feat_files)],
                      "site_types": site_types,
                      "calibrated": use_calibration,
                      "n_features": X_eng.shape[1]}
        joblib.dump(bundle_out, out_pkl)
        global model_bundle, model_status
        model_bundle = bundle_out
        model_status = (f"✅ {model_mode}  AUC={auc:.3f}  AP={apr:.3f}  "
                        f"{'ASTER+S2' if has_aster else 'S2 only'}")
        logs.append(f"\n💾 Saved → {out_pkl}\n🔁 App model updated.")

        cv_row = (f"| **5-fold CV AUC** | {cvs.mean():.4f} ± {cvs.std():.4f} |"
                  if cvs is not None else "")
        shap_note = "✅ SHAP beeswarm computed" if (use_shap and _HAS_SHAP) else ("⚠ not installed" if use_shap else "—")
        summary = f"""### ✅ Training Complete — {VERSION}

| Parameter | Value |
|---|---|
| **Model** | {model_mode} |
| **AUC** | **{auc:.4f}** |
| **Average Precision** | {apr:.4f} |
{cv_row}
| **Youden-J threshold** | {youden_thr:.4f} |
| **F1-optimal threshold** | {f1_thr:.4f} |
| **Total pixels** | {len(y):,} |
| **Features** | {X.shape[1]} raw → {X_eng.shape[1]} engineered |
| **ASTER SWIR** | {'✅ 6 bands + 12 indices' if has_aster else '❌ Not used'} |
| **Calibration** | {'✅ CalibratedClassifierCV (isotonic)' if use_calibration else '❌ Not applied'} |
| **Optuna HPO** | {'✅ 50 trials' if (use_optuna and _HAS_OPTUNA) else ('⚠ not installed' if use_optuna else '—')} |
| **SHAP** | {shap_note} |
| **Saved** | `{out_pkl}` |

```
{report}
```

**Log:**
```
{''.join(logs)}
```"""
        return tmp_png_path, out_pkl, summary

    except Exception as _e:
        return None, None, f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── TAB 8: STACK BUILDER (Sentinel-2 + ASTER SWIR + DEM) ─────────────

def run_converter(s10_files, s20_files, dem_file, aster_swir_file=None):
    import shutil
    logs = []
    if not s10_files or not dem_file: return None, None, "❌ Need 10m bands and DEM."
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()

        def _copy_to_temp(f):
            src_path = f if isinstance(f, str) else f.name
            p = os.path.join(temp_dir, os.path.basename(src_path))
            shutil.copy(src_path, p); return p

        s10_paths = [_copy_to_temp(f) for f in s10_files]
        s20_paths = [_copy_to_temp(f) for f in s20_files] if s20_files else []
        dem_path  = _copy_to_temp(dem_file)
        aster_path = _copy_to_temp(aster_swir_file) if aster_swir_file else None

        with rasterio.open(s10_paths[0]) as ref:
            h, w = ref.height, ref.width
            ref_transform = ref.transform
            ref_crs = ref.crs
            _raw_profile = ref.profile.copy()

        # Build a clean float32 output profile (strip dtype/count/compress/nodata
        # from the source so they do not conflict with our float32 multi-band stack)
        _STRIP_KEYS = {"dtype", "count", "compress", "nodata", "photometric",
                       "predictor", "zlevel", "tiled", "blockxsize", "blockysize",
                       "interleave"}
        profile = {k: v for k, v in _raw_profile.items() if k not in _STRIP_KEYS}
        profile.update({
            "driver"   : "GTiff",
            "dtype"    : "float32",
            "nodata"   : float("nan"),   # NaN nodata so border pixels are masked
            "compress" : "lzw",
            "tiled"    : True,
            "blockxsize": 256,
            "blockysize": 256,
            "interleave": "band",
        })

        def reproj_band(src_path, band_idx=1):
            with rasterio.open(src_path) as src:
                raw = src.read(band_idx).astype(np.float32)
                nd  = src.nodata
                # Convert nodata / sentinel values to NaN before any ratio computation
                if nd is not None:
                    raw = np.where(raw == float(nd), np.nan, raw)
                # Treat common Sentinel-2 integer fill value (0 for uint16/uint8)
                if src.dtypes[band_idx - 1] in ("uint16", "int16", "uint8"):
                    raw = np.where(raw == 0, np.nan, raw)
                same_transform = np.allclose(list(src.transform), list(ref_transform), rtol=1e-6)
                same_crs = (src.crs is not None and ref_crs is not None and src.crs == ref_crs)
                if same_transform and same_crs and raw.shape == (h, w):
                    return raw
                # Fill destination with NaN so border pixels outside source extent stay NaN
                dest = np.full((h, w), np.nan, dtype=np.float32)
                reproject(
                    raw, dest,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=ref_transform, dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan, dst_nodata=np.nan,
                )
                return dest

        def safe_div(a, b):
            """Divide a by b, returning NaN where b==0 or either input is NaN."""
            with np.errstate(divide="ignore", invalid="ignore"):
                valid = (~np.isnan(a)) & (~np.isnan(b)) & (b != 0)
                return np.where(valid, a / b, np.nan).astype(np.float32)

        BAND_ORDER = ["B02","B03","B04","B05","B06","B8A","B08","B11","B12"]

        def band_key(path):
            bn = os.path.basename(path).upper()
            for i, b in enumerate(BAND_ORDER):
                if b in bn: return i
            return 99

        all_refl_paths = (sorted(s10_paths, key=band_key) + sorted(s20_paths, key=band_key))
        refl = {}
        for p in all_refl_paths:
            bn = os.path.basename(p).upper()
            for b in BAND_ORDER:
                if b in bn and b not in refl:
                    refl[b] = reproj_band(p)
                    logs.append(f"  ✔ {b} ← {os.path.basename(p)}")
                    break

        missing = [b for b in BAND_ORDER if b not in refl]
        for b in missing:
            # Fill missing bands with NaN so they don't silently zero-out index ratios
            refl[b] = np.full((h, w), np.nan, dtype=np.float32)
            logs.append(f"  ⚠️ {b} missing → NaN (will be masked)")
        b02,b03,b04 = refl["B02"],refl["B03"],refl["B04"]
        b05,b06,b8a = refl["B05"],refl["B06"],refl["B8A"]
        b08,b11,b12 = refl["B08"],refl["B11"],refl["B12"]

        IO   = safe_div(b04, b02); CM   = safe_div(b11, b8a)
        FI   = safe_div(b11, b08); GS   = safe_div(b04, b08)
        NDVI = safe_div(b08 - b04, b08 + b04)

        dem_arr = reproj_band(dem_path)
        # DEM NaN → interpolate from neighbours so terrain bands don't kill valid mask
        if np.any(np.isnan(dem_arr)):
            from scipy.ndimage import generic_filter as _gf
            _fill = _gf(np.nan_to_num(dem_arr, nan=0.0),
                        lambda v: np.nanmean(v) if np.any(np.isfinite(v)) else 0.0,
                        size=3)
            dem_arr = np.where(np.isnan(dem_arr), _fill, dem_arr).astype(np.float32)

        px = float(abs(ref_transform.a))
        dz_dx = np.gradient(dem_arr, axis=1) / px
        dz_dy = np.gradient(dem_arr, axis=0) / px
        slope     = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)
        aspect    = np.degrees(np.arctan2(-dz_dy, dz_dx)).astype(np.float32) % 360
        roughness = (maximum_filter(dem_arr, 3) - minimum_filter(dem_arr, 3)).astype(np.float32)

        # Build a common valid mask from reflectance bands only;
        # index / terrain NaN pixels are set to 0 (not NaN) to avoid killing valid pixels.
        # The valid mask is embedded in band 0 (b02): any pixel that is NaN in b02 is masked.
        refl_mask = np.isnan(b02) | np.isnan(b03) | np.isnan(b04)  # nodata footprint

        def _fill_nan_band(arr):
            """Replace NaN with 0.0 for index bands; nodata will be masked via refl_mask."""
            out = arr.copy()
            out[np.isnan(out)] = 0.0
            return out.astype(np.float32)

        # 18-band Sentinel-2 + DEM stack
        # Reflectance bands: keep NaN (drives valid mask in predict_gold via _read_s2)
        # Index / terrain bands: replace NaN with 0 (avoids spurious invalid pixels)
        s2_stack = [b02, b03, b04, b05, b06, b8a, b08, b11, b12,
                    _fill_nan_band(IO), _fill_nan_band(CM),
                    _fill_nan_band(FI), _fill_nan_band(GS),
                    _fill_nan_band(NDVI),
                    dem_arr, slope, aspect, roughness]

        # Log valid pixel stats
        _valid_px = int(np.sum(~refl_mask))
        _total_px = h * w
        logs.append(f"  📊 Valid pixels: {_valid_px:,} / {_total_px:,} "
                    f"({_valid_px/_total_px*100:.1f}%)")
        if _valid_px == 0:
            return None, None, (
                "❌ No valid pixels in the reflectance bands.\n\n"
                "Possible causes:\n"
                "• All band files have nodata=0 covering the entire scene\n"
                "• Wrong CRS — bands do not overlap after reprojection\n"
                "• Input files are empty or corrupt\n\n"
                "Check log for band detection details:\n" + "\n".join(logs)
            )

        # ASTER SWIR processing
        aster_stack_out = None
        aster_out_path = None
        if aster_path:
            logs.append("\n🛰 Processing ASTER SWIR stack…")
            try:
                with rasterio.open(aster_path) as asrc:
                    raw_a = asrc.read().astype("float32")
                    a_nb = asrc.count
                    if a_nb >= 6:
                        raw_a = raw_a[:6]
                    else:
                        pad = np.zeros((6 - a_nb, asrc.height, asrc.width), dtype="float32")
                        raw_a = np.vstack([raw_a, pad])
                    a_t = asrc.transform; a_c = asrc.crs

                # Resample ASTER to Sentinel-2 grid (chunked for memory)
                aster_aligned = resample_aster_to_sentinel(raw_a, a_t, a_c, ref_transform, ref_crs, h, w)
                logs.append(f"  ✔ ASTER resampled {raw_a.shape[1]}×{raw_a.shape[2]} → {h}×{w}")

                # Compute ASTER indices
                a_idx = compute_aster_indices(aster_aligned)
                aster_bands = list(aster_aligned)   # 6 bands
                aster_idx_arrs = [a_idx[k] for k in sorted(a_idx.keys())]  # 12 indices
                aster_stack_out = aster_bands + aster_idx_arrs    # 18 bands total

                # Save ASTER stack to structured output directory
                aster_out_path = _out_path("stack", "aster_swir_stack", ".tif")
                a_profile = profile.copy()
                n_aster_bands = len(aster_stack_out)
                a_profile.update({"count": n_aster_bands, "dtype": "float32", "nodata": float("nan"), "compress": "lzw"})
                with rasterio.open(aster_out_path, "w", **a_profile) as dst:
                    for bi, arr in enumerate(aster_stack_out, start=1):
                        dst.write(arr, bi)
                logs.append(f"  ✔ ASTER stack ({n_aster_bands} bands) → aster_swir_stack.tif")
            except Exception as ae:
                logs.append(f"  ⚠️ ASTER processing failed: {ae}")

        arr = np.stack(s2_stack, axis=0)
        out_path = _out_path("stack", "full_features_stack", ".tif")
        profile.update({"count": 18})  # dtype/nodata/compress already set in base profile
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)

        logs.append(f"\n✅ Sentinel-2 stack 18×{h}×{w} → full_features_stack.tif")
        if aster_out_path:
            logs.append(f"✅ ASTER SWIR stack 18×{h}×{w} → aster_swir_stack.tif")
        return out_path, aster_out_path, "\n".join(logs)

    except Exception as _e:
        return None, None, f"❌\n{traceback.format_exc()}"
    finally:
        try:
            if temp_dir is not None:
                import shutil as _sh
                _sh.rmtree(temp_dir, ignore_errors=True)
        except Exception as _e:
            import logging; logging.getLogger(__name__).debug(f"temp dir cleanup: {_e}")


# ── THREE.JS 3D VIEWER ────────────────────────────────────────────────

CMAP_HEX = {
    "hot":        ["#000000","#300000","#600000","#900800","#c02000",
                   "#e83800","#ff6000","#ff9400","#ffcc40","#ffffff"],
    "RdYlGn_r":   ["#006837","#1a7f3a","#56ae58","#aad488","#ffffbf",
                   "#fdc87a","#fd8d3c","#e3361c","#c0000a","#800026"],
    "YlOrBr":     ["#ffffe5","#fff7bc","#fee391","#fed976","#fec44f",
                   "#fe9929","#ec7014","#cc4c02","#993404","#662506"],
    "Reds":       ["#fff5f0","#ffdcd0","#fcbba1","#fc8f6f","#fc6141",
                   "#f03020","#d01010","#b00000","#880000","#670000"],
    "Blues":      ["#f7fbff","#deebf7","#c6dbef","#9ecae1","#6baed6",
                   "#4292c6","#2171b5","#08519c","#08306b","#041a3f"],
    "terrain":    ["#1a1a6e","#1050a0","#2080c0","#40a8d4","#a0d4e4",
                   "#e4e4b4","#c8d890","#90b870","#508040","#284018"],
    "plasma":     ["#0d0887","#3b049a","#6a00a8","#8f0da4","#b12a90",
                   "#cc4778","#e16462","#f2844b","#fca636","#f0f921"],
    "inferno":    ["#000004","#1b0c41","#420a68","#6b166e","#932667",
                   "#bb3754","#dd513a","#f37819","#fca50a","#fcffa4"],
    "viridis":    ["#440154","#472c7a","#3b528b","#2c728e","#21918c",
                   "#28ae80","#5ec962","#addc30","#fde725","#ffffff"],
    "copper":     ["#000000","#220e00","#441c00","#663000","#8a4400",
                   "#b05800","#cc6a10","#e07a30","#f09060","#ffc090"],
}


def _build_threejs_html(dem_s, col_s, color_label, cmap_name, exag, stride, geo, vis_mode):
    import base64, json as _json_mod
    r_s, c_s = dem_s.shape
    dem_min = float(np.nanmin(dem_s)); dem_max = float(np.nanmax(dem_s))
    dem_norm = np.clip((dem_s - dem_min) / max(dem_max - dem_min, 1e-9), 0, 1)
    dem_norm = np.nan_to_num(dem_norm, nan=0.0)
    vmin = float(np.nanpercentile(col_s, 2)); vmax = float(np.nanpercentile(col_s, 98))
    col_norm = np.clip((col_s - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    col_norm = np.nan_to_num(col_norm, nan=0.0)
    dem_b64 = base64.b64encode((dem_norm * 255).astype(np.uint8).tobytes()).decode("ascii")
    col_b64 = base64.b64encode((col_norm * 255).astype(np.uint8).tobytes()).decode("ascii")
    stops    = CMAP_HEX.get(cmap_name, CMAP_HEX["hot"])
    stops_js = _json_mod.dumps(stops)
    geo_str  = geo.get("geo_str", "—"); area_km2 = geo.get("area_km2", 0)
    vmid     = (vmin + vmax) / 2

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"/><title>Gold Prospectivity 3D | ASTER Edition</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;background:#070709;overflow:hidden}}
canvas{{display:block}}
.panel{{position:absolute;z-index:10;background:rgba(5,5,10,0.90);
  border:1px solid rgba(240,192,64,.18);border-radius:18px;
  backdrop-filter:blur(18px);font-family:'Share Tech Mono',monospace;
  color:#d4c9a8;font-size:11px;line-height:1.75}}
#controls{{top:14px;right:14px;padding:14px 16px;min-width:235px}}
.ctit{{font-family:'Orbitron',monospace;font-size:9px;letter-spacing:3px;
  color:#b8860b;margin-bottom:10px;text-transform:uppercase}}
.cgrp{{margin-bottom:10px}}
.cgrp label{{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:4px;font-size:10px;color:#8a8070}}
.cgrp label .v{{color:#f0c040;font-weight:bold}}
input[type=range]{{-webkit-appearance:none;width:100%;height:3px;
  background:linear-gradient(to right,#b8860b,#f0c040);border-radius:2px;cursor:pointer}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:13px;height:13px;
  border-radius:50%;background:#f0c040;cursor:pointer}}
.trow{{display:flex;gap:6px;margin-top:4px}}
.tbtn{{flex:1;padding:5px 0;border:1px solid rgba(240,192,64,.18);background:transparent;
  color:#6a6050;font-family:'Share Tech Mono',monospace;font-size:10px;
  border-radius:3px;cursor:pointer;transition:all .18s}}
.tbtn:hover{{border-color:#f0c040;color:#d4c9a8}}
.tbtn.on{{background:rgba(240,192,64,.12);border-color:#f0c040;color:#f0c040}}
#legend{{bottom:50px;right:14px;padding:12px 16px;min-width:192px}}
.ltit{{font-family:'Orbitron',monospace;font-size:8px;color:#b8860b;
  letter-spacing:3px;text-transform:uppercase;margin-bottom:10px}}
.zrow{{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:10px}}
.zsw{{width:28px;height:8px;border-radius:2px;flex-shrink:0}}
#hint{{bottom:12px;left:50%;transform:translateX(-50%);padding:5px 18px;
  white-space:nowrap;font-size:10px;color:#504840;letter-spacing:1px;border-radius:20px}}
#loading{{position:fixed;inset:0;z-index:100;background:#07070a;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:22px}}
.lt{{font-family:'Orbitron',monospace;font-size:21px;font-weight:900;color:#f0c040;
  letter-spacing:4px;text-shadow:0 0 40px rgba(240,192,64,.5)}}
.ls{{font-family:'Share Tech Mono',monospace;font-size:12px;color:#504840;letter-spacing:3px}}
.lbo{{width:280px;height:2px;background:rgba(240,192,64,.10);border-radius:1px;overflow:hidden}}
.lbi{{height:100%;width:0;background:#f0c040;border-radius:1px;transition:width .3s}}
</style></head><body>
<div id="loading">
  <div class="lt">GOLDGIS ASTER EDITION</div>
  <div class="ls">3D TERRAIN · HYDROTHERMAL MAPPING</div>
  <div class="lbo"><div class="lbi" id="lbi"></div></div>
</div>
<div class="panel" id="controls">
  <div class="ctit">⚙ Controls</div>
  <div class="cgrp">
    <label>Vertical Exaggeration <span class="v" id="ev">{exag}×</span></label>
    <input type="range" id="exag" min="1" max="50" value="{exag}">
  </div>
  <div class="cgrp">
    <label>Opacity <span class="v" id="ov">100%</span></label>
    <input type="range" id="opa" min="10" max="100" value="100">
  </div>
  <div class="cgrp">
    <label>Ambient Light <span class="v" id="av">50%</span></label>
    <input type="range" id="amb" min="5" max="100" value="50">
  </div>
  <div class="cgrp">
    <label>Sun Azimuth <span class="v" id="sv">315°</span></label>
    <input type="range" id="sun" min="0" max="360" value="315">
  </div>
  <div class="trow">
    <button class="tbtn" id="bwire" onclick="toggleWire()">Wire</button>
    <button class="tbtn on" id="bauto" onclick="toggleAuto()">AutoRot</button>
    <button class="tbtn" id="bfog" onclick="toggleFog()">Fog</button>
  </div>
</div>
<div class="panel" id="legend">
  <div class="ltit">ASTER Layer</div>
  <div class="zrow"><div class="zsw" style="background:linear-gradient(90deg,#900018,#ff2040)"></div><span>High</span></div>
  <div class="zrow"><div class="zsw" style="background:linear-gradient(90deg,#806000,#f0c040)"></div><span>Moderate</span></div>
  <div class="zrow"><div class="zsw" style="background:linear-gradient(90deg,#0c3818,#28b050)"></div><span>Low</span></div>
  <div style="font-size:9px;color:#504840;margin-top:8px">Layer: {color_label}<br>{geo_str[:40]}</div>
</div>
<div class="panel" id="hint">🖱 Left-drag: rotate | Scroll: zoom | Touch: 1-finger rotate · 2-finger zoom</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const W={c_s},H={r_s};const STOPS={stops_js};
function sp(p){{document.getElementById('lbi').style.width=p+'%'}}
sp(20);
function b64u8(b64){{const bin=atob(b64),a=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i);return a;}}
const DEM=b64u8("{dem_b64}");const COL=b64u8("{col_b64}");
function hexRgb(h){{return[parseInt(h.slice(1,3),16)/255,parseInt(h.slice(3,5),16)/255,parseInt(h.slice(5,7),16)/255];}}
function sampleCmap(t){{const n=STOPS.length-1,i=Math.min(Math.floor(t*n),n-1),f=t*n-i;const a=hexRgb(STOPS[i]),b=hexRgb(STOPS[i+1]);return[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}}
sp(35);
const renderer=new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(window.innerWidth,window.innerHeight);renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
renderer.shadowMap.enabled=true;document.body.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x070709);
const camera=new THREE.PerspectiveCamera(45,window.innerWidth/window.innerHeight,0.1,18000);
const ambLight=new THREE.AmbientLight(0xfff5e0,0.50);scene.add(ambLight);
const sunLight=new THREE.DirectionalLight(0xfff5cc,1.40);sunLight.position.set(W*0.30,W*1.50,H*0.50);sunLight.castShadow=true;scene.add(sunLight);
scene.add(new THREE.HemisphereLight(0x1a2255,0x3a2508,0.32));
sp(55);
const geom=new THREE.PlaneGeometry(W-1,H-1,W-1,H-1);geom.rotateX(-Math.PI/2);
const pos=geom.attributes.position;const vc=new Float32Array(pos.count*3);
geom.setAttribute('color',new THREE.BufferAttribute(vc,3));
let exagMul={exag};
function rebuild(){{for(let j=0;j<H;j++){{for(let i=0;i<W;i++){{const idx=j*W+i;pos.setY(idx,DEM[idx]/255*exagMul*3.2);const[r,g,b]=sampleCmap(COL[idx]/255);vc[idx*3]=r;vc[idx*3+1]=g;vc[idx*3+2]=b;}}}}pos.needsUpdate=true;geom.computeVertexNormals();geom.attributes.color.needsUpdate=true;}}
rebuild();
const mat=new THREE.MeshPhongMaterial({{vertexColors:true,side:THREE.FrontSide,shininess:38,specular:new THREE.Color(0x1a1a00)}});
const mesh=new THREE.Mesh(geom,mat);mesh.position.set(W*.5,0,H*.5);mesh.receiveShadow=true;scene.add(mesh);
const wireMat=new THREE.MeshBasicMaterial({{color:0x404020,wireframe:true,opacity:.10,transparent:true}});
const wireMesh=new THREE.Mesh(geom,wireMat);wireMesh.position.copy(mesh.position);wireMesh.visible=false;scene.add(wireMesh);
const gridH=new THREE.GridHelper(Math.max(W,H)*1.25,22,0x181808,0x0e0e04);gridH.position.set(W*.5,-0.8,H*.5);scene.add(gridH);
sp(75);
let theta=Math.PI*1.15,phi=.52,radius=Math.max(W,H)*1.42,panX=W*.5,panY=0,panZ=H*.5;
let isDrag=false,isRight=false,lx=0,ly=0,autoRot=true;
function updateCam(){{camera.position.set(panX+radius*Math.sin(phi)*Math.sin(theta),panY+radius*Math.cos(phi),panZ+radius*Math.sin(phi)*Math.cos(theta));camera.lookAt(panX,panY,panZ);}}
updateCam();
const cvs=renderer.domElement;
cvs.addEventListener('mousedown',e=>{{isDrag=true;isRight=e.button===2;lx=e.clientX;ly=e.clientY;e.preventDefault();}});
cvs.addEventListener('mousemove',e=>{{if(!isDrag)return;const dx=e.clientX-lx,dy=e.clientY-ly;if(isRight){{radius=Math.max(28,Math.min(radius-dy*radius*.006,14000));}}else{{theta-=dx*.005;phi=Math.max(.04,Math.min(Math.PI*.48,phi+dy*.005));}}lx=e.clientX;ly=e.clientY;updateCam();}});
cvs.addEventListener('mouseup',()=>isDrag=false);
cvs.addEventListener('wheel',e=>{{radius=Math.max(28,Math.min(radius+e.deltaY*radius*.001,14000));updateCam();e.preventDefault();}},{{passive:false}});
document.getElementById('exag').addEventListener('input',function(){{exagMul=+this.value;document.getElementById('ev').textContent=this.value+'×';rebuild();}});
document.getElementById('opa').addEventListener('input',function(){{mat.opacity=this.value/100;mat.transparent=mat.opacity<1;document.getElementById('ov').textContent=this.value+'%';}});
document.getElementById('amb').addEventListener('input',function(){{ambLight.intensity=this.value/100;document.getElementById('av').textContent=this.value+'%';}});
document.getElementById('sun').addEventListener('input',function(){{const az=+this.value,r=Math.max(W,H)*1.6;document.getElementById('sv').textContent=az+'°';sunLight.position.set(W*.5+r*Math.cos(az*Math.PI/180),r,H*.5+r*Math.sin(az*Math.PI/180));}});
window.addEventListener('resize',()=>{{renderer.setSize(window.innerWidth,window.innerHeight);camera.aspect=window.innerWidth/window.innerHeight;camera.updateProjectionMatrix();}});
function setBtn(id,on){{document.getElementById(id).classList.toggle('on',on);}}
function toggleWire(){{wireMesh.visible=!wireMesh.visible;setBtn('bwire',wireMesh.visible);}}
function toggleAuto(){{autoRot=!autoRot;setBtn('bauto',autoRot);}}
function toggleFog(){{const on=!scene.fog;scene.fog=on?new THREE.FogExp2(0x070709,.0014):null;setBtn('bfog',on);}}
sp(100);
let frame=0;
function animate(){{requestAnimationFrame(animate);frame++;if(autoRot){{theta+=0.0025;updateCam();}}renderer.render(scene,camera);}}
setTimeout(()=>{{const l=document.getElementById('loading');l.style.transition='opacity .8s';l.style.opacity='0';setTimeout(()=>l.remove(),900);animate();}},380);
</script></body></html>"""
    return html


def make_3d_visualization(s2_file, aster_file, vis_mode, z_index, cmap_3d, stride_val, exag_val):
    if s2_file is None:
        return None, "### ❌ Upload a GeoTIFF first."
    try:
        stack, transform, crs, rows, cols, profile = _read_s2(_fpath(s2_file))
        n_bands = stack.shape[0]
        if n_bands < REQUIRED_BANDS:
            return None, f"### ❌ {n_bands} bands found, need {REQUIRED_BANDS}."
        geo = geo_info_dict(_make_mock_src(profile, transform, crs, rows, cols))

        # Load ASTER if available
        aster = None
        if aster_file is not None:
            try:
                with rasterio.open(_fpath(aster_file)) as asrc:
                    raw = asrc.read().astype("float32")[:6]
                    aster = resample_aster_to_sentinel(raw, asrc.transform, asrc.crs,
                                                      transform, crs, rows, cols)
            except Exception as _e:
                aster = None

        if vis_mode == "Probability (RF Model)":
            if model_bundle is None:
                return None, f"### ❌ {model_status}"
            pixels    = stack.reshape(n_bands, -1).T
            valid_mask= ~np.any(np.isnan(pixels), axis=1)
            prob_flat = np.full(rows * cols, np.nan, dtype="float32")
            prob_flat[valid_mask] = _get_model_predict_proba(pixels[valid_mask])
            color_arr = prob_flat.reshape(rows, cols); color_label = "Gold Prospectivity"
        elif vis_mode == "ASTER Hydrothermal" and aster is not None:
            idx = compute_aster_indices(aster)
            color_arr = idx["AST_HydAlt"]; color_label = "Hydrothermal Alteration"
        elif vis_mode == "ASTER Silica" and aster is not None:
            idx = compute_aster_indices(aster)
            color_arr = idx["AST_Silica"]; color_label = "Silica Index"
        elif vis_mode == "Custom Index":
            arr = get_band_or_index(z_index.strip(), stack)
            if arr is None and aster is not None:
                arr = compute_aster_index_by_name(z_index.strip(), aster)
            if arr is None:
                return None, f"### ❌ Unknown index: {z_index}"
            color_arr = arr; color_label = z_index.strip()
        else:
            color_arr = stack[9]; color_label = "Iron Oxide Index"

        dem = stack[14]
        stride = max(1, int(stride_val))
        dem_s = dem      [::stride, ::stride]
        col_s = color_arr[::stride, ::stride]
        r_s, c_s = dem_s.shape

        html_content = _build_threejs_html(dem_s, col_s, color_label, cmap_3d,
                                           int(exag_val), stride, geo, vis_mode)
        import base64
        encoded = base64.b64encode(html_content.encode("utf-8")).decode("ascii")
        iframe_html = (f'<iframe src="data:text/html;base64,{encoded}" '
                       f'style="width:100%;height:720px;border:none;border-radius:8px;" '
                       f'allowfullscreen></iframe>')

        stats_md = f"""
### 🏔️ 3D Viewer Ready — {VERSION}

| Parameter | Value |
|---|---|
| **Layer** | {color_label} |
| **ASTER available** | {'✅' if aster is not None else '❌'} |
| **Grid size** | {c_s} × {r_s} (stride ×{stride}) |
| **DEM range** | {float(np.nanmin(dem_s)):.0f} – {float(np.nanmax(dem_s)):.0f} m |
| **Scene** | {geo.get('geo_str','—')} |
"""
        return iframe_html, stats_md
    except Exception as _e:
        return "<p>❌ Error — see stats panel.</p>", f"### ❌\n```\n{traceback.format_exc()}\n```"


# ── CSS ───────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  /* ══ PREMIUM DARK GOLD GEOLOGICAL PALETTE ══ */
  --gold:           #d4962a;
  --gold-bright:    #f5c842;
  --gold-pale:      #fde68a;
  --gold-dim:       #8a5e0f;
  --gold-glow:      rgba(212,150,42,.22);
  --gold-glow2:     rgba(212,150,42,.08);
  --gold-line:      rgba(212,150,42,.32);
  --gold-subtle:    rgba(212,150,42,.05);

  /* Deep void backgrounds */
  --bg:             #030508;
  --surface:        #080b12;
  --surface2:       #0c1019;
  --surface3:       #111520;
  --surface4:       #161c2a;
  --surface5:       #1c2436;

  /* Glass effect helpers */
  --glass:          rgba(12,16,28,.72);
  --glass-border:   rgba(212,150,42,.12);
  --glass-highlight:rgba(255,255,255,.03);

  /* Borders — fine metallic lines */
  --border:         #161d2e;
  --border-mid:     #1e2840;
  --border-light:   #283452;
  --border-gold:    rgba(212,150,42,.28);

  /* Typography — readable sizes */
  --text:           #c8d4ec;
  --text-dim:       #5a6e96;
  --text-muted:     #323e5a;
  --text-bright:    #e4edff;
  --text-gold:      #e8b842;

  /* Accent spectrum */
  --green:          #0fd4a0;
  --green-dim:      rgba(15,212,160,.14);
  --cyan:           #00ddd0;
  --cyan-dim:       rgba(0,221,208,.12);
  --blue:           #4a9eff;
  --violet:         #a07ef5;
  --violet-dim:     rgba(160,126,245,.12);
  --red:            #f06060;
  --orange:         #f5903c;

  /* Accent aliases */
  --accent-cyan:    #00ddd0;
  --accent-green:   #0fd4a0;
  --accent-violet:  #a07ef5;

  /* Geometry */
  --radius:         6px;
  --radius-sm:      4px;
  --radius-lg:      10px;
  --radius-xl:      16px;

  /* Motion */
  --transition:     all 0.22s cubic-bezier(.4,0,.2,1);
  --transition-slow:all 0.45s cubic-bezier(.4,0,.2,1);

  /* Shadows */
  --shadow-sm:      0 2px 8px rgba(0,0,0,.6);
  --shadow-md:      0 4px 20px rgba(0,0,0,.7);
  --shadow-lg:      0 8px 36px rgba(0,0,0,.8);
  --shadow-gold:    0 0 20px rgba(212,150,42,.14), 0 0 40px rgba(212,150,42,.06);
  --shadow-inset:   inset 0 1px 0 rgba(255,255,255,.03), inset 0 -1px 0 rgba(0,0,0,.3);
}


/* ══ KEYFRAME ANIMATIONS ══════════════════════════════════════════════ */

@keyframes blink {
  0%,100%{opacity:1} 50%{opacity:.2}
}
@keyframes pulse-gold {
  0%,100%{box-shadow:0 0 8px rgba(212,150,42,.3), 0 0 16px rgba(212,150,42,.1)}
  50%{box-shadow:0 0 20px rgba(212,150,42,.6), 0 0 40px rgba(212,150,42,.2)}
}
@keyframes radar-sweep {
  0%{transform:rotate(0deg)}
  100%{transform:rotate(360deg)}
}
@keyframes scan-pulse {
  0%{opacity:0; transform:scaleY(0)}
  20%{opacity:.6}
  80%{opacity:.6}
  100%{opacity:0; transform:scaleY(1)}
}
@keyframes shimmer {
  0%{background-position:-200% center}
  100%{background-position:200% center}
}
@keyframes float-up {
  0%{opacity:0; transform:translateY(8px)}
  100%{opacity:1; transform:translateY(0)}
}
@keyframes glow-pulse {
  0%,100%{opacity:.6}
  50%{opacity:1}
}
@keyframes scanline {
  from{background-position:0 0}
  to{background-position:0 100px}
}
@keyframes border-flow {
  0%,100%{border-color:var(--border-gold)}
  50%{border-color:rgba(212,150,42,.55)}
}

/* ══ RESET & BASE ═════════════════════════════════════════════════════ */

*,*::before,*::after { box-sizing:border-box; margin:0; padding:0 }

html,body,.gradio-container,.gradio-container * {
  background-color:transparent;
  font-family:'Rajdhani','Space Grotesk','Segoe UI',system-ui,sans-serif !important;
}
html,body {
  background:#030508 !important;
}
.gradio-container {
  background:#030508 !important;
  color:var(--text) !important;
  font-size:13px !important;
}
footer,.svelte-1ipelgc,footer.svelte-1rjryqp { display:none !important }

/* ── Premium scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width:3px; height:3px }
::-webkit-scrollbar-track { background:transparent }
::-webkit-scrollbar-thumb {
  background:linear-gradient(180deg,var(--gold-dim),var(--border-mid));
  border-radius:3px;
}
::-webkit-scrollbar-thumb:hover { background:var(--gold-dim) }

/* ══ TOP BAR ══════════════════════════════════════════════════════════ */

.gis-topbar {
  display:flex; align-items:center; gap:12px;
  height:48px;
  background:linear-gradient(90deg,#040609 0%, #07090f 35%, #080c15 70%, #050709 100%);
  border-bottom:1px solid rgba(212,150,42,.18);
  padding:0 18px;
  position:sticky; top:0; z-index:300;
  box-shadow:0 1px 0 rgba(212,150,42,.08), 0 4px 40px rgba(0,0,0,.95), 0 0 120px rgba(212,150,42,.03);
  background-image:linear-gradient(90deg,#040609 0%, #07090f 35%, #080c15 70%, #050709 100%),
    repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,255,255,.003) 2px,rgba(255,255,255,.003) 4px);
}

/* Animated accent line on top bar */
.gis-topbar::after {
  content:'';
  position:absolute; bottom:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,
    transparent 0%,
    rgba(212,150,42,.05) 10%,
    rgba(212,150,42,.35) 30%,
    rgba(245,200,66,.5) 50%,
    rgba(212,150,42,.35) 70%,
    rgba(212,150,42,.05) 90%,
    transparent 100%);
}

.gis-logo-area {
  display:flex; align-items:center; gap:13px;
  padding-right:20px;
  border-right:1px solid rgba(212,150,42,.15);
  flex-shrink:0;
}

.gis-logo-icon {
  width:36px; height:36px; border-radius:9px;
  background:linear-gradient(145deg,#1a0e00, #5a3400, var(--gold));
  display:flex; align-items:center; justify-content:center;
  font-size:17px;
  border:1px solid rgba(212,150,42,.4);
  box-shadow:0 0 0 1px rgba(0,0,0,.5), var(--shadow-gold);
  flex-shrink:0;
  position:relative; overflow:hidden;
  animation:pulse-gold 4s ease-in-out infinite;
}
.gis-logo-icon::before {
  content:'';
  position:absolute; top:-50%; left:-50%; width:200%; height:200%;
  background:conic-gradient(transparent 270deg, rgba(212,150,42,.25) 360deg);
  animation:radar-sweep 8s linear infinite;
}
.gis-logo-icon::after {
  content:'';
  position:absolute; inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.18) 0%,transparent 55%);
  pointer-events:none;
}

.gis-app-name {
  font-family:'Orbitron','Rajdhani',sans-serif !important;
  font-weight:700; font-size:.82rem;
  color:var(--gold-bright);
  letter-spacing:5px; text-transform:uppercase;
  line-height:1;
  text-shadow:0 0 12px rgba(212,150,42,.5), 0 0 24px rgba(212,150,42,.2);
}
.gis-app-sub {
  font-size:.5rem; color:var(--text-muted);
  letter-spacing:2.5px; text-transform:uppercase; margin-top:5px;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  opacity:.8;
}

.gis-topbar-chips {
  display:flex; align-items:center; gap:5px; flex-wrap:wrap;
}

/* ══ CHIPS & BADGES ══════════════════════════════════════════════════ */

.gis-chip, .gis-badge {
  display:inline-flex; align-items:center; gap:5px;
  padding:3px 11px 3px 9px;
  background:rgba(255,255,255,.025);
  border:1px solid var(--border-mid);
  border-radius:var(--radius-xl);
  font-size:.56rem; font-weight:600; letter-spacing:.6px;
  color:var(--text-dim);
  white-space:nowrap;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  transition:var(--transition);
  backdrop-filter:blur(8px);
}
.gis-chip:hover, .gis-badge:hover {
  border-color:var(--border-gold);
  background:var(--gold-subtle);
}

.gis-chip-dot { width:5px; height:5px; border-radius:50%; flex-shrink:0; }

.gis-chip.gold, .gis-badge.gold {
  border-color:rgba(212,150,42,.35);
  color:var(--gold-bright);
  background:rgba(212,150,42,.07);
  text-shadow:0 0 8px rgba(212,150,42,.3);
}
.gis-chip.green, .gis-badge.green {
  border-color:rgba(15,212,160,.28);
  color:var(--green);
  background:rgba(15,212,160,.05);
}
.gis-chip.violet, .gis-badge.violet {
  border-color:rgba(160,126,245,.25);
  color:var(--violet);
  background:rgba(160,126,245,.05);
}
.gis-chip.cyan, .gis-badge.cyan {
  border-color:rgba(0,221,208,.22);
  color:var(--cyan);
  background:rgba(0,221,208,.05);
}

.gis-chip-dot.green {
  background:var(--green);
  box-shadow:0 0 6px var(--green);
  animation:blink 2s infinite;
}
.gis-chip-dot.gold { background:var(--gold); box-shadow:0 0 5px var(--gold); }
.gis-chip-dot.violet { background:var(--violet); }
.gis-chip-dot.cyan { background:var(--cyan); }

.gis-topbar-right { margin-left:auto; display:flex; align-items:center; gap:8px; }
.gis-institution {
  font-size:.55rem; color:var(--text-muted); letter-spacing:.5px;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  padding:3px 10px;
  border:1px solid rgba(212,150,42,.12);
  border-radius:var(--radius-sm);
  background:rgba(212,150,42,.03);
  white-space:nowrap;
}

/* ══ UPLOAD STRIP ═════════════════════════════════════════════════════ */

.gis-upload-strip {
  display:flex !important; align-items:center !important; gap:10px !important;
  padding:5px 18px !important;
  background:linear-gradient(90deg, var(--surface) 0%, var(--surface2) 50%, var(--surface) 100%) !important;
  border-bottom:1px solid var(--border-mid) !important;
  box-shadow:inset 0 -1px 0 rgba(0,0,0,.3) !important;
  position:relative;
}
.gis-upload-strip::before {
  content:'';
  position:absolute; left:0; top:0; bottom:0; width:2px;
  background:linear-gradient(180deg, var(--gold), transparent);
}

.gis-upload-label {
  font-size:.6rem; font-weight:700; letter-spacing:2px;
  text-transform:uppercase; color:var(--text-muted);
  white-space:nowrap;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
}
.gis-upload-status {
  font-size:.6rem; color:var(--green);
  padding:3px 12px;
  background:rgba(15,212,160,.05);
  border:1px solid rgba(15,212,160,.2);
  border-radius:var(--radius-xl); white-space:nowrap;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  transition:var(--transition);
  backdrop-filter:blur(4px);
}
.gis-upload-status.warn {
  color:var(--gold); background:rgba(212,150,42,.05);
  border-color:rgba(212,150,42,.22);
}

/* ══ TABS ═════════════════════════════════════════════════════════════ */

#gis-tabs-root .tab-nav,
#gis-tabs-root [role="tablist"],
.tabs .tab-nav {
  position:sticky !important; top:0 !important; z-index:200 !important;
  background:linear-gradient(180deg, var(--surface2) 0%, var(--surface) 100%) !important;
  border-bottom:2px solid rgba(212,150,42,.2) !important;
  border-top:none !important;
  padding:0 8px !important; margin:0 !important;
  display:flex !important;
  flex-wrap:nowrap !important;
  overflow:visible !important;
  justify-content:space-between !important;
  align-items:stretch !important;
  box-shadow:0 4px 24px rgba(0,0,0,.7) !important;
  gap:0px !important;
  width:100% !important;
  box-sizing:border-box !important;
}

#gis-tabs-root .tab-nav button,
#gis-tabs-root [role="tab"] {
  background:transparent !important;
  border:none !important;
  border-bottom:3px solid transparent !important;
  border-radius:0 !important;
  color:var(--text-muted) !important;
  font-size:.58rem !important;
  font-weight:700 !important;
  letter-spacing:.4px !important;
  text-transform:uppercase !important;
  padding:0 6px !important;
  cursor:pointer !important;
  white-space:nowrap !important;
  transition:var(--transition) !important;
  height:40px !important;
  flex:1 1 0 !important;
  min-width:0 !important;
  font-family:'Rajdhani','IBM Plex Mono',sans-serif !important;
  position:relative !important;
  display:flex !important;
  align-items:center !important;
  justify-content:center !important;
  text-align:center !important;
  overflow:hidden !important;
  text-overflow:ellipsis !important;
}

#gis-tabs-root .tab-nav button::after,
#gis-tabs-root [role="tab"]::after {
  content:none !important;
}

#gis-tabs-root .tab-nav button:hover,
#gis-tabs-root [role="tab"]:hover {
  color:var(--gold-bright) !important;
  background:rgba(212,150,42,.07) !important;
  border-bottom-color:rgba(212,150,42,.4) !important;
}

#gis-tabs-root .tab-nav button.selected,
#gis-tabs-root [role="tab"][aria-selected="true"] {
  color:var(--gold-bright) !important;
  border-bottom-color:var(--gold-bright) !important;
  background:rgba(212,150,42,.10) !important;
  text-shadow:0 0 10px rgba(212,150,42,.5) !important;
  font-weight:800 !important;
}

/* ══ FORM ELEMENTS ════════════════════════════════════════════════════ */

label, .gr-form label {
  color:var(--text-dim) !important;
  font-size:.63rem !important; font-weight:700 !important;
  letter-spacing:.8px !important; text-transform:uppercase !important;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  margin-bottom:4px !important;
}

input[type=text],input[type=number],textarea,select,.gr-input {
  background:linear-gradient(135deg, var(--surface3), var(--surface4)) !important;
  border:1px solid var(--border-mid) !important;
  border-radius:var(--radius-sm) !important;
  color:var(--text) !important;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  font-size:.78rem !important;
  transition:var(--transition) !important;
  padding:6px 10px !important;
  box-shadow:var(--shadow-inset) !important;
}
input[type=text]:focus,input[type=number]:focus,textarea:focus {
  border-color:var(--gold-dim) !important;
  outline:none !important;
  box-shadow:0 0 0 3px rgba(212,150,42,.1), var(--shadow-inset) !important;
  background:linear-gradient(135deg, var(--surface4), var(--surface5)) !important;
}

/* Dropdown */
.gr-dropdown select, select.gr-input {
  background:linear-gradient(135deg, var(--surface3), var(--surface4)) !important;
  border:1px solid var(--border-mid) !important;
  color:var(--text) !important;
}

/* Checkbox */
input[type=checkbox] { accent-color:var(--gold) !important }

/* Range */
input[type=range] { accent-color:var(--gold) !important }
.gr-slider .range { background:var(--surface4) !important }

/* ══ BUTTONS ══════════════════════════════════════════════════════════ */

.gr-button-primary,
button[data-testid="primary"],
.primary {
  background:linear-gradient(135deg, #5a3000 0%, #8a5200 35%, var(--gold) 80%, var(--gold-bright) 100%) !important;
  color:#02040a !important;
  font-weight:800 !important; font-size:.7rem !important;
  letter-spacing:2px !important; text-transform:uppercase !important;
  border:none !important;
  border-radius:var(--radius-sm) !important;
  box-shadow:0 2px 16px rgba(212,150,42,.25), 0 0 0 1px rgba(212,150,42,.2), inset 0 1px 0 rgba(255,255,255,.18) !important;
  transition:var(--transition) !important;
  font-family:'Rajdhani','IBM Plex Mono',sans-serif !important;
  height:36px !important;
  position:relative; overflow:hidden;
}
.gr-button-primary::before,
button[data-testid="primary"]::before {
  content:'';
  position:absolute; inset:0;
  background:linear-gradient(135deg, rgba(255,255,255,.12) 0%, transparent 50%, rgba(0,0,0,.1) 100%);
  pointer-events:none;
}
.gr-button-primary::after,
button[data-testid="primary"]::after {
  content:'';
  position:absolute; inset:0;
  background:linear-gradient(90deg,
    transparent 0%,
    rgba(255,255,255,.15) 50%,
    transparent 100%);
  transform:translateX(-100%);
  transition:transform 0.6s ease;
}
.gr-button-primary:hover::after,
button[data-testid="primary"]:hover::after {
  transform:translateX(100%);
}
.gr-button-primary:hover,
button[data-testid="primary"]:hover {
  box-shadow:0 6px 32px rgba(212,150,42,.4), 0 0 0 1px rgba(212,150,42,.35), inset 0 1px 0 rgba(255,255,255,.2) !important;
  filter:brightness(1.08) !important;
  transform:translateY(-2px) !important;
}
.gr-button-primary:active,
button[data-testid="primary"]:active {
  transform:translateY(0) !important;
  filter:brightness(.95) !important;
  box-shadow:0 2px 10px rgba(212,150,42,.2) !important;
}

.gr-button-secondary,
button[data-testid="secondary"] {
  background:linear-gradient(135deg, var(--surface3), var(--surface4)) !important;
  color:var(--text-dim) !important;
  font-weight:700 !important; font-size:.66rem !important;
  letter-spacing:1px !important; text-transform:uppercase !important;
  border:1px solid var(--border-mid) !important;
  border-radius:var(--radius-sm) !important;
  transition:var(--transition) !important;
  font-family:'Rajdhani','IBM Plex Mono',sans-serif !important;
  box-shadow:var(--shadow-inset) !important;
  height:32px !important;
}
.gr-button-secondary:hover,
button[data-testid="secondary"]:hover {
  border-color:var(--border-gold) !important;
  color:var(--gold-bright) !important;
  background:linear-gradient(135deg, var(--surface4), rgba(212,150,42,.08)) !important;
  box-shadow:0 2px 12px rgba(212,150,42,.12), var(--shadow-inset) !important;
  transform:translateY(-1px) !important;
}

/* ══ PANELS & ACCORDIONS ═════════════════════════════════════════════ */

.gr-panel,.gr-box,.gr-group,
.gradio-container .block {
  background:linear-gradient(145deg, var(--surface), var(--surface2)) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius) !important;
  box-shadow:var(--shadow-sm), var(--shadow-inset) !important;
  transition:var(--transition-slow) !important;
}
.gr-panel:hover,.gr-group:hover {
  border-color:var(--border-mid) !important;
}

/* Glassmorphism cards */
.gradio-container .block.padded {
  background:linear-gradient(145deg,
    rgba(12,16,28,.9) 0%,
    rgba(8,11,18,.95) 100%) !important;
  backdrop-filter:blur(12px);
  border:1px solid var(--glass-border) !important;
}

/* Accordion */
.gr-accordion, details.gr-accordion {
  background:linear-gradient(135deg, var(--surface2), var(--surface3)) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius-sm) !important;
  margin-bottom:6px !important;
  overflow:hidden !important;
  transition:var(--transition) !important;
  box-shadow:var(--shadow-inset) !important;
}
.gr-accordion:hover, details.gr-accordion:hover {
  border-color:var(--border-mid) !important;
}
.gr-accordion summary, details.gr-accordion summary {
  padding:8px 12px !important;
  font-size:.64rem !important; font-weight:700 !important;
  letter-spacing:1.2px !important; text-transform:uppercase !important;
  color:var(--text-dim) !important;
  background:transparent !important;
  border-bottom:1px solid var(--border) !important;
  cursor:pointer !important;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  transition:var(--transition) !important;
  display:flex; align-items:center; gap:8px;
}
.gr-accordion summary:hover { color:var(--text) !important; background:rgba(212,150,42,.03) !important; }
.gr-accordion summary::marker,
.gr-accordion summary::-webkit-details-marker { color:var(--gold-dim) !important }

/* ══ MARKDOWN ══════════════════════════════════════════════════════════ */

.gr-markdown { color:var(--text) !important }
.gr-markdown h1 {
  color:var(--gold-bright); font-weight:700; margin-bottom:8px; font-size:1.05rem;
  font-family:'Orbitron','Rajdhani',sans-serif !important;
  text-shadow:0 0 12px rgba(212,150,42,.3);
}
.gr-markdown h2 {
  color:var(--gold); font-weight:700; margin:10px 0 6px; font-size:.9rem;
  font-family:'Rajdhani',sans-serif !important; letter-spacing:1px;
}
.gr-markdown h3 {
  color:var(--text-bright); font-weight:600; margin:8px 0 5px; font-size:.82rem;
  border-left:2px solid var(--gold-dim); padding-left:7px;
}
.gr-markdown p  { color:var(--text); line-height:1.6; margin:3px 0; font-size:.8rem; }
.gr-markdown ul, .gr-markdown ol { padding-left:16px; color:var(--text) }
.gr-markdown li { margin-bottom:2px; line-height:1.5; font-size:.8rem; }
.gr-markdown strong { color:var(--text-bright) }
.gr-markdown em { color:var(--text-dim) }

/* Tables */
.gr-markdown table {
  border-collapse:collapse; width:100%;
  font-size:.75rem;
  border-radius:var(--radius-sm); overflow:hidden;
  box-shadow:var(--shadow-sm);
}
.gr-markdown td, .gr-markdown th {
  padding:6px 12px; border:1px solid var(--border);
  color:var(--text);
}
.gr-markdown tr:nth-child(even) { background:rgba(255,255,255,.018); }
.gr-markdown tr:hover { background:rgba(212,150,42,.04); }
.gr-markdown th {
  background:linear-gradient(135deg, var(--surface3), var(--surface4));
  color:var(--gold-bright);
  font-size:.62rem; letter-spacing:1.5px; text-transform:uppercase;
  font-family:'IBM Plex Mono','Courier New',monospace;
  border-bottom:1px solid var(--border-gold);
}
.gr-markdown code {
  background:rgba(0,221,208,.07); color:var(--cyan);
  padding:1px 6px; border-radius:3px;
  font-family:'IBM Plex Mono','Courier New',monospace; font-size:.82em;
  border:1px solid rgba(0,221,208,.18);
}
.gr-markdown pre {
  background:linear-gradient(135deg, var(--surface3), var(--surface4));
  border:1px solid var(--border-mid);
  border-radius:var(--radius-sm); padding:10px; overflow-x:auto;
  box-shadow:var(--shadow-inset);
}
.gr-markdown pre code { background:none; border:none; padding:0 }

/* ══ PANEL HEADER ══════════════════════════════════════════════════════ */

.gis-panel-header {
  display:flex; align-items:center; gap:8px;
  padding:8px 12px 7px;
  background:linear-gradient(90deg, rgba(212,150,42,.06) 0%, transparent 80%);
  border-bottom:1px solid rgba(212,150,42,.12);
  border-radius:var(--radius) var(--radius) 0 0;
  font-size:.63rem; font-weight:700; letter-spacing:2px; text-transform:uppercase;
  color:var(--text-dim);
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  position:relative;
}
.gis-panel-header::before {
  content:''; width:3px; height:14px;
  background:linear-gradient(180deg, var(--gold-bright), var(--gold-dim));
  border-radius:3px; flex-shrink:0;
  box-shadow:0 0 10px rgba(212,150,42,.5);
}
.gis-panel-header::after {
  content:'';
  position:absolute; bottom:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg, var(--gold-dim), transparent 60%);
}

/* ══ OUTPUT AREA ══════════════════════════════════════════════════════ */

.out-label {
  font-size:.58rem; letter-spacing:2.5px; text-transform:uppercase;
  color:var(--text-muted); padding:4px 0 10px;
  display:flex; align-items:center; gap:8px;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
}
.out-label::before {
  content:''; display:inline-block; width:3px; height:13px;
  background:linear-gradient(180deg, var(--cyan), transparent);
  border-radius:3px;
  box-shadow:0 0 8px rgba(0,221,208,.4);
}

.stats-area {
  background:linear-gradient(145deg, var(--surface), var(--surface2)) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius) !important;
  padding:12px !important;
  min-height:280px;
  box-shadow:var(--shadow-sm), var(--shadow-inset) !important;
}
.tab-content-row { padding:10px !important; gap:10px !important; }

/* ══ DIVIDERS ══════════════════════════════════════════════════════════ */

.gis-divider {
  height:1px;
  background:linear-gradient(90deg, transparent, rgba(212,150,42,.2), var(--border-mid), rgba(212,150,42,.2), transparent);
  margin:12px 0;
}

/* ══ STATUS STRIP ═════════════════════════════════════════════════════ */

.gis-status-strip {
  display:flex; align-items:center; gap:12px;
  padding:4px 18px;
  background:linear-gradient(90deg, var(--surface) 0%, var(--surface2) 50%, var(--surface) 100%);
  border-top:1px solid rgba(212,150,42,.15);
  position:sticky; bottom:0; z-index:100;
  font-size:.56rem; letter-spacing:1px;
  overflow-x:auto; scrollbar-width:none;
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  box-shadow:0 -6px 28px rgba(0,0,0,.7);
}
.gis-status-strip::before {
  content:'';
  position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,
    transparent 0%,
    rgba(212,150,42,.08) 20%,
    rgba(212,150,42,.25) 50%,
    rgba(212,150,42,.08) 80%,
    transparent 100%);
}
.gis-status-strip::-webkit-scrollbar { display:none }
.gis-status-item {
  display:flex; align-items:center; gap:6px; white-space:nowrap;
  color:var(--text);
}
.gis-status-label { color:var(--text-muted) }
.gis-status-dot {
  width:6px; height:6px; border-radius:50%;
  background:var(--green);
  box-shadow:0 0 8px var(--green), 0 0 16px rgba(15,212,160,.3);
  animation:blink 2.5s infinite; flex-shrink:0;
}
.gis-status-dot.off {
  background:var(--text-muted); box-shadow:none; animation:none;
}
.gis-status-sep { color:var(--border-mid); user-select:none }

/* ══ UPLOAD BUTTON ═════════════════════════════════════════════════════ */

.gis-upload-btn button {
  background:linear-gradient(135deg, var(--surface3), var(--surface4)) !important;
  border:1px solid var(--border-mid) !important;
  color:var(--text-dim) !important;
  font-size:.63rem !important; font-weight:700 !important;
  letter-spacing:.8px !important; text-transform:uppercase !important;
  border-radius:var(--radius-sm) !important;
  padding:4px 12px !important;
  transition:var(--transition) !important;
  height:28px !important;
  font-family:'Rajdhani',sans-serif !important;
  box-shadow:var(--shadow-inset) !important;
}
.gis-upload-btn button:hover {
  border-color:var(--border-gold) !important;
  color:var(--gold-bright) !important;
  background:rgba(212,150,42,.07) !important;
  box-shadow:0 2px 10px rgba(212,150,42,.15), var(--shadow-inset) !important;
  transform:translateY(-1px) !important;
}

/* ══ FILE UPLOADS ══════════════════════════════════════════════════════ */

.compact-file-upload .upload-container,
.compact-file-upload label {
  min-height:40px !important; max-height:70px !important;
  background:linear-gradient(135deg, var(--surface3), var(--surface4)) !important;
  border:1px dashed var(--border-mid) !important;
  border-radius:var(--radius-sm) !important;
  transition:var(--transition) !important;
  box-shadow:var(--shadow-inset) !important;
}
.compact-file-upload .upload-container:hover {
  border-color:var(--border-gold) !important;
  background:rgba(212,150,42,.04) !important;
  box-shadow:0 0 16px rgba(212,150,42,.06), var(--shadow-inset) !important;
}
.compact-file-upload svg {
  width:12px !important; height:12px !important;
  color:var(--text-muted) !important;
}

/* Image output */
.gr-image {
  border-radius:var(--radius) !important;
  overflow:hidden !important;
  box-shadow:var(--shadow-md) !important;
  border:1px solid var(--border-mid) !important;
}
.gr-image img {
  border-radius:var(--radius) !important;
  width:100% !important;
}

/* ══ LOG / TEXTBOX ══════════════════════════════════════════════════════ */

.log-box textarea {
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  font-size:.72rem !important; color:var(--text-dim) !important;
  background:linear-gradient(135deg, var(--surface3), var(--surface4)) !important;
  line-height:1.5 !important;
  box-shadow:var(--shadow-inset) !important;
}

/* ══ 3D CONTROL BAR ════════════════════════════════════════════════════ */

.v3d-control-bar {
  display:flex !important; align-items:center !important; gap:10px !important;
  padding:8px 18px !important;
  background:linear-gradient(90deg, var(--surface) 0%, var(--surface2) 100%) !important;
  border-bottom:1px solid var(--border-mid) !important;
  flex-wrap:nowrap !important; overflow-x:auto !important;
  min-height:52px !important;
  box-shadow:var(--shadow-inset) !important;
}
.v3d-fullscreen {
  position:relative !important; width:100% !important;
  height:calc(100vh - 200px) !important; min-height:580px !important;
  background:#020305 !important;
  border-radius:var(--radius) !important;
  overflow:hidden !important;
  border:1px solid var(--border-mid) !important;
  box-shadow:var(--shadow-lg) !important;
}
#v3d-frame { position:absolute; inset:0; width:100% !important; height:100% !important; border:none }

/* ══ GRADIO FIXES ══════════════════════════════════════════════════════ */

.gradio-container .block { border-radius:var(--radius) !important }
.gradio-container .wrap { gap:8px !important }
.gradio-container .gap { gap:8px !important }
.gr-padded { padding:10px !important }

/* Progress bar — gold theme */
.progress-bar { background:var(--gold) !important; border-radius:2px !important }
.progress-level-inner {
  background:rgba(212,150,42,.1) !important;
  border:1px solid rgba(212,150,42,.15) !important;
}

/* File download */
.file-preview {
  background:linear-gradient(135deg, var(--surface2), var(--surface3)) !important;
  border:1px solid var(--border) !important;
  border-radius:var(--radius-sm) !important;
}

/* ══ GIS CARD ══════════════════════════════════════════════════════════ */

.gis-card {
  background:linear-gradient(145deg, var(--surface2), var(--surface3));
  border:1px solid var(--glass-border);
  border-radius:var(--radius-lg);
  padding:16px;
  position:relative; overflow:hidden;
  box-shadow:var(--shadow-md), var(--shadow-inset);
  transition:var(--transition-slow);
  backdrop-filter:blur(10px);
}
.gis-card:hover {
  border-color:rgba(212,150,42,.25);
  box-shadow:var(--shadow-md), var(--shadow-gold), var(--shadow-inset);
  transform:translateY(-1px);
}
.gis-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,
    transparent 0%,
    var(--gold-dim) 20%,
    var(--gold-bright) 50%,
    var(--gold-dim) 80%,
    transparent 100%);
  animation:glow-pulse 4s ease-in-out infinite;
}
.gis-card::after {
  content:''; position:absolute; inset:0;
  background:linear-gradient(135deg, var(--glass-highlight) 0%, transparent 60%);
  pointer-events:none;
}

/* ══ HIGHLIGHT BOX ═════════════════════════════════════════════════════ */

.gis-highlight-box {
  background:rgba(212,150,42,.04);
  border:1px solid rgba(212,150,42,.16);
  border-left:3px solid var(--gold);
  border-radius:0 var(--radius-sm) var(--radius-sm) 0;
  padding:9px 13px; margin:6px 0;
  font-size:.75rem; color:var(--text-dim);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.02);
}

/* ══ TOOLTIP-STYLE INFO TAGS ═══════════════════════════════════════════ */

.gis-tag {
  display:inline-flex; align-items:center; gap:4px;
  padding:1px 9px;
  background:var(--surface4);
  border:1px solid var(--border-mid);
  border-radius:14px;
  font-size:.57rem; font-weight:700; letter-spacing:.5px;
  color:var(--text-dim);
  font-family:'IBM Plex Mono','Courier New',monospace !important;
  transition:var(--transition);
}

/* ══ RESPONSIVE ════════════════════════════════════════════════════════ */

@media (max-width:860px) {
  .gis-topbar { padding:0 14px; gap:10px }
  .gis-topbar-chips { display:none }
  .gis-institution { display:none }
  .tab-content-row { flex-direction:column !important }
  .gis-app-name { font-size:.75rem }
}

/* ══ GRADIO INTERNAL PATCHES ═══════════════════════════════════════════ */

.svelte-1gfkn6j { color:var(--text-dim) !important }
[data-testid="textbox"] textarea { color:var(--text) !important }
[data-testid="dropdown"] select  { color:var(--text) !important }
.wrap.svelte-12cmxck { background:var(--surface) !important }

/* ══ FULL-SCREEN LAYOUT ════════════════════════════════════════════════ */

html,body{width:100%!important;height:100%!important;overflow-x:hidden!important}
.gradio-container{max-width:100%!important;width:100%!important;min-height:100vh!important;padding:0!important;margin:0!important}
.main,.contain,.wrap,#component-0,div[id^="component-"].gradio-container>div{max-width:100%!important;width:100%!important}
.app.svelte-182fdeq,.app{max-width:100%!important}
"""



# ── HELPER ────────────────────────────────────────────────────────────

def _resolve_shared(shared_state, per_tab_file):
    p = _fpath(per_tab_file)
    if p: return p
    if shared_state: return shared_state
    return None


# ── UI ────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title=f"Gold Prospectivity System {VERSION}", fill_width=True) as app:

    shared_s2_state    = gr.State(value=None)
    shared_aster_state = gr.State(value=None)

    # ── TOP BAR ──────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="gis-topbar">
      <div class="gis-logo-area">
        <div class="gis-logo-icon">⛏</div>
        <div>
          <div class="gis-app-name">GOLDGIS</div>
          <div class="gis-app-sub">Prospectivity Platform {VERSION}</div>
        </div>
      </div>
      <span class="gis-badge">🛰 S2 + ASTER + DEM</span>
      <span class="gis-badge gold">RF Calibrated</span>
      <span class="gis-badge violet">ASTER B4–B9</span>
      <span class="gis-badge green">{'● ' + model_status[:28] if '✅' in model_status else '○ No Model'}</span>
      <span class="gis-badge" style="margin-left:auto">Beni-Suef University · Nader Safwat Ayed Hanna</span>
    </div>""")

    # ── GLOBAL UPLOAD STRIP ───────────────────────────────────────────
    with gr.Row(elem_classes="gis-upload-strip"):
        gr.HTML('<span class="gis-upload-label">📂 Sentinel-2 :</span>')
        global_s2 = gr.UploadButton("⬆ S2 GeoTIFF", file_types=[".tif",".tiff"],
                                     size="sm", elem_classes="gis-upload-btn")
        gr.HTML('<span class="gis-upload-label" style="margin-left:14px">🛰 ASTER SWIR :</span>')
        global_aster = gr.UploadButton("⬆ ASTER GeoTIFF", file_types=[".tif",".tiff"],
                                        size="sm", elem_classes="gis-upload-btn")
        global_status = gr.HTML(
            '<span class="gis-upload-status warn">No files loaded — upload Sentinel-2 (18-band) and optionally ASTER SWIR (6-band)</span>')

    def _on_s2_upload(f):
        if f is None: return None, '<span class="gis-upload-status warn">No S2 file</span>'
        p = _fpath(f)
        try:
            with rasterio.open(p) as src:
                nb = src.count; h = src.height; w = src.width
            msg = (f'<span class="gis-upload-status">✔ S2: {os.path.basename(p)} &nbsp;·&nbsp; '
                   f'{nb} bands &nbsp;·&nbsp; {w}×{h} px</span>')
        except Exception as e:
            msg = f'<span class="gis-upload-status warn">⚠ {e}</span>'
        return p, msg

    def _on_aster_upload(f):
        if f is None: return None, '<span class="gis-upload-status warn">No ASTER file</span>'
        p = _fpath(f)
        try:
            with rasterio.open(p) as src:
                nb = src.count; h = src.height; w = src.width
            msg = (f'<span class="gis-upload-status" style="border-color:rgba(155,89,255,.3);'
                   f'color:#9b59ff;background:rgba(155,89,255,.08);">'
                   f'✔ ASTER: {os.path.basename(p)} &nbsp;·&nbsp; {nb} bands &nbsp;·&nbsp; {w}×{h} px</span>')
        except Exception as e:
            msg = f'<span class="gis-upload-status warn">⚠ {e}</span>'
        return p, msg

    global_s2.upload(fn=_on_s2_upload, inputs=[global_s2],
                     outputs=[shared_s2_state, global_status])
    global_aster.upload(fn=_on_aster_upload, inputs=[global_aster],
                        outputs=[shared_aster_state, global_status])

    # ── TABS ──────────────────────────────────────────────────────────
    with gr.Tabs(elem_id="gis-tabs-root"):

        # ══ TAB 1: GOLD DETECTOR ══════════════════════════════════════
        with gr.Tab("🔍 Detector"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Detection Parameters</div>')
                    with gr.Accordion("⚙️  Detection Mode", open=True):
                        use_model_chk = gr.Checkbox(label="Use RF Model  (uncheck = index-only)", value=True)
                        comparison_dd = gr.Dropdown(
                            choices=COMPARISON_MODES, value="Hybrid Sentinel-2 + ASTER",
                            label="Data source comparison mode")
                        extra_idx_txt = gr.Textbox(label="Index-only — indices to combine",
                                                   value="IO, CM, GS, AST_HydAlt")
                    with gr.Accordion("🗺️  Display Settings", open=True):
                        composite_dd  = gr.Dropdown(choices=COMPOSITE_CHOICES, value="Iron Oxide Index",
                                                    label="Reference panel")
                        coord_dd      = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                    label="Coordinate grid")
                        threshold_sl  = gr.Slider(0.30, 0.90, 0.60, step=0.05,
                                                  label="High-priority threshold")
                    with gr.Accordion("🏷️  Label Overlay", open=False):
                        show_lbl_chk  = gr.Checkbox(label="Show known gold pixels in cyan", value=False)
                        label_file_in = gr.File(label="Label GeoTIFF (0/1)", file_types=[".tif",".tiff"])
                    with gr.Accordion("💾  Export", open=False):
                        save_tif_chk  = gr.Checkbox(label="Export probability map as GeoTIFF", value=True)
                    det_btn = gr.Button("▶  Run Gold Analysis", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Analysis Map</div>')
                    det_map     = gr.Image(label="", type="filepath", show_label=False)
                    det_tif_out = gr.File(label="📥 Download GeoTIFF")
                    det_stats   = gr.Markdown("*Run analysis to see the full report.*")

            def _run_det(s2_shared, aster_shared, thresh, use_m, comp, coord, extra,
                         show_lbl, lbl_f, save_t, cmp_mode):
                return predict_gold(s2_shared, aster_shared, thresh, use_m, comp, coord,
                                    extra, show_lbl, lbl_f, save_t, cmp_mode)

            det_btn.click(
                fn=_run_det,
                inputs=[shared_s2_state, shared_aster_state,
                        threshold_sl, use_model_chk, composite_dd,
                        coord_dd, extra_idx_txt, show_lbl_chk,
                        label_file_in, save_tif_chk, comparison_dd],
                outputs=[det_map, det_tif_out, det_stats])

        # ══ TAB 2: INDEX EXPLORER ═════════════════════════════════════
        with gr.Tab("🗺️ Index"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Explorer Parameters</div>')
                    with gr.Accordion("📈  Index & Colour", open=True):
                        exp_idx  = gr.Dropdown(choices=ALL_INDEX_NAMES, value="IO",
                                               label="Band / Index")
                        exp_cmap = gr.Dropdown(choices=CMAP_OPTIONS, value="hot",
                                               label="Colour map")
                    with gr.Accordion("🔧  Stretch & Grid", open=True):
                        exp_lo    = gr.Slider(0, 10, 2, step=0.5, label="Stretch low %")
                        exp_hi    = gr.Slider(90, 100, 98, step=0.5, label="Stretch high %")
                        exp_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                               label="Coordinate grid")
                    with gr.Accordion("💾  Options", open=False):
                        exp_hist = gr.Checkbox(label="Show histogram", value=True)
                        exp_save = gr.Checkbox(label="Export as GeoTIFF", value=False)
                    exp_btn = gr.Button("▶  Explore Index", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Index Map</div>')
                    exp_map   = gr.Image(label="", type="filepath", show_label=False)
                    exp_out   = gr.File(label="📥 Download GeoTIFF")
                    exp_stats = gr.Markdown("*Results here.*")

            exp_btn.click(
                fn=lambda s2, idx, cmap, lo, hi, hist, coord, save:
                    explore_index(s2, idx, cmap, lo, hi, hist, coord, save),
                inputs=[shared_s2_state, exp_idx, exp_cmap, exp_lo, exp_hi,
                        exp_hist, exp_coord, exp_save],
                outputs=[exp_map, exp_out, exp_stats])

        # ══ TAB 3: MULTI-INDEX COMPARE ════════════════════════════════
        with gr.Tab("📊 Compare"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Comparison Parameters</div>')
                    with gr.Accordion("📊  Index Selection", open=True):
                        cmp_names = gr.Textbox(label="Indices (comma-separated, max 6)",
                                               value="IO, CM, GS, AST_HydAlt, AST_AlOH, AST_Silica")
                        cmp_cmaps = gr.Textbox(label="Colormaps (optional)",
                                               value="hot, YlOrBr, Reds, RdYlGn_r, YlOrBr, plasma")
                    with gr.Accordion("🗺️  Grid Settings", open=True):
                        cmp_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                               label="Coordinate grid")
                    with gr.Accordion("ℹ️  Index Groups", open=False):
                        gr.Markdown("""
**Sentinel-2 Iron:** `IO, FI, IronEx, GosEx, Ferric`

**Sentinel-2 Hydrothermal:** `CM, AlOH, MgOH, Silica, GS`

**ASTER SWIR:** `AST_Ferric, AST_AlOH, AST_MgOH, AST_Silica, AST_Clay, AST_Quartz, AST_HydAlt`

**Terrain:** `DEM, SLP, ASP, RGH`
""")
                    cmp_btn = gr.Button("▶  Compare Indices", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Comparison Grid</div>')
                    cmp_map   = gr.Image(label="", type="filepath", show_label=False)
                    cmp_stats = gr.Markdown("*Results here.*")

            cmp_btn.click(
                fn=lambda s2, aster, names, cmaps, coord:
                    multi_index_compare(s2, aster, names, cmaps, coord),
                inputs=[shared_s2_state, shared_aster_state, cmp_names, cmp_cmaps, cmp_coord],
                outputs=[cmp_map, cmp_stats])

        # ══ TAB 4: ASTER SWIR EXPLORER ════════════════════════════════
        with gr.Tab("🌋 ASTER"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">ASTER SWIR Explorer</div>')
                    with gr.Accordion("🛰  Band / Index", open=True):
                        ast_sel = gr.Dropdown(
                            choices=ALL_ASTER_NAMES,
                            value="AST_HydAlt",
                            label="ASTER band or index")
                        ast_cmap = gr.Dropdown(choices=CMAP_OPTIONS, value="RdYlGn_r",
                                               label="Colour map")
                        ast_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                               label="Coordinate grid")
                    with gr.Accordion("🗺️  Options", open=True):
                        ast_all = gr.Checkbox(
                            label="Show all 6 ASTER bands + 12 indices (overview mode)", value=False)
                        ast_save = gr.Checkbox(label="Export index as GeoTIFF", value=False)
                    with gr.Accordion("ℹ️  ASTER Band Guide", open=False):
                        gr.Markdown("""
| Band | Wavelength | Key Minerals |
|---|---|---|
| B4 | 1.60-1.70 µm | References |
| B5 | 2.145-2.185 µm | Al-OH (kaolinite) |
| B6 | 2.185-2.225 µm | Al-OH (dickite) |
| B7 | 2.235-2.285 µm | Al-OH / carbonate |
| B8 | 2.295-2.365 µm | Fe-Mg clay, talc |
| B9 | 2.360-2.430 µm | Carbonate, talc |

**Key Indices:**
- `AST_AlOH` → kaolinite, alunite (epithermal)
- `AST_MgOH` → chlorite, carbonate (propylitic)
- `AST_HydAlt` → composite alteration signature
- `AST_Silica` → silicification / quartz veins
- `AST_Clay` → general clay discrimination
""")
                    ast_btn = gr.Button("▶  Explore ASTER", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — ASTER SWIR Map</div>')
                    ast_map  = gr.Image(label="", type="filepath", show_label=False)
                    ast_out  = gr.File(label="📥 Download GeoTIFF")
                    ast_stat = gr.Markdown("*Upload ASTER SWIR stack and run.*")

            ast_btn.click(
                fn=lambda aster, s2, sel, cmap, coord, all_b, save:
                    explore_aster(aster, s2, sel, cmap, coord, all_b, save),
                inputs=[shared_aster_state, shared_s2_state,
                        ast_sel, ast_cmap, ast_coord, ast_all, ast_save],
                outputs=[ast_map, ast_out, ast_stat])

        # ══ TAB 5: HYDROTHERMAL MAP ════════════════════════════════════
        with gr.Tab("🔥 Hydro"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Hydrothermal Alteration Mapping</div>')
                    with gr.Accordion("⚙️  Parameters", open=True):
                        hyd_thresh = gr.Slider(50, 95, 75, step=5,
                                              label="Alteration threshold percentile (p)")
                        hyd_coord  = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                label="Coordinate grid")
                    with gr.Accordion("💾  Export", open=False):
                        hyd_mask = gr.Checkbox(label="Export hydrothermal mask GeoTIFF", value=True)
                    with gr.Accordion("ℹ️  Interpretation Guide", open=False):
                        gr.Markdown("""
**Hydrothermal Alteration Index** = (B5+B7+B9)/(B4+B6+B8)

Values > 1.0 indicate hydrothermal alteration.  
The intensity map classifies pixels into 4 levels:
- **Intense** (top 5%) → likely alteration core
- **High** (p80-p95) → strong alteration halo
- **Moderate** (p60-p80) → weak alteration fringe
- **Low** (< p60) → background / unaltered

**Key minerals detected:**
- AlOH → kaolinite, alunite (argillic/advanced argillic)
- MgOH → chlorite, talc, carbonate (propylitic)
- Ferric → iron oxides, gossans
- Silica → silicification, quartz veins
""")
                    hyd_btn = gr.Button("▶  Map Hydrothermal Alteration", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Hydrothermal Alteration Map</div>')
                    hyd_map  = gr.Image(label="", type="filepath", show_label=False)
                    hyd_out  = gr.File(label="📥 Download Hydrothermal Mask")
                    hyd_stat = gr.Markdown("*Upload ASTER SWIR and run.*")

            hyd_btn.click(
                fn=lambda aster, s2, thresh, coord, mask:
                    map_hydrothermal(aster, s2, thresh, coord, mask),
                inputs=[shared_aster_state, shared_s2_state,
                        hyd_thresh, hyd_coord, hyd_mask],
                outputs=[hyd_map, hyd_out, hyd_stat])

        # ══ TAB 6: MINERAL MAPPING ═════════════════════════════════════
        with gr.Tab("💎 Minerals"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Quartz/Silica & Clay Mapping</div>')
                    with gr.Accordion("⚙️  Thresholds", open=True):
                        min_qtz  = gr.Slider(50, 98, 80, step=2,
                                            label="Quartz threshold percentile")
                        min_clay = gr.Slider(50, 98, 75, step=2,
                                            label="Clay threshold percentile")
                        min_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                               label="Coordinate grid")
                    with gr.Accordion("💾  Export", open=False):
                        min_poly = gr.Checkbox(label="Export mineral classes as GeoTIFF", value=True)
                    with gr.Accordion("ℹ️  Mineral Guide", open=False):
                        gr.Markdown("""
**Quartz Index** = (B5×B7)/(B6²)

**Clay Ratio** = (B5+B7)/B6

**Combined map classes:**
- 🔴 Quartz + Clay → hydrothermal core (gold target zone)
- 🟣 Quartz/Silica only → silicification / vein quartz
- 🟠 Clay only → argillic alteration halo
- ⬛ Background → unaltered country rock

High Quartz + AlOH + MgOH overlap = **orogenic gold target**
""")
                    min_btn = gr.Button("▶  Map Minerals", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Mineral Discrimination Map</div>')
                    min_map  = gr.Image(label="", type="filepath", show_label=False)
                    min_out  = gr.File(label="📥 Download Mineral Classes GeoTIFF")
                    min_stat = gr.Markdown("*Upload ASTER SWIR and run.*")

            min_btn.click(
                fn=lambda aster, s2, qtz, clay, coord, poly:
                    map_minerals(aster, s2, qtz, clay, coord, poly),
                inputs=[shared_aster_state, shared_s2_state,
                        min_qtz, min_clay, min_coord, min_poly],
                outputs=[min_map, min_out, min_stat])

        # ══ TAB 7: TRAIN MODEL ════════════════════════════════════════
        with gr.Tab("🧠 Train"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=300):
                    gr.HTML('<div class="gis-panel-header">Training Configuration</div>')
                    with gr.Accordion("📂  Training Data", open=True):
                        tr_feat = gr.File(label="Sentinel-2 Feature GeoTIFFs (one per site)",
                                         file_types=[".tif",".tiff"], file_count="multiple",
                                         elem_classes="compact-file-upload")
                        tr_aster = gr.File(label="ASTER SWIR GeoTIFFs (optional, per site)",
                                          file_types=[".tif",".tiff"], file_count="multiple",
                                          elem_classes="compact-file-upload")
                        tr_use_lbl = gr.Checkbox(label="Use label TIFs (supervised)", value=True)
                        tr_lbl = gr.File(label="Label GeoTIFFs (0=bg / 1=gold) — optional",
                                        file_types=[".tif",".tiff"], file_count="multiple",
                                        elem_classes="compact-file-upload")
                    with gr.Accordion("📍  Site Configuration", open=True):
                        tr_sites = gr.Textbox(label="Site names (comma-separated)",
                                              placeholder="Sukari, Fatira, Haimur")
                        tr_types = gr.Textbox(label="Site types (gold / no-gold / proxy)",
                                              placeholder="gold, gold, no-gold", value="gold")
                    with gr.Accordion("⚙️  Model Hyperparameters", open=True):
                        tr_trees    = gr.Slider(50, 1000, 300, step=50, label="Trees (RF/base estimator)")
                        tr_depth    = gr.Textbox(label="Max depth (number or None)", value="None")
                        tr_algo     = gr.Dropdown(
                            choices=["Random Forest", "Ensemble RF+GB",
                                     "XGBoost", "LightGBM", "Full Ensemble RF+GB+XGB+LGBM"],
                            value="Ensemble RF+GB",
                            label="🧠 Algorithm / Ensemble")
                        tr_ensemble = gr.Checkbox(label="🧠 Ensemble mode (for backward compat.)", value=True)
                        tr_calibrate= gr.Checkbox(label="📐 Calibrate probabilities (CalibratedClassifierCV)", value=True)
                        tr_shap     = gr.Checkbox(
                            label=f"🔍 SHAP explainability {'✅ installed' if _HAS_SHAP else '⚠ install: pip install shap'}",
                            value=_HAS_SHAP)
                        tr_optuna   = gr.Checkbox(
                            label=f"🔬 Optuna HPO (50 trials) {'✅ installed' if _HAS_OPTUNA else '⚠ install: pip install optuna'}",
                            value=False)
                    with gr.Accordion("💾  Save Model", open=True):
                        tr_save = gr.Textbox(label="Save path (.pkl)", value=MODEL_PATH, placeholder=MODEL_PATH)
                    tr_btn = gr.Button("▶  Start Training", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Training Report</div>')
                    tr_plot  = gr.Image(label="", type="filepath", show_label=False)
                    tr_model = gr.File(label="📥 Download .pkl")
                    tr_stats = gr.Markdown("*Results here.*")

            tr_btn.click(
                fn=run_training,
                inputs=[tr_feat, tr_aster, tr_lbl, tr_use_lbl,
                        tr_sites, tr_types, tr_trees, tr_depth,
                        tr_ensemble, tr_calibrate, tr_algo, tr_shap,
                        tr_optuna, tr_save],
                outputs=[tr_plot, tr_model, tr_stats])

        # ══ TAB N1: SPECTRAL ANGLE MAPPER ═════════════════════════════
        with gr.Tab("🔬 SAM"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Spectral Angle Mapper (SAM)</div>')
                    with gr.Accordion("🔬  Target Endmember", open=True):
                        sam_em   = gr.Dropdown(
                            choices=list(SAM_ENDMEMBERS.keys()),
                            value=list(SAM_ENDMEMBERS.keys())[0],
                            label="Target mineral endmember")
                        sam_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                label="Coordinate grid")
                    with gr.Accordion("💾  Export", open=False):
                        sam_save = gr.Checkbox(label="Export SAM angle GeoTIFF", value=False)
                    with gr.Accordion("ℹ️  About SAM", open=False):
                        gr.Markdown("""
**Spectral Angle Mapper (SAM)** measures the angle between an image pixel spectrum
and a reference mineral spectrum (endmember). Lower angle = better spectral match.

**Endmembers used:**
- **Kaolinite** — Al-OH argillic alteration (advanced argillic / epithermal)
- **Goethite** — Iron oxide / gossan zones
- **Chlorite** — Propylitic alteration (greenschist facies)
- **Quartz Vein** — Silicification / quartz vein systems

*Best-match map highlights the most probable mineral identity per pixel.*
""")
                    sam_btn = gr.Button("▶  Run SAM Analysis", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — SAM Mineral Map</div>')
                    sam_map  = gr.Image(label="", type="filepath", show_label=False)
                    sam_tif  = gr.File(label="📥 Download SAM GeoTIFF")
                    sam_stat = gr.Markdown("*Upload Sentinel-2 and run.*")

            sam_btn.click(
                fn=lambda s2, aster, em, coord, save:
                    run_sam_analysis(s2, aster, em, coord, save),
                inputs=[shared_s2_state, shared_aster_state, sam_em, sam_coord, sam_save],
                outputs=[sam_map, sam_tif, sam_stat])

        # ══ TAB N2: PCA ANOMALY DETECTION ═════════════════════════════
        with gr.Tab("🧮 PCA"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">PCA Spectral Anomaly Detection</div>')
                    with gr.Accordion("⚙️  Parameters", open=True):
                        pca_comp  = gr.Slider(2, 8, 4, step=1, label="Number of PCA components")
                        pca_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                label="Coordinate grid")
                    with gr.Accordion("💾  Export", open=False):
                        pca_save = gr.Checkbox(label="Export PC maps + anomaly as GeoTIFF", value=False)
                    with gr.Accordion("ℹ️  About PCA Anomaly", open=False):
                        gr.Markdown("""
**PCA Anomaly Detection** decomposes the multi-band image into uncorrelated principal
components. Pixels with high Euclidean distance from the scene centroid in PC-space
are spectral anomalies — often mineralised zones that don't match the regional background.

**Outputs:**
- **PC1–PC3 maps** — major scene variance components
- **Anomaly map** — spectral distance from centroid (top 10% highlighted)
- **Scree plot** — variance explained per component

*Include ASTER for a 24-band PCA (18 S2 + 6 ASTER bands).*
""")
                    pca_btn = gr.Button("▶  Run PCA Anomaly", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — PCA Anomaly Map</div>')
                    pca_map  = gr.Image(label="", type="filepath", show_label=False)
                    pca_tif  = gr.File(label="📥 Download PCA GeoTIFF")
                    pca_stat = gr.Markdown("*Upload Sentinel-2 and run.*")

            pca_btn.click(
                fn=lambda s2, aster, nc, coord, save:
                    run_pca_anomaly(s2, aster, nc, coord, save),
                inputs=[shared_s2_state, shared_aster_state, pca_comp, pca_coord, pca_save],
                outputs=[pca_map, pca_tif, pca_stat])

        # ══ TAB N3: STRUCTURAL ANALYSIS ════════════════════════════════
        with gr.Tab("🏔️ Structure"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Lineament & Structural Analysis</div>')
                    with gr.Accordion("⚙️  Parameters", open=True):
                        str_kern  = gr.Slider(3, 31, 7, step=2,
                                              label="Lineament kernel window (px, odd)")
                        str_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                label="Coordinate grid")
                    with gr.Accordion("💾  Export", open=False):
                        str_save = gr.Checkbox(label="Export structural maps GeoTIFF", value=False)
                    with gr.Accordion("ℹ️  Method", open=False):
                        gr.Markdown("""
**Lineament Density** uses Sobel gradient edge detection on the DEM to identify
topographic lineaments (faults, shear zones, lithological contacts).

**Bearing Map** weights N–S structural trends (0–180°) higher than E–W trends,
reflecting the dominant NNW-SSE structural fabric of the Eastern Desert.

**Fe-Oxide × Structure** product identifies fault-controlled iron enrichment —
the classic orogenic gold structural target environment.

*High structural density + high iron oxide = optimal structural trap for gold.*
""")
                    str_btn = gr.Button("▶  Run Structural Analysis", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Structural Map</div>')
                    str_map  = gr.Image(label="", type="filepath", show_label=False)
                    str_tif  = gr.File(label="📥 Download Structural GeoTIFF")
                    str_stat = gr.Markdown("*Upload Sentinel-2 (with DEM bands) and run.*")

            str_btn.click(
                fn=lambda s2, aster, kern, coord, save:
                    run_structural_analysis(s2, aster, kern, coord, save),
                inputs=[shared_s2_state, shared_aster_state, str_kern, str_coord, str_save],
                outputs=[str_map, str_tif, str_stat])

        # ══ TAB N4: MCDA ═══════════════════════════════════════════════
        with gr.Tab("🎯 MCDA"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Multi-Criteria Decision Analysis</div>')
                    with gr.Accordion("⚖️  Criterion Weights", open=True):
                        mcda_w_io    = gr.Slider(0.0, 5.0, 2.0, step=0.5, label="Iron Oxide weight")
                        mcda_w_cm    = gr.Slider(0.0, 5.0, 1.5, step=0.5, label="Clay Minerals weight")
                        mcda_w_aster = gr.Slider(0.0, 5.0, 2.5, step=0.5, label="ASTER Alteration weight")
                        mcda_w_str   = gr.Slider(0.0, 5.0, 1.0, step=0.5, label="Structural Density weight")
                        mcda_w_elev  = gr.Slider(0.0, 5.0, 0.5, step=0.5, label="Elevation Inverse weight")
                    with gr.Accordion("⚙️  Output Settings", open=True):
                        mcda_thr   = gr.Slider(0.30, 0.90, 0.60, step=0.05,
                                               label="High-prospectivity threshold")
                        mcda_coord = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                 label="Coordinate grid")
                    with gr.Accordion("💾  Export", open=False):
                        mcda_save = gr.Checkbox(label="Export MCDA score GeoTIFF", value=True)
                    with gr.Accordion("ℹ️  About MCDA", open=False):
                        gr.Markdown("""
**Weighted Overlay MCDA** combines multiple independent evidence layers into a single
composite prospectivity score. Each layer is normalised 0–1 before weighting.

**Layers:**
- 🔴 **Iron Oxide** (IO = B04/B02) — direct iron alteration evidence
- 🟠 **Clay Minerals** (CM = B11/B8A) — argillic alteration evidence
- 🌋 **ASTER Alteration** (HydAlt) — hydrothermal composite index
- 🏔️ **Structural Density** — lineament / fault proximity
- 📏 **Elevation Inverse** — preference for lower-lying placer/oxidised zones

*Adjust weights based on your geological model for the target area.*
""")
                    mcda_btn = gr.Button("▶  Run MCDA", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — MCDA Composite Score</div>')
                    mcda_map  = gr.Image(label="", type="filepath", show_label=False)
                    mcda_tif  = gr.File(label="📥 Download MCDA GeoTIFF")
                    mcda_stat = gr.Markdown("*Set weights and run.*")

            mcda_btn.click(
                fn=lambda s2, aster, w_io, w_cm, w_ast, w_str, w_el, coord, thr, save:
                    run_mcda(s2, aster, w_io, w_cm, w_ast, w_str, w_el, coord, thr, save),
                inputs=[shared_s2_state, shared_aster_state,
                        mcda_w_io, mcda_w_cm, mcda_w_aster, mcda_w_str, mcda_w_elev,
                        mcda_coord, mcda_thr, mcda_save],
                outputs=[mcda_map, mcda_tif, mcda_stat])

        # ══ TAB N5: ZONE CSV EXPORT ═════════════════════════════════════
        with gr.Tab("📋 Export"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">Export High-Probability Zone to CSV</div>')
                    with gr.Accordion("⚙️  Export Settings", open=True):
                        exp_thr = gr.Slider(0.30, 0.95, 0.60, step=0.05,
                                            label="Probability threshold (pixels above exported)")
                    with gr.Accordion("ℹ️  Output Columns", open=False):
                        gr.Markdown("""
**Each exported pixel row contains:**

`row, col` — pixel coordinates  
`lat, lon` — WGS84 geographic coordinates  
`probability` — RF model probability (if model loaded)  
`iron_oxide, clay_minerals, ferrous_iron, gossan, ndvi` — Sentinel-2 indices  
`elevation_m, slope_deg, aspect_deg, roughness_m` — terrain metrics  
`AST_AlOH, AST_Carb, AST_Clay, ...` — all 12 ASTER indices (if ASTER uploaded)

*Output is limited to 50,000 pixels (random sample for very large zones).*
*Import CSV into QGIS/ArcGIS for field verification planning.*
""")
                    zexp_btn = gr.Button("▶  Export Zone CSV", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Zone Data Table</div>')
                    zexp_csv  = gr.File(label="📥 Download Zone CSV")
                    zexp_stat = gr.Markdown("*Upload Sentinel-2 (and optionally ASTER), then export.*")

            zexp_btn.click(
                fn=lambda s2, aster, thr: export_zones_csv(s2, aster, thr),
                inputs=[shared_s2_state, shared_aster_state, exp_thr],
                outputs=[zexp_csv, zexp_stat])

        # ══ TAB 8: STACK BUILDER (imported from STACK_BUILDER.py) ═══════════
        with gr.Tab("🛰️ Stack"):
            # ── Import functions from STACK_BUILDER.py if it exists ───
            # Fall back to the built-in run_converter defined in this file.
            import importlib.util as _ilu, os as _osb
            _sb_path = _osb.path.join(_osb.path.dirname(_osb.path.abspath(__file__)), "STACK_BUILDER.py")
            if _osb.path.exists(_sb_path):
                _sb_spec = _ilu.spec_from_file_location("STACK_BUILDER", _sb_path)
                _sb_mod  = _ilu.module_from_spec(_sb_spec)
                _sb_spec.loader.exec_module(_sb_mod)
                _sb_run_converter    = _sb_mod.run_converter
                _sb_run_aster        = _sb_mod.run_aster_converter
                _sb_run_aster_18band = _sb_mod.run_aster_18band
            else:
                # Use the built-in run_converter (defined in this file)
                _sb_run_converter    = run_converter
                def _sb_run_aster(*a, **k):        return None, "⚠️ STACK_BUILDER.py not found — ASTER-only builder unavailable."
                def _sb_run_aster_18band(*a, **k): return None, "⚠️ STACK_BUILDER.py not found — ASTER 18-band builder unavailable."

            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):

                    # ── Section 1: Sentinel-2 + DEM ───────────────────
                    with gr.Accordion("🛰️  Sentinel-2 + DEM  →  18-band stack", open=True):
                        sb_10  = gr.File(label="10m bands (.tif) — B02 B03 B04 B08",
                                         file_types=[".tif", ".tiff"], file_count="multiple",
                                         elem_classes="compact-file-upload")
                        sb_20  = gr.File(label="20m bands (.tif) — B05 B06 B8A B11 B12 [optional]",
                                         file_types=[".tif", ".tiff"], file_count="multiple",
                                         elem_classes="compact-file-upload")
                        sb_dem = gr.File(label="DEM (.tif)", file_types=[".tif", ".tiff"],
                                         elem_classes="compact-file-upload")
                        sb_aster = gr.File(label="ASTER SWIR stack (.tif) [optional]",
                                           file_types=[".tif", ".tiff"],
                                           elem_classes="compact-file-upload")
                        gr.Markdown("`B02 B03 B04 B05 B06 B8A B08 B11 B12 · IO CM FI GS NDVI · DEM Slope Aspect Roughness`")
                        sb_btn = gr.Button("⚙️  Build Sentinel-2 Stack", variant="primary", size="lg")

                    # ── Section 2: ASTER 6-band ────────────────────────
                    with gr.Accordion("🌋  ASTER  →  6-band stack (B04–B09)", open=False):
                        ast_files = gr.File(label="ASTER band files (.tif) — B04 B05 B06 B07 B08 B09",
                                            file_types=[".tif", ".tiff"], file_count="multiple",
                                            elem_classes="compact-file-upload")
                        ast_px = gr.Dropdown(label="Output pixel size (m)",
                                             choices=["15", "30"], value="15", interactive=True)
                        gr.Markdown("`aster_stack.tif` — 6 bands, float32")
                        ast_btn = gr.Button("⚙️  Build ASTER Stack", variant="primary", size="lg")

                    # ── Section 3: ASTER 6-band → 18-band ─────────────
                    with gr.Accordion("🔬  ASTER 6-band  →  18-band stack", open=False):
                        a18_file = gr.File(label="6-band ASTER stack (.tif)",
                                           file_types=[".tif", ".tiff"],
                                           elem_classes="compact-file-upload")
                        a18_px = gr.Dropdown(label="Output pixel size (m)",
                                             choices=["15", "30"], value="15", interactive=True)
                        gr.Markdown("`aster_18band_stack.tif` — 18 bands · AST_ALOH · AST_CLAY · AST_FERRIC · AST_HYDALT · AST_SILICA …")
                        a18_btn = gr.Button("⚙️  Build ASTER 18-Band Stack", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.HTML('<div class="out-label">◈  Build Log & Downloads</div>')
                    sb_log = gr.Textbox(label="Build Log", lines=14, interactive=False,
                                        elem_classes="log-box")
                    sb_out     = gr.File(label="📥 Download full_features_stack.tif")
                    sb_ast_out = gr.File(label="📥 Download aster_swir_stack.tif", visible=True)
                    ast_log    = gr.Textbox(label="ASTER Build Log", lines=6, interactive=False,
                                            elem_classes="log-box")
                    ast_out    = gr.File(label="📥 Download aster_stack.tif")
                    a18_log    = gr.Textbox(label="ASTER 18-Band Log", lines=6, interactive=False,
                                            elem_classes="log-box")
                    a18_out    = gr.File(label="📥 Download aster_18band_stack.tif")

            sb_btn.click(fn=run_converter,
                         inputs=[sb_10, sb_20, sb_dem, sb_aster],
                         outputs=[sb_out, sb_ast_out, sb_log])
            ast_btn.click(fn=_sb_run_aster,
                          inputs=[ast_files, ast_px],
                          outputs=[ast_out, ast_log])
            a18_btn.click(fn=_sb_run_aster_18band,
                          inputs=[a18_file, a18_px],
                          outputs=[a18_out, a18_log])

        # ══ TAB 9: 3D VIEW ════════════════════════════════════════════
        with gr.Tab("🏔️ 3D View", elem_id="tab-3d"):
            with gr.Row(elem_classes="v3d-control-bar"):
                with gr.Column(scale=1, min_width=180):
                    v3_mode = gr.Dropdown(
                        choices=["Iron Oxide", "Probability (RF Model)",
                                 "ASTER Hydrothermal", "ASTER Silica", "Custom Index"],
                        value="ASTER Hydrothermal", label="Colour layer", container=False)
                with gr.Column(scale=1, min_width=120):
                    v3_idx  = gr.Textbox(label="Custom index", placeholder="AST_AlOH…",
                                        value="IO", container=False)
                with gr.Column(scale=1, min_width=120):
                    v3_cmap = gr.Dropdown(choices=CMAP_OPTIONS, value="RdYlGn_r",
                                         label="Colour map", container=False)
                with gr.Column(scale=1, min_width=120):
                    v3_stride = gr.Slider(1, 20, 6, step=1, label="Stride", container=False)
                with gr.Column(scale=1, min_width=120):
                    v3_exag = gr.Slider(1, 50, 8, step=1, label="V-exag ×", container=False)
                with gr.Column(scale=1, min_width=160):
                    v3_btn = gr.Button("▶  Launch 3D", variant="primary")

            v3_html = gr.HTML(
                value="""<div style='width:100%;height:620px;background:#040508;
                   display:flex;align-items:center;justify-content:center;
                   border-radius:8px;color:#5a6485;font-family:"Share Tech Mono",monospace;
                   font-size:.85rem;letter-spacing:1.5px;flex-direction:column;gap:12px'>
                   <div style='font-size:3rem;opacity:.12'>🏔️</div>
                   Upload S2 + ASTER, then click ▶ Launch 3D
                   </div>""",
                label="", show_label=False, elem_id="v3d-wrapper")
            v3_stats = gr.Markdown("", visible=False)

            v3_btn.click(
                fn=lambda s2, aster, mode, idx, cmap, stride, exag:
                    make_3d_visualization(s2, aster, mode, idx, cmap, stride, exag),
                inputs=[shared_s2_state, shared_aster_state,
                        v3_mode, v3_idx, v3_cmap, v3_stride, v3_exag],
                outputs=[v3_html, v3_stats])

        # ══ TAB 11: SPATIAL STATISTICS ════════════════════════════════
        with gr.Tab("📐 Spatial"):
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">📐 Spatial Autocorrelation & Hotspot Analysis</div>')
                    with gr.Accordion("⚙️  Parameters", open=True):
                        sp_index = gr.Dropdown(
                            choices=["Iron Oxide (IO)", "Clay Minerals (CM)",
                                     "Gossan (GS)", "Ferrous Iron (FI)",
                                     "NDVI", "Elevation", "Slope",
                                     "AST_HydAlt", "AST_AlOH"],
                            value="Iron Oxide (IO)",
                            label="Index to analyse")
                        sp_kernel = gr.Slider(3, 31, 9, step=2,
                                             label="Kernel window (px, odd number)")
                    with gr.Accordion("ℹ️  Method Guide", open=False):
                        gr.Markdown("""
**Moran's I** — global spatial autocorrelation:
- `I > 0`: clustered pattern (typical for mineralisation)
- `I ≈ 0`: random
- `I < 0`: dispersed / negative association

**Getis-Ord Gi*** — local hotspot Z-score:
- `|z| > 1.96` → statistically significant (p < 0.05)
- Red zones = high-value clusters (ore target hotspots)
- Blue zones = low-value clusters (background)

**LISA Classification:**
- High-High → core alteration / mineralisation hotspot
- Uncertain high → fringe / margin zone
- Cold spot → background country rock
- Not significant → statistically neutral

**Optimal Thresholds** — data-driven probability cutoffs:
- Youden-J: maximises sensitivity + specificity simultaneously
- F1-optimal: maximises F1 score (balanced precision/recall)
- p80/p90/p95: percentile-based (no label required)
""")
                    sp_btn = gr.Button("▶  Run Spatial Analysis", variant="primary", size="lg")

                with gr.Column(scale=3, elem_classes="stats-area"):
                    gr.HTML('<div class="out-label">◈  Output — Spatial Statistics</div>')
                    sp_map  = gr.Image(label="", type="filepath", show_label=False)
                    sp_stat = gr.Markdown("*Upload Sentinel-2 (and optionally ASTER) then run.*")

            sp_btn.click(
                fn=lambda s2, aster, idx, kern:
                    run_spatial_analysis(s2, aster, idx, kern),
                inputs=[shared_s2_state, shared_aster_state, sp_index, sp_kernel],
                outputs=[sp_map, sp_stat])

        # ══ TAB 12: SETTINGS ══════════════════════════════════════════
        with gr.Tab("📁 Files"):
            import os as _os_fm, shutil as _shutil_fm, datetime as _dt_fm

            # ── Helper: human-readable file size ─────────────────────
            def _fmt_size(nb):
                if nb < 1024:        return f"{nb} B"
                if nb < 1024**2:     return f"{nb/1024:.1f} KB"
                if nb < 1024**3:     return f"{nb/1024**2:.1f} MB"
                return f"{nb/1024**3:.2f} GB"

            # ── Helper: file-type icon ────────────────────────────────
            def _ftype_icon(name):
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                return {
                    "tif":"🗺️","tiff":"🗺️","geotiff":"🗺️",
                    "png":"🖼️","jpg":"🖼️","jpeg":"🖼️",
                    "pkl":"🤖","pkl2":"🤖",
                    "md":"📝","txt":"📝","csv":"📊",
                    "py":"🐍","json":"📋","zip":"🗜️",
                }.get(ext, "📄")

            # ── Helper: scan a directory recursively (depth 1 = flat) ─
            def _scan_dir(path, recursive=False):
                rows = []
                try:
                    entries = sorted(_os_fm.scandir(path),
                                     key=lambda e: (not e.is_dir(), e.name.lower()))
                    for e in entries:
                        st = e.stat()
                        mtime = _dt_fm.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                        if e.is_dir():
                            # Count children
                            try:
                                n_ch = len(list(_os_fm.scandir(e.path)))
                            except Exception:
                                n_ch = 0
                            rows.append(("dir", e.path, e.name, n_ch, mtime))
                            if recursive:
                                for sub in _scan_dir(e.path, recursive=True):
                                    rows.append(sub)
                        else:
                            rows.append(("file", e.path, e.name, st.st_size, mtime))
                except PermissionError:
                    pass
                return rows

            # ── HTML renderer for the file browser panel ──────────────
            def _render_browser(path, filter_ext="", search_q="", recursive=False):
                path = path.strip() or OUTPUT_ROOT
                if not _os_fm.path.isdir(path):
                    return f"<p style='color:#f06060'>❌ Not a directory: {path}</p>", path

                rows = _scan_dir(path, recursive=recursive)

                # filter
                if filter_ext and filter_ext != "All":
                    rows = [r for r in rows
                            if r[0] == "dir" or r[2].lower().endswith(f".{filter_ext.lower().lstrip('.')}")]
                if search_q.strip():
                    q = search_q.strip().lower()
                    rows = [r for r in rows if q in r[2].lower()]

                # stats
                n_files = sum(1 for r in rows if r[0] == "file")
                n_dirs  = sum(1 for r in rows if r[0] == "dir")
                total_b = sum(r[3] for r in rows if r[0] == "file")

                # ── Build HTML table ──────────────────────────────────
                html = f"""
<style>
.fm-wrap{{font-family:'DejaVu Sans',monospace;color:#c8d4ec;font-size:13px}}
.fm-toolbar{{display:flex;align-items:center;gap:10px;padding:8px 12px;
             background:#080b12;border-bottom:1px solid #1e2840;flex-wrap:wrap}}
.fm-stat{{color:#5a6e96;font-size:11px}}
.fm-stat b{{color:#d4962a}}
.fm-table{{width:100%;border-collapse:collapse;margin:0}}
.fm-table th{{background:#0c1020;color:#5a6e96;font-size:11px;text-transform:uppercase;
              letter-spacing:.8px;padding:6px 10px;text-align:left;
              border-bottom:1px solid #1e2840;position:sticky;top:0}}
.fm-table td{{padding:5px 10px;border-bottom:1px solid #0e1525;vertical-align:middle;
              white-space:nowrap}}
.fm-table tr:hover td{{background:#0f1828}}
.fm-dir  td{{color:#a07ef5}}
.fm-file td{{color:#c8d4ec}}
.fm-size{{color:#5a6e96;text-align:right;font-variant-numeric:tabular-nums}}
.fm-mtime{{color:#3d5080;font-size:11px}}
.fm-path{{font-size:11px;color:#3d5080;max-width:340px;overflow:hidden;
          text-overflow:ellipsis;white-space:nowrap}}
.fm-head{{background:#040710;padding:8px 12px;border-bottom:1px solid #1e2840;
          display:flex;align-items:center;gap:8px}}
.fm-breadcrumb{{color:#d4962a;font-size:12px;font-weight:bold;
                overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}}
.fm-empty{{padding:24px;text-align:center;color:#3d5080}}
</style>
<div class="fm-wrap">
<div class="fm-head">
  <span style="color:#d4962a;font-size:16px">📂</span>
  <span class="fm-breadcrumb">{path}</span>
</div>
<div class="fm-toolbar">
  <span class="fm-stat">📁 <b>{n_dirs}</b> folders</span>
  <span class="fm-stat">📄 <b>{n_files}</b> files</span>
  <span class="fm-stat">💾 <b>{_fmt_size(total_b)}</b> total</span>
</div>
<div style="overflow-x:auto;overflow-y:auto;max-height:480px">
<table class="fm-table">
<thead><tr>
  <th style="width:24px"></th>
  <th>Name</th>
  <th>Full Path</th>
  <th style="text-align:right">Size</th>
  <th>Modified</th>
</tr></thead>
<tbody>"""
                # parent folder nav row
                parent = _os_fm.path.dirname(path)
                if parent != path:
                    html += f"""<tr class="fm-dir">
  <td>⬆️</td>
  <td colspan="2"><span style="color:#5a6e96">.. (parent: {parent})</span></td>
  <td class="fm-size">—</td><td class="fm-mtime">—</td></tr>"""

                if not rows:
                    html += f'<tr><td colspan="5" class="fm-empty">(empty)</td></tr>'

                for rtype, rpath, rname, rsize, rmtime in rows:
                    if rtype == "dir":
                        indent = "&nbsp;&nbsp;" * rpath.replace(path, "").count(_os_fm.sep) if recursive else ""
                        html += f"""<tr class="fm-dir">
  <td>📁</td>
  <td>{indent}<b>{rname}/</b></td>
  <td class="fm-path">{rpath}</td>
  <td class="fm-size">{rsize} items</td>
  <td class="fm-mtime">{rmtime}</td></tr>"""
                    else:
                        indent = "&nbsp;&nbsp;" * (rpath.replace(path, "").count(_os_fm.sep) - 1) if recursive else ""
                        html += f"""<tr class="fm-file">
  <td>{_ftype_icon(rname)}</td>
  <td>{indent}{rname}</td>
  <td class="fm-path" title="{rpath}">{rpath}</td>
  <td class="fm-size">{_fmt_size(rsize)}</td>
  <td class="fm-mtime">{rmtime}</td></tr>"""

                html += "</tbody></table></div></div>"
                return html, path

            # ── Folder stats for the summary panel ───────────────────
            def _folder_stats(path):
                """Return a markdown summary of all output sub-folders."""
                lines = ["### 📊 Output Folder Summary\n",
                         "| Folder | Files | Total Size |",
                         "|--------|-------|-----------|"]
                for key, subdir in sorted(_OUTPUT_SUBDIRS.items()):
                    if not _os_fm.path.isdir(subdir):
                        lines.append(f"| **{key}** | — | — |")
                        continue
                    files = [e for e in _os_fm.scandir(subdir) if e.is_file()]
                    total = sum(e.stat().st_size for e in files)
                    lines.append(f"| **{key}** | {len(files)} | {_fmt_size(total)} |")
                grand = sum(
                    e.stat().st_size
                    for subdir in _OUTPUT_SUBDIRS.values()
                    if _os_fm.path.isdir(subdir)
                    for e in _os_fm.scandir(subdir)
                    if e.is_file()
                )
                lines.append(f"\n**Grand total:** {_fmt_size(grand)}")
                return "\n".join(lines)

            # ─────────────────────────────────────────────────────────
            # UI LAYOUT
            # ─────────────────────────────────────────────────────────

            # Row 1 — Quick-access shortcut buttons + stats
            with gr.Row(elem_classes="tab-content-row"):
                gr.HTML('<div class="gis-panel-header">📁 Gold Outputs File Manager</div>')

            with gr.Row():
                fm_btn_root    = gr.Button("📂 All Outputs",    size="sm")
                fm_btn_maps    = gr.Button("🗺️ Maps",           size="sm")
                fm_btn_geotiff = gr.Button("🌍 GeoTIFFs",       size="sm")
                fm_btn_stack   = gr.Button("🛰️ Stacks",          size="sm")
                fm_btn_models  = gr.Button("🤖 Models",          size="sm")
                fm_btn_shap    = gr.Button("🔍 SHAP",            size="sm")
                fm_btn_reports = gr.Button("📝 Reports",         size="sm")
                fm_btn_hdf     = gr.Button("💽 HDF Convert",     size="sm")
                fm_btn_refresh = gr.Button("🔄 Refresh",         size="sm", variant="primary")

            # Row 2 — Browser + controls side by side
            with gr.Row(equal_height=False, elem_classes="tab-content-row"):

                # ── LEFT: controls ────────────────────────────────────
                with gr.Column(scale=1, min_width=300):

                    with gr.Accordion("🔎 Browse & Filter", open=True):
                        fm_path_box = gr.Textbox(
                            label="Current path",
                            value=OUTPUT_ROOT,
                            placeholder="Absolute folder path…",
                            interactive=True)
                        with gr.Row():
                            fm_filter_dd = gr.Dropdown(
                                label="Type filter",
                                choices=["All", "tif", "tiff", "png", "pkl", "md", "txt", "csv", "py"],
                                value="All", interactive=True)
                            fm_recursive = gr.Checkbox(label="Recursive", value=False)
                        fm_search = gr.Textbox(
                            label="Search filename (contains…)",
                            placeholder="e.g. gold_prob or 2025",
                            interactive=True)
                        fm_browse_btn = gr.Button("🔍 Browse", variant="primary")

                    with gr.Accordion("📥 Download a File", open=True):
                        fm_dl_path = gr.Textbox(
                            label="Paste full file path here",
                            placeholder="Copy path from table above…",
                            interactive=True)
                        fm_dl_btn  = gr.Button("📥 Prepare Download", variant="primary")
                        fm_dl_out  = gr.File(label="⬇️ Click to download", interactive=False)

                    with gr.Accordion("⬆️ Upload Files", open=False):
                        fm_up_files = gr.File(
                            label="Select files to upload",
                            file_count="multiple")
                        fm_up_dest  = gr.Dropdown(
                            label="Destination folder",
                            choices=[OUTPUT_ROOT] + list(_OUTPUT_SUBDIRS.values()),
                            value=OUTPUT_ROOT,
                            interactive=True,
                            allow_custom_value=True)
                        fm_up_btn   = gr.Button("⬆️ Upload", variant="primary")

                    with gr.Accordion("✏️ Rename / Copy / Move", open=False):
                        fm_src_path  = gr.Textbox(label="Source path", placeholder="Full path to file…", interactive=True)
                        fm_dst_path  = gr.Textbox(label="Destination path / new name", placeholder="Full path…", interactive=True)
                        with gr.Row():
                            fm_rename_btn = gr.Button("✏️ Rename/Move")
                            fm_copy_btn   = gr.Button("📋 Copy")

                    with gr.Accordion("🗑️ Delete", open=False):
                        fm_del_path = gr.Textbox(
                            label="File path to delete",
                            placeholder="Paste full path — this is permanent!",
                            interactive=True)
                        fm_del_btn  = gr.Button("🗑️ Delete File", variant="stop")

                    with gr.Accordion("🧹 Clean Old Outputs", open=False):
                        fm_clean_age = gr.Slider(1, 30, 7, step=1,
                            label="Delete outputs older than N days")
                        fm_clean_type = gr.Dropdown(
                            label="File type to clean",
                            choices=["All outputs", "maps (PNG)", "geotiff (TIF)",
                                     "shap (PNG)", "reports (MD)", "hdf_convert (TIF)"],
                            value="maps (PNG)")
                        fm_clean_btn = gr.Button("🧹 Clean Old Files", variant="stop")

                # ── RIGHT: file browser HTML + status ─────────────────
                with gr.Column(scale=3):
                    fm_browser_html = gr.HTML(label="", value="<div style='color:#5a6e96;padding:20px'>Click a folder shortcut or Browse to explore files.</div>")
                    fm_status_md    = gr.Markdown(value=_folder_stats(OUTPUT_ROOT))
                    fm_action_log   = gr.Textbox(label="Action Log", lines=4, interactive=False, elem_classes="log-box")

            # ─────────────────────────────────────────────────────────
            # BACKEND FUNCTIONS
            # ─────────────────────────────────────────────────────────

            def _fm_browse_fn(path, filt, search, recursive):
                html, resolved = _render_browser(path, filt, search, recursive)
                return html, resolved, _folder_stats(OUTPUT_ROOT)

            def _fm_dl_fn(file_path):
                file_path = (file_path or "").strip()
                if not file_path:
                    return None, "❌ Paste a file path first."
                if not _os_fm.path.isfile(file_path):
                    return None, f"❌ Not a file: {file_path}"
                return file_path, f"✅ Ready: {_os_fm.path.basename(file_path)}  ({_fmt_size(_os_fm.path.getsize(file_path))})"

            def _fm_upload_fn(files, dest):
                if not files:
                    return "❌ No files selected.", _folder_stats(OUTPUT_ROOT)
                dest = (dest or OUTPUT_ROOT).strip()
                _os_fm.makedirs(dest, exist_ok=True)
                saved, failed = [], []
                for f in files:
                    src = f if isinstance(f, str) else f.name
                    dst = _os_fm.path.join(dest, _os_fm.path.basename(src))
                    try:
                        _shutil_fm.copy(src, dst)
                        saved.append(f"  ✅ {_os_fm.path.basename(src)} → {dst}")
                    except Exception as _ue:
                        failed.append(f"  ❌ {_os_fm.path.basename(src)}: {_ue}")
                log = "\n".join(saved + failed)
                return log, _folder_stats(OUTPUT_ROOT)

            def _fm_rename_fn(src, dst):
                src = (src or "").strip(); dst = (dst or "").strip()
                if not src or not dst: return "❌ Fill both Source and Destination paths.", _folder_stats(OUTPUT_ROOT)
                if not _os_fm.path.exists(src): return f"❌ Source not found: {src}", _folder_stats(OUTPUT_ROOT)
                try:
                    _os_fm.makedirs(_os_fm.path.dirname(_os_fm.path.abspath(dst)), exist_ok=True)
                    _shutil_fm.move(src, dst)
                    return f"✅ Moved/Renamed:\n  {src}\n→ {dst}", _folder_stats(OUTPUT_ROOT)
                except Exception as _re:
                    return f"❌ {_re}", _folder_stats(OUTPUT_ROOT)

            def _fm_copy_fn(src, dst):
                src = (src or "").strip(); dst = (dst or "").strip()
                if not src or not dst: return "❌ Fill both Source and Destination paths.", _folder_stats(OUTPUT_ROOT)
                if not _os_fm.path.isfile(src): return f"❌ Source not a file: {src}", _folder_stats(OUTPUT_ROOT)
                try:
                    _os_fm.makedirs(_os_fm.path.dirname(_os_fm.path.abspath(dst)), exist_ok=True)
                    _shutil_fm.copy2(src, dst)
                    return f"✅ Copied:\n  {src}\n→ {dst}", _folder_stats(OUTPUT_ROOT)
                except Exception as _ce:
                    return f"❌ {_ce}", _folder_stats(OUTPUT_ROOT)

            def _fm_delete_fn(file_path):
                file_path = (file_path or "").strip()
                if not file_path: return "❌ Paste a file path first.", _folder_stats(OUTPUT_ROOT)
                if not _os_fm.path.isfile(file_path):
                    return f"❌ Not a file: {file_path}", _folder_stats(OUTPUT_ROOT)
                try:
                    _os_fm.remove(file_path)
                    return f"🗑️ Deleted: {file_path}", _folder_stats(OUTPUT_ROOT)
                except Exception as _de:
                    return f"❌ {_de}", _folder_stats(OUTPUT_ROOT)

            def _fm_clean_fn(age_days, clean_type):
                """Delete output files older than age_days days in the chosen category."""
                import time as _time_fm
                cutoff = _time_fm.time() - age_days * 86400
                folder_map = {
                    "All outputs":          [p for p in _OUTPUT_SUBDIRS.values()],
                    "maps (PNG)":           [_OUTPUT_SUBDIRS["maps"]],
                    "geotiff (TIF)":        [_OUTPUT_SUBDIRS["geotiff"]],
                    "shap (PNG)":           [_OUTPUT_SUBDIRS["shap"]],
                    "reports (MD)":         [_OUTPUT_SUBDIRS["reports"]],
                    "hdf_convert (TIF)":    [_OUTPUT_SUBDIRS["hdf_convert"]],
                }
                folders = folder_map.get(clean_type, [])
                removed, skipped, errors = [], [], []
                for folder in folders:
                    if not _os_fm.path.isdir(folder): continue
                    for e in _os_fm.scandir(folder):
                        if not e.is_file(): continue
                        if e.stat().st_mtime < cutoff:
                            try:
                                _os_fm.remove(e.path)
                                removed.append(f"  🗑️ {e.name}")
                            except Exception as _ce:
                                errors.append(f"  ❌ {e.name}: {_ce}")
                        else:
                            skipped.append(e.name)
                log_parts = [f"✅ Cleaned {len(removed)} files (>{age_days}d old) from '{clean_type}'"]
                if removed: log_parts += removed[:20]
                if len(removed) > 20: log_parts.append(f"  … and {len(removed)-20} more")
                if errors: log_parts += errors
                log_parts.append(f"  (kept {len(skipped)} recent files)")
                return "\n".join(log_parts), _folder_stats(OUTPUT_ROOT)

            # ─────────────────────────────────────────────────────────
            # WIRE EVENTS
            # ─────────────────────────────────────────────────────────

            # Quick-folder shortcut buttons
            _fm_folder_btns = [
                (fm_btn_root,    OUTPUT_ROOT),
                (fm_btn_maps,    _OUTPUT_SUBDIRS["maps"]),
                (fm_btn_geotiff, _OUTPUT_SUBDIRS["geotiff"]),
                (fm_btn_stack,   _OUTPUT_SUBDIRS["stack"]),
                (fm_btn_models,  _OUTPUT_SUBDIRS["models"]),
                (fm_btn_shap,    _OUTPUT_SUBDIRS["shap"]),
                (fm_btn_reports, _OUTPUT_SUBDIRS["reports"]),
                (fm_btn_hdf,     _OUTPUT_SUBDIRS["hdf_convert"]),
            ]
            for _fbtn, _fdir in _fm_folder_btns:
                _fbtn.click(
                    fn=lambda p=_fdir: _fm_browse_fn(p, "All", "", False),
                    inputs=None,
                    outputs=[fm_browser_html, fm_path_box, fm_status_md])

            fm_btn_refresh.click(
                fn=lambda p, fi, sq, rc: _fm_browse_fn(p, fi, sq, rc),
                inputs=[fm_path_box, fm_filter_dd, fm_search, fm_recursive],
                outputs=[fm_browser_html, fm_path_box, fm_status_md])

            fm_browse_btn.click(
                fn=_fm_browse_fn,
                inputs=[fm_path_box, fm_filter_dd, fm_search, fm_recursive],
                outputs=[fm_browser_html, fm_path_box, fm_status_md])

            # Live filter/search on dropdown or text change
            fm_filter_dd.change(
                fn=_fm_browse_fn,
                inputs=[fm_path_box, fm_filter_dd, fm_search, fm_recursive],
                outputs=[fm_browser_html, fm_path_box, fm_status_md])

            fm_dl_btn.click(
                fn=_fm_dl_fn,
                inputs=[fm_dl_path],
                outputs=[fm_dl_out, fm_action_log])

            fm_up_btn.click(
                fn=_fm_upload_fn,
                inputs=[fm_up_files, fm_up_dest],
                outputs=[fm_action_log, fm_status_md])

            fm_rename_btn.click(
                fn=_fm_rename_fn,
                inputs=[fm_src_path, fm_dst_path],
                outputs=[fm_action_log, fm_status_md])

            fm_copy_btn.click(
                fn=_fm_copy_fn,
                inputs=[fm_src_path, fm_dst_path],
                outputs=[fm_action_log, fm_status_md])

            fm_del_btn.click(
                fn=_fm_delete_fn,
                inputs=[fm_del_path],
                outputs=[fm_action_log, fm_status_md])

            fm_clean_btn.click(
                fn=_fm_clean_fn,
                inputs=[fm_clean_age, fm_clean_type],
                outputs=[fm_action_log, fm_status_md])

        with gr.Tab("⚙️ Settings"):
            settings_css_out = gr.HTML("", visible=True)

            # ── Row 1: Appearance ─────────────────────────────────────
            with gr.Row(elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML('<div class="gis-panel-header">🎨  Colour Scheme</div>')
                    gr.Markdown("**Quick Presets**")
                    with gr.Row():
                        preset_dark   = gr.Button("⬛ Dark Gold",    size="sm", variant="secondary")
                        preset_space  = gr.Button("🌌 Deep Space",   size="sm", variant="secondary")
                        preset_earth  = gr.Button("🌍 Earth Tones",  size="sm", variant="secondary")
                        preset_reset  = gr.Button("↺ Reset",         size="sm", variant="secondary")
                    gr.Markdown("**Custom Colours**")
                    s_bg     = gr.ColorPicker(label="Background",    value="#0a0c10")
                    s_surf   = gr.ColorPicker(label="Surface",       value="#0e1016")
                    s_gold   = gr.ColorPicker(label="Accent / Gold", value="#e8b84b")
                    s_text   = gr.ColorPicker(label="Text Primary",  value="#d8dce8")
                    s_border = gr.ColorPicker(label="Border",        value="#1e2232")
                    s_green  = gr.ColorPicker(label="Accent Green",  value="#3bcc7a")

                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">📐  Layout & Typography</div>')
                    s_fontsize = gr.Slider(10, 20, 14, step=1,  label="Base Font Size (px)")
                    s_radius   = gr.Slider(0,  16,  6, step=1,  label="Border Radius (px)")
                    s_headerh  = gr.Slider(28, 72, 42, step=2,  label="Tab Header Height (px)")
                    s_no_anim  = gr.Checkbox(label="Disable animations (performance mode)", value=False)
                    s_compact  = gr.Checkbox(label="Compact panel spacing", value=False)
                    gr.Markdown("---")
                    gr.HTML('<div class="gis-panel-header">🖥️  Display</div>')
                    s_map_dpi   = gr.Slider(72, 300, 140, step=10,
                                            label="Map render DPI (higher = slower)")
                    s_map_size  = gr.Dropdown(
                        choices=["Small (16×12)", "Medium (20×16)", "Large (26×20)", "XL (32×24)"],
                        value="Large (26×20)", label="Default map figure size")
                    s_interp    = gr.Dropdown(
                        choices=["bilinear", "nearest", "bicubic", "lanczos"],
                        value="bilinear", label="Image interpolation method")

                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">🗺️  Cartographic Defaults</div>')
                    s_default_cmap   = gr.Dropdown(choices=CMAP_OPTIONS, value="RdYlGn_r",
                                                    label="Default probability colourmap")
                    s_default_coord  = gr.Dropdown(choices=COORD_CHOICES, value="WGS84",
                                                    label="Default coordinate grid")
                    s_default_thresh = gr.Slider(0.30, 0.90, 0.60, step=0.05,
                                                 label="Default detection threshold")
                    s_show_scalebar  = gr.Checkbox(label="Always show scale bar",   value=True)
                    s_show_north     = gr.Checkbox(label="Always show north arrow", value=True)
                    s_show_stamp     = gr.Checkbox(label="Show map stamp / watermark", value=True)
                    s_coord_n        = gr.Slider(4, 10, 6, step=1,
                                                 label="Coordinate grid lines (count)")

            # ── Row 2: Analysis & Processing ──────────────────────────
            gr.HTML('<div class="gis-divider"></div>')
            with gr.Row(elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">🔬  Analysis Defaults</div>')
                    s_pixel_size    = gr.Slider(5, 60, 20, step=5,
                                                label="Pixel size (metres) — used in area calculations")
                    s_norm_lo       = gr.Slider(0, 10, 2, step=0.5,
                                                label="Stretch low percentile (%)")
                    s_norm_hi       = gr.Slider(90, 100, 98, step=0.5,
                                                label="Stretch high percentile (%)")
                    s_morans_sample = gr.Slider(1000, 50000, 10000, step=1000,
                                                label="Moran's I sample size (pixels)")
                    s_hotspot_ker   = gr.Slider(3, 31, 9, step=2,
                                                label="Default Getis-Ord kernel (px)")
                    s_hyd_thresh    = gr.Slider(50, 95, 75, step=5,
                                                label="Default hydrothermal threshold %ile")

                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">🤖  Model & Training Defaults</div>')
                    s_default_algo  = gr.Dropdown(
                        choices=MODEL_ALGO_CHOICES,
                        value="Ensemble RF+GB",
                        label="Default training algorithm")
                    s_default_trees = gr.Slider(50, 1000, 300, step=50,
                                                label="Default number of trees")
                    s_default_mode  = gr.Dropdown(
                        choices=COMPARISON_MODES,
                        value="Hybrid Sentinel-2 + ASTER",
                        label="Default data source mode")
                    s_auto_calibrate = gr.Checkbox(label="Auto-enable probability calibration", value=True)
                    s_auto_shap      = gr.Checkbox(
                        label=f"Auto-enable SHAP {'(installed ✅)' if _HAS_SHAP else '(not installed ⚠)'}",
                        value=_HAS_SHAP)
                    s_auto_optuna    = gr.Checkbox(
                        label=f"Auto-enable Optuna HPO {'(installed ✅)' if _HAS_OPTUNA else '(not installed ⚠)'}",
                        value=False)
                    s_cv_folds       = gr.Slider(3, 10, 5, step=1,
                                                 label="Cross-validation folds")

                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">💾  Export & Paths</div>')
                    s_model_path    = gr.Textbox(label="Model save path (.pkl)",
                                                 value=MODEL_PATH, placeholder=MODEL_PATH)
                    s_export_compress = gr.Dropdown(
                        choices=["lzw", "deflate", "zstd", "none"],
                        value="lzw", label="GeoTIFF compression codec")
                    s_export_dtype   = gr.Dropdown(
                        choices=["float32", "float16", "uint8"],
                        value="float32", label="GeoTIFF output dtype")
                    s_auto_geotiff   = gr.Checkbox(label="Auto-export GeoTIFF after detection", value=True)
                    s_auto_report    = gr.Checkbox(label="Auto-generate summary report", value=True)
                    s_report_detail  = gr.Dropdown(
                        choices=["Brief", "Standard", "Full"],
                        value="Full", label="Report verbosity level")

            # ── Row 3: Performance & Cache ─────────────────────────────
            gr.HTML('<div class="gis-divider"></div>')
            with gr.Row(elem_classes="tab-content-row"):
                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">⚡  Performance & Cache</div>')
                    s_cache_slots   = gr.Slider(2, 16, _CACHE_MAX, step=1,
                                                label="Raster LRU cache slots")
                    s_resample_algo = gr.Dropdown(
                        choices=["lanczos", "bilinear", "nearest", "cubic"],
                        value="lanczos", label="ASTER resampling algorithm")
                    s_moran_threads = gr.Checkbox(label="Multi-thread Moran's I (faster, ~10% error)",
                                                  value=False)
                    s_simplify_path = gr.Slider(0.1, 1.0, 0.5, step=0.1,
                                                label="Matplotlib path simplification")
                    s_chunk_size    = gr.Slider(5000, 100000, 20000, step=5000,
                                                label="Pixel chunk size for large rasters")
                    s_max_threads   = gr.Slider(1, 16, 4, step=1,
                                                label="Max Gradio server threads")

                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">🌐  ASTER Processing</div>')
                    s_aster_bands_in = gr.CheckboxGroup(
                        choices=["B4", "B5", "B6", "B7", "B8", "B9"],
                        value=["B4", "B5", "B6", "B7", "B8", "B9"],
                        label="ASTER bands to load by default")
                    s_aster_indices  = gr.CheckboxGroup(
                        choices=list(ASTER_INDICES.keys()),
                        value=list(ASTER_INDICES.keys()),
                        label="ASTER indices to compute")
                    s_aster_nan_fill = gr.Dropdown(
                        choices=["local_mean", "zero", "global_mean"],
                        value="local_mean", label="ASTER NaN fill strategy")

                with gr.Column(scale=1, min_width=260):
                    gr.HTML('<div class="gis-panel-header">🔧  Apply & Info</div>')

                    with gr.Row():
                        apply_btn  = gr.Button("✔  Apply Settings", variant="primary",  size="lg")
                        export_btn = gr.Button("📥 Export Config",  variant="secondary", size="sm")

                    settings_status = gr.Markdown("*Adjust settings above and click Apply.*")

                    gr.HTML('<div class="gis-divider"></div>')
                    gr.HTML('<div class="gis-panel-header">ℹ️  About</div>')
                    gr.Markdown(f"""
**{VERSION}** — AI-Powered Hydrothermal Alteration & Mineral Prospectivity Mapping

**Author:** Nader Safwat Ayed Hanna  
**Institution:** Beni-Suef University, Faculty of Earth Sciences

| Component | Status |
|---|---|
| XGBoost | {"✅ installed" if _HAS_XGB else "⚠ not installed"} |
| LightGBM | {"✅ installed" if _HAS_LGB else "⚠ not installed"} |
| SHAP | {"✅ installed" if _HAS_SHAP else "⚠ not installed"} |
| Optuna | {"✅ installed" if _HAS_OPTUNA else "⚠ not installed"} |

**Features:** up to 38 (S2+ASTER) + 20 engineered = **58 total**  
**Indices:** 10 Sentinel-2 + 12 ASTER SWIR = **22 geological indices**
""")

            # ── Settings logic ─────────────────────────────────────────
            PRESETS = {
                "dark-gold":   dict(bg="#030508", surface="#080b12", gold="#d4962a",
                                    text="#c8d4ec", border="#161d2e", green="#0fd4a0"),
                "deep-space":  dict(bg="#02030a", surface="#060810", gold="#7c6fff",
                                    text="#c4cbdc", border="#10142a", green="#00ddd0"),
                "earth-tones": dict(bg="#080601", surface="#100e06", gold="#c49030",
                                    text="#d8cdb8", border="#201808", green="#5aaa30"),
                "default":     dict(bg="#030508", surface="#080b12", gold="#d4962a",
                                    text="#c8d4ec", border="#161d2e", green="#0fd4a0"),
            }

            def _build_css(bg, surf, gold, text, border, green, fontsize, radius,
                           headerh, no_anim, compact):
                tr = "none" if no_anim else "all 0.18s cubic-bezier(.4,0,.2,1)"
                pad = "6px" if compact else "12px"

                def hex_to_rgba(hex_color, alpha):
                    """Convert #rrggbb to rgba(r,g,b,alpha)."""
                    try:
                        h = hex_color.lstrip("#")
                        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                        return f"rgba({r},{g},{b},{alpha})"
                    except Exception:
                        return hex_color

                return f"""<style>
:root {{
  --bg:{bg}; --surface:{surf}; --surface2:{hex_to_rgba(surf, 0.87)}; --surface3:{hex_to_rgba(surf, 0.73)};
  --gold:{gold}; --gold-dim:{hex_to_rgba(gold, 0.67)}; --gold-bright:{gold};
  --text:{text}; --text-dim:#6070a0; --text-muted:#5a6485;
  --border:{border}; --border-mid:{hex_to_rgba(border, 0.87)};
  --accent-green:{green}; --accent-cyan:#00e5cc;
  --header-h:{headerh}px; --radius:{radius}px; --radius-sm:{max(2,radius-2)}px;
  --transition:{tr};
}}
html,body,.gradio-container{{font-size:{fontsize}px!important}}
.tab-content-row,.gr-padded{{padding:{pad}!important}}
</style>"""

            def _apply_settings(bg, surf, gold, text, border, green,
                                 fs, ra, hh, na, compact,
                                 dpi, fig_size, interp,
                                 def_cmap, def_coord, def_thresh,
                                 show_sb, show_na, show_stamp, coord_n,
                                 px_size, norm_lo, norm_hi,
                                 moran_samp, hk, hyd_thr,
                                 def_algo, def_trees, def_mode,
                                 auto_cal, auto_shap, auto_optuna, cv_folds,
                                 mdl_path, compress, dtype, auto_tif, auto_rep, rep_detail,
                                 cache, resamp, moran_mt, simp, chunk, threads,
                                 aster_bands, aster_idx, nan_fill):
                global PIXEL_SIZE_M, _CACHE_MAX
                # Apply runtime-mutable globals
                PIXEL_SIZE_M = int(px_size)
                _CACHE_MAX   = int(cache)
                plt.rcParams["path.simplify_threshold"] = float(simp)

                css = _build_css(bg, surf, gold, text, border, green, fs, ra, hh, na, compact)
                status = f"""### ✔ Settings Applied

| Setting | Value |
|---|---|
| Pixel size | **{int(px_size)} m** |
| Map DPI | {int(dpi)} |
| Figure size | {fig_size} |
| Cache slots | {int(cache)} |
| Resampling | {resamp} |
| Compression | {compress} |
| Default threshold | {def_thresh:.0%} |
| Default colourmap | {def_cmap} |
| ASTER bands active | {', '.join(aster_bands) if aster_bands else '—'} |
| ASTER indices active | {len(aster_idx)} / {len(ASTER_INDICES)} |
"""
                return css, gr.update(value=status, visible=True)

            def _preset_fn(name, fs, ra, hh, na, compact,
                           dpi, fig_size, interp,
                           def_cmap, def_coord, def_thresh,
                           show_sb, show_na, show_stamp, coord_n,
                           px_size, norm_lo, norm_hi,
                           moran_samp, hk, hyd_thr,
                           def_algo, def_trees, def_mode,
                           auto_cal, auto_shap, auto_optuna, cv_folds,
                           mdl_path, compress, dtype, auto_tif, auto_rep, rep_detail,
                           cache, resamp, moran_mt, simp, chunk, threads,
                           aster_bands, aster_idx, nan_fill):
                p = PRESETS.get(name, PRESETS["default"])
                css = _build_css(p["bg"], p["surface"], p["gold"], p["text"],
                                 p["border"], p["green"], fs, ra, hh, na, compact)
                return (css,
                        p["bg"], p["surface"], p["gold"],
                        p["text"], p["border"], p["green"])

            # All settings inputs list (used for apply + live updates)
            _all_settings = [
                s_bg, s_surf, s_gold, s_text, s_border, s_green,
                s_fontsize, s_radius, s_headerh, s_no_anim, s_compact,
                s_map_dpi, s_map_size, s_interp,
                s_default_cmap, s_default_coord, s_default_thresh,
                s_show_scalebar, s_show_north, s_show_stamp, s_coord_n,
                s_pixel_size, s_norm_lo, s_norm_hi,
                s_morans_sample, s_hotspot_ker, s_hyd_thresh,
                s_default_algo, s_default_trees, s_default_mode,
                s_auto_calibrate, s_auto_shap, s_auto_optuna, s_cv_folds,
                s_model_path, s_export_compress, s_export_dtype,
                s_auto_geotiff, s_auto_report, s_report_detail,
                s_cache_slots, s_resample_algo, s_moran_threads,
                s_simplify_path, s_chunk_size, s_max_threads,
                s_aster_bands_in, s_aster_indices, s_aster_nan_fill,
            ]
            _css_inputs = [
                s_bg, s_surf, s_gold, s_text, s_border, s_green,
                s_fontsize, s_radius, s_headerh, s_no_anim, s_compact,
            ]

            # Live CSS preview on colour/layout changes only
            for _inp in _css_inputs:
                _inp.change(
                    fn=lambda *a: _build_css(*a),
                    inputs=_css_inputs,
                    outputs=[settings_css_out])

            # Apply button applies everything
            apply_btn.click(
                fn=_apply_settings,
                inputs=_all_settings,
                outputs=[settings_css_out, settings_status])

            # Export config as markdown summary
            def _export_config(*args):
                labels = [
                    "Background", "Surface", "Gold", "Text", "Border", "Green",
                    "Font size", "Radius", "Header height", "No animations", "Compact",
                    "Map DPI", "Figure size", "Interpolation",
                    "Default colourmap", "Default coord", "Default threshold",
                    "Show scalebar", "Show north arrow", "Show stamp", "Coord grid n",
                    "Pixel size (m)", "Norm lo%", "Norm hi%",
                    "Moran sample", "Hotspot kernel", "Hydro threshold",
                    "Default algo", "Default trees", "Default mode",
                    "Auto calibrate", "Auto SHAP", "Auto Optuna", "CV folds",
                    "Model path", "Compress", "Dtype", "Auto GeoTIFF",
                    "Auto report", "Report detail",
                    "Cache slots", "Resample", "Moran threads",
                    "Path simplify", "Chunk size", "Max threads",
                    "ASTER bands", "ASTER indices", "NaN fill",
                ]
                rows = "\n".join(f"| {l} | `{v}` |" for l, v in zip(labels, args))
                return f"### ⚙️ Exported Configuration\n\n| Setting | Value |\n|---|---|\n{rows}"

            export_btn.click(fn=_export_config, inputs=_all_settings,
                             outputs=[settings_status])

            # Preset buttons
            # Preset buttons — pass only non-colour settings (colours are overridden by preset)
            _preset_out = [settings_css_out, s_bg, s_surf, s_gold, s_text, s_border, s_green]
            _preset_in = _all_settings[6:]  # skip s_bg/surf/gold/text/border/green

            preset_dark .click(fn=lambda *a: _preset_fn("dark-gold",   *a),
                               inputs=_preset_in, outputs=_preset_out)
            preset_space.click(fn=lambda *a: _preset_fn("deep-space",  *a),
                               inputs=_preset_in, outputs=_preset_out)
            preset_earth.click(fn=lambda *a: _preset_fn("earth-tones", *a),
                               inputs=_preset_in, outputs=_preset_out)
            preset_reset.click(fn=lambda *a: _preset_fn("default",     *a),
                               inputs=_preset_in, outputs=_preset_out)

        # ── Tab 13: Literature Review ──────────────────────────────────
        if _HAS_LIT:
            build_literature_tab()
        else:
            with gr.Tab("📚  Literature Review"):
                gr.Markdown(
                    "### ⚠️ الملف `literature_review_tab.py` مش موجود\n\n"
                    "حط الملف في نفس مجلد التطبيق وأعد التشغيل."
                )

    # ── STATUS BAR ────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="gis-status-strip">
      <div class="gis-status-item">
        <span class="gis-status-dot {'off' if not model_bundle else ''}"></span>
        <span style="color:var(--text-muted)">MODEL</span>
        <span style="color:var(--gold);font-weight:600">{model_status[:40]}</span>
      </div>
      <span style="color:var(--border-mid)">│</span>
      <div class="gis-status-item">
        <span style="color:var(--text-muted)">SENTINEL-2</span>
        <span>18 bands + 20 engineered</span>
      </div>
      <span style="color:var(--border-mid)">│</span>
      <div class="gis-status-item">
        <span style="color:var(--text-muted)">ASTER SWIR</span>
        <span style="color:var(--accent-violet)">B4-B9 + 12 geological indices</span>
      </div>
      <span style="color:var(--border-mid)">│</span>
      <div class="gis-status-item">
        <span style="color:var(--text-muted)">FEATURES</span>
        <span>Up to 38+20 = 58 total</span>
      </div>
      <span style="color:var(--border-mid)">│</span>
      <div class="gis-status-item">
        <span style="color:var(--text-muted)">ENGINE</span>
        <span style="color:var(--accent-cyan)">RF{'✚XGB' if _HAS_XGB else ''}{'✚LGBM' if _HAS_LGB else ''} + CalibratedCV</span>
      </div>
      <span style="color:var(--border-mid)">│</span>
      <div class="gis-status-item">
        <span style="color:var(--text-muted)">SPATIAL</span>
        <span style="color:var(--accent-green)">Moran·I + Gi* + LISA</span>
      </div>
      <span style="color:var(--border-mid)">│</span>
      <div class="gis-status-item" style="margin-left:auto">
        <span style="color:var(--text-muted)">SYSTEM</span>
        <span>Gold Prospectivity {VERSION} · SAM · PCA · MCDA · Structure · Eastern Desert</span>
      </div>
    </div>""")

if __name__ == "__main__":
    print("\n" + "═"*68)
    print(f"  GOLD PROSPECTIVITY SYSTEM  {VERSION}")
    print(f"  AI-Powered Hydrothermal Alteration & Mineral Prospectivity Mapping")
    print("  Beni-Suef University — Faculty of Earth Sciences — Nader Safwat Ayed Hanna")
    print("═"*68)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Status : {model_status}")
    print(f"  Stack  : ASTER SWIR B4-B9 · 12 geological indices · CalibratedClassifierCV")
    print(f"  Engine : RF{'+ XGBoost' if _HAS_XGB else ' (XGB: not installed)'} "
          f"{'+ LightGBM' if _HAS_LGB else '(LGBM: not installed)'}")
    print(f"  SHAP   : {'✅ installed' if _HAS_SHAP else '⚠ not installed (pip install shap)'}")
    print(f"  Optuna : {'✅ installed' if _HAS_OPTUNA else '⚠ not installed (pip install optuna)'}")
    print(f"  Spatial: Moran's I + Getis-Ord Gi* + LISA hotspot clustering")
    print(f"  Cache  : {_CACHE_MAX}-slot thread-safe raster LRU cache")
    print(f"  Features: 18 S2 + 20 engineered + 6 ASTER bands + 12 ASTER indices = 56 max")
    print("  URL    : http://localhost:7860")
    print("═"*68+"\n")
    app.queue(max_size=8).launch(
        server_name="0.0.0.0", server_port=7860,
        share=False, show_error=True,
        max_threads=4,
    )
