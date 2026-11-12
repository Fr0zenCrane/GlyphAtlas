"""Deterministic, CPU-only 48/64/80 Atlas planning shared by train/inference.

This module does not infer character size from a word polygon. The global glyph
renderer must provide final-resolution, per-character/shaping-cluster ink extents.
It does not change the legacy 64px encoder or automatically enable new training.

Caller contract (same code path for training, inference and preview)::

    ink = measure_global_ink(final_cluster_layers, baseline_degrees=angle)
    unit = GlyphUnit(key, text, region_id, width_at64, ink, resolved_font_path)
    plan = plan_atlas(units, budget=256)  # omit budget for automatic bucket
    rasters = [render_unit(u, a) for u, a in zip(units, plan.allocations)]

The native draw_glyph2 currently returns only the merged raster. It must expose
its actual per-character/shaping-cluster layers before this policy can be enabled
there. Do not synthesize extents by polygon-width/text-length. Then adapt the VAE
packing, owner repetitions, position IDs and padding attention to actual unit
shapes:48/64/80 square units produce9/16/25 packed tokens. Legacy AtlasEncoder
still hardcodes16 tokens per64x64 tile; sending these arrays into it is NOT valid.
The default64 raster helper is not claimed byte-identical to the historical
renderer. Real encoder/model quality and throughput remain untested.
"""
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Sequence


LEVELS = (48, 64, 80)


@dataclass(frozen=True)
class InkExtent:
    """Ink width/height in final global-glyph pixels, measured along text axes.

    For joining scripts these must refer to shaping clusters, not detached code
    points. Measurements come after all scene transforms, resizing and clipping.
    """
    width: float
    height: float

    def __post_init__(self):
        if not all(isfinite(x) and x >= 0 for x in (self.width, self.height)):
            raise ValueError('Global glyph ink extents must be finite and nonnegative')

    @property
    def short_side(self):
        return min(self.width, self.height)


def measure_global_ink(cluster_masks, *, baseline_degrees: float = 0.0,
                       ink_threshold: float = 0.0) -> tuple[InkExtent, ...]:
    """Measure per-cluster ink in the FINAL global raster, not polygon boxes.

    Each mask is a separate character/shaping-cluster layer on the final global
    canvas, already transformed/resized/clipped exactly like that canvas. Do not
    pass an entire word mask as a substitute for its character/cluster layers.
    Project ink pixels along the baseline so rotated axis-aligned boxes cannot
    make tiny glyphs appear large. An empty layer yields0, hence highest need.
    """
    import numpy as np
    if not isfinite(baseline_degrees) or not isfinite(ink_threshold):
        raise ValueError('Measurement parameters must be finite')
    angle = np.deg2rad(baseline_degrees)
    c, s = np.cos(angle), np.sin(angle)
    pixel_extent = abs(c) + abs(s)
    extents = []
    for mask in cluster_masks:
        mask = np.asarray(mask)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise ValueError('Expected a finite2D alpha/ink layer at final resolution')
        y, x = np.nonzero(mask > ink_threshold)
        if not len(x):
            extents.append(InkExtent(0.0, 0.0))
            continue
        u = c*x + s*y
        v = -s*x + c*y
        extents.append(InkExtent(float(np.ptp(u)+pixel_extent),
                                 float(np.ptp(v)+pixel_extent)))
    if not extents:
        raise ValueError('No per-character/cluster raster layers supplied')
    return tuple(extents)


@dataclass(frozen=True)
class GlyphUnit:
    key: str
    text: str
    owner: int
    width64: int
    global_ink: tuple[InkExtent, ...]
    font_path: str

    def __post_init__(self):
        if not self.key or not self.text or self.text.isspace():
            raise ValueError('A unit needs a stable key and non-whitespace text')
        if self.owner < 0 or self.width64 < 64 or self.width64 % 64:
            raise ValueError('Expected nonnegative owner and original 64px tile width')
        if not self.global_ink:
            raise ValueError('Missing measured per-character/cluster global glyph ink; no bbox proxy allowed')
        if not self.font_path:
            raise ValueError('Use the same resolved font as the global glyph renderer')

    @property
    def global_min_px(self):
        return min(x.short_side for x in self.global_ink)

    def shape(self, height: int):
        if height not in LEVELS:
            raise ValueError('Only one step below/above64 is supported:48/64/80')
        content_width = self.width64 * height // 64
        return 16 * ceil(content_width / 16), height

    def cost(self, height: int):
        w, h = self.shape(height)
        return (w // 16) * (h // 16)


@dataclass(frozen=True)
class Allocation:
    key: str
    owner: int
    height: int
    width: int
    tokens: int
    global_min_px: float
    reason: str


@dataclass(frozen=True)
class BudgetPlan:
    budget: int
    baseline_tokens: int
    used_tokens: int
    allocations: tuple[Allocation, ...]
    downgrade_order: tuple[str, ...]
    upgrade_order: tuple[str, ...]

    @property
    def padding_tokens(self):
        return self.budget - self.used_tokens


class BudgetTooSmall(ValueError):
    def __init__(self, budget, minimum):
        self.budget, self.minimum = budget, minimum
        super().__init__(f'Budget{budget} cannot retain all units at48px; need at least{minimum}. '
                         'Use a larger bucket; no silent32px fallback or text truncation.')


def plan_atlas(units: Sequence[GlyphUnit], *, budget: int | None = None,
               buckets: Sequence[int] = (128, 256, 512),
               clear_global_px: float = 80.0, fixed64_keys: Sequence[str] = ()) -> BudgetPlan:
    """Start at64; lower global-clear units, fit budget, then promote small glyphs.

    Ranking is per-unit minimum measured global-character short side. Smallest
    ranks first for upgrades; largest ranks first for downgrades. Equal sizes
    break ties by stable key, independent of input enumeration. Whole word units
    share one size. The80px threshold is a configurable heuristic, not a proven
    perceptual threshold. Every unit is retained; no upgrades above80/down below48.
    """
    units = tuple(units)
    if len({u.key for u in units}) != len(units):
        raise ValueError('Unit keys must be unique')
    if not isfinite(clear_global_px) or clear_global_px <= 0:
        raise ValueError('clear_global_px must be positive and finite')
    protected = set(fixed64_keys)
    if not protected.issubset({u.key for u in units}):raise ValueError('Unknown fixed64 key')
    minimum = sum(u.cost(64 if u.key in protected else 48) for u in units)
    if budget is None:
        if not buckets or any(type(b) is not int or b <= 0 for b in buckets):
            raise ValueError('Buckets must be positive integers')
        ordered = sorted(set(buckets))
        budget = next((b for b in ordered if b >= minimum), None)
        if budget is None:
            raise BudgetTooSmall(ordered[-1], minimum)
    if type(budget) is not int or budget <= 0:
        raise ValueError('Budget must be a positive integer')
    if minimum > budget:
        raise BudgetTooSmall(budget, minimum)
    down = sorted((u for u in units if u.key not in protected), key=lambda u: (-u.global_min_px, u.key))
    up = sorted((u for u in units if u.key not in protected), key=lambda u: (u.global_min_px, u.key))
    height = {u.key: 64 for u in units}
    reason = {u.key: 'punctuation_fixed64' if u.key in protected else 'default64' for u in units}
    baseline = sum(u.cost(64) for u in units)
    used = baseline
    # Global >=80 means the supplementary Atlas is least necessary. Release
    # its redundant detail even when baseline fits, to fund small global glyphs.
    for u in down:
        if u.global_min_px >= clear_global_px:
            height[u.key] = 48
            reason[u.key] = 'global_clear_downgrade'
            used -= u.cost(64) - u.cost(48)
    for u in down:
        if used <= budget:
            break
        if height[u.key] == 64:
            height[u.key] = 48
            reason[u.key] = 'budget_downgrade'
            used -= u.cost(64) - u.cost(48)
    # Most underserved glyph gets first choice of80, then64 if80 cannot fit.
    # No low-priority upgrades solely to fill padding; >=80-global units stay48.
    for u in up:
        if u.global_min_px >= clear_global_px:
            continue
        current = height[u.key]
        for target in (80, 64):
            if target <= current:
                continue
            extra = u.cost(target) - u.cost(current)
            if used + extra <= budget:
                used += extra
                height[u.key] = target
                reason[u.key] = 'global_small_upgrade' if target == 80 else 'priority_restore64'
                break
    allocations = tuple(Allocation(u.key, u.owner, height[u.key], u.shape(height[u.key])[0],
                                   u.cost(height[u.key]), u.global_min_px, reason[u.key])
                        for u in units)
    assert sum(a.tokens for a in allocations) == used <= budget
    return BudgetPlan(budget, baseline, used, allocations,
                      tuple(u.key for u in down), tuple(u.key for u in up))


def render_unit(unit: GlyphUnit, allocation: Allocation):
    """Re-rasterize at48/64/80 from font outlines; never upsample a64px bitmap.

    Uses the selected whole-unit width/aspect and7/8 ink margins. Returns a uint8
    grayscale canvas; the caller must VAE-encode at this actual shape, not resize
    back to64. Layout padding on the right is counted in the token budget.
    """
    from PIL import Image, ImageFont, ImageDraw
    import numpy as np
    if (allocation.key != unit.key or allocation.owner != unit.owner
            or (allocation.width, allocation.height) != unit.shape(allocation.height)
            or allocation.tokens != unit.cost(allocation.height)):
        raise ValueError('Allocation does not match this unit')
    w, h = allocation.width, allocation.height
    content_w = unit.width64 * h // 64
    max_w, max_h = max(1, content_w - h / 8), h * 7 / 8
    base = ImageFont.truetype(unit.font_path, 56)
    l, t, r, b = base.getbbox(unit.text)
    scale = min(max_w / max(1, r-l), max_h / max(1, b-t))
    size = max(1, int(56 * scale))
    font = base.font_variant(size=size)
    l, t, r, b = font.getbbox(unit.text)
    while size > 1 and (r-l > max_w or b-t > max_h):
        size -= 1
        font = base.font_variant(size=size)
        l, t, r, b = font.getbbox(unit.text)
    canvas = Image.new('L', (w, h), 0)
    ImageDraw.Draw(canvas).text(((content_w-(r-l))/2-l, (h-(b-t))/2-t),
                                unit.text, font=font, fill=255)
    return np.asarray(canvas).copy()
