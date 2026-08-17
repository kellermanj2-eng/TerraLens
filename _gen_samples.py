"""Generate synthetic sample satellite image pair for demo/testing."""
import numpy as np
import cv2
import os

rng = np.random.default_rng(42)
H, W = 480, 640

def make_base():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Muddy river / water (bottom-left)
    img[300:, :220] = [100, 120, 140]
    noise = rng.integers(0, 15, (180, 220, 3))
    img[300:, :220] = (img[300:, :220].astype(int) + noise).clip(0, 255).astype(np.uint8)
    # Dark forest (top half) — low brightness so burn stands out clearly
    img[:240, :] = [25, 70, 30]
    noise = rng.integers(0, 18, (240, W, 3))
    img[:240, :] = (img[:240, :].astype(int) + noise).clip(0, 255).astype(np.uint8)
    # Agriculture (middle)
    img[240:320, :] = [45, 115, 50]
    noise = rng.integers(0, 20, (80, W, 3))
    img[240:320, :] = (img[240:320, :].astype(int) + noise).clip(0, 255).astype(np.uint8)
    # Bare soil (bottom-right)
    img[300:, 220:] = [100, 130, 150]
    noise = rng.integers(0, 20, (180, 420, 3))
    img[300:, 220:] = (img[300:, 220:].astype(int) + noise).clip(0, 255).astype(np.uint8)
    # Roads
    cv2.line(img, (0, 310), (W, 295), (70, 80, 90), 3)
    cv2.line(img, (320, 0), (305, H), (70, 80, 90), 3)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    return img

before = make_base()

# After: bright ash/burn scar (high luminance) over dark forest (low luminance)
# This maximises grayscale difference → clear change detection
after = before.copy()

# Primary burn ellipse centred at (470, 110)
burn = np.zeros((H, W), dtype=np.uint8)
cv2.ellipse(burn, (470, 110), (140, 88), 10, 0, 360, 255, -1)
# Secondary burn patch (rectangular)
burn[35:180, 345:590] = 255

burn_mask = burn == 255
# Bright grey-white ash colour (BGR ~180,175,170) — very high contrast to forest
ash_base = np.array([165, 170, 175], dtype=np.uint8)
noise = rng.integers(0, 25, (int(burn_mask.sum()), 3)).astype(np.uint8)
after[burn_mask] = (ash_base.astype(int) + noise).clip(0, 255).astype(np.uint8)

# Fringe: slightly lighter than original forest but not as bright as core burn
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
dilated = cv2.dilate(burn, kernel)
fringe_mask = (dilated == 255) & ~burn_mask
after[fringe_mask] = (after[fringe_mask].astype(float) * 0.5 +
                       np.array([120, 125, 130]) * 0.5).clip(0, 255).astype(np.uint8)

# Very light blur to remove pixel-perfect edges
after = cv2.GaussianBlur(after, (3, 3), 1)

os.makedirs("data", exist_ok=True)
cv2.imwrite("data/sample_before.png", before)
cv2.imwrite("data/sample_after.png", after)

# Verify
b = cv2.imread("data/sample_before.png")
a = cv2.imread("data/sample_after.png")
print(f"before: {b.shape}, after: {a.shape}")
diff = cv2.absdiff(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                   cv2.cvtColor(a, cv2.COLOR_BGR2GRAY))
blurred = cv2.GaussianBlur(diff, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
pct = thresh.sum() / 255 / thresh.size * 100
print(f"approx change at threshold=40: {pct:.1f}%  (target 15-35%)")
print(f"diff max={diff.max()}, mean-in-burn={diff[50:190,350:585].mean():.1f}")
