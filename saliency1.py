#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path("/home/uctzyro/Scratch/data/attributions")      
OUT_DIR = Path("/home/uctzyro/Scratch/data/group_all")     
PATTERN = "*_gradcam.nii.gz"                             

def normalize01(v):
    v = v.astype(np.float32)
    mn, mx = float(v.min()), float(v.max())
    return (v - mn) / (mx - mn + 1e-6)

def save_previews_all_axes(base, heat, pfx, title, alpha=0.35):
    """
    Save 9 PNGs total: 3 evenly spaced slices from each axis (Z, Y, X).
    Files will be named like: {pfx}_z{idx}.png, {pfx}_y{idx}.png, {pfx}_x{idx}.png
    """
    base01 = normalize01(base)
    axes = [
        (0, 'z', 'Axial (Z)'), #not exact due to the original files not in standard direction
        (1, 'y', 'Coronal (Y)'),
        (2, 'x', 'Sagittal (X)'),
    ]
    for axis, code, plane in axes:
        D = heat.shape[axis]
        idxs = [max(0, min(D-1, int(D*0.25))), D//2, max(0, min(D-1, int(D*0.75)))]
        #here shows slicers with 0.25, 0.5 and 0.75, it changed to show 0.15 and 0.85 slicer for some saliency maps
        for i in idxs:
            if axis == 0:
                b = base01[i, :, :]
                h = heat[i, :, :]
            elif axis == 1:
                b = base01[:, i, :]
                h = heat[:, i, :]
            else:  # axis == 2
                b = base01[:, :, i]
                h = heat[:, :, i]

            plt.figure(figsize=(7,7), dpi=220)
            plt.title(f"{title} | {plane} | {code}={i}")
            plt.imshow(b, origin="lower", cmap="gray", interpolation="none")
            plt.imshow(h, origin="lower", alpha=alpha, interpolation="none")
            thr = np.quantile(h, 0.90)
            plt.contour(h >= thr, levels=[0.5], linewidths=1.0)
            plt.axis("off")
            plt.savefig(f"{pfx}_{code}{i}.png", bbox_inches="tight")
            plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC_DIR.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files matching {PATTERN} under {SRC_DIR}")

    stack = []
    ref_img = None
    for f in files:
        img = nib.load(str(f))
        data = img.get_fdata(dtype=np.float32)
        stack.append(normalize01(data))           # per-case normalize to avoid scale bias
        if ref_img is None:
            ref_img = img

    group = np.mean(np.stack(stack, axis=0), axis=0).astype(np.float32)  # [D,H,W]

    #save nifti
    out_nii = OUT_DIR / "group_mean.nii.gz"
    nib.save(nib.Nifti1Image(group, ref_img.affine, ref_img.header), str(out_nii))
    print(f"[ok] wrote {out_nii}")

    #PNG previews 
    #PNG previews
    save_previews_all_axes(group, group, str(OUT_DIR / "group_mean"), "Group ALL Grad-CAM", alpha=0.35)
    print(f"[ok] wrote 9 PNG previews (3 per axis) in {OUT_DIR}")


if __name__ == "__main__":
    main()
