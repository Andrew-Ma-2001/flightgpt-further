def fixed_variance_gaussian_accuracy_reward(completions, solution, **kwargs):
    """
    固定方差版高斯点奖励 —— 仅关注目标点定位精度。
    R_total = nu * R_point   （不含 coverage，不含自适应方差）
    """
    import re
    import numpy as np
    import ast
    import os
    from datetime import datetime

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    # ===== 配置参数（可按需调整） =====
    SIGMA = 20.0          # 固定像素级标准差 (σ_x = σ_y = 20)
    NU = 1.0              # 点奖励权重（通常保持 1.0 即可）

    for content, sol in zip(contents, solution):
        # 解析 GT
        try:
            sol = ast.literal_eval(sol)
        except Exception:
            rewards.append(0.0)
            continue

        gt_target = sol['target_position']   # [x, y]

        # 解析模型输出
        target_matches = re.findall(
            r'"target_location"\s*:\s*\[(\d+),\s*(\d+)\]', content
        )
        if not target_matches:
            rewards.append(0.0)
            continue

        pred_target = list(map(int, target_matches[0]))
        pred_x, pred_y = pred_target
        gt_x, gt_y = gt_target

        # ===== 固定方差高斯点奖励 =====
        # σ_x = σ_y = SIGMA
        r_point = np.exp(
            -0.5 * (((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) / (SIGMA ** 2))
        )

        reward = NU * r_point
        rewards.append(reward)

        # 调试日志（可选）
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            if log_path:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path, "a") as f:
                    f.write(
                        f"--- {current_time} ---\n"
                        f"pred=({pred_x},{pred_y}), gt=({gt_x},{gt_y}), "
                        f"dist={np.hypot(pred_x - gt_x, pred_y - gt_y):.1f}px, "
                        f"reward={reward:.4f}\n"
                    )

    return rewards


def gaussian_accuracy_reward(completions, solution, **kwargs):
    """
    GUI-G² style Gaussian reward for CityNav.
 
    R_total = v * R_point + gamma * R_coverage + R_format
    """
    import ast
    import os
    import re
    from datetime import datetime

    import numpy as np

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    # Hyperparameters
    alpha = 0.5    # adaptive variance scaling factor (paper default)
    nu = 1.0       # weight for point reward
    gamma = 1.0    # weight for coverage reward
    for content, sol in zip(contents, solution):
        try:
            sol = ast.literal_eval(sol)
        except Exception as e:
            print(e)
            rewards.append(0.0)
            continue
        gt_target = sol['target_position']          # [x, y]
        gt_landmark_bboxes = sol['landmark_bbox']   # [[x1,y1,x2,y2], ...]
        # ========== Parse model output ==========
        target_matches = re.findall(
            r'"target_location"\s*:\s*\[(\d+),\s*(\d+)\]', content
        )
        bbox_matches = re.findall(
            r'"landmark_bbox"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', content
        )
        if not target_matches:
            rewards.append(0.0)
            continue
        pred_target = list(map(int, target_matches[0]))
        pred_bbox = list(map(int, bbox_matches[0])) if bbox_matches else None
        # ========== R_point: Gaussian Point Reward ==========
        # Find the closest GT landmark bbox to compute adaptive sigma
        # σx = α * bbox_width, σy = α * bbox_height
        best_r_point = 0.0
        for gt_bbox in gt_landmark_bboxes:
            bbox_w = gt_bbox[2] - gt_bbox[0]
            bbox_h = gt_bbox[3] - gt_bbox[1]
            # Adaptive variance based on landmark size
            sigma_x = max(alpha * bbox_w, 1.0)   # avoid division by zero
            sigma_y = max(alpha * bbox_h, 1.0)
            # GT center (use target_position as the point GT)
            gt_x, gt_y = gt_target
            pred_x, pred_y = pred_target
            # Gaussian point reward
            r_point = np.exp(
                -0.5 * (
                    ((pred_x - gt_x) ** 2) / (sigma_x ** 2) +
                    ((pred_y - gt_y) ** 2) / (sigma_y ** 2)
                )
            )
            best_r_point = max(best_r_point, r_point)
        # ========== R_coverage: Gaussian Coverage Reward ==========
        best_r_coverage = 0.0
        if pred_bbox is not None:
            for gt_bbox in gt_landmark_bboxes:
                r_cov = _bhattacharyya_reward(pred_bbox, gt_bbox)
                best_r_coverage = max(best_r_coverage, r_cov)
        # ========== Total reward ==========
        reward = nu * best_r_point + gamma * best_r_coverage
        rewards.append(reward)
        # Debug logging
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a") as f:
                f.write(f"--- {current_time} ---\n")
                f.write(f"R_point={best_r_point:.4f}, R_coverage={best_r_coverage:.4f}, "
                        f"total={reward:.4f}\n")
                f.write(f"pred_target={pred_target}, gt_target={gt_target}\n")
                f.write(f"pred_bbox={pred_bbox}\n\n")
    return rewards

def _bhattacharyya_reward(pred_bbox, gt_bbox):
    """
    Gaussian Coverage Reward using Bhattacharyya coefficient.
    Models each bbox as a 2D Gaussian:
      mean = bbox center, covariance = diag(σx², σy²)
      where σx = 0.5 * width, σy = 0.5 * height
    Bhattacharyya coefficient:
      BC = exp(-1/8 * (μp-μg)^T Σ^{-1} (μp-μg) - 1/2 * ln(det(Σ)/sqrt(det(Σp)*det(  Σg))))
      where Σ = (Σp + Σg) / 2
    """
    import numpy as np

    alpha = 0.5
  
    # Predicted bbox → Gaussian
    pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2.0
    pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2.0
    pred_sx = max(alpha * (pred_bbox[2] - pred_bbox[0]), 1.0)
    pred_sy = max(alpha * (pred_bbox[3] - pred_bbox[1]), 1.0)
  
    # GT bbox → Gaussian
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2.0
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2.0
    gt_sx = max(alpha * (gt_bbox[2] - gt_bbox[0]), 1.0)
    gt_sy = max(alpha * (gt_bbox[3] - gt_bbox[1]), 1.0)
  
    # Averaged covariance: Σ = (Σp + Σg) / 2
    # Since both are diagonal: Σ_avg_x = (σpx² + σgx²) / 2
    avg_var_x = (pred_sx ** 2 + gt_sx ** 2) / 2.0
    avg_var_y = (pred_sy ** 2 + gt_sy ** 2) / 2.0
  
    # Term 1: Mahalanobis distance
    mahal = (1.0 / 8.0) * (
        ((pred_cx - gt_cx) ** 2) / avg_var_x +
        ((pred_cy - gt_cy) ** 2) / avg_var_y
    )
  
    # Term 2: Covariance normalization
    # det(Σ_avg) = avg_var_x * avg_var_y  (diagonal)
    # det(Σp) = pred_sx² * pred_sy²
    # det(Σg) = gt_sx² * gt_sy²
    det_avg = avg_var_x * avg_var_y
    det_pred = (pred_sx ** 2) * (pred_sy ** 2)
    det_gt = (gt_sx ** 2) * (gt_sy ** 2)
  
    log_term = 0.5 * np.log(det_avg / np.sqrt(det_pred * det_gt))
  
    # Bhattacharyya coefficient
    r_coverage = np.exp(-mahal - log_term)
  
    return r_coverage


def gaussian_accuracy_r_point(
    pred_x,
    pred_y,
    gt_x: float,
    gt_y: float,
    bbox_w: float,
    bbox_h: float,
    alpha: float = 0.5,
):
    import numpy as np

    pred_x = np.asarray(pred_x)
    pred_y = np.asarray(pred_y)

    sigma_x = max(alpha * bbox_w, 1.0)
    sigma_y = max(alpha * bbox_h, 1.0)
    return np.exp(-0.5 * (((pred_x - gt_x) ** 2) / (sigma_x**2) + ((pred_y - gt_y) ** 2) / (sigma_y**2)))


def _bbox_from_center(cx: float, cy: float, w: float, h: float):
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return [x1, y1, x2, y2]


def plot_gaussian_accuracy_reward_heatmaps(
    map_w: int = 1000,
    map_h: int = 1000,
    bbox_frac: float | None = 0.1,
    bbox_w_frac: float | None = None,
    bbox_h_frac: float | None = None,
    gt_point_x: int = 0,
    gt_point_y: int = 0,
    point_radius: int = 250,
    point_step: int = 5,
    bbox_step: int = 10,
    alpha: float = 0.5,
    nu: float = 1.0,
    gamma: float = 1.0,
    # When visualizing "total over point", decide how pred_bbox moves relative to pred_point.
    couple_point_bbox: bool = False,
    bbox_offset_x: float = 0.0,
    bbox_offset_y: float = 0.0,
    fixed_pred_bbox_cx: float | None = None,
    fixed_pred_bbox_cy: float | None = None,
    out_path: str | None = None,
):
    """
    Visualize GUI-G² style reward components as heatmaps.

    - Point heatmap: vary pred_target=(x,y), keep GT point fixed.
    - Coverage heatmap: vary pred_bbox center, keep bbox size fixed and GT bbox fixed at:
        (a) map center, (b) map right side.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if map_w <= 0 or map_h <= 0:
        raise ValueError("map_w/map_h must be > 0")
    if bbox_w_frac is None or bbox_h_frac is None:
        if bbox_frac is None:
            raise ValueError("Provide either bbox_frac or both bbox_w_frac and bbox_h_frac")
        if not (0 < bbox_frac < 1):
            raise ValueError("bbox_frac must be in (0, 1)")
        bbox_w_frac = float(bbox_frac)
        bbox_h_frac = float(bbox_frac)
    if not (0 < float(bbox_w_frac) < 1) or not (0 < float(bbox_h_frac) < 1):
        raise ValueError("bbox_w_frac/bbox_h_frac must be in (0, 1)")
    if point_step <= 0 or bbox_step <= 0:
        raise ValueError("point_step and bbox_step must be > 0")

    bbox_w = map_w * float(bbox_w_frac)
    bbox_h = map_h * float(bbox_h_frac)

    # --- R_point heatmap over predicted target ---
    pxs = np.arange(gt_point_x - point_radius, gt_point_x + point_radius + 1, point_step)
    pys = np.arange(gt_point_y - point_radius, gt_point_y + point_radius + 1, point_step)
    PX, PY = np.meshgrid(pxs, pys, indexing="xy")
    R_POINT = gaussian_accuracy_r_point(PX, PY, gt_point_x, gt_point_y, bbox_w=bbox_w, bbox_h=bbox_h, alpha=alpha)

    # --- R_coverage heatmap over predicted bbox center ---
    cxs = np.arange(0, map_w + 1, bbox_step)
    cys = np.arange(0, map_h + 1, bbox_step)
    CX, CY = np.meshgrid(cxs, cys, indexing="xy")

    pred_bbox = np.empty(CX.shape + (4,), dtype=float)
    pred_bbox[..., 0] = CX - bbox_w / 2.0
    pred_bbox[..., 1] = CY - bbox_h / 2.0
    pred_bbox[..., 2] = CX + bbox_w / 2.0
    pred_bbox[..., 3] = CY + bbox_h / 2.0

    gt_bboxes = [
        _bbox_from_center(map_w / 2.0, map_h / 2.0, bbox_w, bbox_h),           # center
        _bbox_from_center(map_w * 0.85, map_h / 2.0, bbox_w, bbox_h),          # right side
    ]
    gt_titles = ["GT bbox @ center", "GT bbox @ right side"]

    def bhatt_grid(gt_bbox):
        # Vectorized wrapper around scalar _bhattacharyya_reward
        # (loops are fine here: grid sizes are moderate with bbox_step>=10)
        Z = np.zeros(CX.shape, dtype=float)
        for iy in range(CX.shape[0]):
            for ix in range(CX.shape[1]):
                Z[iy, ix] = _bhattacharyya_reward(pred_bbox[iy, ix].tolist(), gt_bbox)
        return Z

    R_COV_0 = bhatt_grid(gt_bboxes[0])
    R_COV_1 = bhatt_grid(gt_bboxes[1])

    # If you want a "total" surface for bbox-centers, assume pred_point fixed at GT (so R_point=1)
    # This matches your note "点一直是 0,0" (point doesn't change), while seeing how bbox location affects reward.
    R_TOTAL_0 = nu * 1.0 + gamma * R_COV_0
    R_TOTAL_1 = nu * 1.0 + gamma * R_COV_1

    # --- Optional: total heatmap over predicted point, with bbox either coupled or fixed ---
    R_TOTAL_OVER_POINT_0 = None
    R_TOTAL_OVER_POINT_1 = None
    if couple_point_bbox or (fixed_pred_bbox_cx is not None and fixed_pred_bbox_cy is not None):
        def total_over_point(gt_bbox):
            Z = np.zeros(PX.shape, dtype=float)
            for iy in range(PX.shape[0]):
                for ix in range(PX.shape[1]):
                    pred_x = float(PX[iy, ix])
                    pred_y = float(PY[iy, ix])
                    if couple_point_bbox:
                        cx = pred_x + float(bbox_offset_x)
                        cy = pred_y + float(bbox_offset_y)
                    else:
                        cx = float(fixed_pred_bbox_cx)
                        cy = float(fixed_pred_bbox_cy)
                    pb = _bbox_from_center(cx, cy, bbox_w, bbox_h)
                    r_cov = _bhattacharyya_reward(pb, gt_bbox)
                    r_point = float(R_POINT[iy, ix])
                    Z[iy, ix] = nu * r_point + gamma * r_cov
            return Z

        R_TOTAL_OVER_POINT_0 = total_over_point(gt_bboxes[0])
        R_TOTAL_OVER_POINT_1 = total_over_point(gt_bboxes[1])

    n_rows = 4 if (R_TOTAL_OVER_POINT_0 is not None) else 3
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 + 5 * n_rows))

    # Row 1: R_point (same for both columns, but kept for side-by-side consistency)
    for j in range(2):
        ax = axes[0, j]
        im = ax.imshow(
            R_POINT,
            origin="lower",
            extent=(pxs.min(), pxs.max(), pys.min(), pys.max()),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.scatter([gt_point_x], [gt_point_y], c="red", s=25, marker="x", label="GT point")
        ax.set_title(f"R_point heatmap (alpha={alpha}, bbox={bbox_w:.0f}x{bbox_h:.0f})")
        ax.set_xlabel("pred_x")
        ax.set_ylabel("pred_y")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Helper to draw GT bbox rectangle overlay
    def draw_bbox(ax, bbox, color="red"):
        import matplotlib.patches as patches

        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

    # Row 2: R_coverage
    for j, (gt_bbox, title, Z) in enumerate([(gt_bboxes[0], gt_titles[0], R_COV_0), (gt_bboxes[1], gt_titles[1], R_COV_1)]):
        ax = axes[1, j]
        im = ax.imshow(
            Z,
            origin="lower",
            extent=(cxs.min(), cxs.max(), cys.min(), cys.max()),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        draw_bbox(ax, gt_bbox, color="cyan")
        ax.set_title(f"R_coverage heatmap ({title})")
        ax.set_xlabel("pred_bbox_center_x")
        ax.set_ylabel("pred_bbox_center_y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 3: Total (with point fixed at (0,0) => R_point=1)
    for j, (title, Z) in enumerate([(gt_titles[0], R_TOTAL_0), (gt_titles[1], R_TOTAL_1)]):
        ax = axes[2, j]
        im = ax.imshow(
            Z,
            origin="lower",
            extent=(cxs.min(), cxs.max(), cys.min(), cys.max()),
            cmap="viridis",
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"Total = nu*1 + gamma*R_cov (nu={nu}, gamma={gamma})\n({title})")
        ax.set_xlabel("pred_bbox_center_x")
        ax.set_ylabel("pred_bbox_center_y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 4: Total over predicted point (bbox coupled or fixed)
    if R_TOTAL_OVER_POINT_0 is not None:
        mode_desc = "bbox center = point + offset" if couple_point_bbox else "bbox center fixed"
        for j, (title, Z) in enumerate([(gt_titles[0], R_TOTAL_OVER_POINT_0), (gt_titles[1], R_TOTAL_OVER_POINT_1)]):
            ax = axes[3, j]
            im = ax.imshow(
                Z,
                origin="lower",
                extent=(pxs.min(), pxs.max(), pys.min(), pys.max()),
                cmap="viridis",
                interpolation="nearest",
                aspect="equal",
            )
            ax.scatter([gt_point_x], [gt_point_y], c="red", s=25, marker="x", label="GT point")
            ax.set_title(f"Total over point ({mode_desc})\n({title})")
            ax.set_xlabel("pred_x")
            ax.set_ylabel("pred_y")
            ax.legend(loc="upper right")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Gaussian accuracy reward visualization", y=0.995)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()


def generate_gaussian_accuracy_gallery(
    out_dir: str,
    map_w: int = 1000,
    map_h: int = 1000,
    gt_point_x: int = 0,
    gt_point_y: int = 0,
    alpha: float = 0.5,
    nu: float = 1.0,
    gamma: float = 1.0,
):
    """
    Batch-generate multiple figures to compare:
    - square bbox vs rectangular bbox
    - point+bbox coupled vs decoupled (bbox fixed or with offset)
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    cases = [
        # square bbox (like before)
        dict(name="square_bbox", bbox_w_frac=0.1, bbox_h_frac=0.1),
        # rectangular bbox: wide
        dict(name="rect_wide_bbox", bbox_w_frac=0.16, bbox_h_frac=0.06),
        # rectangular bbox: tall
        dict(name="rect_tall_bbox", bbox_w_frac=0.06, bbox_h_frac=0.16),
    ]

    # 1) Baseline: only bbox-center sweeps (point fixed at GT)
    for c in cases:
        plot_gaussian_accuracy_reward_heatmaps(
            map_w=map_w,
            map_h=map_h,
            bbox_frac=None,
            bbox_w_frac=c["bbox_w_frac"],
            bbox_h_frac=c["bbox_h_frac"],
            gt_point_x=gt_point_x,
            gt_point_y=gt_point_y,
            alpha=alpha,
            nu=nu,
            gamma=gamma,
            out_path=os.path.join(out_dir, f"gaussian_accuracy_{c['name']}_bbox_sweep.png"),
        )

    # 2) Coupled: bbox center follows point (offset 0,0)
    for c in cases:
        plot_gaussian_accuracy_reward_heatmaps(
            map_w=map_w,
            map_h=map_h,
            bbox_frac=None,
            bbox_w_frac=c["bbox_w_frac"],
            bbox_h_frac=c["bbox_h_frac"],
            gt_point_x=gt_point_x,
            gt_point_y=gt_point_y,
            alpha=alpha,
            nu=nu,
            gamma=gamma,
            couple_point_bbox=True,
            bbox_offset_x=0.0,
            bbox_offset_y=0.0,
            out_path=os.path.join(out_dir, f"gaussian_accuracy_{c['name']}_coupled_offset0.png"),
        )

    # 3) Not together: bbox center follows point with offset (e.g. shift to the right)
    for c in cases:
        plot_gaussian_accuracy_reward_heatmaps(
            map_w=map_w,
            map_h=map_h,
            bbox_frac=None,
            bbox_w_frac=c["bbox_w_frac"],
            bbox_h_frac=c["bbox_h_frac"],
            gt_point_x=gt_point_x,
            gt_point_y=gt_point_y,
            alpha=alpha,
            nu=nu,
            gamma=gamma,
            couple_point_bbox=True,
            bbox_offset_x=map_w * 0.3,
            bbox_offset_y=0.0,
            out_path=os.path.join(out_dir, f"gaussian_accuracy_{c['name']}_coupled_offset_right.png"),
        )

    # 4) Decoupled: bbox center fixed (e.g. map center) while point moves
    for c in cases:
        plot_gaussian_accuracy_reward_heatmaps(
            map_w=map_w,
            map_h=map_h,
            bbox_frac=None,
            bbox_w_frac=c["bbox_w_frac"],
            bbox_h_frac=c["bbox_h_frac"],
            gt_point_x=gt_point_x,
            gt_point_y=gt_point_y,
            alpha=alpha,
            nu=nu,
            gamma=gamma,
            couple_point_bbox=False,
            fixed_pred_bbox_cx=map_w / 2.0,
            fixed_pred_bbox_cy=map_h / 2.0,
            out_path=os.path.join(out_dir, f"gaussian_accuracy_{c['name']}_decoupled_bbox_fixed_center.png"),
        )

def gaussian_reward(pred_x, pred_y, gt_x, gt_y, sigma: float = 20.0, nu: float = 1.0):
    import numpy as np

    pred_x = np.asarray(pred_x)
    pred_y = np.asarray(pred_y)
    gt_x = np.asarray(gt_x)
    gt_y = np.asarray(gt_y)

    r_point = np.exp(-0.5 * (((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) / (sigma ** 2)))
    return nu * r_point


def plot_reward_heatmap(
    gt_x: int = 0,
    gt_y: int = 0,
    sigma: float = 20.0,
    nu: float = 1.0,
    radius: int = 200,
    step: int = 2,
    out_path: str | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if step <= 0:
        raise ValueError("step must be > 0")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    xs = np.arange(gt_x - radius, gt_x + radius + 1, step)
    ys = np.arange(gt_y - radius, gt_y + radius + 1, step)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = gaussian_reward(X, Y, gt_x, gt_y, sigma=sigma, nu=nu)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        Z,
        origin="lower",
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        cmap="viridis",
        vmin=0.0,
        vmax=float(nu),
        interpolation="nearest",
        aspect="equal",
    )
    ax.scatter([gt_x], [gt_y], c="red", s=25, marker="x", label="GT")
    ax.set_title(f"Reward heatmap (sigma={sigma}, nu={nu})")
    ax.set_xlabel("pred_x")
    ax.set_ylabel("pred_y")
    ax.legend(loc="upper right")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("reward")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize reward functions as heatmaps.")
    parser.add_argument(
        "--mode",
        type=str,
        default="fixed",
        choices=["fixed", "gaussian-accuracy", "gaussian-accuracy-gallery"],
        help="Which reward visualization to run.",
    )
    parser.add_argument("--gt-x", type=int, default=0)
    parser.add_argument("--gt-y", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=20.0)
    parser.add_argument("--nu", type=float, default=1.0)
    parser.add_argument("--radius", type=int, default=200, help="Half-width (in pixels) around GT to plot.")
    parser.add_argument("--step", type=int, default=2, help="Grid step size (in pixels).")
    parser.add_argument("--out", type=str, default=None, help="If set, save figure to this path instead of showing.")

    # gaussian-accuracy visualization args
    parser.add_argument("--map-w", type=int, default=1000)
    parser.add_argument("--map-h", type=int, default=1000)
    parser.add_argument("--bbox-frac", type=float, default=0.1, help="Shortcut: bbox_w_frac=bbox_h_frac=bbox_frac.")
    parser.add_argument("--bbox-w-frac", type=float, default=None, help="BBox width as fraction of map width.")
    parser.add_argument("--bbox-h-frac", type=float, default=None, help="BBox height as fraction of map height.")
    parser.add_argument("--point-radius", type=int, default=250)
    parser.add_argument("--point-step", type=int, default=5)
    parser.add_argument("--bbox-step", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--couple-point-bbox", action="store_true", help="Make pred_bbox center follow pred_point (+offset).")
    parser.add_argument("--bbox-offset-x", type=float, default=0.0)
    parser.add_argument("--bbox-offset-y", type=float, default=0.0)
    parser.add_argument("--fixed-pred-bbox-cx", type=float, default=None)
    parser.add_argument("--fixed-pred-bbox-cy", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="gaussian_accuracy_gallery")
    args = parser.parse_args()

    if args.mode == "fixed":
        plot_reward_heatmap(
            gt_x=args.gt_x,
            gt_y=args.gt_y,
            sigma=args.sigma,
            nu=args.nu,
            radius=args.radius,
            step=args.step,
            out_path=args.out,
        )
    elif args.mode == "gaussian-accuracy":
        plot_gaussian_accuracy_reward_heatmaps(
            map_w=args.map_w,
            map_h=args.map_h,
            bbox_frac=args.bbox_frac,
            bbox_w_frac=args.bbox_w_frac,
            bbox_h_frac=args.bbox_h_frac,
            gt_point_x=args.gt_x,
            gt_point_y=args.gt_y,
            point_radius=args.point_radius,
            point_step=args.point_step,
            bbox_step=args.bbox_step,
            alpha=args.alpha,
            nu=args.nu,
            gamma=args.gamma,
            couple_point_bbox=args.couple_point_bbox,
            bbox_offset_x=args.bbox_offset_x,
            bbox_offset_y=args.bbox_offset_y,
            fixed_pred_bbox_cx=args.fixed_pred_bbox_cx,
            fixed_pred_bbox_cy=args.fixed_pred_bbox_cy,
            out_path=args.out,
        )
    else:
        generate_gaussian_accuracy_gallery(
            out_dir=args.out_dir,
            map_w=args.map_w,
            map_h=args.map_h,
            gt_point_x=args.gt_x,
            gt_point_y=args.gt_y,
            alpha=args.alpha,
            nu=args.nu,
            gamma=args.gamma,
        )