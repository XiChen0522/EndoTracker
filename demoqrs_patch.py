
# demoqrs_batch.py
import os, sys, json, torch, argparse, numpy as np
from pathlib import Path
from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def delta_avg(pred_last, gt_last, H, W):
    diag = np.sqrt(H**2 + W**2)
    thresholds = np.array([1, 2, 4, 8, 16]) / 256.0 * diag
    errors = np.linalg.norm(pred_last - gt_last, axis=1)
    accs = [(errors < t).mean() for t in thresholds]
    print(f"  δ<1:{accs[0]*100:.1f}% δ<2:{accs[1]*100:.1f}% δ<4:{accs[2]*100:.1f}% δ<8:{accs[3]*100:.1f}% δ<16:{accs[4]*100:.1f}%")
    print(f"  δ_avg: {np.mean(accs)*100:.1f}%")
    return float(np.mean(accs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",   required=True,  help="视频文件夹路径")
    parser.add_argument("--ann_dir",     required=True,  help="标注json文件夹路径")
    parser.add_argument("--output_dir",  default="./saved_videos_batch")
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--max_frames",  type=int, default=50)
    parser.add_argument("--use_v2_model", action="store_true")
    parser.add_argument("--grid_size",   type=int, default=0,  help="0=只用标注点，>0=额外加稠密网格")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 加载模型 ──────────────────────────────────────
    if args.checkpoint:
        model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2")
    model = model.to(DEFAULT_DEVICE)
    model.eval()
    print("✅ CoTracker2 加载完成")

    vis = Visualizer(save_dir=args.output_dir, pad_value=120, linewidth=2, fps=10)
    COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]

    # ── 批量处理 ──────────────────────────────────────
    video_files = sorted(Path(args.video_dir).glob("*.mp4"))
    print(f"共找到 {len(video_files)} 个视频\n")

    all_results = {}

    for video_path in video_files:
        seq_name = video_path.stem
        ann_path = Path(args.ann_dir) / f"annotation_{seq_name}.json"

        if not ann_path.exists():
            print(f"[跳过] {seq_name}，无标注文件")
            continue

        print(f"\n{'='*50}\n处理: {seq_name}")

        # 读视频
        video_np = read_video_from_path(str(video_path))
        video_np = video_np[:args.max_frames]
        T, H, W  = video_np.shape[0], video_np.shape[1], video_np.shape[2]
        print(f"  帧数={T}, 分辨率={H}x{W}")

        video = torch.from_numpy(video_np).permute(0,3,1,2)[None].float().to(DEFAULT_DEVICE)

        # 读标注
        with open(ann_path) as f:
            ann = json.load(f)
        f0_pts  = np.array(ann["f0"])
        gt_last = np.array(ann["fLast"])
        N = len(f0_pts)

        # 构造 queries [1, N, 3] 格式 [帧号, x, y]
        queries = torch.tensor(
            [[0.0, x, y] for x, y in f0_pts],
            dtype=torch.float32
        ).unsqueeze(0).to(DEFAULT_DEVICE)

        # ── 推理 ──────────────────────────────────────
        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                video,
                queries=queries,
                grid_size=args.grid_size,
            )
        # pred_tracks: [1, T, N, 2]
        # pred_visibility: [1, T, N]

        all_trajs = pred_tracks[0].cpu().numpy()    # [T, N, 2]
        pred_last = all_trajs[-1]                   # [N, 2]

        # ── 指标 ──────────────────────────────────────
        errors = np.linalg.norm(pred_last - gt_last, axis=1)
        ATE    = errors.mean()
        for i in range(N):
            print(f"  P{i+1}: 误差={errors[i]:.1f}px")
        print(f"  平均ATE: {ATE:.1f}px")
        davg = delta_avg(pred_last, gt_last, H, W)

        # ── 官方追踪视频 ──────────────────────────────
        vis.visualize(
            video=video,
            tracks=pred_tracks,
            visibility=pred_visibility,
            filename=f"tracking_{seq_name}"
        )
        print(f"  ✅ 追踪视频 → {args.output_dir}/tracking_{seq_name}.mp4")

        # ── 最后一帧对比图 ────────────────────────────
        import cv2
        last_frame = video_np[-1].copy()
        last_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)

        for i in range(N):
            color = COLORS[i % len(COLORS)]
            px = int(round(pred_last[i, 0]))
            py = int(round(pred_last[i, 1]))
            gx = int(round(gt_last[i, 0]))
            gy = int(round(gt_last[i, 1]))

            if 0<=px<W and 0<=py<H:
                cv2.circle(last_frame, (px,py), 8, color, -1)
                cv2.putText(last_frame, f"P{i+1}", (px+10,py-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.drawMarker(last_frame,(gx,gy),(255,255,255),cv2.MARKER_CROSS,20,2)
            cv2.circle(last_frame,(gx,gy),8,(255,255,255),2)
            cv2.putText(last_frame,f"GT{i+1}",(gx+10,gy+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            if 0<=px<W and 0<=py<H:
                cv2.line(last_frame,(px,py),(gx,gy),(200,200,200),1)
                mx,my = (px+gx)//2,(py+gy)//2
                cv2.putText(last_frame,f"{errors[i]:.1f}px",(mx+4,my),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,200,200),1)

        cv2.putText(last_frame,f"ATE={ATE:.1f}px",(10,H-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
        out_img = str(Path(args.output_dir) / f"lastframe_{seq_name}.png")
        cv2.imwrite(out_img, last_frame)
        print(f"  ✅ 对比图   → {out_img}")

        # 保存 json
        all_results[seq_name] = {
            "ATE": float(ATE), "delta_avg": davg,
            "per_point_error": errors.tolist(),
            "pred_last": pred_last.tolist(),
            "gt_last": gt_last.tolist()
        }

    # ── 汇总 ──────────────────────────────────────────
    print(f"\n{'='*50}\n汇总 (CoTracker2):")
    ATEs  = [r['ATE']       for r in all_results.values()]
    davgs = [r['delta_avg'] for r in all_results.values()]
    for seq, res in all_results.items():
        print(f"  {seq}: ATE={res['ATE']:.1f}px  δ_avg={res['delta_avg']*100:.1f}%")
    print(f"\n平均 ATE:   {np.mean(ATEs):.1f}px")
    print(f"平均 δ_avg: {np.mean(davgs)*100:.1f}%")

    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump({"sequences": all_results,
                   "mean_ATE": float(np.mean(ATEs)),
                   "mean_delta_avg": float(np.mean(davgs))}, f, indent=2)
    print(f"\n结果保存至 {args.output_dir}")