# report_module.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ReportRecorder:
    """
    收集滚动窗口评估结果并生成报告。
    会在 reports/ 目录下输出：
      - window_metrics.csv / .json
      - rolling_loss_lr_all_windows.png
      - run_log.txt 嵌入 Markdown
      - report.md
    """
    def __init__(self, out_dir="reports"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.window_results = []
        self.train_curves   = []
        self.val_curves     = []
        self.lr_curves      = []
        self.max_epochs     = 0

    def record_window(self, window_id, rmse, mae, tr_losses, val_losses, lr_list):
        # 存 per-window metrics
        self.window_results.append({
            "window_id": int(window_id),
            "rmse": float(rmse),
            "mae": float(mae),
            "epochs": len(tr_losses)
        })
        # 存曲线
        self.train_curves.append(tr_losses)
        self.val_curves.append(val_losses)
        self.lr_curves.append(lr_list)
        self.max_epochs = max(self.max_epochs, len(tr_losses))

    def save_metrics_csv(self):
        df = pd.DataFrame(self.window_results)
        path = os.path.join(self.out_dir, "window_metrics.csv")
        df.to_csv(path, index=False)
        print(f"Saved per-window metrics to {path}")

    def save_metrics_json(self):
        path = os.path.join(self.out_dir, "window_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.window_results, f, indent=2, ensure_ascii=False)
        print(f"Saved per-window metrics to {path}")

    def plot_aggregate_curves(self, filename="rolling_loss_lr_all_windows.png"):
        # pad curves to max_epochs
        def pad(lst):
            return lst + [np.nan] * (self.max_epochs - len(lst))
        trains = np.array([pad(c) for c in self.train_curves])
        vals   = np.array([pad(c) for c in self.val_curves])
        lrs    = np.array([pad(c) for c in self.lr_curves])
        epochs = np.arange(self.max_epochs)

        cmap = plt.get_cmap("tab10")
        fig, ax1 = plt.subplots(figsize=(8,5))
        for i in range(len(trains)):
            color = cmap(i % 10)
            ax1.plot(epochs, trains[i], color=color, alpha=0.4, lw=1)
            ax1.plot(epochs, vals[i],   color=color, alpha=0.8, lw=1.5, ls="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (Huber)")
        ax1.legend(["Train","Val"], loc="upper right")

        ax2 = ax1.twinx()
        for i in range(len(lrs)):
            ax2.plot(epochs, lrs[i], color=cmap(i % 10), alpha=0.3, ls=":")
        ax2.set_ylabel("Learning Rate", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        fig.tight_layout()
        path = os.path.join(self.out_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved aggregate curves to {path}")

    def summary(self):
        rmses = [w["rmse"] for w in self.window_results]
        maes  = [w["mae"]  for w in self.window_results]
        summary = {
            "n_windows": len(rmses),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std":  float(np.std(rmses)),
            "mae_mean":  float(np.mean(maes)),
            "mae_std":   float(np.std(maes))
        }
        path = os.path.join(self.out_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {path}")
        return summary

    def generate_markdown_report(self):
        summary = self.summary()
        lines = [
            "# Rolling‑Window CNN‑LSTM 评估报告",
            "",
            f"- 窗口数量: **{summary['n_windows']}**",
            f"- RMSE 平均 ± 标准差: **{summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}**",
            f"- MAE  平均 ± 标准差: **{summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}**",
            "",
            "## 每窗口指标（示例前 10 条）",
            "| window_id |   RMSE |   MAE | epochs |",
            "|----------:|-------:|------:|-------:|"
        ]
        for w in self.window_results[:10]:
            lines.append(f"| {w['window_id']:9d} | {w['rmse']:6.4f} | {w['mae']:5.4f} | {w['epochs']:6d} |")

        lines += [
            "",
            "## 曲线汇总",
            "![Loss & LR](rolling_loss_lr_all_windows.png)",
            "",
            "## 运行日志",
            "```",
        ]
        # 嵌入 run_log.txt
        log_path = os.path.join(self.out_dir, "run_log.txt")
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    lines.append(line.rstrip("\n"))
        lines.append("```")

        md = "\n".join(lines)
        path = os.path.join(self.out_dir, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved markdown report to {path}")
