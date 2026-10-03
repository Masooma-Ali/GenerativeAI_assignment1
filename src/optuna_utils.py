"""Shared Optuna reporting (summary.txt, trials.csv, plots, best_config.json)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna


def make_report(study, out, space, extra_cfg=None, metric_name="objective"):
    df = study.trials_dataframe(); df.to_csv(os.path.join(out, "trials.csv"), index=False)
    comp = df[df.state == "COMPLETE"]
    lines = [f"Optuna study: {study.study_name}", f"trials by state: {df['state'].value_counts().to_dict()}",
             "", "Search space:"] + [f"  {k:12s} {v}" for k, v in space.items()]
    if len(comp):
        bt = study.best_trial
        lines += ["", f"Best trial: #{bt.number}  {metric_name}={bt.value:.4f}", "Best params:"]
        lines += [f"  {k}: {v}" for k, v in bt.params.items()]
        try:
            imp = optuna.importance.get_param_importances(study)
            lines += ["", "Parameter importance (fANOVA):"] + [f"  {k}: {v:.3f}" for k, v in imp.items()]
        except Exception as e:
            imp = None; lines += ["", f"(importance not computed: {e})"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(comp.number, comp.value, label="completed trial")
        ax.plot(comp.number, comp.value.cummax(), "r-", label="best so far")
        ax.set_xlabel("trial"); ax.set_ylabel(metric_name); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_history.png"), dpi=140); plt.close(fig)
        pc = [c for c in comp.columns if c.startswith("params_")]
        cols = 3; rows = (len(pc) + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(11, 3 * rows)); axs = axs.ravel()
        for ax, c in zip(axs, pc):
            ax.scatter(comp[c], comp.value, s=18); ax.set_xlabel(c.replace("params_", "")); ax.set_ylabel(metric_name)
            if c in ("params_lr", "params_weight_decay"): ax.set_xscale("log")
            ax.grid(alpha=.3)
        for ax in axs[len(pc):]: ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_params.png"), dpi=140); plt.close(fig)
        if imp:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_xlabel("importance")
            fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_importance.png"), dpi=140); plt.close(fig)
        json.dump(dict(bt.params, **(extra_cfg or {})), open(os.path.join(out, "best_config.json"), "w"), indent=2)
    txt = "\n".join(lines); open(os.path.join(out, "summary.txt"), "w").write(txt); print("\n" + txt)
