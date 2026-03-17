import json
import os
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd


def load_training_folder(folder: str) -> dict[str, dict]:
    """
    Scan *folder* for every ``*.json`` file and load each one.

    Returns
    -------
    dict[str, dict]
        Keys are the model name (``nombre_modelo`` field inside the JSON,
        falling back to the filename stem if the field is absent).
        Values are the raw parsed dictionaries.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    ValueError
        If no JSON files are found in *folder*.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    json_files = sorted(folder_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {folder_path.resolve()}")

    runs: dict[str, dict] = {}
    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Use the internal model name when present, else the filename
        name = data.get("nombre_modelo") or filepath.stem

        # Handle duplicate names by appending the filename stem
        if name in runs:
            name = f"{name}__{filepath.stem}"

        runs[name] = data
        print(f"  ✔ Loaded '{name}' from {filepath.name}")

    print(f"\n{len(runs)} run(s) loaded from '{folder_path.resolve()}'.\n")
    return runs


def runs_to_dataframe(runs: dict[str, dict]) -> pd.DataFrame:
    """
    Flatten the *runs* dictionary into a single tidy ``pd.DataFrame``
    with one row per epoch per model.

    Columns: ``model``, ``epoch``, ``time_s``, ``acc_train``, ``loss``
    """
    frames = []
    for name, data in runs.items():
        info = data.get("info_extra", {})
        df = pd.DataFrame({
            "model"    : name,
            "epoch"    : info.get("historial_intervalo_epochs", []),
            "time_s"   : info.get("historial_intervalo_times", []),
            "acc_train": info.get("historial_intervalo_acc_train", []),
            "loss"     : info.get("historial_intervalo_loss", []),
        })
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def runs_metadata(runs: dict[str, dict]) -> pd.DataFrame:
    """
    Return a summary ``pd.DataFrame`` with one row per model and high-level
    hyperparameters / results.
    """
    rows = []
    for name, data in runs.items():
        info = data.get("info_extra", {})
        arch = data.get("arquitectura", {})
        rows.append({
            "model"          : name,
            "test_acc (%)"   : data.get("precision_test"),
            "epochs"         : data.get("epocas"),
            "learning_rate"  : data.get("learning_rate"),
            "train_time (s)" : data.get("training_time_seconds"),
            "arch_hidden"    : arch.get("oculta"),
            "num_partitions" : info.get("num_particiones"),
            "architecture"   : info.get("architecture"),
        })
    return pd.DataFrame(rows).set_index("model")


def compare_runs(
    runs: dict[str, dict],
    keys: Optional[list[str]] = None,
    ) -> alt.VConcatChart:
        """
        Compare N training runs visually with Altair.

        Parameters
        ----------
        runs : dict[str, dict]
            The dictionary returned by :func:`load_training_folder`, or any subset.
        keys : list[str] | None
            If given, only the runs whose names are in *keys* are compared.
            When ``None`` (default) all runs are compared.

        Returns
        -------
        alt.VConcatChart
            The combined Altair chart (ready to display in a Jupyter notebook or
            to save programmatically).
        """
        # ── filter ──────────────────────────────────────────────────────────────
        if keys:
            missing = [k for k in keys if k not in runs]
            if missing:
                raise KeyError(f"Keys not found in runs: {missing}")
            selected = {k: runs[k] for k in keys}
        else:
            selected = runs

        if len(selected) < 1:
            raise ValueError("Need at least one run to compare.")

        # ── tidy data ────────────────────────────────────────────────────────────
        df = runs_to_dataframe(selected)
        meta = runs_metadata(selected).reset_index()

        # ── colour scale shared across all sub-charts ────────────────────────────
        colour = alt.Color(
            "model:N",
            legend=alt.Legend(title="Model", orient="right"),
            scale=alt.Scale(scheme="category10"),
        )

        # ── shared tooltip fields ────────────────────────────────────────────────
        tooltip_epoch = [
            alt.Tooltip("model:N",     title="Model"),
            alt.Tooltip("epoch:Q",     title="Epoch"),
            alt.Tooltip("acc_train:Q", title="Train Acc (%)", format=".2f"),
            alt.Tooltip("loss:Q",      title="Loss",          format=".4f"),
        ]
        tooltip_time = [
            alt.Tooltip("model:N",     title="Model"),
            alt.Tooltip("time_s:Q",    title="Time (s)", format=".2f"),
            alt.Tooltip("epoch:Q",     title="Epoch"),
            alt.Tooltip("acc_train:Q", title="Train Acc (%)", format=".2f"),
        ]

        # ── Chart A: Accuracy vs Epochs ──────────────────────────────────────────
        acc_epoch = (
            alt.Chart(df)
            .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
            .encode(
                x=alt.X("epoch:Q",     title="Epoch",                scale=alt.Scale(nice=False)),
                y=alt.Y("acc_train:Q", title="Training Accuracy (%)", scale=alt.Scale(zero=False)),
                color=colour,
                tooltip=tooltip_epoch,
            )
            .properties(
                title=alt.TitleParams("Training Accuracy vs Epochs", fontSize=14),
                width=1000, height=500,
            )
            .interactive()
        )

        # ── Chart B: Accuracy vs Time ────────────────────────────────────────────
        acc_time = (
            alt.Chart(df)
            .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
            .encode(
                y=alt.Y("time_s:Q",    title="Elapsed Time (s)",      scale=alt.Scale(nice=False)),
                x=alt.X("epoch:Q", title="Epoch",  scale=alt.Scale(nice=False)),
                color=colour,
                tooltip=tooltip_time,
            )
            .properties(
                title=alt.TitleParams("Training epochs vs Time", fontSize=14),
                width=1000, height=500,
            )
            .interactive()
        )

        # ── Chart C: Loss vs Epochs ──────────────────────────────────────────────
        loss_epoch = (
            alt.Chart(df)
            .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
            .encode(
                x=alt.X("epoch:Q", title="Epoch",  scale=alt.Scale(nice=False)),
                y=alt.Y("loss:Q",  title="Loss",   scale=alt.Scale(zero=False)),
                color=colour,
                tooltip=tooltip_epoch,
            )
            .properties(
                title=alt.TitleParams("Loss vs Epochs", fontSize=14),
                width=1000, height=500,
            )
            .interactive()
        )

        # ── Combine ──────────────────────────────────────────────────────────────
        combined = (
            alt.vconcat(acc_epoch, acc_time, loss_epoch)
            .resolve_scale(color="shared")
            .properties(
                title=alt.TitleParams(
                    text=f"Training Comparison — {len(selected)} model(s)",
                    fontSize=18,
                )
            )
        )

        return combined
