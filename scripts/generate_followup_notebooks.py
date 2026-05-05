#!/usr/bin/env python3
"""Generate follow-up notebooks for city-swap comparison and English pilot work."""

from __future__ import annotations

import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def lines(text: str) -> list[str]:
    text = inspect.cleandoc(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


comparison_cells = [
    md_cell(
        """
        # 73 Cross-Run City Swap Comparison

        One notebook to compare all city-swap runs we already have:
        - `70_city_swap_batch_eval.ipynb` main models
        - `72_restored_models_city_swap.ipynb` restored adversarial / attr_reg attempts
        - `challengers/c10_challenger_city_swap_eval.ipynb` selected challenger winners

        Goal: compare existing swap evidence without retraining anything.
        """
    ),
    code_cell(
        """
        import json
        from pathlib import Path

        import numpy as np
        import pandas as pd

        try:
            import matplotlib.pyplot as plt
            HAS_MPL = True
        except ModuleNotFoundError:
            HAS_MPL = False

        CWD = Path.cwd()
        NOTEBOOK_DIR = CWD if (CWD / "results").exists() else (CWD / "notebooks")
        REPO_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR.parent

        MAIN_SWAP_JSON = REPO_ROOT / "notebooks" / "results" / "city_swap_all" / "city_swap_all_models.json"
        UNIFIED_TABLE_CSV = REPO_ROOT / "notebooks" / "results" / "unified_comparison" / "c71_unified_models_table.csv"
        C10_SUMMARY_CSV = REPO_ROOT / "results" / "challengers_city_swap" / "c10_selected_family_city_swap" / "c10_selected_family_city_swap_summary.csv"
        R72_SUMMARY_CSV = REPO_ROOT / "notebooks" / "results" / "two-models-restore" / "city_swap_restored_models" / "72_restored_models_city_swap_summary.csv"
        OUTPUT_DIR = REPO_ROOT / "notebooks" / "results" / "cross_run_city_swap"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Repo root:", REPO_ROOT)
        print("Output dir:", OUTPUT_DIR)
        """
    ),
    code_cell(
        """
        unified = pd.read_csv(UNIFIED_TABLE_CSV)
        unified = unified.rename(columns={"display_name": "display_name_c71"})

        with MAIN_SWAP_JSON.open("r", encoding="utf-8") as f:
            main_swap_raw = json.load(f)

        main_rows = []
        for run_name, payload in main_swap_raw.items():
            row = {
                "source_run": "70_batch",
                "expected_track": "main",
                "run_name": run_name,
                "status": "error" if "error" in payload else "ok",
                "error": payload.get("error"),
                "model_dir": payload.get("model_dir"),
                "overall_flip_rate": payload.get("overall_flip_rate"),
                "swap_accuracy": payload.get("accuracy"),
                "swap_macro_f1": payload.get("macro_f1"),
                "uses_scrubbing": payload.get("uses_scrubbing"),
            }
            if row["model_dir"]:
                row["model_name"] = Path(row["model_dir"]).name
            else:
                row["model_name"] = run_name

            per_city = payload.get("per_city_flip_rate", {})
            per_class = payload.get("per_class_flip_rate", {})
            if per_city:
                city_rates = [v.get("flip_rate", v) if isinstance(v, dict) else v for v in per_city.values()]
                row["max_city_flip_rate"] = max(city_rates)
                row["min_city_flip_rate"] = min(city_rates)
            if per_class:
                class_rates = [v.get("flip_rate", v) if isinstance(v, dict) else v for v in per_class.values()]
                row["max_class_flip_rate"] = max(class_rates)
            main_rows.append(row)

        main_df = pd.DataFrame(main_rows)

        c10_df = pd.read_csv(C10_SUMMARY_CSV).rename(
            columns={
                "family": "family_from_swap",
                "accuracy": "swap_accuracy",
                "macro_f1": "swap_macro_f1",
            }
        )
        c10_df["source_run"] = "c10_challengers"
        c10_df["expected_track"] = "challenger"
        c10_df["run_name"] = c10_df["model_name"]
        c10_df["status"] = c10_df["status"].fillna("ok")
        city_flip_cols = [c for c in c10_df.columns if c.startswith("flip_")]
        if city_flip_cols:
            c10_df["max_city_flip_rate"] = c10_df[city_flip_cols].max(axis=1)
            c10_df["min_city_flip_rate"] = c10_df[city_flip_cols].min(axis=1)

        restored_df = pd.read_csv(R72_SUMMARY_CSV).rename(
            columns={
                "family": "family_from_swap",
                "accuracy": "swap_accuracy",
                "macro_f1": "swap_macro_f1",
            }
        )
        restored_df["source_run"] = "72_restored"
        restored_df["expected_track"] = "main"
        restored_df["run_name"] = restored_df["model_name"]

        keep_cols = [
            "source_run",
            "run_name",
            "expected_track",
            "model_name",
            "family_from_swap",
            "status",
            "error",
            "swap_accuracy",
            "swap_macro_f1",
            "overall_flip_rate",
            "max_city_flip_rate",
            "min_city_flip_rate",
            "model_dir",
            "model_path",
            "uses_scrubbing",
        ]

        main_df = main_df.reindex(columns=keep_cols)
        c10_df = c10_df.reindex(columns=keep_cols)
        restored_df = restored_df.reindex(columns=keep_cols)

        swap_all = pd.concat([main_df, c10_df, restored_df], ignore_index=True, sort=False)
        swap_all
        """
    ),
    code_cell(
        """
        compare_df = swap_all.merge(
            unified[
                [
                    "track",
                    "family_group",
                    "model_name",
                    "display_name_c71",
                    "method",
                    "accuracy",
                    "macro_f1",
                    "worst_gap",
                    "macro_gap",
                ]
            ],
            left_on=["model_name", "expected_track"],
            right_on=["model_name", "track"],
            how="left",
        )

        compare_df["family"] = compare_df["family_from_swap"].fillna(compare_df["family_group"])
        compare_df["display_name"] = compare_df["display_name_c71"].fillna(compare_df["model_name"])
        compare_df["rank_flip_then_f1"] = compare_df.sort_values(
            ["overall_flip_rate", "swap_macro_f1"],
            ascending=[True, False],
            na_position="last",
        ).reset_index().index + 1
        compare_df["rank_gap_then_f1"] = compare_df.sort_values(
            ["worst_gap", "macro_f1"],
            ascending=[True, False],
            na_position="last",
        ).reset_index().index + 1

        compare_df = compare_df[
            [
                "rank_flip_then_f1",
                "rank_gap_then_f1",
                "source_run",
                "track",
                "family",
                "model_name",
                "display_name",
                "status",
                "accuracy",
                "macro_f1",
                "worst_gap",
                "macro_gap",
                "swap_accuracy",
                "swap_macro_f1",
                "overall_flip_rate",
                "max_city_flip_rate",
                "min_city_flip_rate",
                "uses_scrubbing",
                "error",
            ]
        ].sort_values(["status", "overall_flip_rate", "swap_macro_f1"], ascending=[True, True, False], na_position="last")

        compare_df.to_csv(OUTPUT_DIR / "73_cross_run_city_swap_table.csv", index=False)
        compare_df
        """
    ),
    code_cell(
        """
ok_df = compare_df[compare_df["status"] == "ok"].copy()

summary = {
    "n_total_rows": int(len(compare_df)),
    "n_ok_rows": int((compare_df["status"] == "ok").sum()),
    "n_failed_rows": int((compare_df["status"] != "ok").sum()),
}

if not ok_df.empty:
    best_flip = ok_df.sort_values(["overall_flip_rate", "swap_macro_f1"], ascending=[True, False]).iloc[0]
    best_macro_f1 = ok_df.sort_values(["swap_macro_f1", "overall_flip_rate"], ascending=[False, True]).iloc[0]
    summary["best_city_swap_stability"] = {
        "model_name": best_flip["model_name"],
        "source_run": best_flip["source_run"],
        "overall_flip_rate": None if pd.isna(best_flip["overall_flip_rate"]) else float(best_flip["overall_flip_rate"]),
        "swap_macro_f1": None if pd.isna(best_flip["swap_macro_f1"]) else float(best_flip["swap_macro_f1"]),
    }
    summary["best_swap_macro_f1"] = {
        "model_name": best_macro_f1["model_name"],
        "source_run": best_macro_f1["source_run"],
        "overall_flip_rate": None if pd.isna(best_macro_f1["overall_flip_rate"]) else float(best_macro_f1["overall_flip_rate"]),
        "swap_macro_f1": None if pd.isna(best_macro_f1["swap_macro_f1"]) else float(best_macro_f1["swap_macro_f1"]),
    }

by_source = (
    compare_df.groupby("source_run", dropna=False)
    .agg(
        rows=("model_name", "size"),
        ok_rows=("status", lambda s: int((s == "ok").sum())),
        median_flip=("overall_flip_rate", "median"),
        best_flip=("overall_flip_rate", "min"),
        best_swap_f1=("swap_macro_f1", "max"),
    )
    .reset_index()
)
by_source.to_csv(OUTPUT_DIR / "73_cross_run_city_swap_by_source.csv", index=False)

report_lines = [
    "=== 73 cross-run city-swap comparison ===",
    f"rows_total: {summary['n_total_rows']}",
    f"rows_ok: {summary['n_ok_rows']}",
    f"rows_failed: {summary['n_failed_rows']}",
    "",
]
if "best_city_swap_stability" in summary:
    item = summary["best_city_swap_stability"]
    report_lines.extend(
        [
            "best_city_swap_stability:",
            f"  model_name: {item['model_name']}",
            f"  source_run: {item['source_run']}",
            f"  overall_flip_rate: {item['overall_flip_rate']}",
            f"  swap_macro_f1: {item['swap_macro_f1']}",
            "",
        ]
    )
if "best_swap_macro_f1" in summary:
    item = summary["best_swap_macro_f1"]
    report_lines.extend(
        [
            "best_swap_macro_f1:",
            f"  model_name: {item['model_name']}",
            f"  source_run: {item['source_run']}",
            f"  overall_flip_rate: {item['overall_flip_rate']}",
            f"  swap_macro_f1: {item['swap_macro_f1']}",
            "",
        ]
    )

failed = compare_df[compare_df["status"] != "ok"][["model_name", "source_run", "status", "error"]]
if not failed.empty:
    report_lines.append("failed_rows:")
    for _, row in failed.iterrows():
        report_lines.append(
            f"  - {row['model_name']} | source={row['source_run']} | status={row['status']} | error={row['error']}"
        )

report_text = "\\n".join(report_lines)
print(report_text)
(OUTPUT_DIR / "73_cross_run_city_swap_report.txt").write_text(report_text, encoding="utf-8")
(OUTPUT_DIR / "73_cross_run_city_swap_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
by_source
        """
    ),
    code_cell(
        """
        leaderboard_cols = [
            "source_run",
            "family",
            "model_name",
            "status",
            "accuracy",
            "macro_f1",
            "swap_accuracy",
            "swap_macro_f1",
            "overall_flip_rate",
            "worst_gap",
            "macro_gap",
        ]
        leaderboard = compare_df[leaderboard_cols].copy()
        leaderboard
        """
    ),
    code_cell(
        """
        if HAS_MPL:
            plot_df = compare_df[compare_df["status"] == "ok"].copy()
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = {"70_batch": "#1f77b4", "c10_challengers": "#2ca02c", "72_restored": "#d62728"}
            for source_run, chunk in plot_df.groupby("source_run"):
                ax.scatter(
                    chunk["overall_flip_rate"],
                    chunk["swap_macro_f1"],
                    s=70,
                    label=source_run,
                    color=colors.get(source_run, "#7f7f7f"),
                    alpha=0.85,
                )
                for _, row in chunk.iterrows():
                    ax.annotate(row["model_name"], (row["overall_flip_rate"], row["swap_macro_f1"]), fontsize=8, alpha=0.8)

            ax.set_xlabel("Overall city-swap flip rate")
            ax.set_ylabel("Macro-F1 on swapped set")
            ax.set_title("Cross-run city-swap comparison")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "73_cross_run_city_swap_scatter.png", dpi=180, bbox_inches="tight")
            plt.show()
        else:
            print("matplotlib is not installed; skipping scatter plot.")
        """
    ),
]


english_cells = [
    md_cell(
        """
        # 02 English Dataset Transfer + City-Swap Pilot

        This notebook turns the English Kaggle dataset into a usable next step, without immediately launching a full retraining loop.

        Recommended order:
        1. load and sanity-check the English dataset,
        2. map raw English categories into the project's 9 target classes,
        3. build an English location-aware evaluation slice,
        4. generate English city-swap counterfactuals,
        5. optionally run already-trained local models on the English slice if checkpoints are available.
        """
    ),
    md_cell(
        """
        ## Why this is the next move

        `city swap` is worth doing on English data, but only after we confirm the dataset can be projected into the same 9-class label space.

        So the practical plan is:
        - do **label mapping first**,
        - create an **English counterfactual benchmark** next,
        - only then evaluate existing models,
        - retrain only if transfer actually looks weak.
        """
    ),
    code_cell(
        """
        import json
        import re
        import zipfile
        from pathlib import Path

        import numpy as np
        import pandas as pd

        try:
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install kagglehub first: pip install kagglehub"
            ) from exc

        try:
            import matplotlib.pyplot as plt
            HAS_MPL = True
        except ModuleNotFoundError:
            HAS_MPL = False

        CWD = Path.cwd()
        NOTEBOOK_DIR = CWD if (CWD / "results").exists() else (CWD / "notebooks")
        REPO_ROOT = NOTEBOOK_DIR.parent if NOTEBOOK_DIR.name == "notebooks" else NOTEBOOK_DIR.parent
        OUTPUT_DIR = REPO_ROOT / "notebooks" / "results" / "english_transfer_pilot"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Repo root:", REPO_ROOT)
        print("Output dir:", OUTPUT_DIR)
        """
    ),
    code_cell(
        """
        KAGGLE_HANDLE = "rayyankauchali0/resume-dataset"
        path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
        print("Downloaded to:", path)

        def load_jsonl(path_obj: Path):
            rows = []
            with path_obj.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return pd.DataFrame(rows)

        def load_dataset(download_dir: Path) -> pd.DataFrame:
            candidates = list(download_dir.rglob("*.csv")) + list(download_dir.rglob("*.jsonl")) + list(download_dir.rglob("*.json"))
            if not candidates:
                zip_candidates = list(download_dir.rglob("*.zip"))
                for archive in zip_candidates:
                    with zipfile.ZipFile(archive) as zf:
                        extract_dir = archive.with_suffix("")
                        zf.extractall(extract_dir)
                candidates = list(download_dir.rglob("*.csv")) + list(download_dir.rglob("*.jsonl")) + list(download_dir.rglob("*.json"))

            if not candidates:
                raise FileNotFoundError(f"No CSV/JSONL/JSON files found under {download_dir}")

            ranked = sorted(candidates, key=lambda p: (p.suffix != ".csv", len(str(p))))
            data_path = ranked[0]
            print("Using data file:", data_path)

            if data_path.suffix == ".csv":
                return pd.read_csv(data_path)
            if data_path.suffix == ".jsonl":
                return load_jsonl(data_path)
            return pd.read_json(data_path)

        df = load_dataset(path)
        print("Shape:", df.shape)
        print("Columns:", list(df.columns))
        df.head(3)
        """
    ),
    code_cell(
        """
        TEXT_CANDIDATES = ["Resume_Text", "resume_text", "Resume", "Text", "Summary", "Experience", "Education", "Skills"]
        CATEGORY_CANDIDATES = ["Category", "category", "Job_Role", "job_role"]
        LOCATION_CANDIDATES = ["Location", "location", "City", "city", "Address", "address"]

        def first_present(columns, candidates):
            for col in candidates:
                if col in columns:
                    return col
            return None

        text_col = first_present(df.columns, TEXT_CANDIDATES)
        category_col = first_present(df.columns, CATEGORY_CANDIDATES)
        location_col = first_present(df.columns, LOCATION_CANDIDATES)

        if text_col is None:
            raise ValueError(f"Could not find a resume text column among: {TEXT_CANDIDATES}")

        df = df.copy()
        df["resume_text_work"] = df[text_col].fillna("").astype(str)
        df["text_len_chars"] = df["resume_text_work"].str.len()
        df["text_len_words"] = df["resume_text_work"].str.split().str.len()

        print("text_col:", text_col)
        print("category_col:", category_col)
        print("location_col:", location_col)
        df[[c for c in [text_col, category_col, location_col, "text_len_words"] if c is not None]].head(5)
        """
    ),
    code_cell(
        """
        TARGET_9_CLASSES = [
            "backend_general_dev",
            "web_frontend",
            "sysadmin_devops_network",
            "project_product",
            "sales_account",
            "tech_support_helpdesk",
            "it_governance_leadership",
            "technical_specialized",
            "generic_it_ops",
        ]

        CATEGORY_TO_TARGET = {
            "Data Science": "technical_specialized",
            "Database": "technical_specialized",
            "DevOps Engineer": "sysadmin_devops_network",
            "DotNet Developer": "backend_general_dev",
            "ETL Developer": "backend_general_dev",
            "Hadoop": "technical_specialized",
            "HR": "generic_it_ops",
            "Advocate": "sales_account",
            "Arts": "generic_it_ops",
            "Automation Testing": "technical_specialized",
            "Blockchain": "technical_specialized",
            "Business Analyst": "project_product",
            "Civil Engineer": "generic_it_ops",
            "Designer": "web_frontend",
            "Electrical Engineering": "generic_it_ops",
            "Health and fitness": "generic_it_ops",
            "Java Developer": "backend_general_dev",
            "Mechanical Engineer": "generic_it_ops",
            "Network Security Engineer": "sysadmin_devops_network",
            "Operations Manager": "it_governance_leadership",
            "PMO": "project_product",
            "Python Developer": "backend_general_dev",
            "SAP Developer": "technical_specialized",
            "Sales": "sales_account",
            "Testing": "tech_support_helpdesk",
            "Web Designing": "web_frontend",
        }

        def normalize_category(value):
            if pd.isna(value):
                return None
            key = str(value).strip()
            if key in CATEGORY_TO_TARGET:
                return CATEGORY_TO_TARGET[key]

            lowered = key.lower()
            if any(token in lowered for token in ["frontend", "web design", "ui", "ux"]):
                return "web_frontend"
            if any(token in lowered for token in ["backend", "python", "java", "dotnet", "developer", "software", "etl"]):
                return "backend_general_dev"
            if any(token in lowered for token in ["devops", "sysadmin", "network", "security", "cloud", "infrastructure"]):
                return "sysadmin_devops_network"
            if any(token in lowered for token in ["product", "project", "business analyst", "pmo"]):
                return "project_product"
            if any(token in lowered for token in ["sales", "account", "advocate", "marketing"]):
                return "sales_account"
            if any(token in lowered for token in ["support", "helpdesk", "testing", "qa"]):
                return "tech_support_helpdesk"
            if any(token in lowered for token in ["manager", "head", "director", "leadership", "operations manager"]):
                return "it_governance_leadership"
            if any(token in lowered for token in ["data", "database", "sap", "blockchain", "hadoop", "ml", "ai"]):
                return "technical_specialized"
            return "generic_it_ops"

        if category_col is not None:
            df["target_9_class"] = df[category_col].apply(normalize_category)
        else:
            df["target_9_class"] = "generic_it_ops"

        mapping_review = (
            df.groupby([category_col, "target_9_class"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            if category_col is not None
            else pd.DataFrame({"target_9_class": df["target_9_class"].value_counts().index, "count": df["target_9_class"].value_counts().values})
        )
        mapping_review.to_csv(OUTPUT_DIR / "02_english_category_mapping_review.csv", index=False)
        mapping_review.head(30)
        """
    ),
    code_cell(
        """
        CITY_PATTERN = re.compile(
            r"\\b("
            r"new york|san francisco|los angeles|seattle|austin|chicago|boston|atlanta|dallas|houston|"
            r"london|berlin|paris|madrid|toronto|vancouver|sydney|melbourne|singapore|dubai|"
            r"bangalore|bengaluru|mumbai|delhi|hyderabad|pune"
            r")\\b",
            flags=re.IGNORECASE,
        )

        def extract_city_mentions(text: str) -> list[str]:
            if not text:
                return []
            return sorted({m.group(0).lower() for m in CITY_PATTERN.finditer(text)})

        df["city_mentions"] = df["resume_text_work"].apply(extract_city_mentions)
        df["n_city_mentions"] = df["city_mentions"].apply(len)
        df["has_city_mention"] = df["n_city_mentions"] > 0

        location_summary = pd.DataFrame(
            {
                "rows_total": [len(df)],
                "rows_with_city_mentions": [int(df["has_city_mention"].sum())],
                "share_with_city_mentions": [float(df["has_city_mention"].mean()) if len(df) else 0.0],
                "median_words": [float(df["text_len_words"].median()) if len(df) else 0.0],
            }
        )
        location_summary.to_csv(OUTPUT_DIR / "02_english_location_summary.csv", index=False)

        city_counts = (
            df.explode("city_mentions")
            .dropna(subset=["city_mentions"])
            .groupby("city_mentions")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        city_counts.to_csv(OUTPUT_DIR / "02_english_city_counts.csv", index=False)

        location_summary
        """
    ),
    code_cell(
        """
        SWAP_CITIES = [
            "new york",
            "san francisco",
            "london",
            "berlin",
            "singapore",
        ]

        def swap_cities_in_text(text: str, target_city: str) -> str:
            if not isinstance(text, str) or not text:
                return text

            def repl(match):
                original = match.group(0)
                if original.isupper():
                    return target_city.upper()
                if original[:1].isupper():
                    return target_city.title()
                return target_city.lower()

            return CITY_PATTERN.sub(repl, text)

        english_eval = df[df["has_city_mention"]].copy()
        english_eval["source_city"] = english_eval["city_mentions"].apply(lambda xs: xs[0] if xs else None)
        english_eval = english_eval[english_eval["source_city"].notna()].copy()

        counterfactual_rows = []
        for _, row in english_eval.iterrows():
            for swap_city in SWAP_CITIES:
                if row["source_city"] == swap_city:
                    continue
                swapped_text = swap_cities_in_text(row["resume_text_work"], swap_city)
                counterfactual_rows.append(
                    {
                        "row_id": int(row.name),
                        "target_9_class": row["target_9_class"],
                        "source_city": row["source_city"],
                        "swap_city": swap_city,
                        "original_text": row["resume_text_work"],
                        "swapped_text": swapped_text,
                    }
                )

        counterfactual_df = pd.DataFrame(counterfactual_rows)
        base_eval_df = english_eval[["resume_text_work", "target_9_class", "source_city"]].rename(columns={"resume_text_work": "original_text"})

        base_eval_df.to_csv(OUTPUT_DIR / "02_english_base_eval_slice.csv", index=False)
        counterfactual_df.to_csv(OUTPUT_DIR / "02_english_city_swap_counterfactuals.csv", index=False)

        print("Base eval rows:", len(base_eval_df))
        print("Counterfactual rows:", len(counterfactual_df))
        counterfactual_df.head(10)
        """
    ),
    code_cell(
        """
        MODEL_CANDIDATES = [
            ("baseline", REPO_ROOT / "models" / "bert_9classes_final"),
            ("groupdro", REPO_ROOT / "models" / "bert_gdro_eta01_2ep"),
            ("label_smoothing", REPO_ROOT / "models" / "bert_label_smoothing"),
            ("combined_best", REPO_ROOT / "models" / "combined_scrubbing_groupdro" / "final"),
        ]

        available_models = [(name, path) for name, path in MODEL_CANDIDATES if path.exists()]
        if not available_models:
            print("No local checkpoints found under repo/models. Dataset preparation is complete; run evaluation on Natasha's machine where checkpoints exist.")
        else:
            print("Found local checkpoints:")
            for name, path in available_models:
                print(" -", name, "->", path)
        """
    ),
    code_cell(
        """
        evaluation_plan = pd.DataFrame(
            [
                {
                    "priority": 1,
                    "step": "Manual review of English->9-class mapping",
                    "why": "Prevents garbage transfer metrics from label mismatch",
                    "artifact": "02_english_category_mapping_review.csv",
                },
                {
                    "priority": 2,
                    "step": "Run existing strongest models on the English base eval slice",
                    "why": "Checks out-of-domain generalization before any retraining",
                    "artifact": "02_english_base_eval_slice.csv",
                },
                {
                    "priority": 3,
                    "step": "Run city-swap on the English counterfactual slice",
                    "why": "Measures location sensitivity on English resumes directly",
                    "artifact": "02_english_city_swap_counterfactuals.csv",
                },
                {
                    "priority": 4,
                    "step": "Only then decide whether to fine-tune challengers on English data",
                    "why": "Avoids another 15 training runs if transfer already looks acceptable",
                    "artifact": "decision_after_eval",
                },
            ]
        )
        evaluation_plan.to_csv(OUTPUT_DIR / "02_english_next_steps_plan.csv", index=False)
        evaluation_plan
        """
    ),
    code_cell(
        """
        quick_answer = {
            "should_we_do_city_swap_on_english": True,
            "should_we_train_again_immediately": False,
            "recommended_first_move": "map English labels into the current 9-class space and evaluate existing strongest checkpoints first",
            "why": [
                "The dataset must be aligned to the current label space before city-swap numbers are interpretable.",
                "Existing models may already transfer well enough to justify evaluation-before-retraining.",
                "If transfer fails, the prepared English counterfactual slice becomes the benchmark for the next training wave.",
            ],
        }
        (OUTPUT_DIR / "02_english_quick_answer.json").write_text(
            json.dumps(quick_answer, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(quick_answer, indent=2, ensure_ascii=False))
        """
    ),
    code_cell(
        """
        if HAS_MPL and not df.empty:
            class_counts = df["target_9_class"].value_counts().sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            class_counts.plot(kind="barh", ax=ax, color="#4c78a8")
            ax.set_title("English dataset mapped into current 9-class space")
            ax.set_xlabel("Rows")
            ax.set_ylabel("Target class")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "02_english_target9_distribution.png", dpi=180, bbox_inches="tight")
            plt.show()
        """
    ),
]


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    write_notebook(NOTEBOOKS / "73_cross_run_city_swap_comparison.ipynb", comparison_cells)
    write_notebook(NOTEBOOKS / "02_english_dataset_transfer_pilot.ipynb", english_cells)


if __name__ == "__main__":
    main()
