
import argparse, json, os, re, sys, textwrap, warnings, difflib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, accuracy_score, f1_score

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_EXCEL_PATH = r"C:\MSProjectPython\D_Building_2000_prediction_dataset.xlsx"
RESULTS_DIR_NAME = "model_outputs_building"

def clean_col(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return re.sub(r"_+", "_", name).strip("_")

def make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

class BuildingPredictor:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.base_dir = os.path.dirname(excel_path) or os.getcwd()
        self.output_dir = os.path.join(self.base_dir, RESULTS_DIR_NAME)
        os.makedirs(self.output_dir, exist_ok=True)

        self.inputs_df = None
        self.targets_df = None
        self.explanations_df = None
        self.data = None
        self.input_columns = []
        self.numeric_columns = []
        self.categorical_columns = []

        self.preprocessor = None
        self.cost_et = None
        self.cost_knn = None
        self.time_et = None
        self.time_knn = None
        self.risk_et = None
        self.risk_rf = None
        self.stage2_costrisk = None
        self.stage2_timerisk = None
        self.stage2_riskscore = None
        self.train_matrix = None
        self.nn = None
        self.train_ref = None
        self.explanation_ref = None

        self.cost_pct_min = None
        self.cost_pct_max = None
        self.time_pct_min = None
        self.time_pct_max = None

    def load_data(self):
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found at: {self.excel_path}")

        xl = pd.ExcelFile(self.excel_path)
        lookup = {s.strip().lower(): s for s in xl.sheet_names}
        def pick(*names):
            for n in names:
                if n.lower() in lookup:
                    return lookup[n.lower()]
            raise ValueError(f"Required sheet not found. Available sheets: {xl.sheet_names}")

        self.inputs_df = pd.read_excel(self.excel_path, sheet_name=pick("Inputs", "Initial_Data", "Initial Data"))
        self.targets_df = pd.read_excel(self.excel_path, sheet_name=pick("Targets", "Results"))
        self.explanations_df = pd.read_excel(self.excel_path, sheet_name=pick("Outcome_Explanation", "Outcome Explanation"))

        self.inputs_df.columns = [clean_col(c) for c in self.inputs_df.columns]
        self.targets_df.columns = [clean_col(c) for c in self.targets_df.columns]
        self.explanations_df.columns = [clean_col(c) for c in self.explanations_df.columns]

        # Keep inputs as the source for planned cost and duration
        self.targets_df = self.targets_df[[c for c in self.targets_df.columns if c not in {"planned_cost", "planned_duration"}]]
        self.data = self.inputs_df.merge(self.targets_df, on="project_id", how="inner").merge(self.explanations_df, on="project_id", how="inner")

        if self.data.empty:
            raise ValueError("Could not merge Inputs, Targets, and Outcome_Explanation on project_id.")

        self.input_columns = [c for c in self.inputs_df.columns if c != "project_id"]
        self.numeric_columns = [c for c in self.input_columns if pd.api.types.is_numeric_dtype(self.inputs_df[c])]
        self.categorical_columns = [c for c in self.input_columns if c not in self.numeric_columns]

        self.cost_pct_min = float(self.data["cost_overrun_percentage"].min())
        self.cost_pct_max = float(self.data["cost_overrun_percentage"].max())
        self.time_pct_min = float(self.data["schedule_overrun_percentage"].min())
        self.time_pct_max = float(self.data["schedule_overrun_percentage"].max())

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for col in self.input_columns:
            if col not in work.columns:
                if col in self.numeric_columns:
                    work[col] = float(self.inputs_df[col].median())
                else:
                    mode = self.inputs_df[col].mode(dropna=True)
                    work[col] = mode.iloc[0] if not mode.empty else ""

        num = lambda c: pd.to_numeric(work[c], errors="coerce")

        work["planned_unit_cost_check"] = num("planned_cost") / num("gross_floor_area_m2")
        work["schedule_days_per_m2"] = num("planned_duration") / num("gross_floor_area_m2")
        work["size_x_floors"] = num("gross_floor_area_m2") * num("floors_count")
        work["avg_floor_plate_check"] = num("avg_floor_plate_m2") * num("floors_count")
        work["climate_stress_index"] = num("temperature") * 0.4 + num("humidity") * 0.6
        work["resource_pressure"] = num("equipment_utilization") * num("labor_intensity_hr_per_m2")
        work["material_total_proxy"] = num("material_intensity_per_m2") * num("gross_floor_area_m2")

        weather = work["weather_condition"].astype(str).str.strip().str.lower()
        work["weather_severity"] = weather.map({"sunny": 0, "cloudy": 1, "rainy": 2, "snowy": 3, "stormy": 4}).fillna(1)

        complexity = work["complexity_level"].astype(str).str.strip().str.lower()
        work["complexity_score"] = complexity.map({"low": 0, "medium": 1, "high": 2, "very high": 3}).fillna(1)

        seismic = work["seismic_zone_or_pga"].astype(str).str.strip().str.lower()
        work["seismic_score"] = seismic.map({"low": 0, "medium": 1, "high": 2}).fillna(1)

        return work.drop(columns=["project_id"], errors="ignore")

    def build_preprocessor(self):
        example = self.engineer_features(self.inputs_df.copy())
        numeric = [c for c in example.columns if pd.api.types.is_numeric_dtype(example[c])]
        categorical = [c for c in example.columns if c not in numeric]
        return ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", make_one_hot())]), categorical),
        ])

    def train(self, train_df: pd.DataFrame):
        X_raw = train_df[self.inputs_df.columns].copy()
        X_eng = self.engineer_features(X_raw)
        self.preprocessor = self.build_preprocessor()
        X = np.asarray(self.preprocessor.fit_transform(X_eng))

        y_cost_pct = train_df["cost_overrun_percentage"].astype(float).values
        y_time_pct = train_df["schedule_overrun_percentage"].astype(float).values
        y_risk = train_df["risk_level"].astype(str).values

        self.cost_et = ExtraTreesRegressor(n_estimators=120, random_state=42, min_samples_leaf=1, n_jobs=1)
        self.cost_knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
        self.time_et = ExtraTreesRegressor(n_estimators=120, random_state=42, min_samples_leaf=1, n_jobs=1)
        self.time_knn = KNeighborsRegressor(n_neighbors=10, weights="distance")
        self.risk_et = ExtraTreesClassifier(n_estimators=220, random_state=42, min_samples_leaf=1, class_weight="balanced", n_jobs=-1)
        self.risk_rf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=1, class_weight="balanced", n_jobs=1)

        self.cost_et.fit(X, y_cost_pct)
        self.cost_knn.fit(X, y_cost_pct)
        self.time_et.fit(X, y_time_pct)
        self.time_knn.fit(X, y_time_pct)
        self.risk_et.fit(X, y_risk)
        self.risk_rf.fit(X, y_risk)

        risk_ord = pd.Series(y_risk).map({"Low": 0, "Medium": 1, "High": 2}).fillna(1).astype(float).values
        stage2 = np.column_stack([X, y_cost_pct, y_time_pct, risk_ord])

        self.stage2_costrisk = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)
        self.stage2_timerisk = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)
        self.stage2_riskscore = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)

        self.stage2_costrisk.fit(stage2, train_df["cost_risk_score"].astype(float).values)
        self.stage2_timerisk.fit(stage2, train_df["schedule_risk_score"].astype(float).values)
        self.stage2_riskscore.fit(stage2, train_df["risk_score"].astype(float).values)

        self.train_matrix = X
        self.nn = NearestNeighbors(n_neighbors=min(8, len(train_df)), metric="euclidean")
        self.nn.fit(X)
        self.train_ref = train_df.reset_index(drop=True).copy()
        self.explanation_ref = self.explanations_df.set_index("project_id").copy()

    def _blend_regression(self, et_model, knn_model, X):
        return et_model.predict(X)

    @staticmethod
    def scenario_from_pct(value: float) -> str:
        if value <= 0:
            return "Optimistic"
        if value <= 0.10:
            return "Most Likely"
        return "Pessimistic"

    def _normalize_text_input(self, col, raw):
        raw = str(raw).strip()
        if raw == "":
            return raw
        options = [str(x) for x in self.inputs_df[col].dropna().astype(str).unique().tolist()]
        # direct / case-insensitive
        for opt in options:
            if opt.strip().lower() == raw.lower():
                return opt
        # common aliases
        alias = raw.lower().replace("-", " ").replace("_", " ")
        for opt in options:
            if opt.lower().replace("-", " ").replace("_", " ") == alias:
                return opt
        close = difflib.get_close_matches(raw.lower(), [o.lower() for o in options], n=1, cutoff=0.72)
        if close:
            idx = [o.lower() for o in options].index(close[0])
            return options[idx]
        return raw

    def prompt_project(self):
        print("Enter the building project's initial data. Press Enter to accept the suggested value.\n")
        row = {"project_id": f"NEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
        for col in self.input_columns:
            label = col.replace("_", " ").title()
            series = self.inputs_df[col]
            if col in self.numeric_columns:
                vals = pd.to_numeric(series, errors="coerce").dropna()
                default = round(float(vals.median()), 2)
                low = round(float(vals.min()), 2)
                high = round(float(vals.max()), 2)
                int_like = np.allclose(vals, np.round(vals)) if len(vals) else False
                prompt = f"{label} [range {low} to {high}, default {int(default) if int_like else default}]: "
                raw = input(prompt).strip()
                if raw == "":
                    row[col] = int(default) if int_like else default
                else:
                    try:
                        row[col] = int(float(raw)) if int_like else float(raw)
                    except ValueError:
                        print("  Invalid numeric input. Using default.")
                        row[col] = int(default) if int_like else default
            else:
                options = [str(x) for x in series.dropna().astype(str).unique().tolist()]
                mode = str(series.mode(dropna=True).iloc[0]) if not series.mode(dropna=True).empty else options[0]
                raw = input(f"{label} [options: {', '.join(options[:12])}] [default {mode}]: ").strip()
                if raw == "":
                    row[col] = mode
                else:
                    row[col] = self._normalize_text_input(col, raw)
        return pd.DataFrame([row])

    def build_input_warnings(self, raw_df: pd.DataFrame):
        warnings_list = []
        row = raw_df.iloc[0]
        for col in self.numeric_columns:
            val = float(row[col])
            series = pd.to_numeric(self.inputs_df[col], errors="coerce").dropna()
            lo = float(series.min())
            hi = float(series.max())
            if val < lo or val > hi:
                warnings_list.append(f"{col}={val} is outside the training range [{round(lo,2)}, {round(hi,2)}].")
        for col in self.categorical_columns:
            val = str(row[col])
            options = {str(x).strip().lower() for x in self.inputs_df[col].dropna().astype(str).unique().tolist()}
            if val.strip().lower() not in options:
                warnings_list.append(f"{col}='{val}' is unusual compared with the training data.")
        return warnings_list

    def _risk_probs(self, X):
        labels = list(self.risk_et.classes_)
        p1 = self.risk_et.predict_proba(X)
        # Align rf probs to same label order
        rf_labels = list(self.risk_rf.classes_)
        p2_raw = self.risk_rf.predict_proba(X)
        p2 = np.zeros_like(p1)
        for i, label in enumerate(labels):
            if label in rf_labels:
                p2[:, i] = p2_raw[:, rf_labels.index(label)]
        probs = 0.6 * p1 + 0.4 * p2
        return labels, probs

    def _driver_flags(self, row):
        flags = []
        def add(cond, text):
            if cond:
                flags.append(text)

        add(float(row["planned_cost"]) >= float(self.inputs_df["planned_cost"].quantile(0.75)), "Large planned budget increases procurement and coordination pressure.")
        add(float(row["planned_duration"]) >= float(self.inputs_df["planned_duration"].quantile(0.75)), "Long planned duration increases exposure to change and disruption.")
        add(str(row["complexity_level"]).lower() in {"high", "very high"}, f"Complexity is {row['complexity_level']}, which raises management and interface risk.")
        add(str(row["building_use_type"]).lower() in {"hospital", "mixed-use"}, f"{row['building_use_type']} projects usually require tighter coordination across systems and stakeholders.")
        add(str(row["structural_system"]).lower() in {"steel", "composite"}, f"{row['structural_system']} structural work can increase sequencing and procurement sensitivity.")
        add(str(row["foundation_type"]).lower() in {"piles", "caisson"}, f"{row['foundation_type']} foundations point to deeper ground/interface complexity.")
        add(str(row["seismic_zone_or_pga"]).lower() == "high", "High seismic demand can increase detailing and quality-control pressure.")
        add(str(row["weather_condition"]).lower() in {"rainy", "snowy", "stormy"}, f"{row['weather_condition']} conditions can reduce site productivity and increase disruption risk.")
        add(float(row["humidity"]) >= float(self.inputs_df["humidity"].quantile(0.75)), "High humidity can affect site productivity and finishing windows.")
        add(float(row["air_quality_index"]) >= float(self.inputs_df["air_quality_index"].quantile(0.75)), "Poor air quality can reduce field efficiency and health-related productivity.")
        add(float(row["equipment_utilization"]) >= float(self.inputs_df["equipment_utilization"].quantile(0.75)), "High equipment utilization suggests limited slack in resources.")
        add(float(row["material_intensity_per_m2"]) >= float(self.inputs_df["material_intensity_per_m2"].quantile(0.75)), "High material intensity increases supply-chain and cost sensitivity.")
        return flags[:6]

    def _similar_case_causes(self, X):
        dists, idxs = self.nn.kneighbors(X)
        candidates = self.train_ref.iloc[idxs[0]].copy()
        def top_mode(col):
            vals = candidates[col].dropna().astype(str).tolist()
            if not vals:
                return ""
            return pd.Series(vals).mode().iloc[0]
        return {
            "cost_primary": top_mode("cost_overrun_primary_cause"),
            "cost_secondary": top_mode("cost_overrun_secondary_cause"),
            "schedule_primary": top_mode("schedule_delay_primary_cause"),
            "schedule_secondary": top_mode("schedule_delay_secondary_cause"),
            "similar_cases": [
                {
                    "project_id": str(r["project_id"]),
                    "risk_level": str(r["risk_level"]),
                    "actual_cost": float(r["actual_cost"]),
                    "actual_duration": float(r["actual_duration"]),
                }
                for _, r in candidates.head(3).iterrows()
            ],
        }

    def predict(self, raw_df: pd.DataFrame):
        X_eng = self.engineer_features(raw_df[self.inputs_df.columns].copy())
        X = np.asarray(self.preprocessor.transform(X_eng))

        cost_pct = float(self._blend_regression(self.cost_et, self.cost_knn, X)[0])
        time_pct = float(self._blend_regression(self.time_et, self.time_knn, X)[0])
        cost_pct = float(np.clip(cost_pct, self.cost_pct_min, self.cost_pct_max))
        time_pct = float(np.clip(time_pct, self.time_pct_min, self.time_pct_max))

        labels, probs = self._risk_probs(X)
        risk_idx = int(np.argmax(probs[0]))
        risk_level = labels[risk_idx]
        risk_ord = {"Low": 0, "Medium": 1, "High": 2}.get(risk_level, 1)

        stage2 = np.column_stack([X, [cost_pct], [time_pct], [risk_ord]])
        risk_score = float(np.clip(self.stage2_riskscore.predict(stage2)[0], 0, 100))
        cost_risk_score = float(np.clip(self.stage2_costrisk.predict(stage2)[0], 0, 100))
        schedule_risk_score = float(np.clip(self.stage2_timerisk.predict(stage2)[0], 0, 100))

        planned_cost = float(raw_df.iloc[0]["planned_cost"])
        planned_duration = float(raw_df.iloc[0]["planned_duration"])
        actual_cost = planned_cost * (1.0 + cost_pct)
        actual_duration = planned_duration * (1.0 + time_pct)

        cost_overrun = actual_cost - planned_cost
        schedule_dev = actual_duration - planned_duration

        cause_block = self._similar_case_causes(X)
        flags = self._driver_flags(raw_df.iloc[0])
        summary = (
            f"Predicted cost pressure is mainly linked to {cause_block['cost_primary'].lower()} and "
            f"{cause_block['cost_secondary'].lower()}. Predicted time pressure is mainly linked to "
            f"{cause_block['schedule_primary'].lower()} and {cause_block['schedule_secondary'].lower()}."
        )

        return {
            "project_id": str(raw_df.iloc[0]["project_id"]),
            "predicted_outputs": {
                "actual_cost": actual_cost,
                "cost_overrun": cost_overrun,
                "cost_overrun_percentage": cost_pct,
                "cost_scenario": self.scenario_from_pct(cost_pct),
                "actual_duration": actual_duration,
                "schedule_deviation": schedule_dev,
                "schedule_overrun_percentage": time_pct,
                "time_scenario": self.scenario_from_pct(time_pct),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "cost_risk_score": cost_risk_score,
                "schedule_risk_score": schedule_risk_score,
            },
            "risk_probabilities": {str(labels[i]): float(probs[0][i]) for i in range(len(labels))},
            "input_warnings": self.build_input_warnings(raw_df),
            "cost_primary_cause": cause_block["cost_primary"],
            "cost_secondary_cause": cause_block["cost_secondary"],
            "schedule_primary_cause": cause_block["schedule_primary"],
            "schedule_secondary_cause": cause_block["schedule_secondary"],
            "result_reason_summary": summary,
            "input_driven_flags": flags,
            "similar_cases": cause_block["similar_cases"],
        }

    def evaluate_last10(self):
        if len(self.data) < 25:
            print("Not enough rows for last-10 holdout.")
            self.train(self.data.copy())
            return None

        train_df = self.data.iloc[:-10].copy()
        test_df = self.data.iloc[-10:].copy()
        self.train(train_df)

        preds = []
        for _, row in test_df[self.inputs_df.columns].iterrows():
            pred = self.predict(pd.DataFrame([row]))
            preds.append(pred["predicted_outputs"])

        pred_df = pd.DataFrame(preds)
        report = pd.DataFrame({
            "project_id": test_df["project_id"].values,
            "actual_risk_level": test_df["risk_level"].values,
            "predicted_risk_level": pred_df["risk_level"].values,
            "actual_actual_cost": test_df["actual_cost"].values,
            "predicted_actual_cost": pred_df["actual_cost"].values,
            "actual_actual_duration": test_df["actual_duration"].values,
            "predicted_actual_duration": pred_df["actual_duration"].values,
            "actual_cost_overrun_percentage": test_df["cost_overrun_percentage"].values,
            "predicted_cost_overrun_percentage": pred_df["cost_overrun_percentage"].values,
            "actual_schedule_overrun_percentage": test_df["schedule_overrun_percentage"].values,
            "predicted_schedule_overrun_percentage": pred_df["schedule_overrun_percentage"].values,
            "actual_completion_placeholder": [np.nan]*len(test_df),  # building file has no completion %
        })

        metrics = {
            "risk_accuracy": float(accuracy_score(test_df["risk_level"], pred_df["risk_level"])),
            "risk_macro_f1": float(f1_score(test_df["risk_level"], pred_df["risk_level"], average="macro")),
            "actual_cost_mae": float(mean_absolute_error(test_df["actual_cost"], pred_df["actual_cost"])),
            "actual_cost_mape": float(mean_absolute_percentage_error(test_df["actual_cost"], pred_df["actual_cost"])),
            "actual_duration_mae": float(mean_absolute_error(test_df["actual_duration"], pred_df["actual_duration"])),
            "cost_overrun_pct_mae": float(mean_absolute_error(test_df["cost_overrun_percentage"], pred_df["cost_overrun_percentage"])),
            "schedule_overrun_pct_mae": float(mean_absolute_error(test_df["schedule_overrun_percentage"], pred_df["schedule_overrun_percentage"])),
        }

        csv_path = os.path.join(self.output_dir, "holdout_last10_predictions_building.csv")
        report.to_csv(csv_path, index=False)

        print("\n" + "=" * 80)
        print("LAST 10 ROWS HOLDOUT CHECK - BUILDING")
        print("=" * 80)
        print(f"Risk accuracy: {metrics['risk_accuracy']:.3f}")
        print(f"Risk macro F1: {metrics['risk_macro_f1']:.3f}")
        print(f"Actual cost MAE: {metrics['actual_cost_mae']:,.3f}")
        print(f"Actual cost MAPE: {metrics['actual_cost_mape']:.3%}")
        print(f"Actual duration MAE: {metrics['actual_duration_mae']:,.3f} days")
        print(f"Cost overrun percentage MAE: {metrics['cost_overrun_pct_mae']:.4f}")
        print(f"Schedule overrun percentage MAE: {metrics['schedule_overrun_pct_mae']:.4f}")
        print(f"Saved holdout comparison to: {csv_path}")
        print("=" * 80 + "\n")

        # Retrain on full dataset for real use
        self.train(self.data.copy())
        return metrics

    def save_prediction_report(self, raw_df: pd.DataFrame, result: dict):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(self.output_dir, f"building_prediction_report_{stamp}.txt")
        json_path = os.path.join(self.output_dir, f"building_prediction_report_{stamp}.json")

        out = result["predicted_outputs"]
        lines = []
        lines.append("BUILDING PROJECT PREDICTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Project ID: {result['project_id']}")
        lines.append("")
        lines.append("Predicted outputs")
        lines.append("-" * 80)
        ordered = [
            "actual_cost", "actual_duration", "cost_overrun_percentage", "schedule_overrun_percentage",
            "cost_scenario", "time_scenario", "cost_overrun", "schedule_deviation",
            "risk_level", "risk_score", "cost_risk_score", "schedule_risk_score"
        ]
        for k in ordered:
            v = out[k]
            if isinstance(v, float):
                if "percentage" in k:
                    lines.append(f"{k}: {v:.4f}")
                else:
                    lines.append(f"{k}: {v:,.3f}")
            else:
                lines.append(f"{k}: {v}")
        lines.append("")
        if result["input_warnings"]:
            lines.append("Input warnings")
            lines.append("-" * 80)
            for item in result["input_warnings"]:
                lines.append(f"- {item}")
            lines.append("")
        lines.append("Likely explanation")
        lines.append("-" * 80)
        lines.append(f"Cost primary cause: {result['cost_primary_cause']}")
        lines.append(f"Cost secondary cause: {result['cost_secondary_cause']}")
        lines.append(f"Schedule primary cause: {result['schedule_primary_cause']}")
        lines.append(f"Schedule secondary cause: {result['schedule_secondary_cause']}")
        lines.append(f"Summary: {result['result_reason_summary']}")
        lines.append("")
        lines.append("Input-driven flags")
        lines.append("-" * 80)
        for item in result["input_driven_flags"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Similar historical cases")
        lines.append("-" * 80)
        for case in result["similar_cases"]:
            lines.append(f"{case['project_id']} | risk={case['risk_level']} | actual_cost={case['actual_cost']:,.0f} | actual_duration={case['actual_duration']:.1f}")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"input": raw_df.iloc[0].to_dict(), "prediction": result}, f, indent=2, ensure_ascii=False)
        return txt_path

def print_prediction(result: dict):
    out = result["predicted_outputs"]
    print("\n" + "=" * 80)
    print("PREDICTION - BUILDING")
    print("=" * 80)
    print(f"Predicted actual cost: {out['actual_cost']:,.3f}")
    print(f"Predicted actual duration: {out['actual_duration']:,.3f}")
    print(f"Predicted cost overrun percentage: {out['cost_overrun_percentage']:.4f}")
    print(f"Predicted schedule overrun percentage: {out['schedule_overrun_percentage']:.4f}")
    print(f"Predicted Cost_Scenario: {out['cost_scenario']}")
    print(f"Predicted Time_Scenario: {out['time_scenario']}")
    print(f"Predicted risk level: {out['risk_level']}")
    print(f"Predicted risk score: {out['risk_score']:.2f}")
    print(f"Predicted cost risk score: {out['cost_risk_score']:.2f}")
    print(f"Predicted schedule risk score: {out['schedule_risk_score']:.2f}")

    if result["risk_probabilities"]:
        print("\nRisk probabilities:")
        for k, v in sorted(result["risk_probabilities"].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {k}: {v:.3f}")

    if result["input_warnings"]:
        print("\nInput warnings:")
        for item in result["input_warnings"]:
            print(f"- {item}")

    print("\nMost likely reasons:")
    print(textwrap.fill("Cost primary cause: " + result["cost_primary_cause"], width=100))
    print(textwrap.fill("Cost secondary cause: " + result["cost_secondary_cause"], width=100))
    print(textwrap.fill("Schedule primary cause: " + result["schedule_primary_cause"], width=100))
    print(textwrap.fill("Schedule secondary cause: " + result["schedule_secondary_cause"], width=100))
    print(textwrap.fill("Summary: " + result["result_reason_summary"], width=100))

    print("\nInput-driven flags:")
    for item in result["input_driven_flags"]:
        print(textwrap.fill("- " + item, width=100))

    print("\nClosest similar historical projects:")
    for case in result["similar_cases"]:
        print(f"  - {case['project_id']} | risk={case['risk_level']} | actual_cost={case['actual_cost']:,.0f} | actual_duration={case['actual_duration']:.1f}")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Building project risk prediction system")
    parser.add_argument("--excel-path", default=DEFAULT_EXCEL_PATH, help="Path to workbook")
    args = parser.parse_args()

    print("\nA Data-Driven Integrated Risk Intelligence Framework for Predictive and Sustainable Construction Project Management")
    print("Building project prediction system\n")

    system = BuildingPredictor(args.excel_path)
    system.load_data()

    answer = input("Run the last-10-rows holdout check first? [Y/n]: ").strip().lower()
    if answer in {"", "y", "yes"}:
        system.evaluate_last10()
    else:
        system.train(system.data.copy())

    while True:
        raw_df = system.prompt_project()
        result = system.predict(raw_df)
        print_prediction(result)
        report_path = system.save_prediction_report(raw_df, result)
        print(f"Prediction report saved to: {report_path}\n")
        again = input("Predict another building project? [y/N]: ").strip().lower()
        if again not in {"y", "yes"}:
            print("Finished.")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
