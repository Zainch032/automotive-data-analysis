"""Insights, notebook problems, and a dollar price model for Car.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).resolve().parent / "Car.csv"

NUMERIC = [
    "Year",
    "Engine HP",
    "Engine Cylinders",
    "Number of Doors",
    "highway MPG",
    "city mpg",
]
CATEGORICAL = [
    "Make",
    "Transmission Type",
    "Driven_Wheels",
    "Vehicle Size",
    "Vehicle Style",
    "Engine Fuel Type",
    "Market Tag",
]
FEATURES = NUMERIC + CATEGORICAL

PRETTY = {
    "Engine HP": "Horsepower",
    "Year": "Model year",
    "Engine Cylinders": "Cylinders",
    "highway MPG": "Highway MPG",
    "city mpg": "City MPG",
    "Number of Doors": "Doors",
    "Make": "Brand",
    "Transmission Type": "Transmission",
    "Driven_Wheels": "Drivetrain",
    "Vehicle Size": "Size",
    "Vehicle Style": "Body style",
    "Engine Fuel Type": "Fuel",
    "Market Tag": "Market tag",
}

NOTEBOOK_SCORES = [
    {"name": "Random Forest", "r2": 0.9741, "mae": 0.0515},
    {"name": "Decision Tree", "r2": 0.9586, "mae": 0.0546},
    {"name": "K Neighbors", "r2": 0.9273, "mae": 0.0767},
    {"name": "Ridge", "r2": 0.5846, "mae": 0.3308},
    {"name": "Linear Regression", "r2": 0.5846, "mae": 0.3309},
]

_STATE: dict | None = None


def money(value: float) -> str:
    return f"${value:,.0f}"


def _prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    work["Market Tag"] = (
        work["Market Category"].fillna("Unknown").astype(str).str.split(",").str[0]
    )
    work["Engine Fuel Type"] = work["Engine Fuel Type"].fillna("unknown")
    # One Audi A6 is listed at 354 highway MPG. Leave it missing so the
    # training imputer can replace it from the training fold only.
    work.loc[work["highway MPG"] > 150, "highway MPG"] = np.nan
    priced = work["MSRP"] > 2000
    return work.loc[priced].reset_index(drop=True)


def _make_pipeline(model) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ]
    )
    return Pipeline([("prep", prep), ("model", model)])


def _candidates() -> list[tuple[str, object]]:
    return [
        ("Linear Regression", LinearRegression()),
        ("Ridge", Ridge(random_state=42)),
        (
            "Decision Tree",
            DecisionTreeRegressor(max_depth=16, random_state=42),
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=80,
                max_depth=16,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ("K Neighbors", KNeighborsRegressor(n_neighbors=5)),
    ]


def _feature_importance(pipeline: Pipeline) -> list[dict]:
    model = pipeline.named_steps["model"]
    encoder = pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    scores = model.feature_importances_
    grouped: dict[str, float] = {}
    offset = 0
    for column in NUMERIC:
        grouped[column] = float(scores[offset])
        offset += 1
    for column, categories in zip(CATEGORICAL, encoder.categories_):
        width = len(categories)
        grouped[column] = float(scores[offset : offset + width].sum())
        offset += width
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    top = ranked[:6]
    peak = top[0][1] or 1.0
    return [
        {
            "label": PRETTY.get(name, name),
            "share": round(value * 100, 1),
            "width": round(value / peak * 100, 1),
        }
        for name, value in top
    ]


def _options(frame: pd.DataFrame) -> dict:
    fuel_counts = frame["Engine Fuel Type"].value_counts()
    fuels = sorted(fuel_counts[fuel_counts >= 20].index.tolist())
    return {
        "makes": sorted(frame["Make"].unique().tolist()),
        "fuels": fuels,
        "transmissions": sorted(frame["Transmission Type"].unique().tolist()),
        "drives": sorted(frame["Driven_Wheels"].unique().tolist()),
        "sizes": ["Compact", "Midsize", "Large"],
        "styles": sorted(frame["Vehicle Style"].unique().tolist()),
        "markets": sorted(frame["Market Tag"].unique().tolist()),
        "cylinders": [0, 3, 4, 5, 6, 8, 10, 12, 16],
        "doors": [2, 3, 4],
    }


def build_state() -> dict:
    raw = pd.read_csv(DATA_PATH)
    clean = _prepare_frame(raw)

    two_door = raw[raw["Number of Doors"] == 2]
    two_real = two_door[two_door["MSRP"] > 2000]
    premium = two_real[two_real["MSRP"] > 100_000]
    brand_counts = premium["Make"].value_counts().head(5)
    brand_peak = int(brand_counts.max())
    brands = [
        {
            "name": name,
            "count": int(count),
            "width": round(count / brand_peak * 100, 1),
        }
        for name, count in brand_counts.items()
    ]

    transmission = raw["Transmission Type"].value_counts()
    transmission_rows = [
        {
            "name": name.replace("_", " ").title(),
            "count": int(count),
            "pct": round(count / len(raw) * 100, 1),
        }
        for name, count in transmission.items()
    ]

    missing = (
        (raw.isnull().mean() * 100)
        .sort_values(ascending=False)
        .loc[lambda series: series > 0]
        .round(2)
    )

    features = clean[FEATURES]
    target = clean["MSRP"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    leaderboard = []
    fitted = {}
    for name, estimator in _candidates():
        pipeline = _make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        leaderboard.append(
            {
                "name": name,
                "r2": round(float(r2_score(y_test, predicted)), 3),
                "mae": int(round(float(mean_absolute_error(y_test, predicted)))),
            }
        )
        fitted[name] = pipeline

    leaderboard.sort(key=lambda row: row["mae"])
    best_name = "Random Forest"
    deployed = _make_pipeline(dict(_candidates())[best_name])
    deployed.fit(features, target)

    holdout = next(row for row in leaderboard if row["name"] == best_name)
    importance = _feature_importance(fitted[best_name])

    placeholder_count = int((raw["MSRP"] == 2000).sum())
    market_missing = int(raw["Market Category"].isna().sum())

    insights = [
        {
            "kicker": "The data",
            "title": f"{len(raw):,} cars, years {int(raw['Year'].min())}–{int(raw['Year'].max())}",
            "body": (
                f"{raw['Make'].nunique()} brands and {raw['Model'].nunique()} model names. "
                f"Most vehicles are 4-door ({int((raw['Number of Doors'] == 4).sum()):,}). "
                f"2-door cars are the smaller group: {len(two_door):,}."
            ),
        },
        {
            "kicker": "2-door prices",
            "title": f"A real 2-door average is {money(two_real['MSRP'].mean())}",
            "body": (
                f"The notebook printed min {money(two_door['MSRP'].min())}, "
                f"max {money(two_door['MSRP'].max())}, average {money(two_door['MSRP'].mean())}. "
                f"After dropping the {int((two_door['MSRP'] == 2000).sum()):,} fake $2,000 rows, "
                f"the 2-door average is {money(two_real['MSRP'].mean())} and the minimum is "
                f"{money(two_real['MSRP'].min())}. The maximum is still the Bugatti Veyron "
                f"at {money(two_real['MSRP'].max())}."
            ),
        },
        {
            "kicker": "Premium 2-door cars",
            "title": f"{len(premium):,} cars priced over $100,000",
            "body": (
                "Counted from the data, the five brands with the most of these cars are "
                + ", ".join(f"{row['name']} ({row['count']})" for row in brands)
                + ". The README lists BMW in that group. BMW is not in the top five."
            ),
        },
        {
            "kicker": "What moves the price",
            "title": f"{importance[0]['label']} is the strongest signal in the corrected model",
            "body": (
                "On a holdout set, with prices left in dollars, "
                + ", ".join(f"{row['label']} ({row['share']}%)" for row in importance[:3])
                + " take the largest shares of the forest. "
                "Tags such as Exotic and Luxury already hint at price, so that top signal is partly circular. "
                "Horsepower is the strongest spec that is not a label."
            ),
        },
    ]

    problems = [
        {
            "level": "High",
            "title": "The error is not in dollars",
            "body": (
                "Random Forest won the notebook with R² 0.974 and MAE 0.0515. "
                "StandardScaler was fit on the whole table, including Price, before the split. "
                "That MAE is a scaled number. The notebook never turns a prediction back into dollars, "
                "so it cannot say what a car should cost. The README’s MAE of $2,500 does not match the printed 0.0515."
            ),
        },
        {
            "level": "High",
            "title": f"{placeholder_count:,} prices are a $2,000 placeholder",
            "body": (
                f"{placeholder_count / len(raw) * 100:.1f}% of rows have MSRP exactly $2,000. "
                "In this dataset that means the price is missing. Those rows set the 2-door minimum "
                "at $2,000 and pull the average down. The predictor on this page drops them."
            ),
        },
        {
            "level": "High",
            "title": f"{market_missing:,} cars were labeled Crossover by mistake",
            "body": (
                "Market Category is empty for 31.4% of the file. Filling the blanks with the mode "
                "wrote “Crossover” into every one of those rows. After that, Crossover becomes the biggest category."
            ),
        },
        {
            "level": "High",
            "title": "The test rows helped train the model",
            "body": (
                "Mean fills, mode fills, LabelEncoder, and StandardScaler all saw every row, "
                "and only then was the data split. The test set leaked into the inputs, so the R² is a bit too kind."
            ),
        },
        {
            "level": "High",
            "title": "Model name lets a tree memorize the price",
            "body": (
                f"There are {raw['Model'].nunique()} model names. LabelEncoder turns them into integers, "
                "and a tree can split on that number until it almost remembers each model’s price. "
                "A new name has no honest code. This page leaves Model out so the estimate is about the car’s specs."
            ),
        },
        {
            "level": "Medium",
            "title": "LabelEncoder pretends brands have a rank",
            "body": (
                "Make, fuel, style, and the other text columns became integers. "
                "Linear Regression and KNN treat a higher code as a bigger value. Brands are not ordered that way. "
                "Trees shrug that off, which is one reason they looked so much better."
            ),
        },
        {
            "level": "Medium",
            "title": "Missing doors were filled with 3.44",
            "body": (
                "Six missing door counts were replaced with the column mean, about 3.44. "
                "That is not a door count. The median is 4, which is a real value. "
                "The same mean was computed before the train/test split."
            ),
        },
        {
            "level": "Medium",
            "title": "Two charts argue the wrong thing",
            "body": (
                "The premium bar chart locks the y-axis from 20 to 100, so the bar heights are cut. "
                "The R² pie chart treats three scores as slices of one pie. R² is not a share of a total. "
                "The transmission pie keeps only the top 3 types, so its percents are not shares of every car."
            ),
        },
        {
            "level": "Low",
            "title": "One Audi A6 claims 354 highway MPG",
            "body": (
                "The 2017 Audi A6 row has highway MPG 354 and city MPG 24. "
                "The highway figure is a bad value, and the notebook left it in. "
                "Electric cars above 100 MPG are real; this one is not electric."
            ),
        },
    ]

    return {
        "pipeline": deployed,
        "catalog": clean[["Make", "Model", "Year", "Engine HP", "MSRP", "Vehicle Style"]].copy(),
        "options": _options(clean),
        "page": {
            "cars": f"{len(raw):,}",
            "two_door_avg": money(two_real["MSRP"].mean()),
            "best_mae": money(holdout["mae"]),
            "best_r2": f"{holdout['r2']:.3f}",
            "placeholder_count": f"{placeholder_count:,}",
            "insights": insights,
            "problems": problems,
            "brands": brands,
            "transmission": transmission_rows,
            "missing": [
                {"name": name, "pct": float(pct)} for name, pct in missing.items()
            ],
            "notebook_scores": NOTEBOOK_SCORES,
            "corrected_scores": leaderboard,
            "importance": importance,
            "trained_on": f"{len(clean):,}",
            "holdout_mae": holdout["mae"],
        },
    }


def get_state() -> dict:
    global _STATE
    if _STATE is None:
        _STATE = build_state()
    return _STATE


def _similar(catalog: pd.DataFrame, make: str, year: int, horsepower: float) -> list[dict]:
    pool = catalog[catalog["Make"] == make]
    if pool.empty:
        pool = catalog
    scored = pool.copy()
    scored["distance"] = (scored["Engine HP"].fillna(horsepower) - horsepower).abs() + (
        scored["Year"] - year
    ).abs() * 15
    nearest = scored.nsmallest(3, "distance")
    rows = []
    for _, row in nearest.iterrows():
        rows.append(
            {
                "name": f"{int(row['Year'])} {row['Make']} {row['Model']}",
                "style": row["Vehicle Style"],
                "hp": "—" if pd.isna(row["Engine HP"]) else f"{int(row['Engine HP'])} hp",
                "price": money(float(row["MSRP"])),
            }
        )
    return rows


def predict(payload: dict) -> dict:
    state = get_state()
    try:
        year = int(payload["year"])
        horsepower = float(payload["engine_hp"])
        cylinders = float(payload["cylinders"])
        doors = float(payload["doors"])
        highway = float(payload["highway_mpg"])
        city = float(payload["city_mpg"])
        row = {
            "Year": year,
            "Engine HP": horsepower,
            "Engine Cylinders": cylinders,
            "Number of Doors": doors,
            "highway MPG": highway,
            "city mpg": city,
            "Make": str(payload["make"]),
            "Transmission Type": str(payload["transmission"]),
            "Driven_Wheels": str(payload["drive"]),
            "Vehicle Size": str(payload["size"]),
            "Vehicle Style": str(payload["style"]),
            "Engine Fuel Type": str(payload["fuel"]),
            "Market Tag": str(payload["market"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Enter a complete car before estimating.") from exc

    frame = pd.DataFrame([row], columns=FEATURES)
    estimate = float(state["pipeline"].predict(frame)[0])
    estimate = max(estimate, 0.0)
    mae = state["page"]["holdout_mae"]
    warnings = []
    if year < 1990 or year > 2017:
        warnings.append("This year is outside the file (1990–2017), so the estimate is weaker.")
    if horsepower < 55 or horsepower > 1001:
        warnings.append("Horsepower is outside the cars in the file (55–1,001).")
    if str(payload["fuel"]) != "electric" and highway > 80:
        warnings.append("Highway MPG looks too high for a non-electric car.")
    if city > highway and str(payload["fuel"]) != "electric":
        warnings.append("City MPG is higher than highway MPG, which is unusual for a gas car.")

    return {
        "price_label": money(estimate),
        "low_label": money(max(estimate - mae, 0)),
        "high_label": money(estimate + mae),
        "mae_label": money(mae),
        "warnings": warnings,
        "similar": _similar(state["catalog"], row["Make"], year, horsepower),
    }
