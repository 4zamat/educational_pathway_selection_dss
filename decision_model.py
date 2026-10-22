"""Transparent WSM and TOPSIS calculations independent of the interface."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Input data cannot be used safely for a ranking."""


@dataclass
class DecisionResult:
    matrix: pd.DataFrame
    criteria: pd.DataFrame
    wsm_normalized: pd.DataFrame
    wsm_contributions: pd.DataFrame
    topsis_normalized: pd.DataFrame
    topsis_weighted: pd.DataFrame
    ideals: pd.DataFrame
    comparison: pd.DataFrame


def _names(series, label):
    if series.isna().any() or series.astype(str).str.strip().eq("").any():
        raise ValidationError(f"Every {label} needs a non-empty name.")
    cleaned = series.astype(str).str.strip()
    if cleaned.str.casefold().duplicated().any():
        raise ValidationError(f"{label.capitalize()} names must be unique (ignoring case).")
    return cleaned


def validate_inputs(matrix, criteria, min_alternatives=5, min_criteria=5):
    """Validate, copy and return canonical inputs; never mutate caller data."""
    if not isinstance(matrix, pd.DataFrame) or not isinstance(criteria, pd.DataFrame):
        raise ValidationError("The matrix and criteria must be tables.")
    if len(matrix) < min_alternatives or len(criteria) < min_criteria:
        raise ValidationError(f"Use at least {min_alternatives} alternatives and {min_criteria} criteria.")
    if not {"Criterion", "Type", "Weight", "Unit"}.issubset(criteria.columns):
        raise ValidationError("Criteria need Criterion, Type, Weight and Unit columns.")
    if "Alternative" not in matrix.columns:
        raise ValidationError("The decision matrix needs an Alternative column.")
    if matrix.columns.duplicated().any() or criteria.columns.duplicated().any():
        raise ValidationError("Table columns must be unique.")
    criteria = criteria.copy().reset_index(drop=True)
    matrix = matrix.copy().reset_index(drop=True)
    criteria["Criterion"] = _names(criteria["Criterion"], "criterion")
    if criteria["Criterion"].str.casefold().eq("alternative").any():
        raise ValidationError("'Alternative' is reserved for the pathway name column.")
    matrix["Alternative"] = _names(matrix["Alternative"], "alternative")
    criteria["Type"] = criteria["Type"].astype(str).str.strip().str.lower()
    if not criteria["Type"].isin(["benefit", "cost"]).all():
        raise ValidationError("Each criterion type must be benefit or cost.")
    if criteria["Unit"].isna().any() or criteria["Unit"].astype(str).str.strip().eq("").any():
        raise ValidationError("Give every criterion a unit, such as KZT, hours/week or 1-10.")
    criteria["Unit"] = criteria["Unit"].astype(str).str.strip()
    weights = pd.to_numeric(criteria["Weight"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValidationError("Weights must be finite, numeric and non-negative.")
    if not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValidationError(f"Weights must sum to 1.0; their current sum is {weights.sum():.6f}.")
    criteria["Weight"] = weights
    names = criteria["Criterion"].tolist()
    if set(matrix.columns) != {"Alternative", *names}:
        raise ValidationError("Matrix columns must match the criterion names exactly. Apply the criteria table first.")
    values = matrix[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValidationError("Fill every matrix cell with a finite number; missing or non-numeric values are not allowed.")
    if (values < 0).any():
        raise ValidationError("This educational model accepts non-negative criterion values only.")
    for index, unit in enumerate(criteria["Unit"]):
        if unit.replace(" ", "") == "1-10" and ((values[:, index] < 1).any() or (values[:, index] > 10).any()):
            raise ValidationError(f"'{names[index]}' uses the 1-10 scale; enter values from 1 to 10.")
    matrix = pd.DataFrame(values, columns=names).assign(Alternative=matrix["Alternative"])
    return matrix[["Alternative", *names]], criteria


def evaluate(matrix, criteria, min_alternatives=5, min_criteria=5):
    matrix, criteria = validate_inputs(matrix, criteria, min_alternatives, min_criteria)
    names = criteria["Criterion"].tolist()
    labels = matrix["Alternative"].tolist()
    values = matrix[names].to_numpy(dtype=float)
    weights = criteria["Weight"].to_numpy(dtype=float)
    benefit = criteria["Type"].eq("benefit").to_numpy()

    # Min-max WSM: all columns have the same higher-is-better interpretation.
    lower, upper = values.min(axis=0), values.max(axis=0)
    span = upper - lower
    varying = span > 0
    normalized = np.ones_like(values)
    normalized[:, varying] = (values[:, varying] - lower[varying]) / span[varying]
    normalized[:, ~benefit & varying] = 1 - normalized[:, ~benefit & varying]
    contributions = normalized * weights
    wsm = contributions.sum(axis=1)

    # Scaling first avoids overflow when squaring unusually large input values.
    scales = values.max(axis=0)
    scaled = np.divide(values, scales, out=np.zeros_like(values), where=scales != 0)
    norms = np.sqrt((scaled ** 2).sum(axis=0))
    vector_normalized = np.divide(scaled, norms, out=np.zeros_like(values), where=norms != 0)
    weighted = vector_normalized * weights
    positive = np.where(benefit, weighted.max(axis=0), weighted.min(axis=0))
    negative = np.where(benefit, weighted.min(axis=0), weighted.max(axis=0))
    distance_best = np.sqrt(((weighted - positive) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted - negative) ** 2).sum(axis=1))
    denominator = distance_best + distance_worst
    closeness = np.divide(distance_worst, denominator, out=np.full(len(values), .5), where=denominator != 0)

    comparison = pd.DataFrame({
        "Alternative": labels,
        "WSM score": wsm,
        "WSM rank": pd.Series(np.round(wsm, 12)).rank(method="min", ascending=False).astype(int),
        "TOPSIS score": closeness,
        "TOPSIS rank": pd.Series(np.round(closeness, 12)).rank(method="min", ascending=False).astype(int),
        "Distance to best": distance_best,
        "Distance to worst": distance_worst,
    })
    comparison["Rank difference"] = comparison["WSM rank"] - comparison["TOPSIS rank"]
    to_frame = lambda a: pd.DataFrame(a, index=pd.Index(labels, name="Alternative"), columns=names)
    return DecisionResult(
        matrix, criteria, to_frame(normalized), to_frame(contributions),
        to_frame(vector_normalized), to_frame(weighted),
        pd.DataFrame([positive, negative], index=["Ideal best", "Ideal worst"], columns=names),
        comparison,
    )


def explain_comparison(result):
    table = result.comparison
    wsm = table.loc[table["WSM rank"].eq(1), "Alternative"].tolist()
    topsis = table.loc[table["TOPSIS rank"].eq(1), "Alternative"].tolist()
    if set(wsm) == set(topsis):
        first = "Both methods select " + ", ".join(wsm) + " as the top-ranked option" + ("s." if len(wsm) > 1 else ".")
    else:
        first = "WSM selects " + ", ".join(wsm) + "; TOPSIS selects " + ", ".join(topsis) + "."
    changed = table.loc[table["Rank difference"].ne(0), "Alternative"].tolist()
    second = (" The rankings differ for " + ", ".join(changed) + "." if changed else " The full rankings agree.")
    return first + second + " WSM adds weighted min-max scores; TOPSIS uses weighted vector-normalized distances to ideal best and worst options. Their score scales should not be interpreted as equivalent probabilities."
