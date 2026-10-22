"""Illustrative classroom data, not measured claims about real programmes."""
import pandas as pd


def sample_criteria():
    return pd.DataFrame([
        {"Criterion": "Total cost", "Type": "cost", "Weight": 0.20, "Unit": "KZT"},
        {"Criterion": "Weekly workload", "Type": "cost", "Weight": 0.15, "Unit": "hours/week"},
        {"Criterion": "Interest fit", "Type": "benefit", "Weight": 0.20, "Unit": "1-10"},
        {"Criterion": "Career fit", "Type": "benefit", "Weight": 0.25, "Unit": "1-10"},
        {"Criterion": "Project quality", "Type": "benefit", "Weight": 0.20, "Unit": "1-10"},
    ])


def sample_matrix():
    return pd.DataFrame([
        ["Data Science", 180000, 10, 9, 9, 8],
        ["AI / ML", 250000, 12, 9, 10, 9],
        ["Cybersecurity", 150000, 8, 7, 7, 8],
        ["Software Engineering", 120000, 10, 8, 8, 10],
        ["Cloud / DevOps", 100000, 6, 6, 7, 7],
    ], columns=["Alternative", "Total cost", "Weekly workload", "Interest fit", "Career fit", "Project quality"])
