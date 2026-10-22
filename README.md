# Educational Pathway Selection DSS

A small Python and Streamlit decision support application. It compares educational pathways using the Weighted Sum Model (WSM) and TOPSIS, with editable inputs and transparent calculations.

The included example compares five hypothetical 12-week learning packages using total cost, weekly workload, interest fit, career fit and project quality. All sample values are illustrative classroom assumptions, not claims about real programmes.

## Run

Use Python 3.11 or later (tested with Python 3.12).

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Stop the server with Ctrl+C. On Windows, after installing the dependencies, you can also double-click `run_app.bat`.

If the default port is occupied:

```sh
python -m streamlit run app.py --server.port 8504
```

## Use

1. **Results** displays the recommendations, ranking table, score chart and normalized-performance heatmap.
2. In **Edit data**, edit criterion names, cost/benefit types, units and weights. Click **Apply criteria**. Weights must be nonnegative and total 1. New or renamed criteria create empty matrix columns.
3. Enter alternative names and every matrix value, then click **Calculate rankings**. Use the editable tables to add or delete rows. Keep at least five alternatives and five criteria.
4. Open **Calculations** to inspect normalization, weighted scores, ideal points, distances and TOPSIS closeness. Results can be downloaded as CSV.
5. **Start blank manual entry** creates an empty scenario. **Reset to sample data** restores the example.

Apply criterion changes before editing the matrix, then calculate. Results reflect the last successful calculation. Invalid calculations clear the previous result. Inputs must be finite and nonnegative; criteria with unit `1-10` require ratings in that range. Names must be unique. Edits last only for the current browser session.

## Example and recalculation

The baseline weights, in criterion order, are `0.20, 0.15, 0.20, 0.25, 0.20`. Both methods select Software Engineering:

| Method | Top alternative | Score |
|---|---|---:|
| WSM | Software Engineering | 0.640000 |
| TOPSIS | Software Engineering | 0.659340 |

Change the weights to `0.10, 0.10, 0.30, 0.30, 0.20`, click **Apply criteria**, then **Calculate rankings**, leaving matrix values unchanged. WSM selects AI / ML (0.733333), while TOPSIS selects Data Science (0.624343). This demonstrates how priorities change the recommendation. The heatmap stays unchanged because it displays normalized performance before weighting.

To demonstrate validation, start a blank scenario and attempt to calculate before filling its matrix. The application rejects the missing values. The complete sample inputs are defined in `sample_data.py`.

## Methods and assumptions

- **WSM:** cost/benefit-aware min-max normalization followed by a weighted sum. A constant column receives a normalized value of 1 for every alternative.
- **TOPSIS:** vector normalization, weighted values, cost/benefit-aware ideal best and worst points, Euclidean distances, and relative closeness. An all-zero column contributes no distance; indistinguishable alternatives receive closeness 0.5.
- Ties use ranks such as `1, 1, 3`; rounding to 12 decimals is used only for tie detection.
- Method scores are not probabilities and should not be compared directly across methods. Rankings can differ because both normalization and aggregation differ.
- The example assumes a 300,000 KZT budget, at most 12 hours per week, and satisfied prerequisites and availability. Feasibility must be checked before entering alternatives; the application does not enforce these scenario limits.
- Weights and ratings are subjective. The system supports the student's judgment and does not replace it.

## Tests

```sh
python -m unittest discover -s tests -v
```

Seven tests cover an independently calculated example, baseline and changed-weight results, cost direction, unit rescaling, ties, zero columns, invalid inputs and rating bounds.

## Files

- `app.py`: Streamlit interface and charts.
- `decision_model.py`: validation, WSM, TOPSIS and comparison explanation.
- `sample_data.py`: illustrative inputs.
- `tests/test_decision_model.py`: model tests.
- `requirements.txt`: runtime dependencies.
- `.streamlit/config.toml`: light theme and usage-statistics preference.
- `run_app.bat`: Windows launcher.
