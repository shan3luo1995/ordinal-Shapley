# main5.tex: 100 simulation experiments

This study checks the identification formulas in `main5.tex`, at source blob
`eac4b16133d05aa09b3438ed6bfe138eaaaa4bfc` (proof revision commit
`29e5c8e9d145154f269c0d36fe1880ca6e0036e7`). Potential outcomes are ordered as
`(Y00,Y01,Y10,Y11)` and take values 0, 1, 2. Each increase of one grade counts
as one score point. The targets are the background and both exposure
responsibilities conditional on `Y11=1` or `Y11=2`.

## Reproduce

Run from the repository root with Python 3.12:

```bash
python -m pip install -r simulations/main5_validation/requirements.txt
python simulations/main5_validation/run_simulation.py
python simulations/main5_validation/make_report.py
cd simulations/main5_validation/results
xelatex -halt-on-error -interaction=nonstopmode report.tex
```

XeLaTeX with Noto CJK or TeX Live Fandol fonts is supported. The report also
contains a pdfLaTeX/CJKutf8 fallback. The simulation produces the two figures
required to compile the report. CSV and JSON results are committed; figure
files and the compiled PDF are reproducible outputs.

The prespecified run uses seed `20261117`, 100 experiments, and sample sizes
2,000, 10,000, and 50,000. These settings can be changed for the simulation:

```bash
python simulations/main5_validation/run_simulation.py --experiments 100 --sample-sizes 2000 10000 50000 --seed 20261117
```

`make_report.py` describes the saved default design and seed; it rejects
other configurations. Library versions used for the saved run are recorded
in `results/summary.json`.

## Design

Each experiment independently draws a population and then a randomised
sample. Within that population, the three sample sizes use nested prefixes
of one sample of 50,000 individuals. Every prefix has exactly one quarter
of its observations in each exposure arm. Across populations, the 100
experiments are independent at each sample size: 300 datasets in total,
not 300 independent datasets. A `SeedSequence` gives population construction
and sample generation separate random streams.

The initial population has independent variables B, D1, and D2:

- `P(B=0,1,2) = 0.1*(1,1,1) + 0.7*V`, with `V ~ Dirichlet(2,2,2)`;
- independent Bernoulli D1 and D2 with probabilities drawn independently
  from `Uniform(0.15,0.85)`;
- `Y00=B`, `Y01=min(2,B+D2)`, `Y10=min(2,B+D1)`, and
  `Y11=min(2,B+D1+D2)`.

Two subsequent probability adjustments broaden this family. Equal mass is
added to types 0001 and 0111 and removed from 0011 and 0101. Independently,
equal mass is added to 1112 and 1222 and removed from 1122 and 1212.
Each adjustment is a random 0.1--0.8 fraction of the smaller removable mass.
These operations preserve both path triple distributions and hence both
Markov conditions. All 12 types allowed by the unit-increment restrictions
have positive mass in every main population.

True responsibilities are computed directly by summing type-specific scores
over the full population law conditional on the final grade. The oracle
identification calculation receives only the four marginal distributions.
Finite-sample estimation receives only arm/outcome counts; it never receives
individual counterfactual outcomes. The primary estimator uses the
prespecified Y10 path to recover theta. The Y01 path is a diagnostic and is
not selected after observing results.

## Checks and results

All 100 populations satisfy both Markov conditions numerically, and both
paths recover all six responsibilities. The largest oracle error is
`3.275e-15`, consistent with floating-point error. Three additional
deterministic boundary examples verify the structural-zero denominator
convention and undefined targets for case events of probability zero.
They are separate from the 100 random experiments.

| Total sample size | Pooled RMSE over six responsibilities |
| ---: | ---: |
| 2,000 | 0.05414 |
| 10,000 | 0.02263 |
| 50,000 | 0.01003 |

All 1,800 estimates are defined. Four raw estimates are negative (three at
2,000 and one at 10,000). No clipping, constrained fitting, or selective
reruns are used. Mean error and RMSE average over population heterogeneity
as well as random sampling; mean error is not Monte Carlo bias for one
fixed data-generating law. Monte Carlo standard errors of the mean errors
are reported, but confidence-interval coverage is not evaluated.

For every main population, two diagnostic populations are also constructed:

- An alternative changes the conditional coupling of Y01 and Y10 while
  preserving both path triples. Its complete joint law differs but all six
  responsibilities agree. This corroborates that the complete joint law
  need not be identified.
- A sensitivity control adds mass to 0112 and 1111 and removes equal mass
  from 0111 and 1112. This preserves all four marginals and unit increments
  but changes theta and violates Markov. The largest discrepancy between
  the original formula and the new truth is 0.11408. These are not part of
  the valid primary sample-generating populations.

These exact checks and simulated trials support the identities for this
family; they do not replace a proof or cover all admissible joint laws.

## Interpreting the two Markov conditions

Under both conditions, the Y10 and Y01 expressions for theta must agree.
This is an observable necessary restriction under randomisation and unit
increments. Equality does not establish that the actual joint law satisfies
Markov: the sensitivity controls preserve that equality while violating
Markov. The report includes an explicit observable counterexample for which
the two expressions differ.

For the present two-exposure, three-grade identification result, one
prespecified path's Markov condition suffices. This differs from asserting
that at least one unknown path satisfies it. The main simulation checks the
stronger two-condition model stated in the current report.

## Saved files

| File | Contents |
| --- | --- |
| `population_checks.csv` | Population parameters, exact checks, both theta formulas |
| `type_probabilities.csv` | Complete probabilities for valid, alternative, and sensitivity laws |
| `replicates.csv` | Every true responsibility, estimate, and error |
| `sample_diagnostics.csv` | Both sample theta estimates and denominator diagnostics |
| `metrics.csv` | Mean error, its Monte Carlo standard error, RMSE, MAE, and range violations |
| `summary.json` | Configuration, versions, and numerical summaries |

In result files, `grade` is the observed final grade (1 or 2), and `cause=0`
denotes background, `cause=1` exposure X1, and `cause=2` exposure X2.
Type probabilities are saved to 15 significant digits; exact checks use
the original floating-point arrays before CSV rounding.
