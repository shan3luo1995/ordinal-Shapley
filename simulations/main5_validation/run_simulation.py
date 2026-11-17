#!/usr/bin/env python3
"""Validate main5.tex using 100 independently generated populations and trials.

Potential-outcome order is (Y00, Y01, Y10, Y11). All primary estimates use
observed (exposure, outcome) data only; truth is enumerated from the full law.
Run: python run_simulation.py --output results
Dependencies: numpy, pandas, matplotlib.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE_BLOB = "eac4b16133d05aa09b3438ed6bfe138eaaaa4bfc"
TYPES = np.array([
    t for t in itertools.product(range(3), repeat=4)
    if all(0 <= d <= 1 for d in
           (t[2]-t[0], t[1]-t[0], t[3]-t[2], t[3]-t[1]))
], dtype=int)
LABELS = ["".join(map(str, t)) for t in TYPES]
INDEX = {s: i for i, s in enumerate(LABELS)}
SCORES = np.column_stack((
    TYPES[:, 0],
    ((TYPES[:, 2]-TYPES[:, 0])+(TYPES[:, 3]-TYPES[:, 1]))/2,
    ((TYPES[:, 1]-TYPES[:, 0])+(TYPES[:, 3]-TYPES[:, 2]))/2,
))
TOL = 1e-12


def base_law(bprob, q1, q2):
    """B, D1, D2 independent; Yab = min(2, B + a D1 + b D2)."""
    law = np.zeros(len(TYPES))
    for b, d1, d2 in itertools.product(range(3), range(2), range(2)):
        t = (b, min(2, b+d2), min(2, b+d1), min(2, b+d1+d2))
        prob = bprob[b]*(q1 if d1 else 1-q1)*(q2 if d2 else 1-q2)
        law[INDEX["".join(map(str, t))]] += prob
    return law


def change_coupling(law, low, high):
    """Preserve both path triples by changing middle-variable coupling."""
    out = law.copy()
    for amount, plus, minus in [
        (low, ("0001", "0111"), ("0011", "0101")),
        (high, ("1112", "1222"), ("1122", "1212")),
    ]:
        for t in plus:
            out[INDEX[t]] += amount
        for t in minus:
            out[INDEX[t]] -= amount
    assert np.all(out >= -TOL)
    assert abs(out.sum()-1) < TOL
    return out


def marginal_law(law):
    return np.array([
        [law[TYPES[:, j] == r].sum() for r in range(3)]
        for j in range(4)
    ])


def triple_law(law, middle):
    out = np.zeros((3, 3, 3))
    for t, w in zip(TYPES, law):
        out[t[0], t[middle], t[3]] += w
    return out


def markov_residual(law, middle):
    """Maximum conditional independence residual, enumerating positive strata."""
    triple = triple_law(law, middle)
    residual = 0.0
    for b in range(3):
        joint = triple[:, b, :]
        prob = joint.sum()
        if prob == 0:
            continue
        conditional = joint/prob
        product = np.outer(conditional.sum(axis=1), conditional.sum(axis=0))
        residual = max(residual, float(np.max(np.abs(conditional-product))))
    return residual


def true_responsibility(law):
    """Definition of the estimand, not the identification formula."""
    truth = np.full((2, 3), np.nan)
    for k in (1, 2):
        take = TYPES[:, 3] == k
        denominator = law[take].sum()
        if denominator > 0:
            truth[k-1] = (law[take, None]*SCORES[take]).sum(axis=0)/denominator
    return truth


def identified_responsibility(marginal, middle=2, structural_zero=False):
    """main5 equations (5)--(7); no clipping or shape-constrained fitting.

    middle=2 uses Y10, as prespecified for the primary estimator.
    middle=1 uses Y01 only for a secondary diagnostic.
    A zero empirical middle-level count does not justify imposing a
    population structural zero; the primary estimator is undefined there.
    """
    xi = marginal[:, 2]
    denominator = marginal[middle, 1]
    if denominator <= 0:
        if structural_zero:
            theta = 0.0
        else:
            return np.full((2, 3), np.nan), np.nan
    else:
        theta = ((marginal[0, 0]-marginal[middle, 0]) *
                 (xi[3]-xi[middle])/denominator)

    grade2_numerator = np.array([
        xi[3]+xi[0]-theta,
        (xi[3]-xi[1]+xi[2]-xi[0]+theta)/2,
        (xi[3]-xi[2]+xi[1]-xi[0]+theta)/2,
    ])
    means = marginal @ np.arange(3)
    unconditional = np.array([
        means[0],
        (means[2]-means[0]+means[3]-means[1])/2,
        (means[1]-means[0]+means[3]-means[2])/2,
    ])
    estimate = np.full((2, 3), np.nan)
    if xi[3] > 0:
        estimate[1] = grade2_numerator/xi[3]
    if marginal[3, 1] > 0:
        estimate[0] = (unconditional-grade2_numerator)/marginal[3, 1]
    return estimate, float(theta)


def draw_population(rng):
    bprob = 0.1+0.7*rng.dirichlet(np.array([2.0, 2.0, 2.0]))
    q1, q2 = rng.uniform(0.15, 0.85, size=2)
    base = base_law(bprob, q1, q2)
    cap_low = min(base[INDEX["0011"]], base[INDEX["0101"]])
    cap_high = min(base[INDEX["1122"]], base[INDEX["1212"]])
    frac_low, frac_high = rng.uniform(0.1, 0.8, size=2)
    low, high = frac_low*cap_low, frac_high*cap_high
    law = change_coupling(base, low, high)
    alternative = change_coupling(base, cap_low-low, cap_high-high)

    # Same observed marginals, still unit monotone, but change theta and
    # violate the Markov assumptions (a negative control, not primary data).
    epsilon = 0.5*min(law[INDEX["0111"]], law[INDEX["1112"]])
    violation = law.copy()
    for t in ("0112", "1111"):
        violation[INDEX[t]] += epsilon
    for t in ("0111", "1112"):
        violation[INDEX[t]] -= epsilon
    assert np.all(violation >= 0)
    return law, alternative, violation, {
        "B0": bprob[0], "B1": bprob[1], "B2": bprob[2],
        "q1": q1, "q2": q2, "coupling_low": low, "coupling_high": high,
        "violation_epsilon": epsilon,
    }


def observed_trials(law, sample_sizes, rng):
    """Nested balanced randomised trials; potential outcomes never enter fitting."""
    largest = max(sample_sizes)
    type_ids = rng.choice(len(TYPES), size=largest, p=law)
    assignments, previous = [], 0
    for n in sample_sizes:
        block = np.repeat(np.arange(4), (n-previous)//4)
        assignments.append(rng.permutation(block))
        previous = n
    arm = np.concatenate(assignments)
    observed_y = TYPES[type_ids, arm]
    for n in sample_sizes:
        counts = np.bincount(3*arm[:n]+observed_y[:n], minlength=12).reshape(4, 3)
        assert np.all(counts.sum(axis=1) == n//4)
        yield n, counts/(n//4)


def create_plots(metrics, estimates, out):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight",
    })
    names = {0: "Background", 1: "Exposure 1", 2: "Exposure 2"}
    colours = {0: "#59616d", 1: "#226a91", 2: "#b66033"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.25), constrained_layout=True)
    for k, ax in enumerate(axes, start=1):
        for j in (0, 1, 2):
            d = metrics[(metrics["grade"] == k) & (metrics["cause"] == j)]
            ax.plot(d["n"], d["rmse"], marker="o", color=colours[j], label=names[j])
        ax.set_xscale("log")
        sizes = sorted(metrics["n"].unique())
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
        ax.set(xlabel="Total sample size", ylabel="RMSE (grade points)",
               title=f"Observed grade {k}")
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(frameon=False)
    fig.savefig(out/"rmse.pdf")
    fig.savefig(out/"rmse.png", dpi=160)
    plt.close(fig)

    n = sorted(estimates["n"].unique())[len(estimates["n"].unique())//2]
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), constrained_layout=True)
    for k in (1, 2):
        for j in (0, 1, 2):
            ax = axes[k-1, j]
            d = estimates[(estimates["n"] == n) & (estimates["grade"] == k)
                          & (estimates["cause"] == j)]
            ax.scatter(d["truth"], d["estimate"], s=12, alpha=.7, color=colours[j])
            limits = [min(0, d["estimate"].min()), max(d["truth"].max(), d["estimate"].max())]
            ax.plot(limits, limits, linestyle="--", color="#8e949b", linewidth=.8)
            ax.set(title=f"Grade {k}: {names[j]}", xlabel="Population truth",
                   ylabel="Observed-data estimate")
    fig.suptitle(f"100 populations, n = {n:,}", fontsize=12)
    fig.savefig(out/"truth_vs_estimate.pdf")
    fig.savefig(out/"truth_vs_estimate.png", dpi=160)
    plt.close(fig)


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted(set(args.sample_sizes))
    if not sizes or any(n <= 0 or n % 4 for n in sizes):
        raise ValueError("Sample sizes must be positive multiples of four.")
    # Deterministic boundary checks are separate from the 100 random experiments.
    boundary_checks = 0
    for labels in [("0001", "2222"), ("0000", "2222"), ("0000", "0001")]:
        boundary_law = np.zeros(len(TYPES))
        for label in labels:
            boundary_law[INDEX[label]] = 0.5
        target = true_responsibility(boundary_law)
        recovered, theta_boundary = identified_responsibility(
            marginal_law(boundary_law), structural_zero=True)
        defined = np.isfinite(target)
        assert np.max(np.abs(target[defined]-recovered[defined])) < TOL
        assert theta_boundary == 0
        assert max(markov_residual(boundary_law, b) for b in (1, 2)) < TOL
        boundary_checks += 1
    root = np.random.SeedSequence(args.seed)
    experiment_seeds = root.spawn(args.experiments)
    rows, checks, populations, sample_diagnostics = [], [], [], []
    for number, seed in enumerate(experiment_seeds, start=1):
        pop_seed, data_seed = seed.spawn(2)
        law, alternative, violation, params = draw_population(np.random.default_rng(pop_seed))
        marg = marginal_law(law)
        truth = true_responsibility(law)
        oracle, theta = identified_responsibility(marg, structural_zero=True)
        oracle_other, theta_other = identified_responsibility(marg, middle=1, structural_zero=True)
        residual1 = markov_residual(law, 2)
        residual2 = markov_residual(law, 1)
        alt_marg = marginal_law(alternative)
        alt_truth = true_responsibility(alternative)
        bad_marg = marginal_law(violation)
        bad_truth = true_responsibility(violation)
        bad_oracle, _ = identified_responsibility(bad_marg, structural_zero=True)
        check = {
            "experiment": number, **params,
            "Pr_Y11_1": marg[3, 1], "Pr_Y11_2": marg[3, 2],
            "Pr_Y10_1": marg[2, 1], "Pr_Y01_1": marg[1, 1],
            "theta_truth": law[INDEX["0112"]], "theta_formula": theta,
            "theta_other_path": theta_other,
            "markov_Y10_residual": residual1, "markov_Y01_residual": residual2,
            "oracle_max_abs_error": np.max(np.abs(oracle-truth)),
            "oracle_other_max_abs_error": np.max(np.abs(oracle_other-truth)),
            "alternative_tv": np.abs(law-alternative).sum()/2,
            "alternative_marginal_error": np.max(np.abs(marg-alt_marg)),
            "alternative_triple_error": max(
                np.max(np.abs(triple_law(law, b)-triple_law(alternative, b))) for b in (1, 2)),
            "alternative_truth_error": np.max(np.abs(truth-alt_truth)),
            "alternative_markov_residual": max(markov_residual(alternative, b) for b in (1, 2)),
            "violation_marginal_error": np.max(np.abs(marg-bad_marg)),
            "violation_markov_residual": max(markov_residual(violation, b) for b in (1, 2)),
            "violation_formula_max_abs_error": np.max(np.abs(bad_oracle-bad_truth)),
        }
        assert max(residual1, residual2) < TOL
        assert check["oracle_max_abs_error"] < TOL
        assert check["oracle_other_max_abs_error"] < TOL
        assert abs(theta-law[INDEX["0112"]]) < TOL
        assert check["alternative_tv"] > TOL
        assert check["alternative_marginal_error"] < TOL
        assert check["alternative_triple_error"] < TOL
        assert check["alternative_truth_error"] < TOL
        assert check["alternative_markov_residual"] < TOL
        assert check["violation_marginal_error"] < TOL
        assert check["violation_markov_residual"] > TOL
        assert check["violation_formula_max_abs_error"] > TOL
        checks.append(check)
        for variant, weights in (("valid", law), ("alternative", alternative), ("violation", violation)):
            populations.append({"experiment": number, "variant": variant,
                                **{f"p_{s}": float(v) for s, v in zip(LABELS, weights)}})

        for n, observed_marginal in observed_trials(law, sizes, np.random.default_rng(data_seed)):
            estimate, theta_hat = identified_responsibility(observed_marginal)
            _, other_theta_hat = identified_responsibility(observed_marginal, middle=1)
            sample_diagnostics.append({
                "experiment": number, "n": n, "theta_hat": theta_hat,
                "theta_hat_other": other_theta_hat, "theta_truth": theta,
                "middle_denominator_zero": observed_marginal[2, 1] == 0,
                "grade1_denominator_zero": observed_marginal[3, 1] == 0,
                "grade2_denominator_zero": observed_marginal[3, 2] == 0,
            })
            for k in (1, 2):
                for j in (0, 1, 2):
                    value = estimate[k-1, j]
                    rows.append({
                        "experiment": number, "n": n, "grade": k, "cause": j,
                        "truth": truth[k-1, j], "estimate": value,
                        "error": value-truth[k-1, j],
                        "outside_score_range": bool(np.isfinite(value) and
                                                    (value < -TOL or value > k+TOL)),
                    })

    estimates = pd.DataFrame(rows)
    exact = pd.DataFrame(checks)
    diag = pd.DataFrame(sample_diagnostics)
    metrics = []
    for (n, k, j), group in estimates.groupby(["n", "grade", "cause"], sort=True):
        e = group["error"].dropna().to_numpy()
        metrics.append({
            "n": int(n), "grade": int(k), "cause": int(j),
            "n_valid": len(e), "mean_error": float(e.mean()),
            "mcse_mean_error": float(e.std(ddof=1)/np.sqrt(len(e))),
            "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e))),
            "max_abs_error": float(np.max(np.abs(e))),
            "outside_score_range": int(group["outside_score_range"].sum()),
        })
    metrics = pd.DataFrame(metrics)
    estimates.to_csv(out/"replicates.csv", index=False, float_format="%.15g")
    exact.to_csv(out/"population_checks.csv", index=False, float_format="%.15g")
    pd.DataFrame(populations).to_csv(out/"type_probabilities.csv", index=False, float_format="%.15g")
    diag.to_csv(out/"sample_diagnostics.csv", index=False, float_format="%.15g")
    metrics.to_csv(out/"metrics.csv", index=False, float_format="%.15g")
    summary = {
        "config": {"experiments": args.experiments, "sample_sizes": sizes, "seed": args.seed,
                   "primary_theta_path": "Y10", "source_main5_blob": SOURCE_BLOB,
                   "design": f"{args.experiments} heterogeneous populations; nested balanced samples within population",
                   "score": "Y=0,1,2; one point per grade"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "exact_checks": {
            "populations_passed": len(exact),
            "deterministic_boundary_checks_passed": boundary_checks,
            "max_markov_residual": float(exact[["markov_Y10_residual","markov_Y01_residual"]].max().max()),
            "max_oracle_error": float(exact["oracle_max_abs_error"].max()),
            "max_other_path_oracle_error": float(exact["oracle_other_max_abs_error"].max()),
            "max_theta_error": float(np.max(np.abs(exact.theta_truth-exact.theta_formula))),
            "alternative_min_tv": float(exact.alternative_tv.min()),
            "alternative_max_marginal_error": float(exact.alternative_marginal_error.max()),
            "alternative_max_triple_error": float(exact.alternative_triple_error.max()),
            "alternative_max_responsibility_error": float(exact.alternative_truth_error.max()),
            "violation_max_marginal_error": float(exact.violation_marginal_error.max()),
            "violation_min_markov_residual": float(exact.violation_markov_residual.min()),
            "violation_mean_max_error": float(exact.violation_formula_max_abs_error.mean()),
            "violation_max_error": float(exact.violation_formula_max_abs_error.max()),
            "min_case_probability": float(exact[["Pr_Y11_1","Pr_Y11_2"]].min().min()),
            "min_middle_probability": float(exact[["Pr_Y10_1","Pr_Y01_1"]].min().min()),
        },
        "finite_sample": {
            "datasets": int(len(diag)), "responsibility_estimates": len(estimates),
            "undefined_estimates": int(estimates.estimate.isna().sum()),
            "outside_score_range": int(estimates.outside_score_range.sum()),
            "zero_denominator_datasets": int(diag[[
                "middle_denominator_zero","grade1_denominator_zero","grade2_denominator_zero"]].any(axis=1).sum()),
            "pooled_rmse_by_n": {
                str(n): float(np.sqrt(np.mean(g.error**2))) for n,g in estimates.groupby("n")
            },
        },
        "metrics": metrics.to_dict(orient="records"),
        "interpretation": (
            "Exact numerical checks corroborate the identities for this family, not a proof over all admissible laws. "
            "Mean error and RMSE average over random populations as well as random samples. "
            "Raw plug-in estimates are not clipped. Observationally identical alternatives verify nonidentification "
            "of the complete joint law; negative controls violate Markov and are not primary simulation data."
        ),
    }
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    create_plots(metrics, estimates, out)
    print(json.dumps({k: summary[k] for k in ("exact_checks","finite_sample")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=int, default=100)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[2000,10000,50000])
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"results")
    args = parser.parse_args()
    if args.experiments < 2:
        parser.error("At least two experiments are required for the error summaries.")
    run(args)

