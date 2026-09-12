# Two-velocity orbit fits for the eccentricity comparison

Code: `scripts/orbit_sampler.py` (`twovel_linear`, `twovel_label_marginal`,
`sample_twovel_refined`), `scripts/ecc_marginal.py`
(`system_loglike_on_egrid_twovel`), `scripts/deep_table.py` (`untied_pairs`,
`q_twovel`, `exchange_pairs`). Scripts select it with `--method twovel` (default);
`--method v1` runs the earlier method unchanged.

## Data

For each visit i the stage-2 fitter also fits that visit alone with two
components and no momentum tie. This gives two velocities (`rv1_untied_per_visit`,
`rv2_untied_per_visit`), which we treat as an unordered pair {a_i, b_i}: the fit of
one visit does not know which component was called 1 at another visit. The pairs
are independent from visit to visit. Visits where either velocity or the epoch is
missing or not finite are dropped (none are in the current deep table). The orbit
step needs at least 6 usable visits and the likelihood step at least 8, as before.
The tied per-visit v2 is not used: it is fixed by v1, gamma and q and carries no
independent information.

## Model

    m1(t) = gamma + K1 g(t),     m2(t) = gamma - (K1/q) g(t),
    g(t)  = cos(nu + omega) + e cos(omega),

with nu the true anomaly for (P, e, omega, M0). At each visit the assignment of
{a_i, b_i} to the two components is unknown, with prior probability 1/2 each way,
and is summed over:

    L_i = 0.5 N(a_i | m1, s) N(b_i | m2, s) + 0.5 N(b_i | m1, s) N(a_i | m2, s).

s is the same for both components: a jitter drawn per trial orbit and
marginalized (log-uniform on 0.5-10 km/s) in the orbit step, as in the earlier
sampler, and the fixed sigma = 1.5 km/s in the e-grid likelihood, the mocks and
the null, as before.

(K1, gamma) enter linearly. For each trial (P, e, omega, M0) they are solved as
follows. Start from the solution that does not depend on the labels: K1 from the
separations, |a_i - b_i| = K1 |g_i| (1 + 1/q), and gamma from the sums,
a_i + b_i = 2 gamma + K1 g_i (1 - 1/q). Then, twice, give each visit the
assignment that fits the current model better and solve the weighted
least-squares problem for (K1, gamma) with both velocities under that assignment
(the same normal equations as the earlier two-velocity sampler). The likelihood
is then evaluated with the label sum above, not with the chosen assignment. In
the e-grid step (K1, gamma) are integrated out as before, by adding
-0.5 ln det of the normal matrix (flat prior; this matrix does not depend on the
data or the labels); P, omega and M0 are averaged over the same draws per grid
point (60,000 by default). K1 > 0 and, in the orbit step, the contact prior
a(1 - e) > R1 + R2 are kept.

## Mass ratio

q is the row's `q_dyn` clipped to [0.1, 1.5] (the fitter's bounds). If `q_dyn`
is missing, not positive, flagged `q_dyn_at_bound`, or within 0.001 of a bound,
we use `q_spec` with the same clip instead (41 of the 748 systems in the earlier
sample). q is held fixed at this value in every step; its error is not
propagated. q > 1 is allowed. It means the component labelled 1 by the joint fit
moves more. Because the labels are summed over, (q, K1, omega) and
(1/q, K1/q, omega + pi) give the same likelihood, so this choice changes the
reported K1 but not L(e).

## Why the fit cannot fold

The earlier likelihood used one velocity per visit, v1 or its reflection about
gamma, whichever lay closer to the model. At q near 1 a small orbit close to
gamma could then match every visit by switching between the two, so the twins
got a K1 about half the amplitude of their measured velocities. In the new model
each visit must match both velocities under either labelling. The separation
|a_i - b_i| is the same under both labellings and equals K1 |g_i| (1 + 1/q), so
an orbit whose amplitude is too small leaves the separations unfit whatever the
labels. The labels only decide the sign of g_i at each visit.

The cost is at q = 1 exactly: the pair is then symmetric about gamma, and each
visit gives |g_i| but not its sign. The eccentricity has to come from the shape
of |g(t)|. For q away from 1 the sums a_i + b_i also carry the sign. The mock
tests below check what this does to K1 and e.

## Mocks

`ecc_val_suite.py` and `ecc_null_full.py` build each mock like a real pair: real
cadence; P, e, omega, M0, K1 and gamma drawn as before; q from the real twin (or
non-twin) q values in the validation suite, and the real system's own q in the
null; m1 and m2 plus Gaussian noise (sigma each); then the two velocities are
exchanged at each visit with probability 0.5. They are fitted with the same
likelihood as the data. Alphas, repeats and output formats are unchanged.

## Tests (local, 2026-09-12; script and outputs in the session scratchpad)

T1, real data. 40 twins and 40 non-twins drawn (seed 20260912) from the 748
systems with status ok and 6 <= P50 < 400 d in the v1-method
`orbit_deep_posteriors.csv`, run with both samplers at the production settings.
The v1 rerun reproduces the file's K1 for all 80. Scale A = q/(1+q) max_i|a_i - b_i|
(label-free, about K1 max|g_i|). Scale B = half the range of the untied primary
velocities after giving each visit the labels preferred by the best two-velocity
draw. Medians of the per-system K1/scale:

| group | K1 v1 | K1 twovel | scale A | K1/A v1 | K1/A twovel | K1/B v1 | K1/B twovel |
|---|---|---|---|---|---|---|---|
| twins | 1.9 | 16.6 | 11.0 | 0.35 | 1.29 | 0.37 | 1.24 |
| non-twins | 9.5 | 19.3 | 16.2 | 0.50 | 0.99 | 0.86 | 1.38 |

The twin ratio is no longer near 0.5, and no twin falls below half of scale A
(0 of 40, against 25 of 40 before). The twin and non-twin ratios agree to 11% on
scale B and differ by 30% on scale A. Both are above 1 mostly because 24 twins
and 13 non-twins move to P50 >= 400 d, where the visits cover part of the orbit
and K1 is extrapolated. For the 14 twins and 16 non-twins that stay in 6-400 d
with at least 8 kept draws, K1/A is 1.08 and 0.89. 15 of the 80 keep fewer than
8 draws (status unconstrained), against 0 with the v1 sampler; 30 refinement
rounds instead of 12 rescue only 6 of them.

T2, mocks. 30 mocks at q = 1 and 30 at q = 0.6, real cadences, K1 = 20 km/s,
P log-uniform 6-400 d, e uniform 0-0.8, sigma 1.5 km/s, labels exchanged with
probability 0.5. Median recovered K1/K1_true: twovel 0.983 (q = 1) and 0.995
(q = 0.6); v1 sampler on the same mocks 0.904 and 1.409 (0.885 and 1.012 with
the true labels given). 77% of the twovel K1 are within 20% of the truth in
both groups; P50 is within 5% of the true P for 40% and 43%.

T3, e-grid likelihood. 12 mocks per cell, K1 = 20, 60,000 draws per grid point:

| q | e true | median argmax e | stacked argmax e | median flat-prior mean e | argmax within 0.15 |
|---|---|---|---|---|---|
| 0.6 | 0.1 | 0.10 | 0.04 | 0.12 | 10/12 |
| 0.6 | 0.6 | 0.59 | 0.64 | 0.59 | 7/12 |
| 1.0 | 0.1 | 0.05 | 0.02 | 0.24 | 8/12 |
| 1.0 | 0.6 | 0.90 | 0.85 | 0.64 | 2/12 |

For q = 0.6 the likelihood peaks at the true e. For q = 1 at e = 0.6 it peaks too
high, although its mean is close: this is the loss of the sign of g at q = 1
described above. A twin-specific shift of this kind is what the injection null
(`ecc_null_full.py`, real q per system) and the validation suite (twin q values
for the twin mocks) are meant to measure; neither has been rerun yet.
