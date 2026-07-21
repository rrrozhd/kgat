# Agreement routing production verdict

**Final verdict: `COST_REDUCTION_NO_GO`; shadow mode remains disabled.**

The canonical max-28 GPU run reproduced the frozen confidence policy exactly:
615 of 2,193 chunks escalated (28.04%), with F1 0.87855, recall 0.87215, and
exact match 0.89694. All input, adapter, outcome, environment, and run-manifest
identity gates passed.

The agreement family is a genuine routing-quality improvement. At an exact 20%
budget, `min_agreement` improves F1 from 0.80548 to 0.85408 and recall from
0.80063 to 0.80570. Its routing AUROC is 0.88710 versus confidence at 0.85723.

That improvement does not convert into the required production economics. The
cheapest policy matching or exceeding the frozen confidence baseline's F1 and
recall while respecting the exact-match tolerance is `conf_x_min_agree` at 612
teacher calls. This is only three fewer calls than confidence (0.49%), far below
the 10% requirement. Its evaluation escalation-token volume also rises from
281,786 to 282,254 tokens (+468, +0.17%), because 509 additional output tokens
more than offset 41 fewer input tokens.

Projected mechanically to 1.16M chunks, calls fall from about 325,308 to
323,721—about 1,587 calls, not the required 10% reduction. Since the point
estimate fails decisively, the 10,000-resample economics bootstrap was not run;
the required dated pricing profile and frozen 1.16M-stratum workload manifest
also do not yet exist. Those missing Alphina/workload inputs cannot rescue this
candidate's failed point gate.

The `ECONOMICS_READY` label in `comparison.json` means only that a signal passed
the predeclared 15%/20%/25% routing-robustness screen and was ready to be tested
for economics. It is not a production or shadow-enable verdict. This document
and `cost_prevalidation.json` apply the subsequent cost gate.

Production action: retain confidence-only routing. Do not enable agreement
shadow emission under the approved fail-closed contract. The result remains
valuable as evidence that grammar agreement improves ranking at fixed budget,
but it does not meet the stated cost-reduction objective.
