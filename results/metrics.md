# Evaluation

[Back to the showcase](../README.md)

| Historical experiment | Result | Denominator |
| :--- | ---: | :--- |
| Proposal precision | 55.56% | 5 matched / 9 proposals |
| Proposal recall | 62.50% | 5 matched / 8 labeled points |
| Winner attribution on matched events | 80.00% | 4 correct / 5 matched |
| Score timeline frame accuracy | 0.00% | 0 / 373 comparable frames |

Timing tolerance: 2 seconds. These are separate saved report views of one local recording, not a general benchmark. The failed score-timeline result is included deliberately. [Sanitized report values →](evaluation.json)

## Interpretation

Automatic scoring remains experimental. Replay from supplied labels or scoreboard observations is not independent scoring accuracy. Fixed-camera calibration, occlusion, player-ID changes, and missed ball events can destabilize downstream statistics.

Existing source reports and newly executed software checks are different evidence types. Model training and full inference were not rerun solely to prepare the README.

## Next useful measurements

Expand the labeled point set, report event timing precision/recall, winner attribution with denominators, and accumulated score error. Evaluate independent gameplay predictions separately from label replay.
