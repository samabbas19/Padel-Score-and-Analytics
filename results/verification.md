# Software verification

Executed while preparing the publication on Windows with Python 3.11 and pandas 3.0.5.

```powershell
python -m unittest discover -s tests -v
```

**21 tests passed.** Coverage includes backend job planning/persistence, point-proposal selection, legal score transitions, score-sequence/checkpoint comparisons, proposal evaluation, winner evaluation, and label replay.

These tests exercise software behavior with fixtures. They do not establish model performance on match footage. Historical event/timeline measurements are presented separately in [metrics.md](metrics.md).
