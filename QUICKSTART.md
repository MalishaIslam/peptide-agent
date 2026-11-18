# Quickstart

Run these in order. Each line says what "done" looks like.

```bash
# 1. install
pip install -r requirements.txt
pytest -q                             # -> 44 passed

# 2. fetch the dataset (needs internet; writes data/peptides.csv)
python scripts/fetch_dataset.py --limit 800
#    -> "wrote N peptides" + a rejection breakdown. Save that breakdown.

# 3. check the batch pipeline reads it
python -c "from peptide_agent.batch import process_csv; r,e=process_csv('data/peptides.csv'); print(len(r),'ok',len(e),'failed')"
#    -> failures should be 0, because fetch_dataset already filtered to scope

# 4. set your key
export ANTHROPIC_API_KEY=sk-...

# 5. see it work
python demo.py                        # scripted questions, prints traces
streamlit run app.py                  # UI

# 6. the number for your resume
python -m evals.run_evals             # -> pass rate + evals/results.json
```
