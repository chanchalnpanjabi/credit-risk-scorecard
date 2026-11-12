# Put this on GitHub (10 minutes)

## 1. Run everything on the real data
```bash
cd credit-risk-scorecard
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
# copy Kaggle's cs-training.csv into data/
python -m pytest -q
python -m src.train --data data/cs-training.csv --out reports
python -m src.render_docs
```
Then open the notebook and use **Kernel > Restart and Run All**, then save, so the committed notebook shows
real outputs rather than the synthetic ones it ships with.
```bash
jupyter notebook notebooks/01_scorecard_walkthrough.ipynb
```

## 2. Check before you publish
- `README.md` Results section is filled and shows your figures.
- `reports/results.md` says **Give Me Some Credit (Kaggle)** as the data source, not SYNTHETIC.
- Notebook's first output line says `DATA SOURCE: ../data/cs-training.csv`.
- `data/*.csv` is NOT staged (`git status` should not list it).

## 3. Create the repo and push
Create an empty repo named `credit-risk-scorecard` on github.com (no README, no .gitignore), then:
```bash
git init
git add .
git status                       # confirm no CSV files are listed
git commit -m "Credit risk scorecard: WoE binning, logistic PD model, PSI, benchmark"
git branch -M main
git remote add origin https://github.com/<your-username>/credit-risk-scorecard.git
git push -u origin main
```
On GitHub: add a one-line description, topics (`credit-risk`, `scorecard`, `logistic-regression`, `python`,
`data-science`) and pin the repo to your profile.

## 4. LinkedIn
Open `docs/linkedin_post.md` (created by `render_docs`), replace `[GitHub link]`, post it, and add the Projects
entry. Update your CV from `docs/cv_bullets.md`.
