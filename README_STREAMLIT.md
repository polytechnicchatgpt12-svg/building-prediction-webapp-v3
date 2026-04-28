# Building Project Prediction Streamlit App

## Files needed on GitHub

Upload these files to the same GitHub repository folder:

- `app.py`
- `building_project_predictor.py`
- `D_Building_2000_prediction_dataset.xlsx`
- `requirements.txt`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push the files to GitHub.
2. Open Streamlit Cloud.
3. Choose the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

## Notes

This app rebuilds the website interface around your existing predictor class. It includes:

- project input form
- prediction result cards
- risk summary
- risk probability chart
- cost and schedule causes
- warning flags
- similar historical projects
- downloadable JSON report
