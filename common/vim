from sklearn.pipeline        import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd

def compare_models(model_dict, X, y, cv=5, scoring="accuracy"):
    """
    Run cross_val_score on each estimator in model_dict,
    return a sorted pd.Series of mean scores.
    """
    results = {}
    for name, est in model_dict.items():
        pipe = Pipeline([("preproc", est["preproc"]), ("clf", est["clf"])])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = scores.mean()
    return pd.Series(results).sort_values(ascending=False)

def grid_search(pipeline, param_grid, X, y, cv=5, scoring="accuracy"):
    gs = GridSearchCV(pipeline, param_grid, cv=cv,
                      scoring=scoring, n_jobs=-1, verbose=1, refit=True)
    gs.fit(X, y)
    return gs

