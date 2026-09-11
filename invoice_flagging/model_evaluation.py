from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score


def train_random_forest(X_train, y_train):

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1
    )

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "criterion": ["gini"]
    }

    scorer = make_scorer(f1_score)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        n_jobs=1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def evaluate_classifier(model, X_test, y_test, model_name):

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    report = classification_report(y_test, preds)

    print(f"\n{model_name} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)