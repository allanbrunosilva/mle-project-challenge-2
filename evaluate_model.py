from sklearn import metrics

def evaluate(model, X_train, X_test, y_train, y_test):
    # Identify model name 
    try:
        model_name = type(model.named_steps.get("kneighborsregressor", model)).__name__
    except AttributeError:
        model_name = type(model).__name__

    print(f"\nEvaluating model: {model_name}")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate performance
    def summarize(split, y_true, y_pred):
        print(f"\n{split} Set Performance")

        r2 = metrics.r2_score(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False) # Since not yet squared, set to False to get RMSE and not only MSE

        # R-squared: how much of the target variation the model explains
        print(f"  R²:   {r2:.4f} | The model explains {r2:.0%} of the variance in the target variable")

        # Mean Absolute Error: The average absolute difference between predicted and actual values
        print(f"  MAE:  {mae:,.0f} | On average, the model's predictions are off by about ${mae:,.0f}")

        # Root Mean Squared Error: The square root of the average squared difference between predictions and true values
        print(f"  RMSE: {rmse:,.0f} | Typical prediction error is around ${rmse:,.0f}, with larger misses penalized more")

        return r2, mae, rmse

    r2_train, _, _ = summarize("Train", y_train, y_train_pred)
    r2_test, _, _ = summarize("Test", y_test, y_test_pred)

    # Simple overfitting/underfitting check
    r2_train = metrics.r2_score(y_train, y_train_pred)
    r2_test = metrics.r2_score(y_test, y_test_pred)
    if abs(r2_train - r2_test) > 0.1:
        print("\nPotential overfitting: train R² much higher than test R².")
    elif r2_test < 0.5:
        print("\nModel may be underfitting: low R² on test data.")
    else:
        print("\nModel generalizes reasonably well.")
