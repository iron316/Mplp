from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             r2_score, mean_squared_error)


def classification_eval(true_label, predict):
    matrix = confusion_matrix(true_label, predict)
    accuracy = accuracy_score(true_label, predict)
    report = classification_report(true_label, predict)
    print("##### test evaluation #####")
    print(report)
    print(f"accuracy score : {accuracy:.3f}")
    print("confusion_matrix")
    print(matrix)


def regression_eval(true_label, predict):
    mae = mean_absolute_error(true_label, predict)
    mse = mean_squared_error(true_label, predict)
    r2 = r2_score(true_label, predict)
    print("##### test evaluation #####")
    print(f"MAE score : {mae:.3f}")
    print(f"MSE score : {mse:.3f}")
    print(f"R2 score : {r2:.3f}")
