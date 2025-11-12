import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

class ModelEvaluation:
    """
    Evaluate regression model predictions with R^2, RMSE, and MAE.
    Usage:
        evaluator = ModelEvaluation(y_true, y_pred)
        r2 = evaluator.r2()
        rmse = evaluator.rmse()
        mae = evaluator.mae()
    """
    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

    def r2(self):
        self.r2_score = r2_score(self.y_true, self.y_pred)
        return self.r2_score

    def rmse(self):
        self.rmse_score = root_mean_squared_error(self.y_true, self.y_pred)
        return self.rmse_score

    def mae(self):
        self.mae_score = mean_absolute_error(self.y_true, self.y_pred)
        return self.mae_score

    def gini(self):
        sorted_list = [x for _,x in sorted(zip(self.y_pred,self.y_true))]
        height, area = 0, 0
        for value in sorted_list:
            height += value
            area += height - value / 2.
        fair_area = height * len(self.y_true) / 2.
        self.gini_score = (fair_area - area) / fair_area
        return self.gini_score