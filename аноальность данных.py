import pandas as pd
import numpy as np

def detect_outliers(data):

    mean = np.mean(data) # ср.зн
    std = np.std(data) # отклон

    # Правило трех сигм. Смотри в тетрадь
    threshold = 3 * std

    outlier_indices = np.where(np.abs(data - mean) > threshold)[0]

    return outlier_indices

advertising_expenses = np.array([100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 500])

outlier_indices = detect_outliers(advertising_expenses)


if len(outlier_indices) > 0:
    print("Обратите внимание на значение под этим индексом:", outlier_indices + 1 )
    print("Значение:", advertising_expenses[outlier_indices])

else:
    print("Все ок")


