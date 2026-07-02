import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Full preprocessing pipeline for the Give Me Some Credit dataset.
    Handles: outlier capping, missing values, feature engineering.
    """

    def __init__(self):
        self.income_median_ = None
        self.dependents_median_ = None
        self.cap_values_ = {}

    def fit(self, X, y=None):
        df = X.copy()

        # Learn cap values from training data only (99th percentile)
        cap_cols = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome']
        for col in cap_cols:
            self.cap_values_[col] = df[col].quantile(0.99)

        # Learn medians from training data only
        self.income_median_ = df['MonthlyIncome'].median()
        self.dependents_median_ = df['NumberOfDependents'].median()

        return self

    def transform(self, X, y=None):
        df = X.copy()

        # 1. Drop age = 0 rows (only during training; test data handled separately)
        df = df[df['age'] > 0]

        # 2. Cap sentinel values in late payment columns
        late_cols = [
            'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate'
        ]
        for col in late_cols:
            df[col] = df[col].clip(upper=10)

        # 3. Cap outliers using values learned from training data
        for col, cap in self.cap_values_.items():
            df[col] = df[col].clip(upper=cap)

        # 4. Impute missing values using training medians
        df['MonthlyIncome'] = df['MonthlyIncome'].fillna(self.income_median_)
        df['NumberOfDependents'] = df['NumberOfDependents'].fillna(self.dependents_median_)

        # 5. Feature engineering
        df['total_late_payments'] = (
            df['NumberOfTime30-59DaysPastDueNotWorse'] +
            df['NumberOfTime60-89DaysPastDueNotWorse'] +
            df['NumberOfTimes90DaysLate']
        )
        df['debt_to_income'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
        df['utilization_flag'] = (
            df['RevolvingUtilizationOfUnsecuredLines'] > 0.75
        ).astype(int)

        return df


def load_and_preprocess(path='../data/raw/cs-training.csv'):
    """Convenience function: load raw data and return cleaned DataFrame."""
    df = pd.read_csv(path, index_col=0)
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit(df)
    return preprocessor.transform(df), preprocessor