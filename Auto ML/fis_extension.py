# fis_extension.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FISIntegration:
    """
    A proof-of-concept fuzzy inference integration for sepsis risk classification.
    Responsible for:
      1) Loading/Preparing data
      2) Defining membership functions and rules
      3) Performing fuzzy inference (predict_fuzzy)
    
    PyCaret typically wants an sklearn-like estimator to do 'fit', 'predict', etc.,
    so we'll wrap this class in a FISClassifier below.
    """

    def __init__(
        self,
        sepsis_file="sepsis.csv",
        no_sepsis_file="no_sepsis.csv",
        sample_size=250,
        threshold=0.40,
        around_0_5_tolerance=0.05
    ):
        """
        :param sepsis_file: Path to sepsis CSV file
        :param no_sepsis_file: Path to no-sepsis CSV file
        :param sample_size: Number of patients to sample from each dataset
        :param threshold: Risk threshold for labeling 1 (sepsis) vs 0 (no sepsis)
        :param around_0_5_tolerance: Tolerance around 0.5 for 'close to half' analysis
        """
        self.sepsis_file = sepsis_file
        self.no_sepsis_file = no_sepsis_file
        self.sample_size = sample_size
        self.threshold = threshold
        self.around_0_5_tolerance = around_0_5_tolerance
        self.combined_data = None
        # Key objects to be populated after data setup
        self.selected_features = [
            'bp_systolic', 'resp', 'bun', 'heart_rate', 'bicarbonate'
        ]
        self.stats = None
        self.antecedents = {}
        self.sepsis_ctrl = None
        self.sepsis_simulation = None

        # Data
        self.X = None
        self.y = None
        self.risk_scores = None
        self.predictions = None

    def load_and_prepare_data(self):
        """
        Loads the sepsis and no-sepsis CSV files, samples data, merges,
        and computes stats needed for fuzzy membership.
        """
        # Load data
        sepsis_data = pd.read_csv(self.sepsis_file)
        no_sepsis_data = pd.read_csv(self.no_sepsis_file)

        # Add target column
        sepsis_data['sepsis_icd'] = 1
        no_sepsis_data['sepsis_icd'] = 0

        # Combine for overall stats
        combined_data = pd.concat([sepsis_data, no_sepsis_data], ignore_index=True)
        self.combined_data = combined_data  # Store for external access

        # Sample from each
        sepsis_sample = sepsis_data.sample(n=self.sample_size, random_state=42)
        no_sepsis_sample = no_sepsis_data.sample(n=self.sample_size, random_state=42)
        selected_combined_data = pd.concat([sepsis_sample, no_sepsis_sample], ignore_index=True)

        # Impute missing values with median
        for feature in self.selected_features:
            median_value = selected_combined_data[feature].median()
            selected_combined_data[feature].fillna(median_value, inplace=True)

        # Define X and y
        self.X = selected_combined_data[self.selected_features]
        self.y = selected_combined_data['sepsis_icd'].tolist()

        # Compute descriptive stats on entire combined dataset
        self.stats = combined_data[self.selected_features].agg(["min", "max", "mean", "median", "std"]).transpose()
        self.stats.columns = ["Min", "Max", "Mean", "Median", "Std Dev"]
        self.stats["Min-Max Range"] = self.stats["Max"] - self.stats["Min"]

    def setup_fis(self):
        """
        Define membership functions, rules (using a rule matrix),
        and create the fuzzy control system.
        """
        # Step sizes for each feature
        feature_steps = {
            'bicarbonate': 0.1,
            'bun': 1.0,
            'heart_rate': 1.0,
            'resp': 0.1,
            'bp_systolic': 1.0
        }

        # Create a dictionary to store the Antecedent objects
        self.antecedents = {}

        # Build universes + membership for each feature
        for feature in self.selected_features:
            step = feature_steps[feature]
            min_val = self.stats.loc[feature, 'Min']
            max_val = self.stats.loc[feature, 'Max']
            median_val = self.stats.loc[feature, 'Median']

            universe = np.arange(min_val, max_val + step, step)
            self.antecedents[feature] = ctrl.Antecedent(universe, feature)

            # Define membership functions for each feature
            self.antecedents[feature]['low'] = fuzz.trimf(
                universe, [min_val, min_val, median_val]
            )
            self.antecedents[feature]['normal'] = fuzz.trimf(
                universe, [min_val, median_val, max_val]
            )
            self.antecedents[feature]['high'] = fuzz.trimf(
                universe, [median_val, max_val, max_val]
            )

        # Define the Consequent for sepsis risk
        sepsis_risk = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'sepsis_risk')
        sepsis_risk['low'] = fuzz.trimf(sepsis_risk.universe, [0, 0, 0.5])
        sepsis_risk['high'] = fuzz.trimf(sepsis_risk.universe, [0.5, 1, 1])

        # -- RULE MATRIX APPROACH --
        # feature_order: the order in which columns appear in each row below
        feature_order = ['bicarbonate', 'bun', 'heart_rate', 'resp', 'bp_systolic']

        # rule_matrix: each row has 1 membership label per feature, then the last item is the outcome label
        rule_matrix = [
            ['normal', 'low',    'normal', 'normal', 'normal', 'low' ],
            ['normal', 'normal', 'low',    'normal', 'normal', 'low' ],
            ['high',   'normal', 'normal', 'normal', 'normal', 'low' ],
            ['normal', 'normal', 'normal', 'low',    'high',   'low' ],
            ['low',    'normal', 'normal', 'normal', 'normal', 'high'],
            ['normal', 'normal', 'normal', 'normal', 'low',    'high'],
            ['normal', 'normal', 'normal', 'high',   'normal', 'high']
        ]

        rules = []
        for row in rule_matrix:
            antecedent_labels = row[:-1]
            outcome_label = row[-1]  # "low" or "high"

            fuzzy_expr = None
            # Build an AND expression across all features
            for i, feature in enumerate(feature_order):
                mf_label = antecedent_labels[i]
                current_expr = self.antecedents[feature][mf_label]
                fuzzy_expr = current_expr if fuzzy_expr is None else (fuzzy_expr & current_expr)

            # The outcome membership on the Consequent
            outcome_expr = sepsis_risk[outcome_label]

            # Create the rule
            rule = ctrl.Rule(fuzzy_expr, outcome_expr)
            rules.append(rule)

        # Build the control system from all rules
        self.sepsis_ctrl = ctrl.ControlSystem(rules)
        self.sepsis_simulation = ctrl.ControlSystemSimulation(self.sepsis_ctrl)

    def predict_fuzzy(self, row):
        """
        Given a single row of data (features), compute the fuzzy-based sepsis risk
        and return the risk score + predicted label (0 or 1).
        """
        # Clip each feature to the [Min, Max] range
        for feature in self.selected_features:
            min_val = self.stats.loc[feature, 'Min']
            max_val = self.stats.loc[feature, 'Max']
            value = np.clip(row[feature], min_val, max_val)
            self.sepsis_simulation.input[feature] = value

        try:
            self.sepsis_simulation.compute()
            risk = self.sepsis_simulation.output['sepsis_risk']
        except Exception as e:
            print(f"Error computing FIS for row:\n{row}\n{e}")
            risk = 0

        # Determine predicted label
        label = 1 if risk >= self.threshold else 0
        return risk, label

    def run_inference(self):
        """
        Run fuzzy inference on all samples in self.X, store risk scores and predictions.
        """
        if self.sepsis_simulation is None:
            raise RuntimeError("Fuzzy system not set up. Call setup_fis() first.")

        self.risk_scores = []
        self.predictions = []

        for _, row in self.X.iterrows():
            self.sepsis_simulation.reset()  # Important for scikit-fuzzy
            risk, label = self.predict_fuzzy(row)
            self.risk_scores.append(risk)
            self.predictions.append(label)

    def evaluate_results(self):
        """
        Display histogram, confusion matrix, and additional metrics
        (accuracy, precision, recall, F1, TPR, TNR, FPR, FNR).
        """
        if not self.predictions:
            raise RuntimeError("No predictions found. Call run_inference() first.")

        # Closeness to 0.5
        close_to_half = [
            score for score in self.risk_scores
            if 0.5 - self.around_0_5_tolerance <= score <= 0.5 + self.around_0_5_tolerance
        ]
        ratio_close_to_half = len(close_to_half) / len(self.risk_scores)

        # Plot histogram of risk scores
        fig, ax = plt.subplots(figsize=(10, 6))
        n, bins, patches = plt.hist(
            self.risk_scores, bins=20,
            color='steelblue', edgecolor='black', alpha=0.7
        )
        plt.title('Distribution of Sepsis Risk Scores')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')

        # Highlight region around 0.5
        lower_bound = 0.5 - self.around_0_5_tolerance
        upper_bound = 0.5 + self.around_0_5_tolerance
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='0.5 mark')
        plt.axvspan(lower_bound, upper_bound, color='orange', alpha=0.2, label='±0.05 around 0.5')
        plt.legend()
        plt.show()

        print(f"Percentage of risk scores close to 0.5: {ratio_close_to_half * 100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(self.y, self.predictions)
        print("Confusion Matrix:")
        print(cm)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (FIS)')
        plt.show()

        # Extract TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

        # TPR, TNR, FPR, FNR
        tpr = recall  # same as recall
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"TPR:       {tpr:.4f}")
        print(f"TNR:       {tnr:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"FNR:       {fnr:.4f}")
    # Properties to access combined data
    @property
    def get_combined_data(self):
        """
        Returns the combined_data DataFrame.
        """
        return self.combined_data

class FISClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible classifier that wraps the FISIntegration class.
    This allows us to treat the Mamdani FIS as a custom model in PyCaret.
    """

    def __init__(self, fis_integration=None):
        """
        :param fis_integration: An instance of FISIntegration, already configured
                                with membership functions and data (optional).
        """
        self.fis_integration = fis_integration
        if self.fis_integration is None:
            self.fis_integration = FISIntegration()  # fallback, but user must call load/setup

    def fit(self, X, y=None):
        """
        In many fuzzy logic systems, we don't 'train' in the typical sense
        but we do need to ensure the FIS is set up and the data is available.
        Here we'll replicate the structure so that PyCaret doesn't complain.
        """

        # If the user hasn't set up data, we attempt to match X, y if possible
        # (But typically you'd call fis_integration.load_and_prepare_data first.)
        if self.fis_integration.X is None:
            # We'll store them in fis_integration so that run_inference can use them
            self.fis_integration.X = X
            self.fis_integration.y = y.tolist() if y is not None else None
        
        # If the FIS control system isn't set up, do it now
        if self.fis_integration.sepsis_simulation is None:
            self.fis_integration.setup_fis()
        
        return self

    def predict(self, X):
        """
        Return class labels (0 or 1). This is required by scikit-learn for classification tasks.
        """
        preds = []
        for _, row in X.iterrows():
            # We must reset the simulation for each row
            self.fis_integration.sepsis_simulation.reset()
            risk, label = self.fis_integration.predict_fuzzy(row)
            preds.append(label)
        return np.array(preds)

    def predict_proba(self, X):
        """
        Return predicted probabilities for each class: [prob_no_sepsis, prob_sepsis].
        This is optional but important if you want AUC, calibration, etc. in PyCaret.
        """
        probs = []
        for _, row in X.iterrows():
            self.fis_integration.sepsis_simulation.reset()
            risk, label = self.fis_integration.predict_fuzzy(row)
            # risk is basically the membership for sepsis = 1
            # so prob_sepsis = risk, prob_no_sepsis = 1 - risk
            probs.append([1 - risk, risk])
        return np.array(probs)
