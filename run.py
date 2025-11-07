# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, List

import random

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar import extract_grammar
from geneticengine.grammar.decorators import weight
from geneticengine.problems import SingleObjectiveProblem, MultiObjectiveProblem
from geneticengine.random.sources import NativeRandomSource
from geneticengine.algorithms.gp.gp import GeneticProgramming
from geneticengine.evaluation.budget import TimeBudget, EvaluationBudget
from geneticengine.representations.tree.initializations import MaxDepthDecider, FullDecider, ProgressivelyTerminalDecider, PositionIndependentGrowDecider
from geneticengine.representations.tree.operators import GrowInitializer, PositionIndependentGrowInitializer, FullInitializer, RampedHalfAndHalfInitializer
from geneticengine.algorithms.gp.operators.initializers import HalfAndHalfInitializer, StandardInitializer
from geneticengine.representations.tree.treebased import TreeBasedRepresentation
from geneticengine.representations.grammatical_evolution.structured_ge import StructuredGrammaticalEvolutionRepresentation
from geneticengine.evaluation.recorder import CSVSearchRecorder
from geneticengine.evaluation.tracker import ProgressTracker
from geneticengine.evaluation.parallel import ParallelEvaluator

from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import LexicaseSelection, TournamentSelection

from geneticengine.solutions.individual import Individual, PhenotypicIndividual
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.problems import Problem
from geneticengine.random.sources import RandomSource
from geneticengine.representations.api import RepresentationWithCrossover, Representation
from geneticengine.evaluation import Evaluator
from typing import Iterator, Any, TypeVar

from sklearn.datasets import load_breast_cancer

import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

import time

import seaborn as sns
import matplotlib.pyplot as plt

import os     

from functools import lru_cache

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
for attemp in range(3):
    try:
        if os.path.exists('Base.csv'):
            df = pd.read_csv('Base.csv')
            print("Dataset loaded successfully")
            break
        else:
            import kagglehub
            import shutil
            path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
            csv_path = os.path.join(path, "Base.csv")
            shutil.copy(csv_path, "Base.csv")
            df = pd.read_csv('Base.csv')
            print("Dataset downloaded and loaded successfully")
            break
    except Exception as e:
        print(f"Attempting again to download dataset due to error: {e}")
        if attemp < 2:
            time.sleep(5)
        else:
            raise e

# %%
df.info()

# %%
df.head(5)

# %%
train_val_df = df[df['month'] <= 5].sample(frac=1, random_state=42)
test_df = df[df['month'] >=6].sample(frac=1, random_state=42)

train_val_df.drop('month', axis=1, inplace=True)
test_df.drop('month', axis=1, inplace=True)

X_train_val = train_val_df.drop('fraud_bool', axis=1)
y_train_val = train_val_df['fraud_bool']
X_test = test_df.drop('fraud_bool', axis=1)
y_test = test_df['fraud_bool']


print(y_train_val.value_counts(), y_test.value_counts())

# %%
categorical_features = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]


encoders = {}
for feat in categorical_features:
    encoder = LabelEncoder()
    X_train_val[feat] = encoder.fit_transform(X_train_val[feat])
    X_test[feat] = encoder.transform(X_test[feat])
    encoders[feat] = encoder

categorical_indices = [X_train_val.columns.get_loc(feat) for feat in categorical_features]

# %%
feature_names = X_train_val.columns.tolist()
n_features = len(feature_names)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fpr_results = []
for train_index, val_index in skf.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
    
    # model_baseline = lgb.LGBMClassifier(max_bins=63,device_type='gpu',n_estimators=350, max_depth=14, learning_rate=0.03, num_leaves=17, boosting_type='gbdt', random_state=42, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
    model_baseline = lgb.LGBMClassifier(n_estimators=350, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
    model_baseline.fit(X_train, y_train, categorical_feature=categorical_indices)

    predictions = model_baseline.predict_proba(X_val)[:,1]
    
    fprs, tprs, thresholds = roc_curve(y_val, predictions)
    threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
    recall = np.max(tprs[fprs==max(fprs[fprs < 0.05])])
    print(recall)
    fpr_results.append(recall)

baseline_tpr = np.mean(fpr_results)
print("Average Recall at FPR < 5%:", baseline_tpr)

# %%
@dataclass
class Value(ABC):
    def evaluate(self):
        pass

class Scalar(ABC):
    pass

@weight(1.2)
@dataclass #Scalar Features (1)
class ScalarVar(Scalar): 
    index: Annotated[int, IntRange(0,n_features-1)]

    def evaluate(self, X_np):
        return X_np[:, self.index]
    
    def __str__(self):
        return feature_names[self.index]
    
#scalar -> scalar
@weight(0.2)
@dataclass 
class Add(Scalar):
    right: Scalar
    left: Scalar

    def evaluate(self, X_np):
        return self.left.evaluate(X_np) + self.right.evaluate(X_np)
    
    def __str__(self):
        return f"({self.left} + {self.right})"

@weight(0.2)
@dataclass
class Subtract(Scalar):
    right: Scalar
    left: Scalar

    def evaluate(self, X_np):
        return (self.left.evaluate(X_np)) - (self.right.evaluate(X_np))
    
    def __str__(self):
        return f"({self.left} - {self.right})"

@weight(0.2)
@dataclass
class Multiply(Scalar):
    right: Scalar
    left: Scalar

    def evaluate(self, X_np):
        return self.left.evaluate(X_np) * self.right.evaluate(X_np)
    
    def __str__(self):
        return f"({self.left} * {self.right})"

@weight(0.2)
@dataclass
class Divide(Scalar):
    right: Scalar
    left: Scalar

    def evaluate(self, X_np):
        denom = self.right.evaluate(X_np)
        denom = np.where(denom == 0, 1e-6, denom)  # Avoid division by zero
        return self.left.evaluate(X_np) / denom
    
    def __str__(self):
        return f"({self.left} / {self.right})"
    
@weight(0.2)
@dataclass
class Sqrt(Scalar):
    value: Scalar

    def evaluate(self, X_np):
        val = self.value.evaluate(X_np)
        val = np.asarray(val)
        val = np.clip(val, a_min=0.0, a_max=None)
        return np.sqrt(val)

    def __str__(self):
        return f"sqrt({self.value})"
    
@weight(0.2)
@dataclass
class Log(Scalar):
    value: Scalar

    def evaluate(self, X_np):
        val = self.value.evaluate(X_np)
        val = np.asarray(val)
        val = np.where(val <= 0, 1e-6, val)
        return np.log(val)
    
    def __str__(self):
        return f"log({self.value})"
    
grammar = extract_grammar([Add, Subtract, Multiply, Divide, Sqrt, Log, ScalarVar], Scalar)
print(f"Grammar: {repr(grammar)}")

# %%
def fitness_function(individual: Individual):
    # Avoid evaluating/adding duplicate engineered features
    ind_str = str(individual)
    if ind_str in X_train_val.columns or ind_str in FEATURE_INDIVIDUALS:
        return [0.0, 0.0, 0.0, 0.0, 1000] 
    
    start = time.perf_counter()
    new_feature = individual.evaluate(X_train_val.to_numpy())
    X_augmented = np.c_[X_train_val.to_numpy(), new_feature]

    cv_score = []
    for train_index, val_index in skf.split(X_train_val, y_train_val):
        X_train, X_val = X_augmented[train_index], X_augmented[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]
        
        # model = lgb.LGBMClassifier(device_type='cpu',n_estimators=350, max_depth=14, learning_rate=0.03, num_leaves=17, boosting_type='gbdt', random_state=42, n_jobs=1, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
        # model = lgb.LGBMClassifier(max_bins=63,device_type='gpu',n_estimators=350, max_depth=14, learning_rate=0.03, num_leaves=17, boosting_type='gbdt', random_state=42, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
        # model = lgb.LGBMClassifier(n_estimators=350, max_depth=14, learning_rate=0.03, num_leaves=17, boosting_type='gbdt', random_state=42, n_jobs=1, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
        # model = lgb.LGBMClassifier(max_bins=63,device_type='gpu',n_estimators=50, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
        model = lgb.LGBMClassifier(n_estimators=50, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, n_jobs=1, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
        model.fit(X_train, y_train, categorical_feature=categorical_indices)
        
        predictions = model.predict_proba(X_val)[:,1]
        
        fprs, tprs, thresholds = roc_curve(y_val, predictions)
        threshold = np.min(thresholds[fprs==max(fprs[fprs < 0.05])])
        recall = np.max(tprs[fprs==max(fprs[fprs < 0.05])])
        cv_score.append(recall)
    end = time.perf_counter()
    elapsed = end - start
    return [np.mean(cv_score), cv_score[0], cv_score[1], cv_score[2], elapsed]

# %%
ELITE_DICT = {}
FEATURE_INDIVIDUALS = {}  # Add this new dictionary to store individuals permanently

class UpdateStep(GeneticStep):
    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random: RandomSource,
        population: Iterator[PhenotypicIndividual],
        target_size: int,
        generation: int,
    ) -> Iterator[PhenotypicIndividual]:
        global X_train_val
        sorted_population = sorted(population, key=lambda ind: ind.get_fitness(problem).fitness_components[0], reverse=True)
        for ind in sorted_population[:5]:
            ind_str = str(ind)
            if ind_str in FEATURE_INDIVIDUALS or ind_str in X_train_val.columns:
                continue
            if ind_str not in ELITE_DICT:
                ELITE_DICT[ind_str] = {"count": 1, "individual": ind}
            else:
                ELITE_DICT[ind_str]["count"] += 1

        if ELITE_DICT:
            top_elite = sorted(ELITE_DICT.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
            print(f"Top 5 elite candidates by count (gen {generation}):")
            for rank, (estr, data) in enumerate(top_elite, start=1):
                print(f"  {rank}. {estr} -> {data['count']}")
        
        if generation % 5 == 0:
            elite_features = [ind_str for ind_str, data in ELITE_DICT.items() if data["count"] >= 4]
            if elite_features:
                for elite_str in elite_features:
                    # Ensure we never add duplicate features to X_train_val
                    if elite_str in FEATURE_INDIVIDUALS or elite_str in X_train_val.columns:
                        # Clean up from ELITE_DICT to avoid reprocessing later
                        del ELITE_DICT[elite_str]
                        continue
                    elite_ind = ELITE_DICT[elite_str]["individual"]
                    print(f" - {elite_str} (appeared {ELITE_DICT[elite_str]['count']} times)")
                    new_feature = elite_ind.get_phenotype().evaluate(X_train_val.to_numpy())
                    X_train_val = pd.DataFrame(
                        np.c_[X_train_val.to_numpy(), new_feature],
                        columns=X_train_val.columns.tolist() + [elite_str]
                    )
                    FEATURE_INDIVIDUALS[elite_str] = elite_ind
                    del ELITE_DICT[elite_str]
            print(f"Updated feature set size: {X_train_val.shape[1]}")
            # Reset elite counts to prioritize stability within the next 5-gen block
            ELITE_DICT.clear()
        return iter(sorted_population)

# %%
def custom_step():
    return SequenceStep(
        UpdateStep(),
        ParallelStep(
            [
                ElitismStep(),
                NoveltyStep(),
                SequenceStep(
                    LexicaseSelection(epsilon=True),
                    # TournamentSelection(tournament_size=3),
                    GenericCrossoverStep(0.9),
                    GenericMutationStep(0.1),
                )
            ],
            weights=[0.04, 0.01, 0.9]
            # weights=[0.05, 0.95]
        ),
    )

prob = MultiObjectiveProblem(
    fitness_function=fitness_function,
    minimize=[False, False, False, False, True]
)

r = NativeRandomSource(123)
alg = GeneticProgramming(
    problem=prob,
    budget=TimeBudget(1800),
    population_size=50,
    representation=TreeBasedRepresentation(grammar, MaxDepthDecider(r, grammar, 5)),
    random=r,
    step=custom_step(),
    tracker=ProgressTracker(
        prob,
        evaluator=ParallelEvaluator(),
        recorders=[CSVSearchRecorder(
            csv_path='output.csv', 
            problem=prob, 
            fields={
                    "TPR": lambda t,i,p: i.get_fitness(p).fitness_components[0],
                    "TPR Diff": lambda t,i,p: i.get_fitness(p).fitness_components[0] - baseline_tpr,
                    "Expression": lambda t, i, p: i.get_phenotype(),
                    'Generation': lambda t,i,p: i.metadata["generation"]
                    },
            only_record_best_individuals=False)]
    )
    
)

solutions = alg.search()


# %%
X_test_augmented = X_test.to_numpy()

print("\nAdding engineered features to test set:")
for feature_name, elite_ind in FEATURE_INDIVIDUALS.items():
    print(f"Adding feature: {feature_name}")
    new_feature = elite_ind.get_phenotype().evaluate(X_test.to_numpy())
    X_test_augmented = np.c_[X_test_augmented, new_feature]

print(f"Total engineered features added: {len(FEATURE_INDIVIDUALS)}")

print("\nTraining final model with 3-fold cross-validation on test set...")

test_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

test_tpr_scores = []
test_auc_scores = []
test_thresholds = []

for fold, (train_idx, test_idx) in enumerate(test_skf.split(X_test_augmented, y_test), 1):
    print(f"\n--- Fold {fold} ---")
    
    X_test_fold = X_test_augmented[test_idx]
    y_test_fold = y_test.iloc[test_idx]

    model_final = lgb.LGBMClassifier(n_estimators=350, max_depth=14, learning_rate=0.03, num_leaves=17, boosting_type='gbdt', random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight=(y_train_val==0).sum() / (y_train_val==1).sum())

    
    model_final.fit(X_train_val, y_train_val, categorical_feature=categorical_indices)
    
    test_predictions = model_final.predict_proba(X_test_fold)[:, 1]
    
    fprs_test, tprs_test, thresholds_test = roc_curve(y_test_fold, test_predictions)
    fpr_threshold = 0.05
    valid_fprs = fprs_test[fprs_test < fpr_threshold]
    
    if len(valid_fprs) > 0:
        max_valid_fpr = max(valid_fprs)
        fold_tpr = np.max(tprs_test[fprs_test == max_valid_fpr])
        fold_threshold = np.min(thresholds_test[fprs_test == max_valid_fpr])
        test_tpr_scores.append(fold_tpr)
        test_thresholds.append(fold_threshold)
        
        print(f"Fold {fold} TPR at FPR < 5%: {fold_tpr:.4f}")
        print(f"Fold {fold} Threshold: {fold_threshold:.4f}")
    else:
        print(f"Warning: Fold {fold} - No FPR values below 5% found")
    
    fold_auc = roc_auc_score(y_test_fold, test_predictions)
    test_auc_scores.append(fold_auc)
    print(f"Fold {fold} AUC-ROC: {fold_auc:.4f}")

print("\n" + "="*50)
print("CROSS-VALIDATION SUMMARY")
print("="*50)

if test_tpr_scores:
    mean_test_tpr = np.mean(test_tpr_scores)
    std_test_tpr = np.std(test_tpr_scores)
    print(f"\nTPR at FPR < 5%:")
    print(f"  Mean: {mean_test_tpr:.4f} Â± {std_test_tpr:.4f}")
    print(f"  Min:  {np.min(test_tpr_scores):.4f}")
    print(f"  Max:  {np.max(test_tpr_scores):.4f}")
    print(f"\nBaseline TPR: {baseline_tpr:.4f}")
    print(f"Improvement: {mean_test_tpr - baseline_tpr:.4f}")

if test_auc_scores:
    mean_test_auc = np.mean(test_auc_scores)
    std_test_auc = np.std(test_auc_scores)
    print(f"\nAUC-ROC:")
    print(f"  Mean: {mean_test_auc:.4f} Â± {std_test_auc:.4f}")
    print(f"  Min:  {np.min(test_auc_scores):.4f}")
    print(f"  Max:  {np.max(test_auc_scores):.4f}")

if test_thresholds:
    mean_threshold = np.mean(test_thresholds)
    print(f"\nMean Threshold: {mean_threshold:.4f}")

print("="*50)

# %%




