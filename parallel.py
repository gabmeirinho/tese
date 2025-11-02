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

import trace

import sys


import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """Load and prepare dataset - only runs in main process"""
    for attemp in range(3):
        try:
            if os.path.exists('base.csv'):
                df_orig = pd.read_csv('base.csv')
                print("Dataset loaded successfully")
                break
            else:
                import kagglehub
                import shutil
                path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
                csv_path = os.path.join(path, "base.csv")
                shutil.copy(csv_path, "base.csv")
                df_orig = pd.read_csv('base.csv')
                print("Dataset downloaded and loaded successfully")
                break
        except Exception as e:
            print(f"Attempting again to download dataset due to error: {e}")
            if attemp < 2:
                time.sleep(5)
            else:
                raise e
    
    df = df_orig
    train_df = df[df['month'] < 5].sample(frac=1, random_state=42)
    val_df = df[df['month'] == 5].sample(frac=1, random_state=42)
    test_df = df[df['month'] >= 6].sample(frac=1, random_state=42)

    train_val_df = pd.concat([train_df, val_df]).sample(frac=1, random_state=42)
    train_val_df.drop('month', axis=1, inplace=True)
    test_df.drop('month', axis=1, inplace=True)
    val_df.drop('month', axis=1, inplace=True)

    X_train_val = train_val_df.drop('fraud_bool', axis=1)
    y_train_val = train_val_df['fraud_bool']
    X_test = test_df.drop('fraud_bool', axis=1)
    y_test = test_df['fraud_bool']
    print(len(df))

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

    print(y_train_val.value_counts(), y_train_val.value_counts(), y_test.value_counts())

    feature_names = X_train_val.columns.tolist()
    n_features = len(feature_names)
    
    return X_train_val, y_train_val, X_test, y_test, feature_names, n_features, encoders, categorical_indices


# These will be set in main
X_train_val = None
y_train_val = None
X_test = None
y_test = None
feature_names = None
n_features = None


@dataclass
class Value(ABC):
    def evaluate(self):
        pass

class Scalar(ABC):
    pass

@weight(1.2)
@dataclass #Scalar Features (1)
class ScalarVar(Scalar): 
    index: Annotated[int, IntRange(0, 30-1)]  # Max features, will be set properly in main
    
    def evaluate(self, X_np):
        return X_np[:, self.index]
    
    def __str__(self):
        if feature_names:
            return feature_names[self.index]
        return f"Feature_{self.index}"
    
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
    
def create_fitness_function(X_tv_np, y_tv_np, ARCHIVE_TV_DF, baseline_tpr_at_fpr, categorical_indices, target_fpr_value = 0.05):
    """Create a picklable fitness function with all data embedded"""
    
    def fitness_function(individual: Scalar):
        # Now these are accessible in worker processes
        if str(individual) in ARCHIVE_TV_DF.columns:
            return [0.0, 0.0, 1000.0, 1000.0]
        
        start = time.perf_counter()
        new_feature = individual.evaluate(X_tv_np)
        if new_feature.ndim == 0:
            new_feature = np.full(X_tv_np.shape[0], new_feature)
    
        X_augmented_full = np.c_[X_tv_np, new_feature]
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
        for train_index, val_index in skf.split(X_tv_np, y_tv_np):
            X_train_fold, X_val_fold = X_augmented_full[train_index], X_augmented_full[val_index]
            y_train_fold, y_val_fold = y_tv_np[train_index], y_tv_np[val_index]
            
            model_fold = lgb.LGBMClassifier(
                n_estimators=50, max_depth=7, learning_rate=0.03, 
                num_leaves=10, boosting_type='gbdt', random_state=42, 
                n_jobs=-1, verbose=-1, 
                scale_pos_weight=(y_train_fold==0).sum() / (y_train_fold==1).sum()
            )
    
            model_fold.fit(X_train_fold, y_train_fold, categorical_feature=categorical_indices)
            
            val_probs = model_fold.predict_proba(X_val_fold)[:, 1]
    
            fpr, tpr, thresholds = roc_curve(y_val_fold, val_probs)
    
            tpr_at_fpr = 0.0
    
            if np.any(fpr <= target_fpr_value):
                valid_indices = np.where(fpr <= target_fpr_value)[0]
                best_index = valid_indices[np.argmax(tpr[valid_indices])]
                tpr_at_fpr = tpr[best_index]
    
            cv_scores.append(tpr_at_fpr)
    
        mean_cv_score = np.mean(cv_scores)
        tpr_diff = mean_cv_score - baseline_tpr_at_fpr
        
        # Analyze complexity
        features, num_operations = analyse_complexity(individual)
        
        end = time.perf_counter()
        elapsed = end - start
        
        # Return 4 values: [TPR, TPR_diff, num_operations, elapsed_time]
        return [mean_cv_score, tpr_diff, num_operations, elapsed]
    
    return fitness_function


def analyse_complexity(individual: Scalar):
    """Analyze complexity of expression"""
    if isinstance(individual, ScalarVar):
        return {individual.index}, 0  # unique features, operations
    
    total_features = set()
    total_operations = 1
    
    if hasattr(individual, 'left') and hasattr(individual, 'right'):
        left_features, left_operations = analyse_complexity(individual.left)
        right_features, right_operations = analyse_complexity(individual.right)
        total_features.update(left_features)
        total_features.update(right_features)
        total_operations += left_operations + right_operations
    elif hasattr(individual, 'value'):
        val_features, val_operations = analyse_complexity(individual.value)
        total_features.update(val_features)
        total_operations += val_operations
    
    return total_features, total_operations

class ArchiveManager:
    """Manages archive state for multiprocessing"""
    def __init__(self, X_tv_np, y_tv_np, X_train_val, y_train_val, baseline_tpr_at_fpr, 
                 categorical_indices, target_fpr_value, skf):
        self.X_tv_np = X_tv_np
        self.y_tv_np = y_tv_np
        self.baseline_tpr_at_fpr = baseline_tpr_at_fpr
        self.ARCHIVE_TV_DF_TEMP = X_train_val.copy()
        self.ARCHIVE_TEMP = []
        self.ARCHIVE_IND_DICT = {}
        self.categorical_indices = categorical_indices
        self.target_fpr_value = target_fpr_value
        self.skf = skf

class ArchiveStep(GeneticStep):
    def __init__(self, archive_manager):
        self.archive_manager = archive_manager
    
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
        for i, individual in enumerate(population):
            if individual.get_fitness(problem).fitness_components[0] > self.archive_manager.baseline_tpr_at_fpr:
                feature_new = individual.get_phenotype().evaluate(self.archive_manager.X_tv_np)
                if feature_new.ndim == 0:
                    feature_new = np.full(self.archive_manager.X_tv_np.shape[0], feature_new)
                self.archive_manager.ARCHIVE_TV_DF_TEMP[str(individual.get_phenotype())] = feature_new
                self.archive_manager.ARCHIVE_IND_DICT[str(individual.get_phenotype())] = individual
                self.archive_manager.ARCHIVE_TEMP.append(individual)
            yield individual
        
        if self.archive_manager.ARCHIVE_TEMP:
            print(f"Archive Size: {len(self.archive_manager.ARCHIVE_TEMP)}")

        if self.archive_manager.ARCHIVE_TEMP and generation % 5 == 0:
            print(f"Archive Size: {len(self.archive_manager.ARCHIVE_TEMP)}")
            
            model_importance = lgb.LGBMClassifier(
                n_estimators=50, max_depth=7, learning_rate=0.03, 
                num_leaves=10, boosting_type='gbdt', random_state=42, 
                n_jobs=-1, verbose=-1, 
                scale_pos_weight=(self.archive_manager.y_tv_np==0).sum() / (self.archive_manager.y_tv_np==1).sum()
            )
            model_importance.fit(
                self.archive_manager.ARCHIVE_TV_DF_TEMP.to_numpy(), 
                self.archive_manager.y_tv_np, 
                categorical_feature=self.archive_manager.categorical_indices
            )
            
            feature_importances = model_importance.feature_importances_
            feature_cols = self.archive_manager.ARCHIVE_TV_DF_TEMP.columns.tolist()
            sorted_indices = np.argsort(-feature_importances)
            sorted_importances = feature_importances[sorted_indices]
            sorted_feature_cols = [feature_cols[i] for i in sorted_indices]
            cumsum_importances = np.cumsum(sorted_importances) / np.sum(sorted_importances)
            threshold_cumsum = 0.8
            num_features_to_keep = np.argmax(cumsum_importances >= threshold_cumsum) + 1
            features_to_keep = sorted_feature_cols[:num_features_to_keep]
            
            print(f"Pruning to top {threshold_cumsum*100}% features: {num_features_to_keep} features kept.")
            self.archive_manager.ARCHIVE_TV_DF_TEMP = self.archive_manager.ARCHIVE_TV_DF_TEMP[features_to_keep]

            self.archive_manager.ARCHIVE_IND_DICT = {
                col: self.archive_manager.ARCHIVE_IND_DICT[col] 
                for col in features_to_keep if col in self.archive_manager.ARCHIVE_IND_DICT
            }

            fold_baseline_scores = []
            for train_idx, val_idx in self.archive_manager.skf.split(
                self.archive_manager.ARCHIVE_TV_DF_TEMP.to_numpy(), 
                self.archive_manager.y_tv_np
            ):
                X_fold_train = self.archive_manager.ARCHIVE_TV_DF_TEMP.to_numpy()[train_idx]
                y_fold_train = self.archive_manager.y_tv_np[train_idx]
                X_fold_val = self.archive_manager.ARCHIVE_TV_DF_TEMP.to_numpy()[val_idx]
                y_fold_val = self.archive_manager.y_tv_np[val_idx]

                model = lgb.LGBMClassifier(
                    n_estimators=50, max_depth=7, learning_rate=0.03, 
                    num_leaves=10, boosting_type='gbdt', random_state=42, 
                    n_jobs=-1, verbose=-1, 
                    scale_pos_weight=(y_fold_train==0).sum() / (y_fold_train==1).sum()
                )
                model.fit(X_fold_train, y_fold_train, categorical_feature=self.archive_manager.categorical_indices)
                
                probs = model.predict_proba(X_fold_val)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_fold_val, probs)
                
                tpr_at_fpr = 0.0
                target_fpr = self.archive_manager.target_fpr_value
                if np.any(fpr <= target_fpr):
                    valid_indices = np.where(fpr <= target_fpr)[0]
                    best_index = valid_indices[np.argmax(tpr[valid_indices])]
                    tpr_at_fpr = tpr[best_index]
                
                fold_baseline_scores.append(tpr_at_fpr)
            
            self.archive_manager.baseline_tpr_at_fpr = np.mean(fold_baseline_scores)
            print(f"New Baseline CV TPR: {self.archive_manager.baseline_tpr_at_fpr}, Archive shape: {self.archive_manager.ARCHIVE_TV_DF_TEMP.shape}")
            self.archive_manager.ARCHIVE_TEMP = []
       

if __name__ == '__main__':
    # Load data only in main process
    X_train_val, y_train_val, X_test, y_test, feature_names, n_features, encoders, categorical_indices = load_and_prepare_data()
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    baseline_top_scores = []

    for train_index, val_index in skf.split(X_train_val, y_train_val):
        X_train_fold, X_val_fold = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train_fold, y_val_fold = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        model_baseline = lgb.LGBMClassifier(n_estimators=50, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight= (y_train_fold==0).sum() / (y_train_fold==1).sum())

        model_baseline.fit(X_train_fold, y_train_fold, categorical_feature=categorical_indices)

        val_probs = model_baseline.predict_proba(X_val_fold)[:,1]

        fpr, tpr, thresholds = roc_curve(y_val_fold, val_probs)

        target_fpr = 0.05

        baseline_tpr_at_fpr = 0.0

        if np.any(fpr <= target_fpr):
            valid_indices = np.where(fpr <= target_fpr)[0]
            best_index = valid_indices[np.argmax(tpr[valid_indices])]
            baseline_tpr_at_fpr = tpr[best_index]

        baseline_top_scores.append(baseline_tpr_at_fpr)

    baseline_tpr_at_fpr = np.mean(baseline_top_scores)

    print(f"Validation TPR: {baseline_tpr_at_fpr}")

    grammar = extract_grammar([Add, Subtract, Multiply, ScalarVar], Scalar)
    print(f"Grammar: {repr(grammar)}")

    ARCHIVE_TV_DF = X_train_val.copy()
    ARCHIVE_TV_DF_TEMP = X_train_val.copy()

    X_tv_np = ARCHIVE_TV_DF.to_numpy()
    y_tv_np = y_train_val.to_numpy()
    
    target_fpr_value = 0.05

    fitness_function = create_fitness_function(
        X_tv_np, y_tv_np, ARCHIVE_TV_DF, 
        baseline_tpr_at_fpr, categorical_indices, target_fpr_value
    )

    archive_manager = ArchiveManager(
        X_tv_np, y_tv_np, X_train_val, y_train_val,
        baseline_tpr_at_fpr, categorical_indices, 
        target_fpr_value, skf
    )


    def lexicase_step():
        return SequenceStep(
            ArchiveStep(archive_manager),  # ← Pass archive manager
            ParallelStep(
                [
                    ElitismStep(),
                    SequenceStep(
                        TournamentSelection(tournament_size=3),
                        GenericCrossoverStep(0.9),
                        GenericMutationStep(0.1),
                    )
                ],
                weights=[0.1, 0.9]
            ),
        )

    prob = MultiObjectiveProblem(
        fitness_function=fitness_function,
        minimize=[False, False, True, True],
    )
    r = NativeRandomSource(123)
    alg = GeneticProgramming(
        problem=prob,
        budget=TimeBudget(600),
        population_size=20,
        representation=TreeBasedRepresentation(grammar, MaxDepthDecider(r, grammar, 5)),
        random=r,
        step=lexicase_step(),
        tracker=ProgressTracker(
            prob,
            evaluator=ParallelEvaluator(),
            recorders=[CSVSearchRecorder(
                csv_path='output.csv', 
                problem=prob, 
                fields={
                        "Eval Time": lambda t,i,p: i.get_fitness(p).fitness_components[3],
                        "TPR Test": lambda t,i,p: i.get_fitness(p).fitness_components[0],
                        "TPR Test Diff": lambda t,i,p: i.get_fitness(p).fitness_components[1],
                        "Expression": lambda t, i, p: i.get_phenotype(),
                        "Num Operations": lambda t,i,p: i.get_fitness(p).fitness_components[2],
                        'Generation': lambda t,i,p: i.metadata["generation"]
                        },
                only_record_best_individuals=False)]
        )
    )

    # tracer = trace.Trace(trace=1, count=0)
    # log_file = open('trace.log', 'w', encoding='utf-8')
    # old_stdout = sys.stdout
    # sys.stdout = log_file
    # tracer.run('alg.search()')
    # sys.stdout = old_stdout
    # log_file.close()
    # print("Trace saved to trace.log")

    solutions = alg.search()



