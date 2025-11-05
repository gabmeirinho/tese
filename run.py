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
model_used = lgb.LGBMClassifier(n_estimators=50, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, n_jobs=-1, verbose=-1)
target_fpr_value = 0.05

# %% [markdown]
# ### Functions and Data Preprocessing

# %%
for attemp in range(3):
    try:
        if os.path.exists('Base.csv'):
            df_orig = pd.read_csv('Base.csv')
            print("Dataset loaded successfully")
            break
        else:
            import kagglehub
            import shutil
            path = kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022")
            csv_path = os.path.join(path, "Base.csv")
            shutil.copy(csv_path, "Base.csv")
            df_orig = pd.read_csv('Base.csv')
            print("Dataset downloaded and loaded successfully")
            break
    except Exception as e:
        print(f"Attempting again to download dataset due to error: {e}")
        if attemp < 2:
            time.sleep(5)
        else:
            raise e

# %%
df_orig.info()

# %%
df_orig.head(2)

# %%
#save the df with the values until month 6
df = df_orig
#train until month 6 and test after month 6
train_df = df[df['month'] < 5].sample(frac=1, random_state=42)
val_df = df[df['month'] == 5].sample(frac=1, random_state=42)
test_df = df[df['month'] >= 6].sample(frac=1, random_state=42)

train_df.drop('month', axis=1, inplace=True)
test_df.drop('month', axis=1, inplace=True)
val_df.drop('month', axis=1, inplace=True)

#split into X and y
X_train = train_df.drop('fraud_bool', axis=1)
y_train = train_df['fraud_bool']
X_val = val_df.drop('fraud_bool', axis=1)
y_val = val_df['fraud_bool']
X_test = test_df.drop('fraud_bool', axis=1)
y_test = test_df['fraud_bool']
print(len(df))

# %%
# X = df.drop(['fraud_bool'], axis=1)
# y = df['fraud_bool']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

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
    X_train[feat] = encoder.fit_transform(X_train[feat])
    X_val[feat] = encoder.transform(X_val[feat])
    X_test[feat] = encoder.transform(X_test[feat])
    encoders[feat] = encoder

categorical_indices = [X_train.columns.get_loc(feat) for feat in categorical_features]

# %%
print(y_train.value_counts(),y_val.value_counts(), y_test.value_counts())

# %% [markdown]
# ### Baseline Model

# %%
feature_names = X_train.columns.tolist()
n_features = len(feature_names)

# %%
model_baseline = lgb.LGBMClassifier(n_estimators=50, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())
model_baseline.fit(X_train, y_train, categorical_feature=categorical_indices)

val_probs = model_baseline.predict_proba(X_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, val_probs)

target_fpr = target_fpr_value
baseline_tpr_at_fpr = 0.0

if np.any(fpr <= target_fpr):
    valid_indices = np.where(fpr <= target_fpr)[0]
    best_index = valid_indices[np.argmax(tpr[valid_indices])]
    baseline_tpr_at_fpr = tpr[best_index]

print(f"Validation TPR: {baseline_tpr_at_fpr}")

# %% [markdown]
# ### Grammar

# %%
@dataclass
class Value(ABC):
    def evaluate(self):
        pass

class Scalar(ABC):
    pass

# %%
@weight(1.2)
@dataclass #Scalar Features (1)
class ScalarVar(Scalar): 
    index: Annotated[int, IntRange(0,n_features-1)]

    def evaluate(self, X_np):
        return X_np[:, self.index]
    
    def __str__(self):
        return feature_names[self.index]

# %%
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

# %%
grammar = extract_grammar([Add, Subtract, Multiply, Divide, Sqrt, Log, ScalarVar], Scalar)
print(f"Grammar: {repr(grammar)}")

# %% [markdown]
# ### Fitness and GP

# %%
ARCHIVE_T_DF_TEMP = X_train.copy()

ARCHIVE_TEMP : list[Individual] = []
ARCHIVE_IND_DICT: dict[str, Individual] = {}

X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()
X_val_np = X_val.to_numpy()
y_val_np = y_val.to_numpy()

# %%
def fitness_function(individual: Scalar): #individual -> expression
    
    if str(individual) in X_train.columns:
        return [0.0, 0.0, 1000.0, 1000.0]
    
    start = time.perf_counter()
    new_feature_train = individual.evaluate(X_train_np)
    if new_feature_train.ndim == 0:
        new_feature_train = np.full(X_train_np.shape[0], new_feature_train)
    X_augmented_train = np.c_[X_train_np, new_feature_train]

    new_feature_val = individual.evaluate(X_val_np)
    if new_feature_val.ndim == 0:
        new_feature_val = np.full(X_val_np.shape[0], new_feature_val)
    X_augmented_val = np.c_[X_val_np, new_feature_val]
        
    model_fold = lgb.LGBMClassifier(n_estimators=50, max_depth=7, learning_rate=0.03, num_leaves=10, boosting_type='gbdt', random_state=42, n_jobs=-1, verbose=-1, scale_pos_weight= (y_train==0).sum() / (y_train==1).sum())

    model_fold.fit(X_augmented_train, y_train_np, categorical_feature=categorical_indices)
        
    val_probs = model_fold.predict_proba(X_augmented_val)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_val_np, val_probs)

    target_fpr = target_fpr_value

    tpr_at_fpr = 0.0

    if np.any(fpr <= target_fpr):
        valid_indices = np.where(fpr <= target_fpr)[0]
        best_index = valid_indices[np.argmax(tpr[valid_indices])]
        tpr_at_fpr = tpr[best_index]

    tpr_diff = tpr_at_fpr - baseline_tpr_at_fpr
        
    features, num_operations = analyse_complexity(individual)

    end = time.perf_counter()
    elapsed = end - start
    return [tpr_at_fpr, tpr_diff, num_operations, elapsed]



# %%
def analyse_complexity(individual: Scalar):
    if isinstance(individual, ScalarVar):
        return {individual.index}, 0 #unique feature
    
    total_features = set()
    total_operations = 1
    if hasattr(individual, 'left') and hasattr(individual, 'right'):
        left_features, left_operations = analyse_complexity(individual.left)
        right_features, right_operations = analyse_complexity(individual.right)
        total_features.update(left_features)
        total_features.update(right_features)
        total_operations += left_operations + right_operations
    elif hasattr(individual, 'value'):
        value_features, value_operations = analyse_complexity(individual.value)
        total_features.update(value_features)
        total_operations += value_operations
    return total_features, total_operations


# %%
from sklearn.feature_selection import RFE

import pickle

class ArchiveStep(GeneticStep):
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
        global ARCHIVE_TEMP, baseline_tpr_at_fpr, ARCHIVE_T_DF_TEMP, ARCHIVE_IND_DICT, ARCHIVE_IND
        
        # Collect individuals with fitness > baseline
        for i, individual in enumerate(population):
            if str(individual.get_phenotype()) in ARCHIVE_T_DF_TEMP.columns:
                yield individual
                continue
            if individual.get_fitness(problem).fitness_components[0] > baseline_tpr_at_fpr:
                feature_new = individual.get_phenotype().evaluate(X_train_np)
                if feature_new.ndim == 0:
                    feature_new = np.full(X_train_np.shape[0], feature_new)
                ARCHIVE_T_DF_TEMP[str(individual.get_phenotype())] = feature_new
                ARCHIVE_IND_DICT[str(individual.get_phenotype())] = individual
                ARCHIVE_TEMP.append(individual)
            yield individual
        
        if ARCHIVE_TEMP:
            print(f"Archive Size: {len(ARCHIVE_TEMP)}")

        # === FIXED 40-FEATURE RFE PRUNING ===
        if ARCHIVE_TEMP and generation % 15 == 0:
            print(f"Archive Size at pruning: {len(ARCHIVE_TEMP)}")
            
            # Store current state for potential rollback
            ARCHIVE_T_DF_BACKUP = ARCHIVE_T_DF_TEMP.copy()
            ARCHIVE_IND_DICT_BACKUP = ARCHIVE_IND_DICT.copy()
            baseline_tpr_backup = baseline_tpr_at_fpr
            
            # === STEP 1: Define Fixed Target Size ===
            num_features = len(ARCHIVE_T_DF_TEMP.columns)
            original_cols = set(X_train.columns)
            num_original = len(original_cols)
            num_new_features = sum(1 for col in ARCHIVE_T_DF_TEMP.columns if col not in original_cols)
            
            # FIXED TARGET: Always select 40 features
            n_features_to_select = 40
            
            # Skip if already at or below target
            if num_features <= n_features_to_select:
                print(f"✅ Archive at or below target ({num_features} ≤ {n_features_to_select}), skipping pruning")
                ARCHIVE_TEMP = []
                return
            
            print(f"Current: {num_features} total ({num_original} original + {num_new_features} new)")
            print(f"⚠️  Pruning to fixed target: {num_features} → {n_features_to_select} features")
            
            # === STEP 2: RFE Feature Selection ===
            estimator = lgb.LGBMClassifier(
                n_estimators=50, 
                max_depth=7, 
                learning_rate=0.03, 
                num_leaves=10, 
                boosting_type='gbdt', 
                random_state=42, 
                n_jobs=-1, 
                verbose=-1,
                scale_pos_weight=(y_train_np==0).sum() / (y_train_np==1).sum()
            )
            
            print(f"Running RFE to select {n_features_to_select} features...")
            
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features_to_select,
                step=max(1, (num_features - n_features_to_select) // 15),
                verbose=0
            )
            
            rfe.fit(ARCHIVE_T_DF_TEMP.to_numpy(), y_train_np)
            
            # Get selected features
            selected_mask = rfe.support_
            feature_cols = ARCHIVE_T_DF_TEMP.columns.tolist()
            features_to_keep = [col for col, selected in zip(feature_cols, selected_mask) if selected]
            
            # === STEP 3: Ensure minimum new features (at least 10) ===
            new_features_kept = sum(1 for col in features_to_keep if col not in original_cols)
            MIN_NEW_FEATURES = 10
            
            if new_features_kept < MIN_NEW_FEATURES and num_new_features >= MIN_NEW_FEATURES:
                print(f"⚠️  Only {new_features_kept} new features selected, adding more to reach {MIN_NEW_FEATURES}")
                
                feature_ranking = list(zip(feature_cols, rfe.ranking_))
                feature_ranking.sort(key=lambda x: x[1])
                
                for col, rank in feature_ranking:
                    if col not in original_cols and col not in features_to_keep:
                        if len(features_to_keep) >= n_features_to_select:
                            break
                        features_to_keep.append(col)
                        new_features_kept += 1
                        if new_features_kept >= MIN_NEW_FEATURES:
                            break
            
            new_features_kept = sum(1 for col in features_to_keep if col not in original_cols)
            print(f"RFE selected: {len(features_to_keep)} features ({new_features_kept} new, {num_original} original)")
            
            # === STEP 4: Prune Archive ===
            ARCHIVE_T_DF_TEMP = ARCHIVE_T_DF_TEMP[features_to_keep]
            
            ARCHIVE_IND = [ARCHIVE_IND_DICT[col] for col in features_to_keep 
                          if col not in original_cols and col in ARCHIVE_IND_DICT]
            ARCHIVE_IND_DICT = {col: ARCHIVE_IND_DICT[col] for col in features_to_keep 
                               if col in ARCHIVE_IND_DICT}
            
            # === STEP 5: Build Validation Set ===
            X_val_archive = pd.DataFrame(index=X_val.index)
            for col in features_to_keep:
                if col in X_val.columns:
                    X_val_archive[col] = X_val[col]
                elif col in ARCHIVE_IND_DICT:
                    val_feature = ARCHIVE_IND_DICT[col].get_phenotype().evaluate(X_val_np)
                    if val_feature.ndim == 0:
                        val_feature = np.full(len(X_val), val_feature)
                    X_val_archive[col] = val_feature
            
            X_train_archive = ARCHIVE_T_DF_TEMP.to_numpy()
            X_val_archive_np = X_val_archive.to_numpy()
            
            # === STEP 6: Recalculate Categorical Indices ===
            categorical_feature_names = [X_train.columns[i] for i in categorical_indices 
                                        if i < len(X_train.columns)]
            new_categorical_indices = [list(ARCHIVE_T_DF_TEMP.columns).index(name) 
                                      for name in categorical_feature_names 
                                      if name in ARCHIVE_T_DF_TEMP.columns]
            
            # === STEP 7: Train Model on Pruned Archive ===
            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=7, learning_rate=0.03, 
                num_leaves=10, boosting_type='gbdt', random_state=42, 
                n_jobs=-1, verbose=-1, 
                scale_pos_weight=(y_train_np==0).sum() / (y_train_np==1).sum()
            )
            
            model.fit(X_train_archive, y_train_np, 
                     categorical_feature=new_categorical_indices if new_categorical_indices else None)
            probs = model.predict_proba(X_val_archive_np)[:, 1]
            
            # === STEP 8: Evaluate New Baseline ===
            fpr, tpr, thresholds = roc_curve(y_val_np, probs)
            
            tpr_at_fpr = 0.0
            target_fpr = target_fpr_value
            if np.any(fpr <= target_fpr):
                valid_indices = np.where(fpr <= target_fpr)[0]
                best_index = valid_indices[np.argmax(tpr[valid_indices])]
                tpr_at_fpr = tpr[best_index]
            else:
                print(f"⚠️  Warning: No FPR points <= {target_fpr}. Min FPR: {fpr.min():.4f}")
            
            # === STEP 9: Rollback if Performance Degrades ===
            performance_threshold = 0.88
            min_acceptable_tpr = baseline_tpr_backup * performance_threshold
            
            if tpr_at_fpr < min_acceptable_tpr:
                print(f"⚠️  ROLLBACK: RFE pruning degraded performance ({tpr_at_fpr:.4f} < {min_acceptable_tpr:.4f})")
                print(f"   Keeping original archive with {len(ARCHIVE_T_DF_BACKUP.columns)} features")
                ARCHIVE_T_DF_TEMP = ARCHIVE_T_DF_BACKUP
                ARCHIVE_IND_DICT = ARCHIVE_IND_DICT_BACKUP
                ARCHIVE_IND = [ARCHIVE_IND_DICT[col] for col in ARCHIVE_T_DF_BACKUP.columns 
                              if col not in original_cols and col in ARCHIVE_IND_DICT]
            else:
                baseline_tpr_at_fpr = tpr_at_fpr
                improvement = ((tpr_at_fpr - baseline_tpr_backup) / baseline_tpr_backup) * 100
                print(f"✅ RFE pruning successful: TPR {baseline_tpr_backup:.4f} → {baseline_tpr_at_fpr:.4f} ({improvement:+.2f}%)")
                print(f"   Archive shape: {ARCHIVE_T_DF_TEMP.shape}")
                
                # === SAVE TO PICKLE (OVERWRITE) ===
                # Only save picklable objects (no Individual objects)
                os.makedirs("checkpoints", exist_ok=True)
                with open("checkpoints/archive.pkl", "wb") as f:
                    pickle.dump({
                        'ARCHIVE_T_DF_TEMP': ARCHIVE_T_DF_TEMP,
                        'feature_expressions': {col: str(ARCHIVE_IND_DICT[col].get_phenotype()) 
                                               for col in ARCHIVE_IND_DICT.keys()},
                        'baseline_tpr_at_fpr': baseline_tpr_at_fpr,
                        'generation': generation
                    }, f)
                print(f"✅ Saved to checkpoints/archive.pkl")
            
            ARCHIVE_TEMP = []

# %%
def lexicase_step():
    return SequenceStep(
        ArchiveStep(),
        ParallelStep(
            [
                ElitismStep(),
                # NoveltyStep(),
                SequenceStep(
                    LexicaseSelection(epsilon=True),
                    # TournamentSelection(tournament_size=3),
                    GenericCrossoverStep(0.8),
                    GenericMutationStep(0.1),
                )
            ],
            # weights=[0.05, 0.05, 0.9]
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
    budget=TimeBudget(1800),
    population_size=50,
    representation=TreeBasedRepresentation(grammar, MaxDepthDecider(r, grammar, 5)),
    random=r,
    step=lexicase_step(),
    tracker=ProgressTracker(
        prob,
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

solutions = alg.search()

# %%
import pickle

with open("checkpoints/archive.pkl", "rb") as f:
    data = pickle.load(f)
    ARCHIVE_T_DF_TEMP = data['ARCHIVE_T_DF_TEMP']
    baseline_tpr_at_fpr = data['baseline_tpr_at_fpr']

print(f"Loaded: {ARCHIVE_T_DF_TEMP.shape}")
print(f"Feature expressions saved: {len(data.get('feature_expressions', {}))}")

# Extract feature names (already in ARCHIVE_T_DF_TEMP)
original_cols = set(X_train.columns)
ARCHIVE_IND = []  # Individual objects lost, but features preserved in dataframe

# Use ARCHIVE_T_DF_TEMP directly for training
print(f"Final ARCHIVE_IND contains {len(ARCHIVE_T_DF_TEMP.columns) - len(original_cols)} new features.")

# === BASELINE MODEL (original features only) ===
model_baseline = lgb.LGBMClassifier(
    n_estimators=350, max_depth=14, learning_rate=0.03, 
    num_leaves=17, boosting_type='gbdt', random_state=42, 
    n_jobs=-1, verbose=-1, 
    scale_pos_weight=(y_train==0).sum() / (y_train==1).sum()
)

model_baseline.fit(X_train, y_train, categorical_feature=categorical_indices)
original_probs = model_baseline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, original_probs)
target_fpr = target_fpr_value
original_tpr_at_fpr = 0.0

if np.any(fpr <= target_fpr):
    valid_indices = np.where(fpr <= target_fpr)[0]
    best_index = valid_indices[np.argmax(tpr[valid_indices])]
    original_tpr_at_fpr = tpr[best_index]

print(f"Original Test TPR at FPR {target_fpr}: {original_tpr_at_fpr}")

# === BUILD AUGMENTED TRAINING SET ===
print("Finalizing feature archive...")
original_cols = set(X_train.columns)
final_feature_cols = ARCHIVE_T_DF_TEMP.columns

ARCHIVE_IND = [ARCHIVE_IND_DICT[col] for col in final_feature_cols 
               if col not in original_cols and col in ARCHIVE_IND_DICT]

print(f"Final ARCHIVE_IND contains {len(ARCHIVE_IND)} new features.")

# === BUILD AUGMENTED TEST SET ===
X_test_enhanced = X_test.copy()
X_test_np = X_test.to_numpy()

for ind in ARCHIVE_IND:
    pheno_str = str(ind.get_phenotype())
    if pheno_str not in X_test_enhanced.columns:
        test_feature = ind.get_phenotype().evaluate(X_test_np)
        if test_feature.ndim == 0:
            test_feature = np.full(X_test_np.shape[0], test_feature)
        X_test_enhanced[pheno_str] = test_feature

X_test_enhanced = X_test_enhanced.loc[:, ~X_test_enhanced.columns.duplicated()]

# === BUILD AUGMENTED TRAINING SET (for model fitting) ===
# ARCHIVE_T_DF_TEMP already has all training samples with new features
X_train_augmented = ARCHIVE_T_DF_TEMP.copy()
y_train_augmented = y_train.copy()

# Recalculate categorical indices for augmented feature set
categorical_feature_names = [X_train.columns[i] for i in categorical_indices 
                            if i < len(X_train.columns)]
new_categorical_indices = [list(X_train_augmented.columns).index(name) 
                          for name in categorical_feature_names 
                          if name in X_train_augmented.columns]

# === AUGMENTED MODEL ===
model_aug = lgb.LGBMClassifier(
    n_estimators=350, max_depth=14, learning_rate=0.03, 
    num_leaves=17, boosting_type='gbdt', random_state=42, 
    n_jobs=-1, verbose=-1, 
    scale_pos_weight=(y_train_augmented==0).sum() / (y_train_augmented==1).sum()
)

model_aug.fit(X_train_augmented, y_train_augmented, 
             categorical_feature=new_categorical_indices if new_categorical_indices else None)

# === PREDICTIONS ON TEST SET ===
# Ensure test set has same column order as training set
final_test_cols = X_train_augmented.columns
X_test_final = X_test_enhanced[final_test_cols]

augmented_probs = model_aug.predict_proba(X_test_final)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, augmented_probs)
target_fpr = target_fpr_value
augmented_tpr_at_fpr = 0.0

if np.any(fpr <= target_fpr):
    valid_indices = np.where(fpr <= target_fpr)[0]
    best_index = valid_indices[np.argmax(tpr[valid_indices])]
    augmented_tpr_at_fpr = tpr[best_index]

print(f"Augmented Test TPR at FPR {target_fpr}: {augmented_tpr_at_fpr}")
print(f"Improvement in TPR at FPR {target_fpr}: {augmented_tpr_at_fpr - original_tpr_at_fpr}")

# %%



