from pathlib import Path
import h_labels
import pandas as pd
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss

LABELS = ['any', 'epidural', 'subdural', 'subarachnoid',
          'intraventricular', 'intraparenchymal', ]
P_LABELS = ['p_any', 'p_epidural', 'p_subdural', 'p_subarachnoid',
            'p_intraventricular', 'p_intraparenchymal', ]


data_path = Path(
    '/mnt/nas/GianlucasLopes/hemorragia/rsna-intracranial-hemorrhage-detection/')

train_df: pd.DataFrame
pkl_output = data_path / 'folds_df_plus.pkl'
with open(pkl_output, 'rb') as f:
    train_df = pickle.load(f)
#############################################################################

path_modelo_salvo = Path('./resultados/treino_8_6/')

results_df: pd.DataFrame
model_results_path = path_modelo_salvo / "model_results.pkl"
with open(model_results_path, 'rb') as f:
    results_df = pickle.load(f)

split_df = pd.DataFrame(results_df['result'].to_list(), columns=P_LABELS)
new_results_df = pd.concat([results_df, split_df], axis=1)
new_results_df = new_results_df.drop(columns=['result'])

#############################################################################

new_train_df = pd.merge(train_df, new_results_df,
                        how='inner', left_on='ID', right_on='id')
new_train_df = new_train_df.drop(columns=['id'])

#############################################################################


def get_or_nan(slices: list[float], index: int):
    try:
        return slices[index]
    except IndexError:
        return np.nan


grouped_df = new_train_df.sort_values(
    'PositionOrd').groupby('StudyInstanceUID')
data_list = []
truth_list = []
for study_uid, group in grouped_df:
    rows = [row for row in group.itertuples()]

    for i, row in enumerate(rows):
        record = {}
        record.update({
            'fold': row.fold,
            'pos': row.PositionOrd,
            'n_slice': len(rows),
        })
        labels = row.labels.split()
        onehot_labels = np.zeros(len(P_LABELS))
        for value in labels:
            onehot_labels[h_labels.label_to_num[value]] = 1.0
        truth_list.append(onehot_labels)
        for j in range(6):
            label = P_LABELS[j]
            slices = [row[label] for row in rows]
            ords = [row.PositionOrd for row in rows]

            right = {f'{label}_r{k}': get_or_nan(
                slices, i+k) for k in range(1, 10)}
            left = {f'{label}_l{k}': get_or_nan(
                slices, i-k) for k in range(1, 10)}
            record.update({
                label: row[label],
            })
            record.update(right)
            record.update(left)
        data_list.append(record)
# return data_list
X_data = pd.DataFrame(data_list)
y_data = pd.DataFrame(truth_list)
y_data['fold'] = X_data['fold']

_params_lgb = {
    'objective': 'binary',
    # "metric": 'auc',
    'num_leaves': 30,
    'min_child_samples': 5,
    'max_depth': 5,
    'learning_rate': 0.02,
    "boosting_type": "gbdt",
    "bagging_freq": 2,
    "bagging_fraction": 0.7,
    "bagging_seed": 11,
    "verbosity": -1,
    'reg_alpha': 0.9,
    'reg_lambda': 0.9,
    'colsample_bytree': 0.9,
    'importance_type': 'gain',
    'random_seed': 20,
    'n_estimators': 5000,
    'n_jobs': 6,
}

train_folds = list(range(0, 6))
valid_folds = list(range(6, 8))
test_folds = list(range(8, 10))


def split_data(X_data: pd.DataFrame, y_data: pd.DataFrame, folds: list[int]):
    X_split = X_data.loc[X_data['fold'].isin(folds)]
    y_split = y_data.loc[y_data['fold'].isin(folds)]
    X_split = X_split.drop(columns=['fold'])
    y_split = y_split.drop(columns=['fold'])
    return X_split, y_split


X_train, y_train = split_data(X_data, y_data, train_folds)
X_valid, y_valid = split_data(X_data, y_data, valid_folds)
X_test, y_test = split_data(X_data, y_data, test_folds)

test_all = []

for i in range(len(LABELS)):

    model = lgb.LGBMClassifier(**_params_lgb)
    model.fit(X_train, y_train[i],
              eval_set=[(X_valid, y_valid[i])],
              eval_metric='multi_logloss',
              verbose=50,
              early_stopping_rounds=100)
    pred_valid = model.predict_proba(
        X_valid, num_iteration=model.best_iteration_)[:, 1]
    pred_test = model.predict_proba(
        X_test, num_iteration=model.best_iteration_)[:, 1]

    eps=1e-6
    logloss = log_loss(y_valid, np.clip(pred_valid, eps, 1-eps))
    print(f'logloss for {LABELS[i]}', logloss)

    accuracy = accuracy_score(pred_test, y_test[i])

    test_all.append(pred_test)