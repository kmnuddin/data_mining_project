from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.svm import SVC

from bson import json_util
import json
import os
import pickle
from hyperopt import STATUS_OK
import numpy as np

RESULTS_DIR = "results/"

def run_svm(args):

    band = args['band']

    data_folder = 'train_test_data'
    data_path = os.path.join(data_folder, '{}.npz'.format(band))

    data = np.load(data_path)

    x_train = data['X_train']
    x_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    C = args['params']['C']
    kernel = args['params']['kernel']
    degree = args['params']['degree']
    gamma = args['params']['gamma']

    svc = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    svc.fit(x_train, y_train)

    y_pred = svc.predict(x_test)

    acc = accuracy_score(y_pred, y_test)
    mse = mean_squared_error(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cfm = confusion_matrix(y_test, y_pred)
    model_name = 'svm_{}_acc_{}'.format(band, str(acc))

    results = {
        'space': args,
        'loss': -acc,

        'C': C,
        'kernel': kernel,
        'degree': degree,
        'gamma': gamma,

        'cr': cr,
        'cfm': cfm,
        'acc': acc,
        'mse': mse,
        'status': STATUS_OK
    }
    pickle.dump(svc, open('models/{}.pkl'.format(model_name), 'wb'))

    return results


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=json_util.default, sort_keys=True,
            indent=4, separators=(',', ': ')
        )

def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=json_util.default, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))
