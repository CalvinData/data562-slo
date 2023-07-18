"""
This module tests the given SVM model using the specified testset.
"""
import logging
import pickle
from pathlib import Path
from fire import Fire
from sklearn.metrics import f1_score

from src.model_utilities import load_dataset, translate_predicted, set_labels

logger = logging.getLogger(__name__)


def model_test(
        dataset_path='.',
        testset_filename='autocode.csv',
        labels=None,
        model_filename='model.pkl',
        encoding='utf-8',
        logging_level=logging.INFO
        ):
    """This tool tests the given stance detection model using the specified testset.

    Keyword Arguments:
        dataset_path -- the system path from which to load the raw JSON files
            (default='.')
        testset_filename -- the name of the training set file
            (default='autocode.csv')
        labels -- the training target labels, which should set
            negative = 0, positive = 1 due to the calculation for macroF measure.
            (default: None, will be set to ['against', 'for', 'neutral', 'na'])
        model_filename -- the name of the model file to test
            (default='model.pkl')
        encoding -- the file encoding to use
            (default: 'utf-8')
        logging_level -- the level of logging to use
            (default: logging.INFO)
    """
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=__name__ + '.log',
        filemode='a'
        )
    logger.info('testing SVM model...')

    testset_filepath = Path(dataset_path, testset_filename)
    model_filepath = Path(dataset_path, model_filename)

    labels = set_labels(labels)
    
    x_test, y_test = load_dataset(testset_filepath, labels, encoding)

    logger.info('\tloading model from %s', model_filepath)
    model = pickle.load(open(model_filepath, 'rb'))

    y_predicted = model.predict(x_test)

    logger.info('\tcorrect labels: %s', translate_predicted(y_test, labels))
    logger.info('\tpredicted labels: %s', translate_predicted(y_predicted, labels))
    results = f1_score(y_test, y_predicted, labels=[0, 1, 2], average='macro')
    logger.info('\tF1 score: %f', results)
    print(f'F1 score: {results}')


if __name__ == '__main__':
    Fire(model_test)
