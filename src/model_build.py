"""
This module builds an SVM model using the specified trainset.
"""
import logging
import pickle
from pathlib import Path
from fire import Fire

from src.model_utilities import load_dataset, set_labels
from src.model_svm import get_model

logger = logging.getLogger(__name__)


def model_build(
        dataset_path='.',
        trainset_filename='autocode.csv',
        word_vectors_filename='wordvec.vec',
        labels=None,
        model_filename='model.pkl',
        profile=True,
        encoding='utf-8',
        logging_level=logging.INFO
        ):
    """This tool builds the stance detection model using the specified trainset
    and word vectors.

    Keyword Arguments:
        dataset_path -- the system path from which to load the raw JSON files
            (default='.')
        trainset_filename -- the name of the training set file
            (default='autocode.csv')
        word_vectors_filename -- the name of the word vectors file
            (default='word_vectors.csv')
        labels -- the training target labels
            (default: ['against', 'for', 'neutral', 'na'])
        model_filename -- the name of the model file to save
            (default='model.pkl')
        profile -- whether to include use profile texts
            (default: True)
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
    logger.info('training SVM model...')

    trainset_filepath = Path(dataset_path, trainset_filename)
    word_vectors_filepath = Path(dataset_path, word_vectors_filename)
    model_filepath = Path(dataset_path, model_filename)

    labels = set_labels(labels)

    logger.info('\tloading training set from %s...', trainset_filepath)
    x_train_arrays, y_train_arrays = load_dataset(trainset_filepath, labels, encoding, profile)

    logger.info('\tbuilding/training SVM model...')
    model = get_model(word_vectors_filepath, profile)
    model.fit(x_train_arrays, y_train_arrays)

    logger.info('\tsaving model in %s...', model_filepath)
    with open(model_filepath, 'wb') as model_fout:
        pickle.dump(model, model_fout)


if __name__ == '__main__':
    Fire(model_build)
