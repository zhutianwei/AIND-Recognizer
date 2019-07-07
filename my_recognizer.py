import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # sequences = test_set.get_all_sequences()
    Xlengths = test_set.get_all_Xlengths()

    for n in list(range(len(Xlengths))):
        model_dict = {}
        optimal_n = float("-inf")
        optimal_word = None

        for word in models.keys():
            model = models.get(word)
            try:
                X, lengths = Xlengths[n]
                model_dict[word] = model.score(X, lengths)

            except Exception:
                model_dict[word] = float("-inf")

            if optimal_n < model_dict[word]:
                optimal_n, optimal_word = model_dict[word], word

        probabilities.append(model_dict)
        guesses.append(optimal_word)

    return probabilities, guesses
