import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        optimal_dic, optimal_model = float("inf"), None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Help from the forum
                # logL = model.score(self.X, self.lengths)
                # p = n**2 + 2 * self.base_model(n).n_features * n - 1
                # logN = np.log(len(self.X))
                # BIC = -2 * logL + p * logN
                BIC = (-2 * self.base_model(n).score(self.X, self.lengths)) + (n**2 + 2 * self.base_model(n).n_features * n - 1) * np.log(len(self.X))

                if BIC < optimal_dic:
                    optimal_dic = BIC
                    optimal_model = self.base_model(n)

            except Exception:
                pass

        return optimal_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        optimal_dic, optimal_model = float("-inf"), None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                for n in list(self.words.keys()):
                    if n == self.this_word:
                        continue
                    X, lengths = self.hwords[n]
                    sum_log_P_X_all_but_i += self.base_model(n).score(X, lengths)

                # Help from the forum
                # log(P(X(i)) = self.base_model(n).score(self.X, self.lengths)
                # M = len(self.words)
                # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                DIC = self.base_model(n).score(self.X, self.lengths) - 1 / (len(self.words) - 1) * sum_log_P_X_all_but_i

                if optimal_dic < DIC:
                    optimal_dic = DIC
                    optimal_model = self.base_model(n)

            except Exception:
                pass

        if not optimal_model:
            optimal_model = self.base_model(self.n_constant)

        return optimal_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        optimal_cv, optimal_model = float("-inf"), None

        for n in range(self.min_n_components, self.max_n_components + 1):
            logL = []
            try:
                for cv_train_idx, cv_test_idx in KFold()(self.sequences):
                    X1, lengths1 = combine_sequences(cv_train_idx, self.sequences)
                    X2, lengths2 = combine_sequences(cv_test_idx, self.sequences)

                    # Reference from the sklearn library
                    model = GaussianHMM(n_components=n, covariance_type='diag', min_covar=0.001,
                                        startprob_prior=1.0, transmat_prior=1.0, means_prior=0,
                                        means_weight=0, covars_prior=0.01, covars_weight=1,
                                        algorithm='viterbi', random_state=self.random_state, n_iter=1000,
                                        tol=0.01, verbose=False, params='stmc', init_params='stmc').fit(X1, lengths1)

                    logL.append(model.score(X2, lengths2))

                    if optimal_cv < np.mean(logL):
                        optimal_cv = np.mean(logL)
                        optimal_model = model

            except Exception:
                pass

        if not optimal_model:
            optimal_model = self.base_model(self.n_constant)

        return optimal_model
