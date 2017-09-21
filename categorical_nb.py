
import sklearn.naive_bayes
import numpy as np
from numpy.matlib import repmat
#from scipy.misc import logsumexp
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.utils.extmath import safe_sparse_dot, logsumexp

class CategoricalNB(sklearn.naive_bayes.BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Categorical models.
    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    CategoricalNB is designed for categorical features.
    Read more in the :ref:`User Guide <categorical_naive_bayes>`.
    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.
    class_prior : array-like, size=[n_classes,], optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.
    Attributes
    ----------
    class_log_prior_ : array, shape = [n_classes]
        Log probability of each class (smoothed).
    feature_log_prob_ : array of dictionaries, shape = [n_classes, n_features, n_feature_vals]
        Empirical log probability of features given a class, P(x_i|y).
    class_count_ : array, shape = [n_classes]
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.
    feature_count_ : array, shape = [n_classes, n_features, num_feature_vals]
        Number of samples encountered for each (class, feature, feature val)
        during fitting. This value is weighted by the sample weight when
        provided.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(2, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import CategoricalNB
    >>> clf = CategoricalNB()
    >>> clf.fit(X, Y)
    CategoricalNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    >>> print(clf.predict(X[2:3]))
    [3]
    References
    ----------
    C.D. Manning, P. Raghavan and H. Schuetze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234-265.
    http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41-48.
    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """

    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True,
                 class_prior=None,
                 # feature_space=None,
                 output_space=None,n_classes=0,max_EM_iter=50):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        if class_prior is None:
            self.class_log_prior_ = None
        else:
            self.class_log_prior_ = np.log(class_prior)

        self.output_space = output_space
        self.map_output_space = {}
        self.n_classes = n_classes

        self.class_count_ = None
        self.feature_count_ = None

        if n_classes > 0:
            self._reset_class_count()
            
        if output_space is not None:
            self._init_output_space()

        #self.feature_space = feature_space
        self.feature_space = None
        self.n_features = None
        self.n_features_binarized_ = None
        self.n_feature_vals = None
        self.feature_indexes_ = None

        # if feature_space is not None:
        #    self._init_feature_space()

        self.delta_ = None
        self.max_EM_iter = max_EM_iter
        self.feature_log_prob_ = None

        self.missing_class_val_itemindex_ = None
        self.available_class_val_itemindex_ = None

    def rand_init_class_log_prior(self):
        self.class_log_prior_ = np.log(self._rand_unif_dist(self.n_classes))

    def rand_init_feature_prob(self):
        self.feature_log_prob_ = np.zeros((self.n_classes, self.n_features_binarized_),dtype=np.float64)
        for c in range(self.n_classes):
            for i in range(self.n_features):
                indexes = self.feature_indexes_[i]
                self.feature_log_prob_[c,indexes] = np.log(self._rand_unif_dist(self.n_feature_vals[i]))
                
    def _set_n_features_binarized(self, dict_vec):
        self.n_features_binarized_ = len(dict_vec.get_feature_names())
        
    def _set_class_val_itemindex(self, y):
        self.missing_class_val_itemindex_ = []
        self.available_class_val_itemindex_ = []
        for l in range(len(y)):
            if y[l] is None:
                self.missing_class_val_itemindex_.append(l)
            else:
                self.available_class_val_itemindex.append(l)
                    
    def _set_n_feature_vals(self):
        self.n_feature_vals = []
        for i in range(self.n_features):
            self.n_feature_vals.append(len(self.feature_space[i]))

    # def _set_map_feature_space(self, dict_vec):
    #     self.map_feature_space = np.empty(self.n_features,dtype=set)
    #     for i in range(self.n_features):
    #         self.map_feature_space[i] = {}
    #     feat_names = dict_vec.get_feature_names()
    #     for r,v in zip(range(len(feat_names)),feat_names):
    #         v_list = v.split('=')
    #         i = int(v_list[0].strip())
    #         val = v_list[1].strip()
    #         self.map_feature_space[i][val] = r

    # def set_feature_space(self, X):
    #     self.n_features = X.shape[1]
    #     self.feature_space = []
    #     for i in range(self.n_features):
    #         self.feature_space.append(list({str(v) for v in frozenset(X[:,i])}))
    #     self._init_feature_space()

    def _init_output_space(self):
        self.n_classes = len(self.output_space)
        self.map_output_space = {}
        for k,s in zip(range(self.n_classes),[str(v) for v in self.output_space]):
            self.map_output_space[s] = k
        self._reset_class_count()

    def _init_feature_space(self, dict_vec):
        self.n_features = len(self.feature_space)
        self._set_n_feature_vals()
        #self._set_map_feature_space(dict_vec)
        self._set_feature_indexes(dict_vec)
        self._set_n_features_binarized(dict_vec)
        self._reset_feature_count()
            
    def set_output_space(self, y):
        S = {str(c) for c in y.unique()}
        self.output_space = list(S)
        self._init_output_space()

    def _set_feature_indexes(self, dict_vec):
        self.feature_indexes_ = np.empty(self.n_features,dtype=list)
        for i in range(self.n_features):
            self.feature_indexes_[i] = []
            vals = self.feature_space[i]
            i_str = str(i)
            for v in vals:
                self.feature_indexes_[i].append(dict_vec.vocabulary_[i_str + '=' + v])

    def set_binarized_input_info(self, dict_vec):
        #print("DictVectorizing input data")
        feature_names = dict_vec.get_feature_names()
        self.n_binarized_features = len(feature_names)
        orig_feat_idx = []
        for v in feature_names:
            v_list = v.split('=')
            i = int(v_list[0].strip())
            orig_feat_idx.append(i)
        self.n_features = np.max(orig_feat_idx) + 1
        self.feature_space = np.empty(self.n_features,dtype=list)
        for i in range(self.n_features):
            self.feature_space[i] = []
            
        feature_names = dict_vec.get_feature_names()
        for v in feature_names:
            v_list = v.split('=')
            i = int(v_list[0].strip())
            val = v_list[1]
            self.feature_space[i].append(val)
        self._init_feature_space(dict_vec)

    def binarize_output(self, y):
        if self.output_space is None:
            self.set_output_space(y)
        n_samples = y.shape[0]
        Y_bin = lil_matrix((n_samples, self.n_classes),dtype=np.float64)
        self._set_class_val_itemindex(y)
        available_indexes = self.available_class_val_itemindex_
        for l in available_indexes:
            Y_bin[l,self.map_output_space[str(y[l])]] = 1.0
        return csc_matrix(Y_bin)

    def _rand_unif_dist(self, n_vals):
        vals = np.zeros(n_vals+1)
        vals[n_vals] = 1.0
        vals[1:n_vals] = np.sort(np.random.rand(n_vals-1))
        return np.diff(vals)
    
    def _reset_count(self):
        self._reset_class_count()
        self._reset_feature_count()

    def _reset_feature_count(self):
        self.feature_count_ = np.zeros((self.n_classes, self.n_features_binarized_),dtype=np.float64)

    def _reset_class_count(self):
        self.class_count_ = np.zeros(self.n_classes,dtype=np.float64)
        
    def _M_step(self):
        self._update_feature_log_prob()
        self._update_class_log_prior(class_prior=self.class_prior)
        #self._update_class_log_prior()

    def _E_step(self,X_bin,Y_bin):
        #m = self.n_classes
        #num_samples = X_bin.shape[0]
        self.delta_ = self.predict_proba(X_bin)
        #self.delta_ = np.exp(self._joint_log_likelihood(X_bin))
        #print(self.delta_.shape)
        available_vals = self.available_class_val_itemindex_
        for l in available_vals:
            self.delta_[l,:] = np.zeros(self.n_classes,dtype=np.float64)
            self.delta_[l,np.argmax(Y_bin[l])] = np.float64(1.0)
        self._count(X_bin)

    def _count(self, X_bin):
        """Count and smooth feature occurrences."""
        self.class_count_ += self._expected_class_count()
        self.feature_count_ += self._expected_class_feature_count(X_bin)

    def _expected_class_feature_count(self, X_bin):
        return safe_sparse_dot(self.delta_.T,X_bin)

    def _expected_class_count(self):
        return np.sum(self.delta_,axis=0)

    # def _update_class_log_prob(self,class_prior=None):
    #     if class_prior is None:
    #         self.class_log_prob_ = (np.log(self.class_count_) - np.log(np.sum(self.class_count_)))
    #     else:
    #         self.class_log_prob_ = np.log(class_prior)
            
    def _update_feature_log_prob(self):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + self.alpha
        for k in range(self.n_classes):
            for i in range(self.n_features):
                idx_list = self.feature_indexes_[i]
                self.feature_log_prob_[k,idx_list] = (np.log(smoothed_fc[k,idx_list]) -
                                                      np.log(np.sum(smoothed_fc[k,idx_list])))

    # def predict_proba(self, X_bin):
    #     return np.exp(self.predict_log_proba(X_bin))

    # def predict_log_proba(self, X_bin):
    #     return self._joint_log_likelihood(X_bin)
                
    def _joint_log_likelihood(self, X_bin):
        """Calculate the posterior log probability of the samples X"""
        # check_is_fitted(self, "classes_")

        print("_joint_log_likelihood...")
        
        n_classes, n_features_bin = self.feature_log_prob_.shape
        n_samples, n_features_X_bin = X_bin.shape

        if n_features_X_bin != n_features_bin:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features_bin, n_features_X_bin))

        jll = (safe_sparse_dot(X_bin,self.feature_log_prob_.T) + repmat(self.class_log_prior_.reshape(1,n_classes),n_samples,1)) 

        print("Avg. log. likelihood: %f" % (np.mean(logsumexp(jll,axis=1))))
        print("done")
        
        return jll
        #return (safe_sparse_dot(X_bin,self.feature_log_prob_.T) + repmat(self.class_log_prior_.reshape(1,n_classes),n_samples,1))

                
    def fit(self, X_bin, Y_bin=None):
        """Fit Naive Bayes classifier according to X, y
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
            Returns self.
        """

        if Y_bin is None:
            num_samples = X_bin.shape[0]
            self.missing_class_val_itemindex_ = range(num_samples)
            self.available_class_val_itemindex_ = []

        self.classes_ = np.array(self.output_space)
        
        if self.class_log_prior_ is None:
            self.rand_init_class_log_prior()

        if self.feature_log_prob_ is None:
            self.rand_init_feature_prob()

        # Run Expectation-Maximization (EM) Algorithm

        self._reset_count()
            
        print(np.exp(self.class_log_prior_))
        
        for t in range(self.max_EM_iter):
            #print t
            self._E_step(X_bin,Y_bin)
            self._M_step()
            print(np.exp(self.class_log_prior_))
            self._reset_count()
            
        return self


    def partial_fit(self, X_bin, Y_bin=None, first_call=False):
        """Update Naive Bayes classifier according to X, y
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : object
            Returns self.
        """

        if Y_bin is None:
            num_samples = X_bin.shape[0]
            self.missing_class_val_itemindex_ = range(num_samples)
            self.available_class_val_itemindex_ = []

        self.classes_ = np.array(self.output_space)
        
        if self.class_log_prior_ is None:
            self.rand_init_class_log_prior()

        if self.feature_log_prob_ is None:
            self.rand_init_feature_prob()
        
        # Run 1 step of Expectation-Maximization (EM) Algorithm

        if first_call:
            self._reset_count()

        #print np.exp(self.class_log_prior_)
        
        self._E_step(X,y)
        self._M_step()
        #print np.exp(self.class_log_prior_)
            
        return self
