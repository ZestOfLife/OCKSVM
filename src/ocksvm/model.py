import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM, SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class OCKSVM(ClassifierMixin, BaseEstimator):
    def __init__(self, n_clusters=5, oc_nu=0.1, gamma='scale', svc_kernel='rbf', random_state=None):
        self.n_clusters = n_clusters
        self.oc_nu = oc_nu
        self.gamma = gamma
        self.svc_kernel = svc_kernel
        self.random_state = random_state
    
    def _get_tags(self):
        return {
            "multiclass": True,
            "requires_y": True,
            "non_deterministic": False,
            "no_validation": False,
            "poor_score": False,
        }

    def __sklearn_tags__(self):
        try:
            from sklearn.utils._estimator_tags import ClassifierTags
            tags = ClassifierTags()
            tags.target_tags.multiclass = True
            return tags
        except (ImportError, ModuleNotFoundError):
            # Fall back to _get_tags() if older version
            return super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else None

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr'], dtype=None)
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # Do Kmeans Clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters, 
            n_init='auto', 
            random_state=self.random_state
        )
        kmeans.fit(X)
        self.clusterer_ = kmeans
        
        cluster_labels = kmeans.labels_
        self.models_ = []

        for i in range(self.n_clusters):
            mask = (cluster_labels == i)
            X_c, y_c = X[mask], y[mask]
            
            if len(X_c) == 0:
                self.models_.append(None)
                continue

            unique_y = np.unique(y_c)
            if len(unique_y) > 1:
                # Soft Convex Hull with One Class SVM to get reduced Support Vectors
                oc_svm = OneClassSVM(nu=self.oc_nu).fit(X_c)
                is_sv = oc_svm.predict(X_c) == 1
                
                X_reduced, y_reduced = X_c[is_sv], y_c[is_sv]
                
                # Check if reduced X,y has multiple classes
                if len(np.unique(y_reduced)) > 1:
                    clf = SVC(kernel=self.svc_kernel, gamma=self.gamma, random_state=self.random_state)
                    clf.fit(X_reduced, y_reduced)
                    self.models_.append(clf)
                else:
                    # If reduced X,y has only one class, use non reduced X,y
                    clf = SVC(kernel=self.svc_kernel, gamma=self.gamma, random_state=self.random_state)
                    clf.fit(X_c, y_c)
                    self.models_.append(clf)
            else:
                # Store the single label for one class clusters
                self.models_.append(unique_y[0])
        
        return self

    def predict(self, X):
        check_is_fitted(self, attributes=['clusterer_'])
        X = check_array(X, accept_sparse=['csr'], dtype=None)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but OCKSVM is expecting "
                f"{self.n_features_in_} features as input."
            )

        if hasattr(self.clusterer_, "cluster_centers_"):
            X = X.astype(self.clusterer_.cluster_centers_.dtype, copy=False)
            
        closest_clusters = self.clusterer_.predict(X)
        preds = []
        
        for i, cluster_idx in enumerate(closest_clusters):
            model = self.models_[cluster_idx]
            if model is None:
                preds.append(self.classes_[0])
            elif isinstance(model, SVC):
                preds.append(model.predict(X[i].reshape(1, -1))[0])
            else:
                preds.append(model)
                
        return np.array(preds)