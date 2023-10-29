import numpy as np


class DataGeneration:
    classes = {
        0: dict(mean=[3, 3, 3, 3], covariance=np.identity(4)),
        1: dict(mean=[-3, -3, -3, -3], covariance=np.identity(4))
    }

    @classmethod
    def generate_data(cls, examples_count_per_class):
        features = None
        labels = np.empty(0)

        # Each iteration relates to class labels 0 and 1 which are assigned to the "key" variable
        for key, parameters in cls.classes.items():
            # Initializing features by taking shape from the mean array before creating class-related data
            features = np.empty((0, len(parameters["mean"]))) if features is None else features

            # Adding newly created class "key" related data to features. Gaussian distribution is used here
            features = np.concatenate((
                features,
                np.random.multivariate_normal(parameters["mean"], parameters["covariance"], examples_count_per_class)
            ), axis=0)

            labels = np.concatenate((
                labels,
                np.full(examples_count_per_class, key)
            ), axis=0)

        return features, labels
