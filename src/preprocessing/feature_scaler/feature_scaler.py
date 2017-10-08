from sklearn.preprocessing import StandardScaler

class FeatureScaler:
    def apply_feature_scaling(self, data):
        sc_X = StandardScaler()
        return sc_X.fit_transform(data)