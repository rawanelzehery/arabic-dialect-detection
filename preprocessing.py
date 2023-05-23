import re
class TextPreprocessor():
    def __init__(self, language='Arabic'):
        self.language = language

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 1. Remove hashtags
        X = [re.sub(r'#\S+', '', text) for text in X]

        # 2. Remove user mentions
        X = [re.sub(r'@\S+', '', text) for text in X]

        # 3. Remove URLs
        X = [re.sub(r'http\S+', '', text) for text in X]

        # 4. Remove non-Arabic characters
        if self.language == 'Arabic':
            X = [re.sub(r'[^\u0600-\u06FF]+', ' ', text) for text in X]

        return X
