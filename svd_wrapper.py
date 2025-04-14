import mlflow.pyfunc

class SurpriseSVDWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.svd_model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        results = []
        for _, row in model_input.iterrows():
            user = row['user_id']
            anime = row['anime_id']
            pred = self.svd_model.predict(user, anime).est
            results.append(pred)
        return results   
