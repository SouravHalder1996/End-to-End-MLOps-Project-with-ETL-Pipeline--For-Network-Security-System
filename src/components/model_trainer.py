import os, sys
from dotenv import load_dotenv
import mlflow
import dagshub
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig

from src.utils.ml_utils.model.estimator import NetworkModel
from src.utils.helper.utils import save_object, load_object
from src.utils.helper.utils import load_numpy_array_data, evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier


class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, best_model_name, classification_train_metric, classification_test_metric):
        with mlflow.start_run():
            train_f1_score = classification_train_metric.f1_score
            train_precision = classification_train_metric.precision_score
            train_recall = classification_train_metric.recall_score

            mlflow.log_metric("Train_F1_Score", train_f1_score)
            mlflow.log_metric("Train_Precision", train_precision)
            mlflow.log_metric("Train_Recall", train_recall)

            test_f1_score = classification_test_metric.f1_score
            test_precision = classification_test_metric.precision_score
            test_recall = classification_test_metric.recall_score

            mlflow.log_metric("Test_F1_Score", test_f1_score)
            mlflow.log_metric("Test_Precision", test_precision)
            mlflow.log_metric("Test_Recall", test_recall)

            mlflow.log_param("Model_Name", best_model_name)
            if hasattr(best_model, 'get_params'):
                for param_name, param_value in best_model.get_params().items():
                    mlflow.log_param(param_name, param_value)

            try:
                mlflow.sklearn.log_model(best_model, artifact_path="sklearn-model")
                logging.info(f"Model successfully logged to MLflow.")
            except Exception as e:
                logging.error(f"Failed to log model to MLflow: {str(e)}")
                raise
        
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(verbose=1),
            "KNN Classifier": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "AdaBoost Classifier": AdaBoostClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(verbose=1),
            "Random Forest Classifier": RandomForestClassifier(verbose=1)
        }

        params = {
            "Logistic Regression": {
                'C': [0.1, 1.0, 10],
                'solver': ['liblinear', 'saga']
            },
            "KNN Classifier": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            "Decision Tree": {
                'criterion': ['gini', 'entropy', 'log_loss'],
                'splitter': ['best', 'random'],
                'max_features': ['sqrt', 'log2']
            },
            "AdaBoost Classifier": {
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'learning_rate': [0.1, 0.01, 0.5, 0.001]
            },
            "Gradient Boosting Classifier": {
                'loss': ['log_loss', 'exponential'],
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'criterion': ['squared_error', 'friedman_mse'],
                'max_features': ['sqrt', 'log2'],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Random Forest Classifier": {
                'criterion': ['gini', 'entropy', 'log_loss']
            }
        }

        model_report, fitted_models = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = fitted_models[best_model_name]

        logging.info(f"Best Model Found: {best_model_name} with accuracy score: {best_model_score}")
        logging.info(f"Best Model Hyperparameters: {best_model.get_params()}")

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        
        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        logging.info(f"Train Metrics - F1: {classification_train_metric.f1_score:.4f}, "
                    f"Precision: {classification_train_metric.precision_score:.4f}, "
                    f"Recall: {classification_train_metric.recall_score:.4f}")
        logging.info(f"Test Metrics - F1: {classification_test_metric.f1_score:.4f}, "
                    f"Precision: {classification_test_metric.precision_score:.4f}, "
                    f"Recall: {classification_test_metric.recall_score:.4f}")

        self.track_mlflow(best_model, best_model_name, classification_train_metric, classification_test_metric)

        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

        os.makedirs("final_model", exist_ok=True)
        model_path = "final_model/model.pkl"
        save_object(model_path, best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            load_dotenv()

            dagshub.init(
                repo_owner='SouravHalder1996', 
                repo_name='End-to-End-MLOps-Project-with-ETL-Pipeline--For-Network-Security-System', 
                mlflow=True
            )

            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
            mlflow_username = os.getenv("MLFLOW_USERNAME")
            mlflow_password = os.getenv("MLFLOW_PASSWORD")
            mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "Default")

            if mlflow_uri:
                if mlflow_username and mlflow_password and "://" in mlflow_uri:
                    scheme, rest = mlflow_uri.split("://", 1)
                    mlflow_uri = f"{scheme}://{mlflow_username}:{mlflow_password}@{rest}"
                mlflow.set_tracking_uri(mlflow_uri)

            mlflow.set_experiment(mlflow_experiment)

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)