from typing import Union
from sklearn.pipeline import Pipeline
from lime.lime_text import LimeTextExplainer


class ClassificationExplainer:

    def __init__(self, model:Pipeline, random_state:int=91, **kwargs) -> None:
        """
        Initialize the Lime explainer object.

        Parameters:
        - model (Pipeline): The machine learning model in which LimeTextExplainer will explain the prediction.
        - random_state (int): The random state to be used for the explainer. Default is 91.
        - kwargs (dict): Additional keyword arguments to be passed to the LimeTextExplainer constructor.
        """
        self.model = model
        self.classes = self.model.classes_.tolist()
        self.explainer = LimeTextExplainer(
            class_names=self.classes, random_state=random_state,**kwargs
        )
        self.explanation = None
   

    def explain_prediction(self, file_id:str, text:str,
                           classification:Union[str,int],
                           num_features:int=10, 
                           show_text_in_notebook:bool=False) -> None:
        """
        Generate explanation visualization in Jupyter Notebook.

        Parameters:
        - file_id (str): The unique identifier of the document/row.
        - text (str): The text content of the document/row (after preprocess - clean text).
        - classification (Union[str, int]): The ground truth classification of the document.
        - num_features (int, optional): The number of features to include in the explanation. Defaults to 10.
        - show_text_in_notebook (bool, optional): Whether to display the text in the notebook. Defaults to False.

        Returns:
        None
        """
        self.pred_class = self.model.predict([text])
        self.explanation = self.explainer.explain_instance(
            text, self.model.predict_proba, num_features=num_features,
            labels=[self.classes.index(self.pred_class)])
        print(f'Document id: {file_id}')
        print(f'Predicted class: {self.pred_class[0]}')
        print(f'Ground Truth class: {classification}')
        self.explanation.show_in_notebook(text=show_text_in_notebook)
