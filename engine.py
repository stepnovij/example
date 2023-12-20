import os
from typing import Any, Dict, Type, List

from processing_functions.utils import create_path_if_not_exists
from .transformations import BaseTransformation


class Runner:
    def __init__(self, component_type: str, manufacturer_name: str, model: str, output_path: str, re_run: bool = False):
        """
        Initialize the Runner with necessary parameters for transformations.

        :param component_type: Type of the component.
        :param manufacturer_name: Name of the manufacturer.
        :param model: Model name.
        :param output_path: Path to store output files.
        :param re_run: Boolean to indicate if the process should be rerun or use cached results.
        """
        self.component_type = component_type
        self.manufacturer_name = manufacturer_name
        self.model = model
        self.re_run = re_run
        self.transformation_results_folder = os.path.join(output_path, 'prompt_results')
        create_path_if_not_exists(self.transformation_results_folder)

    def run_transformations(self,
                            data: Dict[str, Any],
                            transformation_sequence: List[Type[BaseTransformation]]) -> Dict[str, Any]:
        """
        Processes data by applying a sequence of transformations.

        :param data: Dictionary containing data to be processed.
        :param transformation_sequence: List of transformation instances to be applied.
        :return: Updated data dictionary after applying all transformations.
        """
        for transformation in transformation_sequence:
            # Initialize transformation with required parameters
            transformation = transformation(
                component_type=self.component_type,
                manufacturer_name=self.manufacturer_name,
                model=self.model,
                output_path=self.transformation_results_folder,
                re_run=self.re_run
            )
            data = transformation.transform(data)
        return data