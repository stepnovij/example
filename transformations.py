import os
import time
import logging
import json
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from .prompts import (
    fill_missing_values,
    get_hiearchy,
    get_set_units,
    parameters_extraction
)
from .utils import execution_specific_code


def add_hierarchy(df: pd.DataFrame, hierarchy_resp: str) -> pd.DataFrame:
    """
    Applies a hierarchical structure to a DataFrame based on a hierarchy response string.

    This function processes a string detailing hierarchy information and adds corresponding
    hierarchy columns to the DataFrame. Each row in the DataFrame is updated based on its
    hierarchical level indicated in the hierarchy response.

    :param df: pd.DataFrame - The DataFrame to be modified.
    :param hierarchy_resp: str - A string representing the hierarchy response, formatted as 'index ## parameter ## level'.
    :return: pd.DataFrame - The DataFrame with added hierarchical columns.
    """
    hierarchy = []
    for x in hierarchy_resp.split('\n'):
        if x and 'Note' not in x and ' ## ' in x:
            idx, param, level = x.split(' ## ')
            if param.startswith('"'):
                param = param.replace('"', '')
            hierarchy.append((int(idx), param.strip(), level.strip()))

    hierarchy_map = dict()
    for i, p, l in hierarchy:
        hierarchy_map[i] = {'param': p, 'level': l}

    # Create new columns
    max_depth = max([int(i[2]) for i in hierarchy])
    min_depth = min([int(i[2]) for i in hierarchy])

    if max_depth == min_depth:
        return df

    for i in range(1, max_depth + 1):
        df[f'parameters_{i}'] = None

    # TODO: HARD-code fix later when mapping is used
    param_column = df.columns[0]

    # Fill the new columns
    for idx, row in df.iterrows():
        if idx in hierarchy_map:
            depth = int(hierarchy_map[idx]['level'])
            df.at[idx, f'parameters_{depth}'] = row[param_column]
    df[f'parameters_{min_depth}'] = df[f'parameters_{min_depth}'].fillna(method='ffill')
    idxs = []
    for i in df[~df[f'parameters_{max_depth}'].isna()].index:
        idxs.append(i)
        idxs.append(i - 1)
    mid_depth = max_depth - 1
    idxs.sort()
    df.loc[list(set(idxs)), f'parameters_{mid_depth}'] = df.loc[list(set(idxs)), f'parameters_{mid_depth}'].fillna(
        method='ffill')
    return df


def load_or_generate_data(file_path: str, generation_function, *args, **kwargs) -> Dict:
    """
    Loads data from a JSON file if it exists, otherwise generates it using the provided function.

    :param file_path: Path to the JSON file.
    :param generation_function: Function to generate data if the file does not exist.
    :param args: Positional arguments for the generation function.
    :param kwargs: Keyword arguments for the generation function.
    :return: Loaded or generated data.
    """
    if os.path.exists(file_path) and not kwargs.get('re_run', False):
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    else:
        data = generation_function(*args)
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file)
        return data


class BaseTransformation:
    def __init__(self, component_type: str, manufacturer_name: str, model: str, output_path: str, re_run: bool = False):
        """
        Initialize the BaseTransformation with necessary parameters common to all transformations.

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

    # TODO: Later introduce Data class that will have all properites instead of having Dict as input/output
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the transformation on a list of DataFrames.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        start_time = time.time()
        # Call the specific transformation logic implemented by the subclass
        result = self.transform(data)

        elapsed_time = round(time.time() - start_time, 2)
        logging.info(f'End {self.__class__.__name__} - Duration: {elapsed_time} seconds')
        return result

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformation logic to be implemented by each subclass.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        raise NotImplementedError("Transform method must be implemented by subclass")


class ColumnMappingTransformation(BaseTransformation):
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get column mappings for each df in a list.

        This function takes a DataFrame and identifies the type of each parameter, which may fall under one of
        four categories: parameters, measurements, units, or conditions. The final output is an array of column types.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        columns_mapping_path = os.path.join(self.transformation_results_folder, 'column_mappings.json')
        tables = data["tables"]
        generate_data_func = lambda: [
            parameters_extraction(
                table,
                self.component_type,
                self.manufacturer_name,
                self.model
            ) for table in tables
        ]
        data["column_mappings"] = load_or_generate_data(columns_mapping_path, generate_data_func, re_run=self.re_run)
        return data


class UnitMappingTransformation(BaseTransformation):

    def _generate_unit_mappings(self, tables: List[pd.DataFrame], column_mappings: List[Dict]) -> Callable:
        return lambda: [
            get_set_units(table, self.component_type, self.manufacturer_name, self.model)[0]
            for idx, table in enumerate(tables) if len(column_mappings[idx]['units']) == 0
        ]

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the unit mapping transformation on a list of DataFrames.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        tables = data['tables']
        column_mappings = data['column_mappings']

        units_mapping_path = os.path.join(self.transformation_results_folder, 'units_mappings.json')
        transformed_tables = [table.copy() for table in tables]
        generate_data_func = self._generate_unit_mappings(transformed_tables, column_mappings)

        units_mappings = load_or_generate_data(units_mapping_path, generate_data_func, re_run=self.re_run)

        # Apply the units mappings to the tables
        for idx, table in enumerate(transformed_tables):
            if len(column_mappings[idx]['units']) == 0:
                new_columns_df = pd.DataFrame.from_dict(units_mappings[idx], orient='index')
                transformed_tables[idx] = table.join(new_columns_df)

        # TODO: Better to create a new property instead of rewriting existing data
        data['tables'] = transformed_tables
        data['units_mappings'] = units_mappings
        return data


class MissingValuesTransformation(BaseTransformation):
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the missing values transformation on a list of DataFrames.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        tables = data['tables']
        tables_cords = data['tables_cords']

        fill_missing_codes_path = os.path.join(self.transformation_results_folder, 'fill_missing_codes.json')
        generate_data_func = lambda: [
            fill_missing_values(table, self.component_type, self.manufacturer_name, self.model)
            for table in tables]
        fill_missing_codes = load_or_generate_data(fill_missing_codes_path, generate_data_func, re_run=self.re_run)

        # Applying missing values logic to tables
        cleaned_tables = []
        cleaned_tables_cord = []

        for idx, df in enumerate(tables):
            df_filled = execution_specific_code(df, fill_missing_codes[idx])
            df_filled_tables_cord = execution_specific_code(tables_cords[idx], fill_missing_codes[idx])
            cleaned_tables.append(df_filled)
            cleaned_tables_cord.append(df_filled_tables_cord)

        data['tables'] = cleaned_tables
        data['tables_cords'] = cleaned_tables_cord
        return data


class HierarchyIdentificationTransformation(BaseTransformation):
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the hierarchy identification transformation on a list of DataFrames.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        column_mappings = data['column_mappings']
        tables = data['tables']

        hierarchy_resps_file = os.path.join(self.transformation_results_folder, 'hierarchy_resps.json')

        generate_data_func = lambda: [self._generate_hierarchy_response(table, column_mappings[i], self.model)
                                      for i, table in enumerate(tables)]

        hierarchy_resps = load_or_generate_data(hierarchy_resps_file, generate_data_func, re_run=self.re_run)

        # Applying hierarchy logic to tables

        # TODO: not modify the input directly:
        for idx, hierarchy_resp in enumerate(hierarchy_resps):
            if hierarchy_resp:
                tables[idx] = add_hierarchy(tables[idx], hierarchy_resp)

        data['tables'] = tables
        return data

    def _generate_hierarchy_response(self, table: pd.DataFrame, column_mapping: Dict, model: str):
        """
        Generate hierarchy response for a given table.

        :param data: Dictionary containing necessary data for transformation.
        :return: Updated dictionary after applying the transformation.
        """
        try:
            columns_valid = column_mapping['parameters'] + column_mapping['measurements']
            return get_hiearchy(table[columns_valid], model)
        except KeyError as exc:
            logging.error(f'An exception has occurred: {exc}')
            return None
