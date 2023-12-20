import pandas as pd
from unittest.mock import patch
from processing_functions.llm.tables_parsing.transformations import ColumnMappingTransformation

mock_column_mapping_response = {
    "parameters": ["param1", "param2"],
    "measurements": ["measure1", "measure2"],
    "units": ["unit1", "unit2"],
    "conditions": ["condition1", "condition2"]
}
mock_load_or_generate_data_response = [mock_column_mapping_response]


@patch('processing_functions.llm.tables_parsing.transformations.load_or_generate_data',
       return_value=mock_load_or_generate_data_response)
def test_column_mapping_with_existing_file(mock_load_or_generate):
    # Setup
    test_output_path = 'test_output'
    test_component_type = 'test_component'
    test_manufacturer_name = 'test_manufacturer'
    test_model = 'test_model'

    # Create a dummy DataFrame
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    tables = [df]
    # Test
    transformation = ColumnMappingTransformation(
        test_component_type,
        test_manufacturer_name,
        test_model,
        test_output_path,
        re_run=False
    )
    data = transformation.transform({'tables': tables})
    transformed_tables = data['tables']
    column_mappings = data['column_mappings']

    # Assertions
    assert len(transformed_tables) == len(tables)
    assert column_mappings == mock_load_or_generate_data_response
    mock_load_or_generate.assert_called_once()  # Ensure load_or_generate_data was called once
