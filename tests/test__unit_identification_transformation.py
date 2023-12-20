import pandas as pd
from unittest.mock import patch
from processing_functions.llm.tables_parsing.transformations import UnitMappingTransformation

column_mappings = [{"units": []}]
mock_unit_mappings = [{"unit_column": "unit_value"}]
mock_tables = [pd.DataFrame({'data': [1, 2, 3]})]


def mock_get_set_units(table, component_type, manufacturer_name, model):
    return (mock_unit_mappings[0], table)


@patch('processing_functions.llm.tables_parsing.transformations.load_or_generate_data', return_value=mock_unit_mappings)
def test_unit_mapping_with_existing_data(mock_load_data):
    transformation = UnitMappingTransformation("component", "manufacturer", "model", "output_path", False)
    data = transformation.transform({'tables': mock_tables, 'column_mappings': column_mappings})
    result_tables = data['tables']
    unit_mappings = data['units_mappings']
    assert mock_load_data.called
    assert len(result_tables) == len(mock_tables)
    assert mock_unit_mappings == unit_mappings
