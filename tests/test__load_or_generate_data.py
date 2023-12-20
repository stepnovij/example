import json
from unittest.mock import patch, mock_open
from processing_functions.llm.tables_parsing.transformations import load_or_generate_data


def mock_generation_function():
    return {"generated": "data"}


mock_response = {"existing": "data"}


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(mock_response))
def test_load_data_without_rerun(mock_file, mock_exists):
    file_path = "file.json"
    result = load_or_generate_data(file_path, mock_generation_function, re_run=False)
    assert result == mock_response
    mock_file.assert_called_with(file_path, 'r')
    mock_exists.assert_called_once_with(file_path)


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
def test_generate_data_with_rerun_or_no_file(mock_file, mock_exists):
    file_path = "file.json"
    result = load_or_generate_data(file_path, mock_generation_function, re_run=True)
    assert result == {"generated": "data"}
    mock_file.assert_called_with(file_path, 'w')
    mock_exists.assert_called_once_with(file_path)


@patch("os.path.exists", return_value=False)
@patch("builtins.open", new_callable=mock_open)
def test_generate_data_no_file(mock_file, mock_exists):
    file_path = "non_existing_file.json"
    result = load_or_generate_data(file_path, mock_generation_function, re_run=False)
    assert result == {"generated": "data"}
    mock_file.assert_called_with(file_path, 'w')
    mock_exists.assert_called_once_with(file_path)
