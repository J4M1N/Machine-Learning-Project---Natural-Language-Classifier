from ingest import download_dataset
import pandas as pd
import os

#test to ensure new file is created
def test_new_dataset_has_been_saved(tmp_path):
    # tmp_path provides a temporary directory unique for this test
    new_dataset = tmp_path/ "new_dataset.json"

    # Ensure the file does not exist before creation
    assert os.path.exists(new_dataset) == False
    
    # Call the function to create the file
    download_dataset(new_dataset)
    assert os.path.exists(new_dataset) # Check if the file now exists
    
    
    # Optional: Verify the file content
    result = pd.read_json(new_dataset)
    assert result.columns[0] == "text"
    assert result.columns[1] == "label"