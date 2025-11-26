from ingest import download_dataset
import pandas as pd

#test to ensure new file is created
def test_new_dataset_has_been_saved(tmp_path):
    # tmp_path provides a temporary directory unique for this test
    test_file = tmp_path / "example_file.txt"
    
    # Ensure the file does not exist before creation
    assert not test_file.exists()
    
    # Call the function to create the file
    create_file = test_file.exists()
    create_file(test_file, content=" ")#content="Hello, pytest!"
    
    # Check if the file now exists
    assert test_file.exists()#, f"File {test_file} was not created"
    
    # Optional: Verify the file content
    with open(test_file, "r") as f:
        content = f.read()
    assert content == "Hello, pytest!"