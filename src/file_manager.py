import os
from typing import Optional

def ensure_directory(path: str) -> None:
    """
    Ensures that a directory exists. If not, creates it.
    
    Args:
        path (str): Path of the directory to ensure exists.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_uploaded_file(file_data, save_path: str) -> None:
    """
    Saves an uploaded file to the specified path.
    
    Args:
        file_data: The file data object from the Streamlit file uploader.
        save_path (str): Path where the file will be saved.
    """
    with open(save_path, "wb") as f:
        f.write(file_data.getbuffer())

def validate_file_extension(file_name: str, allowed_extensions: Optional[list] = None) -> bool:
    """
    Validates if a file has an allowed extension.
    
    Args:
        file_name (str): Name of the file to validate.
        allowed_extensions (list, optional): List of allowed file extensions. Defaults to ["jpg", "png"].
    
    Returns:
        bool: True if the file extension is valid, False otherwise.
    """
    if allowed_extensions is None:
        allowed_extensions = ["jpg", "png"]
    
    file_extension = file_name.split(".")[-1].lower()
    return file_extension in allowed_extensions

def cleanup_directory(directory_path: str, keep_files: Optional[list] = None) -> None:
    """
    Deletes all files in a directory except those specified in keep_files.
    
    Args:
        directory_path (str): Path of the directory to clean up.
        keep_files (list, optional): List of files to keep. Defaults to None (deletes all files).
    """
    if keep_files is None:
        keep_files = []
    
    for file_name in os.listdir(directory_path):
        if file_name not in keep_files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

def get_file_size(file_path: str) -> Optional[float]:
    """
    Retrieves the size of a file in megabytes (MB).
    
    Args:
        file_path (str): Path of the file.
    
    Returns:
        float: File size in MB or None if file does not exist.
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return None

def list_files_in_directory(directory_path: str) -> list:
    """
    Lists all files in a directory.
    
    Args:
        directory_path (str): Path of the directory to list files from.
    
    Returns:
        list: List of file names in the directory.
    """
    if os.path.exists(directory_path):
        return [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
    return []

# Example usage (for testing or understanding):
if __name__ == "__main__":
    # Ensure directories exist
    data_path = "./data/test"
    ensure_directory(data_path)
    
    # Example of saving a dummy file
    dummy_file = "example.jpg"
    with open(os.path.join(data_path, dummy_file), "w") as f:
        f.write("This is a test file.")
    
    # Validate file extension
    print(validate_file_extension(dummy_file))  # Output: True
    
    # Get file size
    print(get_file_size(os.path.join(data_path, dummy_file)))  # Output: File size in MB
    
    # List files
    print(list_files_in_directory(data_path))  # Output: ['example.jpg']
    
    # Cleanup directory (deleting all files)
    cleanup_directory(data_path)
    print(list_files_in_directory(data_path))  # Output: []
