import os
import random

def delete_random_files(directory, percentage=10):
    """
    Delete a random percentage of files in a directory.

    Parameters:
    directory (str): The path of the directory from which files will be deleted.
    percentage (int): The percentage of files to delete. Default is 10.

    Returns:
    None: The function deletes files in-place and does not return any value.
    """
    
    # List all files in directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Calculate how many files to delete
    num_to_delete = int(len(files) * percentage / 100)
    
    # Randomly choose files to delete
    files_to_delete = random.sample(files, num_to_delete)
    
    # Delete files
    for file_name in files_to_delete:
        file_path = os.path.join(directory, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")

# Directories to clean up
src_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\training\\non-human'
val_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\validation\\non-human'
test_dir = 'E:\\TensorFlow\\Front Door Vids\\training-files\\testing\\non-human'

# Apply the function to each directory
for directory in [src_dir, val_dir, test_dir]:
    delete_random_files(directory, percentage=10)
