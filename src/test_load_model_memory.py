import sys
import time
import joblib
import subprocess
from memory_profiler import profile


def run_memory_profiler(script_path:str, file_path:str) -> None:
    """
    Run memory profiler on a given script and file path.

    Parameters:
    - script_path (str): Path to the Python script to perform the memory profile.
    - file_path (str): Path to the trained model to be used as input for the script.
    """
    print(f'[INFO] Testing memory for model {file_path}')
    cmd = f'{sys.executable} -m memory_profiler {script_path} {file_path}'
    cmd_output = subprocess.check_output(cmd, shell=True).decode('utf-8')
    print(cmd_output)


@profile
def main(filename:str):
    """
    Main function to load a model from a filename.
    Objective: measure the Used Memory when loading the model

    Parameters:
    filename (str): The path to the file containing the saved model.
    """
    start = time.time()
    _ = joblib.load(filename)
    load_duration = time.time() - start
    print(f'Model Load duration (s): {load_duration}')

if __name__ == "__main__":
    main(sys.argv[1])
