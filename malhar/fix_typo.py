import re
import os

def fix_typo_in_file(file_path):
    """
    Reads a file, replaces the whole word "औरर" with "और",
    and writes the content back to the file.

    Args:
        file_path (str): The path to the file to be processed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # CORRECTED LINE:
        # Use re.subn() which returns a tuple of (new_string, number_of_subs_made)
        new_content, count = re.subn(r'\bऔरर\b', 'और', content, flags=re.UNICODE)

        if count > 0:
            print(f"Found and replaced {count} instance(s) in {file_path}")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
        else:
            print(f"No typos found in {file_path}")

    except Exception as e:
        print(f"Could not process file {file_path}: {e}")

if __name__ == "__main__":
    # Get the directory where the script is located.
    current_directory = os.getcwd()
    print(f"Searching for .html files in: {current_directory}\n")

    # Loop through all files in the directory
    for filename in os.listdir(current_directory):
        if filename.endswith(".html"):
            fix_typo_in_file(os.path.join(current_directory, filename))

    print("\nScript finished.")
