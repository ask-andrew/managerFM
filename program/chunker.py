import os

def split_python_file(input_filepath, output_dir="split_files", chunk_size=10000):
    """
    Splits a Python file into multiple smaller files, each of a specified chunk size.

    Args:
        input_filepath (str): The path to the input Python file.
        output_dir (str): The directory where the split files will be saved.
                          Defaults to "split_files".
        chunk_size (int): The maximum number of characters per output file.
                          Defaults to 10000.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_filepath}'")
        return
    except Exception as e:
        print(f"Error reading file '{input_filepath}': {e}")
        return

    file_base_name = os.path.basename(input_filepath)
    name_without_ext, ext = os.path.splitext(file_base_name)

    num_chunks = (len(content) + chunk_size - 1) // chunk_size

    print(f"Splitting '{input_filepath}' into {num_chunks} chunks...")

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(content))
        chunk = content[start:end]

        output_filename = f"{name_without_ext}_part{i+1}{ext}"
        output_filepath = os.path.join(output_dir, output_filename)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(chunk)
            print(f"Created '{output_filepath}' ({len(chunk)} characters)")
        except Exception as e:
            print(f"Error writing to file '{output_filepath}': {e}")

# --- How to use the script ---
if __name__ == "__main__":
    input_file_path = "/Users/andrewledet/Desktop/gmailapi/analyze4.py"
    split_python_file(input_file_path)
    print("\nFile splitting complete! 🎉")
    print(f"Check the '{os.path.join(os.getcwd(), 'split_files')}' directory for the split files.")