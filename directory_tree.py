import os
from pathlib import Path
import argparse

def generate_tree(directory: Path, prefix: str = "", ignore_patterns: list = None) -> None:
    """
    Generate and print a tree structure of the given directory.
    
    Args:
        directory (Path): The directory path to start from
        prefix (str): Prefix for the current line (used for recursion)
        ignore_patterns (list): List of patterns to ignore (e.g., ['.git', '__pycache__'])
    """
    if ignore_patterns is None:
        ignore_patterns = []
    
    # Get all entries in the directory
    entries = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    
    # Count how many entries we'll actually process (excluding ignored ones)
    entries = [e for e in entries if not any(pattern in str(e) for pattern in ignore_patterns)]
    total_entries = len(entries)
    
    for index, entry in enumerate(entries):
        is_last = index == total_entries - 1
        
        # Create the appropriate connector
        connector = "└── " if is_last else "├── "
        
        # Print the current entry
        print(f"{prefix}{connector}{entry.name}")
        
        # If it's a directory, recurse into it
        if entry.is_dir():
            # Create the appropriate prefix for nested items
            extension = "    " if is_last else "│   "
            generate_tree(entry, prefix + extension, ignore_patterns)

def main():
    parser = argparse.ArgumentParser(description="Generate a tree view of a directory structure")
    parser.add_argument("path", nargs="?", default=".", help="Directory path to start from (default: current directory)")
    parser.add_argument("--ignore", "-i", nargs="+", default=[".git", "__pycache__", ".pytest_cache", ".venv"],
                      help="Patterns to ignore (default: .git __pycache__ .pytest_cache .venv)")
    
    args = parser.parse_args()
    
    # Convert the path to a Path object
    start_path = Path(args.path).resolve()
    
    if not start_path.exists():
        print(f"Error: Path '{start_path}' does not exist")
        return
    
    if not start_path.is_dir():
        print(f"Error: Path '{start_path}' is not a directory")
        return
    
    print(f"\nDirectory Tree for: {start_path}\n")
    generate_tree(start_path, ignore_patterns=args.ignore)

if __name__ == "__main__":
    main()