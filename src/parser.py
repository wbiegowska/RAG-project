import argparse
from src.docling_parser import parse_document

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("parse")
    arg_parser.add_argument('--source_file',required=False, default="data/Core Rules.pdf", help="Path to the source document file.") ###CHANGE DEFAULT 
    arg_parser.add_argument('--target_file', required=False, default="data/processed/Core_rules_parsed.md", help="Path to the output file.") ###CHANGE DEFAULT
    args = arg_parser.parse_args()

    md = parse_document(args.source_file)
    with open(args.target_file, "w") as f:
        f.write(md)