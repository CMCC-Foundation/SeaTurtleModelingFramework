import argparse
import sys
import warnings

from src.main import analysis, training

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run preprocessing, training, and/or analysis.")
    parser.add_argument("-p", "--preprocessing", action="store_true", help="Run preprocessing")
    parser.add_argument("-t", "--training", action="store_true", help="Run training")
    parser.add_argument("-a", "--analysis", action="store_true", help="Run analysis")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
    
    if args.preprocessing:
        # ignore everything except the message
        warnings.formatwarning = lambda msg, *args, **kwargs: str(msg) + '\n'
        warnings.warn("Warning: Unable to execute preprocessing. Exiting. Use -t or -a")
        sys.exit()

    if args.training:
        training()

    if args.analysis:
        analysis()




