import argparse
import datetime
import json
import os
import shutil

import models

# GLOBAL VARIABLES
TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.dirname(os.path.realpath(__file__))
CNFG = os.path.join(os.path.join(PATH, "data"), "config.json")


def output_json(data, file):
    """ Output data to json """
    with open(file, "w") as output:
        output.write(json.dumps(data, indent=4))


def get_args():
    """ Process args from command line """
    parser = argparse.ArgumentParser()

    # Set classifier to run
    parser.add_argument(
        "-m", "--model",
    )
    # Set data set to run
    parser.add_argument(
        "-d","--data",
    )
    # Generate plots or not
    parser.add_argument(
        "-p", "--plots", action="store_true",
    )
    # Set name of the output file
    parser.add_argument(
        "-n", "--name", default=f'RUN_DATA_{TIME}',
    )
    # Single parameter set run
    parser.add_argument(
        "-s", "--single", action="store_true",
    )
    return parser.parse_args()


def main():
    """ Main """
    # Process command line
    args = get_args()

    # Process directories and config
    dir_name = os.path.join(os.path.join(PATH, "out"), args.name)
    os.makedirs(dir_name)
    shutil.copy(CNFG, dir_name)

    # Open config
    with open(CNFG, "r") as open_file:
        config = json.load(open_file)

    # Run datasets in config
    for data_name, info in config.items():

        # If its not the data set we want, skip
        if args.data and data_name not in args.data:
            continue

        # Create output directory
        out_dir = os.path.join(dir_name, data_name)
        os.makedirs(out_dir)

        # Initialize data class
        dataset = models.DataSet(data_name, os.path.join(os.path.join(PATH, "data"), info["file"]), info)
        dataset_data = list()

        # Pre-process data
        dataset.process(info["label"])

        # Create model instance
        model_name = args.model
        model_inst = models.generate_classes(dataset, model_name)

        # Loop through model instances
        for model in model_inst:

            # If its a single run, input hyperparams and run
            if args.single:
                model.update_params()

            # Run single run of classifier
            model.run(generate_plots=True)

            # If running range of params, run grid search
            if not args.single:
                model.grid_search_cv()
                model.run()

            # Output results to json file
            model.to_json(os.path.join(out_dir, f"{model.name}.json"))

            # Append to results to data
            dataset_data.append(model.output_dict(for_json=True))

            # Create plots for model
            if args.plots:
                model.plot(out_dir)

        # Output final results to json
        output_json(dataset_data, os.path.join(out_dir, f"{data_name}.json"))
        print(f"Processed dataset: {data_name}")


if __name__ == "__main__":
    main()
