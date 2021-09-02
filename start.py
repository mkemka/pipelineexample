import pandas as pd
import yaml
import logging
import support
import datetime
import argparse



def main():
    logging.info("main() has started...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="This is the config file.")
    args = parser.parse_args()

    config_file = yaml.safe_load(open(args.config).read())
    print(config_file)

    # Read in the files
    ds_names, ds_ratings = support.import_data(config_file['folder'])
    logging.info("Datasets have been imported.")
    # Filter the files
    ds_names_filtered = support.filter_names(ds_names)
    ds_ratings_filtered = support.filter_ratings(ds_ratings)
    logging.info("Datasets have been filtered")
    # Join the files
    ds_join = support.join_datasets(ds_names_filtered, ds_ratings_filtered)
    logging.info("Datasets have been joined.")
    # Run the bag of words
    ds_join_words, ds_model = support.generate_bag_of_words(ds_join)
    logging.info("Bag of words has been generated.")

    # Save the files
    #support.save_parquet(ds_join_words, config_file['output_filename'])
    logging.info("File has been saved.")



if __name__ == '__main__':
    start_process = datetime.datetime.now()

    logging.basicConfig(level = logging.INFO, filename = 'log.txt')
    
    main()