import train_reader
import generate_model
import argparse
import pathlib

print('Running a test version of the Deep Learning Cell Classifier')

parser = argparse.ArgumentParser(description='user inputs')
parser.add_argument('--path', type = str, required = True)
parser.add_argument('--file_type', type = str, required = True)
parser.add_argument('--input', type = int, required = True)
parser.add_argument('--truth', type = int, required = True)
parser.add_argument('--sgmt', type = int, required = True)
parser.add_argument('--use_GPU', type = bool, required = True)
parser.add_argument('--diameter', type = int, required = True)
parser.add_argument('--threshold', type = float, required = True)



args = parser.parse_args()

def main(args):

    # data_x, data_y = train_reader.run_pipeline(img_dir, 'tif', 0, 1, -1, True, 20, 2000)
    data_x, data_y = train_reader.run_pipeline(args.path, args.file_type, args.input, args.truth, args.sgmt, args.use_GPU, args.diameter, args.threshold)
    x = generate_model.init_model(40)
    generate_model.fit_model(x, data_x, data_y, 40)
    print(args)

if __name__ == '__main__':
    main(args)
