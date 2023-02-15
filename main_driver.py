import segment
import generate_model
import argparse
import os

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main(args):
    data_x, data_y = segment.run_pipeline(args.path, args.file_type, args.input, args.truth, args.sgmt, args.use_GPU, args.diameter, args.threshold)
    x = generate_model.init_model(args.diameter*2, args.threshold)
    generate_model.fit_model(x, data_x, data_y, args.diameter*2)
    print(args)

if __name__ == '__main__':
    main(args)
