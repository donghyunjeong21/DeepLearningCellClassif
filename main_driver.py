import train_reader
import generate_model

def main():
    print('Running a test version of the Deep Learning Cell Classifier')
    data_x, data_y = train_reader.testrun()
    x = generate_model.init_model(40)
    generate_model.fit_model(x, data_x, data_y, 40)

if __name__ == '__main__':
    main()
