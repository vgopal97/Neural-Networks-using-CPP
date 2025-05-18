#include <iostream>
#include<vector>
#include<algorithm>
#include<random>
#include<numeric>
#include<string>
#include <fstream>

#include "tensor.h"
#include "functionality.h"
//#define DEBUG_LOGS

// Compilation Command g++ -std=c++17 -g -fsanitize=address iris.cpp -o iris

int main()
{
    /* Configurations */
    int epochs = 100;
    int batch_size = 4;
    double learning_rate = 0.01;
    int output_dim = 3;
    int iters = 0;

    /* Prepare the model */
    Network NN(batch_size);
    NN.add_linear(4, output_dim);
    NN.add_relu();
    normal_loss LOSS(&NN, output_dim);

    /* Iris Dataset Handling */
    std::string myText;
    std::ifstream MyReadFile("/Users/bandhakavi/Downloads/iris/iris.data");
    std::vector< std::vector<tensor_double> > X;
    std::vector<tensor_double> Y;
    std::unordered_map<std::string, int> strtoi;

    strtoi["Iris-virginica"] = 2;
    strtoi["Iris-versicolor"] = 1;
    strtoi["Iris-setosa"] = 0;

    while (std::getline (MyReadFile, myText)) 
    {
        std::vector<float> temp(4, 0.0);
        std::vector<tensor_double> data(4, 0.0);
        const char* data_str = myText.c_str();
        char iris_type[20];

        sscanf(data_str, "%f,%f,%f,%f,%s", &temp[0], &temp[1], &temp[2], &temp[3], iris_type);
        data[0] = temp[0];
        data[1] = temp[1];
        data[2] = temp[2];
        data[3] = temp[3];

        std::cout<<data[0]<<", "<<data[1]<<", "<<data[2]<<", "<<data[3]<<std::endl;

        X.push_back(data);
        Y.push_back(strtoi[std::string(iris_type)]);
    }
    X.pop_back();
    Y.pop_back();

    MyReadFile.close();

    Dataset iris_dataset(batch_size, X, Y);
    iters = X.size() / batch_size + 1;
    for(int epoch=0; epoch<epochs; epoch++)
    {
        for(auto iter = 0; iter < iters; iter++)
        {
            /* get the train and dev data */
            auto x_train = iris_dataset.get_X(TRAIN);
            auto y_train = iris_dataset.get_Y(TRAIN);
            auto x_dev  = iris_dataset.get_X(DEV);
            auto y_dev  = iris_dataset.get_Y(DEV);

            /* Do the forward propoagation on train data */
            auto y_pred = NN.forward(x_train);
            auto train_loss = LOSS.forward(y_train, y_pred);

            /* Do the backward propogation on train data */
            auto dLdZ = LOSS.backward();
            auto backprop = NN.backward(dLdZ);
            NN.update_weights(learning_rate);

            /* Do the forward propogation on dev data */
            y_pred = NN.forward(x_dev);
            auto dev_loss = LOSS.forward(y_dev, y_pred);

            if(iter % 3 == 0)
            {
                std::cout<<"Epoch: "<<epoch<<", Iteration: "<<iter<<", Train Loss: "<<train_loss<<", Dev Loss: "<<dev_loss<<std::endl;
            }

            /* Update the train and dev dataset */
            iris_dataset.next(TRAIN);
            iris_dataset.next(DEV);
        } 
    }

    return 0;
}