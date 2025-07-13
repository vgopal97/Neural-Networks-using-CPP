#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <string>
#include <fstream>

#define PYTORCH_IMPLEMENTATION_CROSS_ENTROPY
#include "tensor.h"
#include "functionality.h"
//#define DEBUG_LOGS
// Compilation Command g++ -std=c++17 -g -fsanitize=address iris.cpp -o iris

int main()
{
    /* Configurations */
    int epochs = 1000;
    int batch_size = 16;
    double learning_rate = 1e-2;
    int hidden_dim = 16;
    int output_dim = 3;
    int iters = 0;

    /* Prepare the model */
    Network NN(batch_size);
    NN.add_linear(4, hidden_dim);
    //NN.add_relu(0.01);
    NN.add_tanh();
    NN.add_linear(hidden_dim, output_dim);
    CrossEntropy LOSS(output_dim);

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

    Dataset iris_dataset(batch_size, X, Y, 12, 2, 1);
    iters = X.size() / batch_size + 1;
    for(int epoch=0; epoch<epochs; epoch++)
    {
        tensor_double acc = 0.0;
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
            auto dev_y_pred = NN.forward(x_dev);
            auto dev_loss = LOSS.forward(y_dev, dev_y_pred);
            acc += accuracy(dev_y_pred, y_dev);

            try
            {
                if(std::isnan(train_loss) || std::isinf(train_loss) || std::isnan(dev_loss) || std::isinf(dev_loss) || backprop.invalid())
                {
                    throw NAN_INF_VALUES_FOUND;
                }
            }
            catch(FunctionalityErrorTypes x)
            {
                std::cout<<"Epoch: "<<epoch<<", Iteration: "<<iter<<", Train Loss: "<<train_loss<<", Dev Loss: "<<dev_loss<<", acc: "<<acc/iters<<std::endl;
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<FunctionalityErrorType[x]<<std::endl;
                exit(0);
            }    

            if(iter%iters == iters-1)
            {   
#ifdef DEBUG_LOGS
                auto predictions = dev_y_pred.argmax();
                for(int i=0; i<batch_size; i++)
                {
                    std::cout<<"Logits : ";
                    for(int j=0; j<dev_y_pred[i].get_shape()[0]; j++)
                    {
                        std::cout<<dev_y_pred[i][j].val()<<",\t\t";
                    }
                    std::cout<<"Prediction : "<<predictions[i].val()<<",\t\t";
                    std::cout<<"Original : "<<y_dev[i].val()<<std::endl;
                }
#endif
                std::cout<<"Epoch: "<<epoch<<", Iteration: "<<iter<<", Train Loss: "<<train_loss<<", Dev Loss: "<<dev_loss<<", acc: "<<acc/iters<<std::endl;
            }

            /* Update the train and dev dataset */
            iris_dataset.next(TRAIN);
            iris_dataset.next(DEV);
        } 
    }

    return 0;
}