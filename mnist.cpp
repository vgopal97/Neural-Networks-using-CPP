#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <string>
#include <fstream>
#include<thread>

#define PYTORCH_IMPLEMENTATION_CROSS_ENTROPY
#include "tensor.h"
#include "functionality.h"
//#define DEBUG_LOGS
// Compilation Command g++ -std=c++17 -g -fsanitize=address iris.cpp -o iris

std::vector<tensor_double> get_mnist_features(std::string str)
{
    std::string cur_num = "";
    std::vector<tensor_double> res;

    for(int i=0; i<str.size(); i++)
    {
        if(str[i] == ',')
        {
            res.push_back(std::stoi(cur_num));
            cur_num = "";
        }
        else
        {
            cur_num += str[i];
        }
    }

    res.push_back(std::stoi(cur_num));

    return res;
}

int main()
{
    /* Configurations */
    int epochs = 100;
    int batch_size = 8;
    double learning_rate = 1e-3;
    int input_dim = 28*28;
    int hidden_dim = 512;
    int output_dim = 16;
    int iters = 0;

    /* MNIST Dataset Handling */
    std::string myText;
    std::ifstream MyReadFile("/Users/bandhakavi/Downloads/mnist/mnist_train.csv");
    std::vector< std::vector<tensor_double> > X;
    std::vector<tensor_double> Y;


    /* First line is. labels of the columns*/
    std::getline(MyReadFile, myText);
    while (std::getline(MyReadFile, myText)) 
    {
        std::vector<tensor_double> features = get_mnist_features(myText);
        /* Normalize the data */
        for(int i=1; i<features.size(); i++)
        {
            features[i] = features[i] / 255;
        }
        Y.push_back(features[0]);
        features.erase(features.begin());
        X.push_back(features);
    }
    MyReadFile.close();

    Dataset mnist_dataset(batch_size, X, Y, 8, 2, 0);

    /* Prepare the model */
    Network NN(batch_size);
    NN.add_linear(input_dim, hidden_dim);
    NN.add_relu(0.01);
    NN.add_linear(hidden_dim, output_dim);
    CrossEntropy LOSS(output_dim);

    iters = X.size() / batch_size + 1;
    for(int epoch=0; epoch<epochs; epoch++)
    {
        tensor_double acc = 0.0;
        for(auto iter = 0; iter < iters; iter++)
        {
            /* get the train and dev data */
            auto x_train = mnist_dataset.get_X(TRAIN);
            auto y_train = mnist_dataset.get_Y(TRAIN);
            

            /* Do the forward propoagation on train data */
            auto y_pred = NN.forward(x_train);
            auto train_loss = LOSS.forward(y_train, y_pred);

            /* Do the backward propogation on train data */
            auto dLdZ = LOSS.backward();
            auto backprop = NN.backward(dLdZ);
            NN.update_weights(learning_rate);

            

            try
            {
                if(std::isnan(train_loss) || std::isinf(train_loss) || backprop.invalid())
                {
                    throw NAN_INF_VALUES_FOUND;
                }
            }
            catch(FunctionalityErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<FunctionalityErrorType[x]<<std::endl;
                exit(0);
            }    

            if(iter%iters == 99)
            {   

                auto x_dev  = mnist_dataset.get_X(DEV);
                auto y_dev  = mnist_dataset.get_Y(DEV);
                /* Do the forward propogation on dev data */
                auto dev_y_pred = NN.forward(x_dev);
                auto dev_loss = LOSS.forward(y_dev, dev_y_pred);
                acc += accuracy(dev_y_pred, y_dev);
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
                std::cout<<"Epoch: "<<epoch<<", Iteration: "<<iter<<", Train Loss: "<<train_loss<<", Dev Loss: "<<dev_loss<<", acc: "<<acc/(iter+1)<<std::endl;
            }
            else
            {
                 std::cout<<"Epoch: "<<epoch<<", Iteration: "<<iter<<", Train Loss: "<<train_loss<<std::endl;
            }

            /* Update the train and dev dataset */
            mnist_dataset.next(TRAIN);
            mnist_dataset.next(DEV);
        } 
    }

    return 0;
}