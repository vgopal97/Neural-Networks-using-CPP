#ifndef FUNCTIONALITY_H 
#define FUNCTIONALITY_H 

#define FAILURE 0
#define SUCCESS 1
#define TRAIN 1
#define DEV 2
#define TEST 3

enum FunctionalityErrorTypes
{
    LAYER_DIMS_MISMATCH,
    NO_WEIGHTS_BEFORE_ACTIVATION,
    FUNCTIONALITY_TOTAL_ERROR_COUNT,
};

const char* FunctionalityErrorType[FUNCTIONALITY_TOTAL_ERROR_COUNT + 1] = 
{
    "LAYER_DIMS_MISMATCH",
    "NO_WEIGHTS_BEFORE_ACTIVATION",
    "FUNCTIONALITY_TOTAL_ERROR_COUNT",
};

tensor unit_tensor(std::vector<int> shape)
{
    tensor res(shape);

    for(int i=0; i<res.get_elem_count(); i++)
    {
        res.data[i] = 1;
    }

    return res;
}

class Dataset
{
    private:
    int size = 0;
    int batch_size = 0;
    std::vector< std::vector<tensor_double> > X;
    std::vector<tensor_double> Y;
    int train_idx = 0;
    std::vector<std::vector<tensor_double> > train_x;
    std::vector<tensor_double> train_y;
    int dev_idx = 0;
    std::vector<std::vector<tensor_double> > dev_x;
    std::vector<tensor_double> dev_y;
    int test_idx = 0;
    std::vector<std::vector<tensor_double> > test_x;
    std::vector<tensor_double> test_y;
    tensor _train_x, _train_y;
    tensor _dev_x, _dev_y;
    tensor _test_x, _test_y;

    public:
    std::vector<tensor_double> temp_train_y;
    std::vector<std::vector<tensor_double> > temp_train_x;
    Dataset(int batch_size, std::vector< std::vector<tensor_double> > X, std::vector<tensor_double> Y)
    {
        this->X = X;
        this->Y = Y;
        this->size = X.size();
        this->batch_size = batch_size;

        /* Randomize the vectors.*/
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::mt19937 f(g());
        std::shuffle(X.begin(), X.end(), f);
        std::shuffle(Y.begin(), Y.end(), f);

        /* Slice the dataset to train, dev and test with 10:2:3 ratio */
        this->train_x = std::vector< std::vector<tensor_double> >(X.begin(), X.begin() + 100);
        this->train_y = std::vector<tensor_double>(Y.begin(), Y.begin() + 100);
        this->dev_x = std::vector< std::vector<tensor_double> >(X.begin() + 101, X.begin() + 120);
        this->dev_y = std::vector<tensor_double>(Y.begin() + 101, Y.begin() + 120);
        this->test_x = std::vector< std::vector<tensor_double> >(X.begin() + 121, X.end());
        this->test_y = std::vector<tensor_double>(Y.begin() + 121, Y.end());
        
        /* Populate the initial tensor values*/
        this->next(TRAIN);
        this->next(TEST);
        this->next(DEV);
    }

    int next(int mode)
    {
        int ret = FAILURE;
        std::vector< std::vector<tensor_double> > temp_x;
        std::vector<tensor_double> temp_y;

        if(mode == TRAIN)
        {
            for(auto i=0; i<this->batch_size; i++)
            {
                temp_x.push_back(this->train_x[(this->train_idx + i) % this->train_x.size()]);
                temp_y.push_back(this->train_y[(this->train_idx + i) % this->train_y.size()]);
            }
            this->_train_x = tensor(temp_x, std::vector<int>{this->batch_size, (int) temp_x[0].size()});
            this->_train_y = tensor(temp_y, std::vector<int>{this->batch_size, 1});

            this->train_idx = (this->train_idx + this->batch_size) % this->train_x.size();
            ret = SUCCESS;
        }
        else if(mode == DEV)
        {
            for(auto i=0; i<this->batch_size; i++)
            {
                temp_x.push_back(this->dev_x[(this->dev_idx + i) % this->dev_x.size()]);
                temp_y.push_back(this->dev_y[(this->dev_idx + i) % this->dev_y.size()]);
            }
            this->_dev_x = tensor(temp_x, std::vector<int>{this->batch_size, (int) temp_x[0].size()});
            this->_dev_y = tensor(temp_y, std::vector<int>{this->batch_size, 1});

            this->dev_idx = (this->dev_idx + this->batch_size) % this->dev_x.size();
            ret = SUCCESS;
        }
        else if(mode == TEST)
        {
            for(auto i=0; i<this->batch_size; i++)
            {
                temp_x.push_back(this->test_x[(this->test_idx + i) % this->test_x.size()]);
                temp_y.push_back(this->test_y[(this->test_idx + i) % this->test_y.size()]);
            }
            this->_test_x = tensor(temp_x, std::vector<int>{this->batch_size, (int) temp_x[0].size()});
            this->_test_y = tensor(temp_y, std::vector<int>{this->batch_size, 1});

            this->test_idx = (this->test_idx + this->batch_size) % this->test_x.size();
            ret = SUCCESS;
        }
        else
        {
            std::cerr<<"Wrong mode used for batch iteration!"<<std::endl;
        }

        this->temp_train_x = temp_x;
        this->temp_train_y = temp_y;

        return ret;
    }

    tensor get_X(int mode)
    {
        if(mode == TRAIN)
        {
            return this->_train_x;
        }
        else if(mode == DEV)
        {
            return this->_dev_x;
        }
        else if(mode == TEST)
        {
            return this->_test_x;
        }
        else
        {
            std::cerr<<"Wrong mode used for batch_iteration!"<<std::endl;
        }

        return tensor();
    }

    tensor get_Y(int mode)
    {
        if(mode == TRAIN)
        {
            return this->_train_y;
        }
        else if(mode == DEV)
        {
            return this->_dev_y;
        }
        else if(mode == TEST)
        {
            return this->_test_y;
        }
        else
        {
            std::cerr<<"Wrong mode used for batch_iteration!"<<std::endl;
        }

        return tensor();
    }

};


/* We are following row wise vectors instead of
 column wise vectors.*/
class Module
{
    protected:
    std::string mod_type;
    std::vector<Module*> mod_list;
    int in, out;
    tensor w, b;
    tensor dLdW, dLdB;
    tensor A;
    int batch = 0;
    public:

        std::string get_mod_type()
        {
            return this->mod_type;
        }

        virtual tensor forward(tensor x)
        {
            return x;
        }

        virtual tensor backward(tensor x)
        {
            return x;
        }

        virtual void update_weights(double l_rate)
        {
            return;
        }

        virtual std::vector<int> get_dims()
        {
            return std::vector<int>{};
        }
};

class Linear : public Module
{
    public:
    Linear(int batch, int in, int out)
    {
        this->in = in;
        this->out = out;
#ifdef DEBUG_LOGS
        std::cout<<"batch: "<<batch<<std::endl;
#endif
        this->w = tensor(std::vector<int>{in, out});
        this->b = tensor(std::vector<int>{1, out});
        w.randomize();
        b.randomize();
        this->mod_type = "Linear";
    }

    tensor forward(tensor x)
    {
        this->A = x;
#ifdef DEBUG_LOGS
        x.print_shape();
        w.print_shape();
#endif
        x = x*this->w + this->b;
        return x;
    }

    tensor backward(tensor dLdZ)
    {
#ifdef DEBUG_LOGS
        dLdZ.print_shape();
        A.print_shape();
#endif
        this->dLdW =  A.transpose() * dLdZ;
        this->dLdB = dLdZ;
        return dLdZ * this->w.transpose();
    }

    void update_weights(double l_rate)
    {
        this->w = this->w - dLdW*l_rate;
        this->b = this->b - dLdB.avg()*l_rate;
    }

    std::vector<int> get_dims()
    {
        std::vector<int> dim;
        dim.push_back(in);
        dim.push_back(out);
        return dim;
    }
};

class ReLu : public Module
{
    private:
    tensor Z;
    
    public:

    ReLu()
    {
        this->mod_type = "ReLu";
    }

    tensor forward(tensor x)
    {
        this->Z= x;
        return x.relu();
    }

    tensor backward(tensor A)
    {
        tensor res = tensor(A.get_shape());

        for(int i=0; i<res.elem_count(); i++)
        {
            res.data[i] = (Z.data[i] > 0) ? (A.data[i]) : (LEAK_RELU_A * A.data[i]);
        }

        return res;
    }

    void update_weights(double l_rate)
    {
        return ;
    }

};

class Tanh : public Module
{
    private:
    tensor Z;

    public:

    Tanh()
    {
        this->mod_type = "Tanh";
    }

    tensor forward(tensor x)
    {
        tensor res;
        res = tensor(x.get_shape());
        this->Z = x;
        for(int i=0; i<x.elem_count(); i++)
        {
            res.data[i] = std::tanh(x.data[i]);
        }

        return res;
    }

    tensor backward(tensor A)
    {
        tensor res;
        res = tensor(this->Z.get_shape());

        for(int i=0; i<A.elem_count(); i++)
        {
            res.data[i] = A.data[i] * (1 - this->Z.data[i] * this->Z.data[i]);
        }

        return res;
    }

    void update_weights(double l_rate)
    {
        return;
    }
};

class SoftMax : public Module
{
    private:
    tensor A;

    public:

    SoftMax()
    {
        this->mod_type = "Softmax";
    }

    tensor forward(tensor x)
    {
        tensor res;
        res = tensor(x.get_shape());
        tensor_double sum = 0;

        for(auto b=0; b<x.get_shape()[0]; b++)
        {
            sum = 0;
            for(auto i=0; i<x[0].get_elem_count(); i++)
            {
                res[b].data[i] = std::pow(2.71, x[b].data[i]);
#ifdef DEBUG_LOGS
                std::cout<<res[b].data[i]<<", ";
#endif
                sum += res[b].data[i];
            }
#ifdef DEBUG_LOGS
            std::cout<<std::endl;
#endif

            for(auto i=0; i<x[0].get_elem_count(); i++)
            {
                res[b].data[i] = res[b].data[i]/sum;
            }

        }
        this->A = A;
        return res;
    }

    /* A`(x) = A(x)*(1-A(x)) */
    tensor backward(tensor A)
    {
        return A;
    }

    void update_weights(double l_rate)
    {
        return;
    }
};

class Network : public Module
{
    public:
    Network()
    {
        this->mod_type = "Network";
    }

    Network(int batch)
    {
        this->batch = batch;
        this->mod_type = "Network";
    }

    tensor forward(tensor x)
    {
        for(auto& mod : this->mod_list)
        {
#ifdef DEBUG_LOGS
            std::cout<<(mod)->get_mod_type()<<std::endl;
#endif
            x = (mod)->forward(x);
        }     
        return x;
    }

    tensor backward(tensor dLdZ)
    {
        for(int i=this->mod_list.size()-2; i>=0; i--)
        {
#ifdef DEBUG_LOGS
            std::cout<<(this->mod_list[i])->get_mod_type()<<", is ref: "<<dLdZ.is_reference<<std::endl;
#endif
            dLdZ = (this->mod_list[i])->backward(dLdZ);
        }
        return dLdZ;
    }

    /* If the last part of the network is an activation function, we
    woruld need the derivative of it as part of the loss function's 
    derivative w.r.t Z.  L(z) = L(A(z)) ==> L^(z) = L^(A(z)) * A^(z). */
    tensor get_dAdZ_for_loss(tensor A)
    {
        if(this->mod_list.back()->get_mod_type() != "Network" || this->mod_list.back()->get_mod_type() != "Linear")
        {
            return this->mod_list.back()->backward(A);
        }

        return A;
    }

    void add_linear(int in, int out)
    {
        try
        {
            if(!this->mod_list.empty() && this->out != in)
            {
                throw LAYER_DIMS_MISMATCH;
            }

            if(this->mod_list.empty())
            {
                this->in = in;
            }
            Linear* layer = new Linear(this->batch, in, out);
            this->mod_list.push_back(layer);
            this->out = out;
        }
        catch(FunctionalityErrorTypes x)
        {
            std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
            exit(0);
            return;
        }    
    }

    void add_tanh()
    {
        try
        {
            if(this->mod_list.empty())
            {
                throw NO_WEIGHTS_BEFORE_ACTIVATION;
            }
            else
            {
                Tanh* act_fn = new Tanh;
                this->mod_list.push_back(act_fn);
            }
        }
        catch(FunctionalityErrorTypes x)
        {
            std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
            exit(0);
            return;
        }
    }

    void add_relu()
    {
        try
        {
            if(this->mod_list.empty())
            {
                throw NO_WEIGHTS_BEFORE_ACTIVATION;
            }
            else
            {
                ReLu* act_fn = new ReLu;
                this->mod_list.push_back(act_fn);
            }
        }
        catch(FunctionalityErrorTypes x)
        {
            std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
            exit(0);
            return;
        }
    }

    void add_softmax()
    {
        try
        {
            if(this->mod_list.empty())
            {
                throw NO_WEIGHTS_BEFORE_ACTIVATION;
            }
            else
            {
                SoftMax* act_fn = new SoftMax;
                this->mod_list.push_back(act_fn);
            }
        }
        catch(FunctionalityErrorTypes x)
        {
            std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
            exit(0);
            return;
        }
    }

    void update_weights(double l_rate)
    {
        for(auto& mod : this->mod_list)
        {
            (mod)->update_weights(l_rate);
        }   
    }
};

class cross_entropy_loss
{

    private:
    tensor y, y_pred;
    int output_dim;
    Network* NN;

    public:

    cross_entropy_loss(int output_dim, Network* NN)
    {
        this->output_dim = output_dim;
        this->NN = NN;
    }

    tensor forward(tensor y, tensor y_pred)
    {
        this->y_pred = y_pred;

        auto y_encoded = y.one_hot_encoding(this->output_dim);
        this->y = y_encoded;

        int batch_size = y_pred.get_shape()[0];

        tensor res(std::vector<int>{batch_size, 1});

        for(int b=0; b<batch_size; b++)
        {
            res[b].data[0] = (-1) * std::log(y_pred[b][])
        }

        return res;
    }

    /* This backward is valid only for softmax function,
        as this would return dLdZ w.r.t Sofrmax Function */
    tensor backward()
    {
        return this->y - this->y_pred;
    }
};

class MSE_loss
{
    private:
    tensor y, y_pred;

    public:

    tensor forward(tensor y, tensor y_pred)
    {
        tensor res(y.get_shape());
        int b = y.get_shape()[0];
        this->y = y;
        this->y_pred = y_pred;
        for(int i=0; i<y.get_elem_count(); i++)
        {
            res.data[i] = (y.data[i] - y_pred.data[i]);
            res.data[i] = (res.data[i] * res.data[i])/b;
        }

        return res;
    }

    tensor backward()
    {
        return (this->y_pred - this->y)*2;
    }
};

class normal_loss
{
    private:
    tensor y_pred;
    tensor tensor_loss;
    int output_dim = 0;
    Network* NN = NULL;

    public:
    normal_loss(Network* NN, int output_dim)
    {
        this->output_dim = output_dim;
        this->NN = NN;
    }

    tensor_double forward(tensor y, tensor y_pred)
    {
        this->y_pred = y_pred;
        this->tensor_loss = y_pred - y.one_hot_encoding(this->output_dim);
        tensor_double res = 0.0;

        for(int i=0; i<y_pred.get_shape()[0]; i++)
        {
            auto iy = (int) y[i][0].val();
            //std::cout<<y_pred[i][iy].val();
            res += (1 - y_pred[i][iy].val());
        }

        return res;
    }

    tensor backward()
    {
        return this->NN->get_dAdZ_for_loss(this->y_pred);
    }
};

#endif