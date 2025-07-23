#ifndef TENSOR_H 
#define TENSOR_H 

typedef  long double tensor_double;

#define LEAK_RELU_A  (0.01)
// Use a reasonable default, but allow runtime adjustment
#define MAX_THREADS_POSSIBLE (16)

enum TensorErrorTypes
{
    INDEX_OUT_OF_BOUNDS,
    TENSOR_SHAPE_NOT_MATCHING,
    TENSOR_NOT_SCALAR,
    TENSOR_DIMENSIONS_NOT_MATCHING,
    TENSOR_SHAPE_IS_NULL,
    TENSOR_SHAPE_LESS_THAN_TWO,
    TENSOR_LAST_TWO_DIMS_MISMATCH_FOR_MAT_MUL,
    TENSOR_NOT_MATRIX,
    TENSOR_DIMENSIONS_OUT_OF_RANGE,
    TENSOR_VALUES_INF_NAN,
    TENSOR_TOTAL_ERROR_COUNT,
};

const char* TensorErrorType[TENSOR_TOTAL_ERROR_COUNT + 1] = 
{
    "INDEX_OUT_OF_BOUNDS",
    "TENSOR_SHAPE_NOT_MATCHING",
    "TENSOR_NOT_SCALAR",
    "TENSOR_DIMENSIONS_NOT_MATCHING",
    "TENSOR_SHAPE_IS_NULL",
    "TENSOR_SHAPE_LESS_THAN_TWO",
    "TENSOR_LAST_TWO_DIMS_MISMATCH_FOR_MAT_MUL",
    "TENSOR_NOT_MATRIX",
    "TENSOR_DIMENSIONS_OUT_OF_RANGE",
    "TENSOR_VALUES_INF_NAN",
    "TENSOR_TOTAL_ERROR_COUNT",
};

class tensor
{
    private:
        int num_elems;
        int num_dims;
        int data_iter = 0;
        std::vector<int> t_shape;

    template<typename T>
    void _add_elems(const std::vector<T>& vec, int cur_dim)
    {

        if constexpr (std::is_same<T, tensor_double>::value)
        {
            for (int i = 0; i < t_shape[cur_dim]; i++)
            {
                data[data_iter++] = vec[i];
            }
        }
        else
        {
            for (int i = 0; i < t_shape[cur_dim]; i++)
            {
                int total_elems = 1;
                _add_elems(vec[i], cur_dim + 1);
            }
        }
    }

        template<typename T>
        void pop_front(std::vector<T>& vec) const
        {
            try
            {
                if(vec.empty())
                {
                    throw TENSOR_SHAPE_IS_NULL;
                }
            }
            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
                return;
            }
            
            // Might have performance impact
            vec.erase(vec.begin());
        }

        bool shape_equal(const std::vector<int>& a, const std::vector<int>& b) const
        {
            if(a.size() != b.size())
            {
                return false;
            }

            for(int i=0; i<a.size(); i++)
            {
                if(a[i] != b[i])
                {
                    return false;
                }
            }

            return true;
        }

        std::vector<int> calc_dims_for_matmul(const std::vector<int>& A, const std::vector<int>& B) const
        {
            std::vector<int> res;
            int dims = A.size();
            res.assign(B.begin(), B.end());
            res[dims-2] = A[dims-2];
            return res;
        }
#ifdef BROADCAST_ENABLED
        bool broadcastable(const std::vector<int>& target, const std::vector<int>& src) const
        {
            if(target.size() != src.size())
            {
                return false;
            }

            for(int i=0; i<target.size(); i++)
            {
                if((src[i] != target[i]) && (src[i] > 1))
                {
                    return false;
                }
            }

            return true;
        }

         void broadcast(tensor res, tensor A, int dim)
        {
            if(dim != this->num_dims-1)
            {
                for(int i=0; i<this->get_shape()[dim]; i++)
                {
                    broadcast(res[i], A[A.get_shape()[dim] > 1 ? i : 0], dim+1);
                }
            }

            if(res.get_shape()[0] != A.t_shape[dim])
            {
                for(int i=0; i<res.get_shape()[dim]; i++)
                {
                    res[i] = A[0];
                }
            }
        }
#endif

        tensor_double random_float(tensor_double min, tensor_double max) {
            return ((tensor_double)rand() / RAND_MAX) * (max - min) + min;
        }

    public:
        tensor_double* data = NULL;
        bool is_reference = false;

        /* Constructor*/
        tensor()
        {
            
        }
       
        /* For a zero tensor initialization*/
        tensor(const std::vector<int>& shape)
        {
            num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            num_dims = shape.size();
            t_shape = shape;
            if(data)
            {
                free(data);
            }
#ifdef DEBUG_LOGS
            //std::cout<<"num_elems: "<<num_elems<<std::endl;
#endif
            data = (tensor_double*)malloc(sizeof(tensor_double)*num_elems);
        }

        /*converting nested vector into a tensor. Nested vector, along with shape as params*/
         template<typename T> tensor(std::vector<T>& vec, std::vector<int> shape)
        {
            num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
            num_dims = shape.size();
            t_shape = shape;
            data = (tensor_double*)malloc(sizeof(tensor_double)*num_elems);
            _add_elems(vec, 0);
        }

        tensor(tensor_double* p_data, std::vector<int>& shape)
        {
            if(shape.empty())
            {
                num_elems = 1;
                num_dims = 1;
                t_shape = std::vector<int>(1,1);
            }
            else
            {
                num_elems = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                num_dims = shape.size();
                t_shape = shape;
            }
            is_reference = true;
            data = p_data;
        }

        tensor(const tensor& A)
        {
            this->t_shape = A.t_shape;
            this->num_elems = A.num_elems;
            this->num_dims = A.num_dims;
            this->data = (tensor_double*) malloc(sizeof(tensor_double)*num_elems);
            for(int i=0; i<num_elems; i++)
            {
                data[i] = A.data[i];
            }
        }
        
        /* Destructor*/
        ~tensor()
        {
            if(!(this->is_reference) && data)
            {
#ifdef DEBUG_LOGS
                //std::cout<<"Tensor Destructed! Addr:"<<this<<std::endl;
#endif
                free(data);
                data = NULL;
            }
        }

        /* Shape Function*/
        std::vector<int> get_shape() const
        {
            if(this->t_shape.empty())
            {
                return std::vector<int>{};
            }
            return t_shape;
        }

        int get_elem_count() const
        {
            return this->num_elems;
        }

        tensor_double val() const
        {
            try
            {
                if(num_elems > 1)
                {
                    std::cout<<num_elems<<std::endl;
                    this->print_shape();
                    throw TENSOR_NOT_SCALAR;
                }
            }
            catch(TensorErrorTypes x)
            {
                std::cout<<num_elems<<std::endl;
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
                return 0;
            }
            
            return data[0];
        }

        void print_shape() const
        {
            std::cout<<"(";
            for(int i=0; i<this->num_dims; i++)
            {
                std::cout<<this->t_shape[i]<<", ";
            }
            std::cout<<")"<<std::endl;
        }

        /* = operator overloading*/
        void operator=(const tensor& A)
        {
            if(this->is_reference)
            {
                try
                {
                    if(!shape_equal(t_shape, A.t_shape))
                    {
                        throw TENSOR_SHAPE_NOT_MATCHING;
                    }
                }
                catch(TensorErrorTypes x)
                {
                    std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                    exit(0);
                    return;
                }
                
                for(int i=0; i < num_elems; i++)
                {
                    data[i] = A.data[i];
                }
                return;
            }

            t_shape = A.t_shape;
            num_elems = A.num_elems;
            num_dims = A.num_dims;
            if(data)
            {
                free(data);
                data = NULL;
            }
            data = (tensor_double*) malloc(sizeof(tensor_double)*num_elems);
            
            for(int i=0; i<num_elems; i++)
            {
                data[i] = A.data[i];
            }
        }

        void operator=(tensor_double x)
        {
            try
            {
                if(this->num_elems != 1 && this->num_dims!=1)
                {
                    throw TENSOR_NOT_SCALAR;
                }
            }
            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
            }

            this->data[0] = x;
        }

        /* [] Operator overloading*/
        tensor operator[](int i)
        {
            try
            {
                if(i >= t_shape[0])
                {
                    throw INDEX_OUT_OF_BOUNDS;
                }
            }

            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
            }
                    
            std::vector<int> sub_tensor_shape = t_shape;
            pop_front(sub_tensor_shape);
            tensor sub_tensor(&(data[i*std::accumulate(sub_tensor_shape.begin(), sub_tensor_shape.end(), 1, std::multiplies<int>())]), sub_tensor_shape);
            return sub_tensor;
        } 

        /* Const version of [] Operator overloading*/
        tensor operator[](int i) const
        {
            try
            {
                if(i >= t_shape[0])
                {
                    throw INDEX_OUT_OF_BOUNDS;
                }
            }

            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
            }
                    
            std::vector<int> sub_tensor_shape = t_shape;
            pop_front(sub_tensor_shape);
            tensor sub_tensor(&(data[i*std::accumulate(sub_tensor_shape.begin(), sub_tensor_shape.end(), 1, std::multiplies<int>())]), sub_tensor_shape);
            return sub_tensor;
        }

        /* + operator overloading*/
        tensor operator+(const tensor& A) const
        {
            tensor res(t_shape);
#ifdef BROADCAST_ENABLED
            bool is_broadcast = false;
#endif
                try
                {
                    if(!shape_equal(t_shape, A.t_shape))
                    {
                        throw TENSOR_SHAPE_NOT_MATCHING;
                    }
                }
                catch(TensorErrorTypes x)
                {
#ifdef BROADCAST_ENABLED
                    if(broadcastable(this->t_shape, A.get_shape()))
                    {
                        is_broadcast = true;
                        broadcast(res, 0);
                    }
                    else
                    {
                        std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                        exit(0);
                    }
#else
                    std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                    exit(0);
#endif
                }

                for(int i=0; i<this->num_elems; i++)
                {
#ifdef BROADCAST_ENABLED
                    res.data[i] = this->data[i] + (is_broadcast ? res.data[i] : A.data[i]);
#else
                     res.data[i] = this->data[i] +  A.data[i];
#endif
                }

            return res;
        }

        /* - operator overloading*/
        tensor operator-(const tensor& A) const
        {
            tensor res(t_shape);
#ifdef BROADCAST_ENABLED
            bool is_broadcast = false;
#endif
            try
                {
                    if(!shape_equal(t_shape, A.t_shape))
                    {
                        throw TENSOR_SHAPE_NOT_MATCHING;
                    }
                }
                catch(TensorErrorTypes x)
                {
#ifdef BROADCAST_ENABLED
                    if(broadcastable(this->t_shape, A.get_shape()))
                    {
                        is_broadcast = true;
                        broadcast(res, A, 0);
                    }
                    else
                    {
                        std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                        exit(0);
                    }
#else
                    std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                     exit(0);
#endif
                }

                for(int i=0; i<this->num_elems; i++)
                {
#ifdef BROADCAST_ENABLED
                    res.data[i] = this->data[i] - (is_broadcast ? res.data[i] : A.data[i]);
#else
                    res.data[i] = this->data[i] - A.data[i];
#endif
                }

            return res;
        }

        void operator+=(const tensor& A)
        {
            (*this) = (*this) + A;
        }

        void operator-=(const tensor& A)
        {
            (*this) = (*this) - A;
        }


        void mul2D(const tensor& A, tensor& res, int m, int n) const
        {
                for(auto r=m; r<n; r++)
                {
                    for(auto c=0; c<A.t_shape[1]; c++)
                    {
                        res.data[r*res.t_shape[1] + c] = 0;
                       for(auto k=0; k<this->t_shape[1]; k++)
                       {
#ifdef DEBUG_LOGS
                            //std::cout<<r<<", "<<c<<std::endl;
                            //std::cout<<r<<", "<<c<<", "<<k<<std::endl;
#endif
                            res.data[r*res.t_shape[1] + c] += (*this)[r][k].val() * A[k][c].val();
                       }
                    }
                }
        }

        /* * operator overloading*/
        tensor operator*(const tensor& A) const
        {
            /* Throw error if the dimesnions are less than 2 and 
               also if both the dimesnions are not compatible*/
            
            try
            {
                if(num_dims < 2)
                {
                    throw TENSOR_SHAPE_LESS_THAN_TWO;
                }

                if(this->num_dims != A.num_dims)
                {
                    throw TENSOR_SHAPE_NOT_MATCHING;
                }

                if(this->t_shape.back() != A.t_shape[A.num_dims-2])
                {
                    throw TENSOR_LAST_TWO_DIMS_MISMATCH_FOR_MAT_MUL;
                }
            }
            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
                return tensor();
            }

            tensor res(calc_dims_for_matmul(this->t_shape, A.t_shape));

            if(this->num_dims > 2)
            {
                for(auto i=0; i<this->t_shape[0]; i++)
                {
                    res[i] = (*this)[i] * A[i];
                }
            }
            else
            {
                /* Do the mutithreading part */
                int threads_called = 0, n = this->get_shape()[0] , work_per_thread = std::ceil((float)(n) / MAX_THREADS_POSSIBLE);
                std::thread th[MAX_THREADS_POSSIBLE];
                for(int m=0; m < n - work_per_thread; m += work_per_thread)
                {
                    th[threads_called++] = std::thread(&tensor::mul2D, this, A, std::ref(res), m, m+work_per_thread);
                }

                /* Do the matrix multiplication */
                mul2D(A, res, n-work_per_thread, n);

                /* Join the Threads after the work done*/
                for(int t=0; t<threads_called; t++)
                {
                    th[t].join();
                }
            }

            return res;
        }

        tensor operator*(double n)
        {
            tensor res(t_shape);
            for(int i=0; i<num_elems; i++)
            {
                res.data[i] = this->data[i]*n;
            }

            return res;
        }


        tensor operator/(tensor_double n)
        {
            tensor res(t_shape);
            for(int i=0; i<num_elems; i++)
            {
                res.data[i] = this->data[i]/n;
            }

            return res;
        }

         void operator*=(const tensor& A)
        {
            (*this) = (*this) * A;
        }

         void operator*=(tensor_double A)
        {
            (*this) = (*this) * A;
        }

        void operator/=(tensor_double A)
        {
            (*this) = (*this) / A;
        }

        tensor relu(tensor_double leak_val)
        {
            tensor res(this->t_shape);
            for(int i=0; i<this->num_elems; i++)
            {
                res.data[i] = this->data[i] > 0.0 ? this->data[i] : (this->data[i] * leak_val);
            }

            return res;
        }

        int elem_count() const
        {
            return this->num_elems;
        }

        tensor transpose()
        {
            try
            {
                if(this->num_dims != 2)
                {
                    throw TENSOR_NOT_MATRIX;
                }
            }
            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
            }

            tensor res(std::vector<int>{this->t_shape[1], this->t_shape[0]});

            for(int i=0; i<res.t_shape[0]; i++)
            {
                for(int j=0; j<res.t_shape[1]; j++)
                {
                    res.data[i*(res.t_shape[1]) + j] = (*this)[j][i].val();
                }
            }
            
            return res;
        }

        tensor avg()
        {
            try
            {
                if(this->num_dims == 1)
                {
                    throw TENSOR_DIMENSIONS_OUT_OF_RANGE;
                }
            }
            catch(TensorErrorTypes x)
            {
                std::cerr<<"File: "<<__FILE__<<", Function: "<<__func__<<", Line: "<<__LINE__<<", ERROR: "<<TensorErrorType[x]<<std::endl;
                exit(0);
            }

            std::vector<int> new_dims(this->t_shape);
            int dims = this->t_shape[0];
            new_dims[0] = 1;
            tensor res(new_dims);
            for(int i=0; i<dims; i++)
            {
                res[0] += (*this)[i];
            }
            
            return res/dims;
        }

        tensor_double sum() const
        {
            tensor_double res = 0;

            for(int i=0; i<this->get_elem_count(); i++)
            {
                res += this->data[i];
            }

            return res;
        }

        tensor one_hot_encoding(int num_classes)
        {
            auto new_shape = this->t_shape;
            new_shape[1] = num_classes;

            tensor Y_encoded = tensor(new_shape);

            for(int i=0; i<new_shape[0]; i++)
            {
                Y_encoded[i][(int)((*this)[i][0].val())].data[0] = 1;
            }

            return Y_encoded;
        }

        void randomize(int size)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            tensor_double stddev = sqrt(2.0f / size);
            std::normal_distribution<tensor_double> normal_dist(0.0f, stddev);

            for(auto i=0; i<this->num_elems; i++)
            {
                this->data[i] = normal_dist(gen);
            }
        }

        bool isInf() const
        {
            for(auto i=0; i<this->num_elems; i++)
            {
                if(std::isinf(this->data[i]))
                {
                    std::cout<<"Caught Nan/Inf value "<<this->data[i]<<std::endl;
                    return true;
                }
            }

            return false;
        }

        tensor argmax()
        {
            tensor res;

            if(this->num_dims > 1)
            {
                auto r_shape = this->get_shape();
                r_shape.pop_back();
                res = tensor(r_shape);
                for(int i=0; i < this->get_shape()[0]; i++)
                {
                    res[i] = (*this)[i].argmax();
                } 
            }
            else
            {
                res = tensor(std::vector<int>{1});
                int max = 0;
                for(int i=1; i<this->elem_count(); i++)
                {
                    max = (*this)[max].val() < (*this)[i].val() ? i : max;
                }
                res.data[0] = max;
            }

            return res;
        }

        tensor_double max() const
        {
            tensor_double max_elem = this->data[0];

            for(int i=1; i<this->get_elem_count(); i++)
            {
                max_elem = (this->data[i] > max_elem) ? this->data[i] : max_elem;
            }
            return max_elem;
        }

        bool invalid() const
        {
            for(int i=0; i<this->get_elem_count(); i++)
            {
                if(std::isnan(this->data[i]) || std::isinf(this->data[i]))
                {
                    return true;
                }
            }

            return false;
        }
        
        tensor extend(int n)
        {
            auto res_shape = this->get_shape();
            res_shape[0] = n;
            tensor res = tensor(res_shape);
            for(int i=0; i<n; i++)
            {
                res[i] = (*this)[0];
            }

            return res;
        }
        /* Reference and dereferencing overloading as required */

        /* Reshape Fucntion*/
};
#endif