# NNet
NNet is a basic deep learning C++ template library that implements different types of layer based neural network architectures.  The goal of the this project is to document an approach to the modern C++ design, implementation, and architecture of a simple yet flexible deep learning framework.  The library is both fast and flexible owing to its use of compile time polymorphism in layer, optimizer, and activation function design and its dynamic dispatch in network training.  NNet allows for the easy construction of dense fully connected feed forward networks.  Data of different ranks and of different numeric type (e.g. float, double) can be handled in a straightforward way by overloading the `NumericTraits` template class. The library takes advantage of modern C++17 programming techniques. Examples of the library's use are provided with sample documentation discussed below.

## Documentation and Derivations
A portion of this project is also concerned with documenting rigorous derivations for various learning algorithms (e.g. back propagation).  Derivations using Einstein summation notation (ESN) are provided in a living LaTeX document (see the [docs](./docs) directory) which will change as new functionality or network architectures are added.  

# Contributions and Contact Information
This project initially started because of an interest in implementing deep net architectures.  Outside contributions as pull requests and discussions are welcome.  Feel free to contact me through my [LinkedIn](http://www.linkedin.com/in/kevin-j-marshall).

## Table of Contents

<!-- toc -->

  * [Getting Started](#getting-started)
    + [Requirements](#requirements)
  * [API](#api)
    + [Data Types and Numeric Precision Controls](#data-types-and-numeric-precision-controls)
    + [Data Handlers](#data-handlers)
    + [Layers](#layers)
      - [Fully Connected Layers](#fully-connected-layers)
      - [Activation Layers](#activation-layers)
    + [Initializers](#initializers)
    + [Network](#network)
    + [Optimizers](#optimizers)
      - [Optimizers with Momentum](#optimizers-with-momentum)
      - [Optimizers with Adaptive Learning Rates](#optimizers-with-adaptive-learning-rates)
      - [Optimizers with Momentum and Adaptive Learning Rates](#optimizers-with-momentum-and-adaptive-learning-rates)
    + [Network Trainer](#network-trainer)
  * [Network Training](#network-training)
  * [Serialization, Saving, and Loading](#serialization-saving-and-loading)
- [Todo List](#todo-list)

<!-- tocstop -->

<!-- npx markdown-toc -i README.md -->

## Getting Started
The easiest way to get started using or understanding NNet is to read the documentation below and examine some of the examples and tests provided in the [tests](./tests) directory.

### Requirements
- A compiler that supports C++17 features (e.g. std::optional, structured bindings, constexpr if )
- CMake >= 3.10
- Boost >= 1.56, serialization
- Eigen >= 3.3
- Google Test >= 1.10

## API
Basic usage of the library is shown in examples
- regression-test.cpp
- minst-test.cpp

The following subsections describe the relevant api, classes, and implementation designs in roughly the order that they would be used or invoked to begin constructing a neural network for modeling a given ML problem.
- Define the data handler, import training and testing data
- Define the layer types (activation, fully connected, etc...) to be used in the neural network
- Define an initializer scheme to seed the network layers
- Define a network that glues together data, layers, and an initializer
- Define an optimizer that performs some sort of gradient descent
- Define a network trainer that glues together the network an optimizer
- Use the network trainer to train the network
- Serialize/save the network or use it to perform predictions
- De-serialize/load the network to re-train to to perform predictions at a later time

### Data Types and Numeric Precision Controls
Data types and numeric precision are controlled by the `NumericTraits` class which is templated on a fundamental data type (usually a POD type) such as float or double.  Most classes take a `NumericTraitsType` template parameter which is intended to alias working data types.  NNet may be extended to more complex data types by providing an overload or specialization to the `NumericTraits` template struct. 

### Data Handlers
Prior to training, all neural nets need some way to import training and testing data.  Data import is handled by a `BaseDataHandler` class template that is parameterized on input and target data types
```c++
template< typename InputDataType, 
          typename TargetDataType >
class BaseDataHandler
```
For most use cases it is general enough to template on a pseudo-vector type e.g. `using InputDataType = Eigen::VectorXd`, that can adequately capture testing and training `(input,target)` pairs.  The `BaseDataHandler` internalizes this pair as an element of an `std::vector< std::pair >`.  Derived classes must implement a `loadData()` member function that populates testing and training data as a container `std::pair` of inputs and targets.

### Layers
This library currently only supports computations on fully connected layers and activation layers.  Layers are templated on numeric type.

#### Fully Connected Layers
Fully connected layers are the main currency by which information propagates through the network net.  
```c++
template< typename NumericTraitsType >
class FullyConnectedLayer
	: public TrainableLayer< NumericTraitsType >
```
Fully connected layers are constructed by specifying the number of inputs, outputs, and layer type
```c++
std::make_shared< FullyConnectedLayerType >( 30, 20, LayerType::HIDDEN );
```
Fully connected layers are trainable layers, notably different from activation layers. This distinction is used during training via dynamic dispatch i.e. downcasting to the derived class.  Fully connected layers store input, output, and weight matrix data members as well as associated gradient information.

#### Activation Layers
Activation layers take inputs and produce a (usually) non-linear output.  They are templated on numeric type and activation function type
```c++
template< typename NumericTraitsType, 
		  template < typename > class ActFun >
class ActivationLayer
	: public BaseLayer< NumericTraitsType >
```
The activation function `ActFun` is a template class with one parameter, in this case numeric type.  The following activation functions are currently supported,
- Identity
- Logistic
- TanH
- ArcTan
- ReLU
- PReLu
- ELU
- SoftMax
- LogSoftMax

Activation layers are constructed by specifying the number of inputs and outputs (the example constructor assumes |inputs| == |outputs|),
```c++
using ActLayerType = ActivationLayer< NumericTraitsType, ArcTanActivation >;
std::make_shared< ActLayerType >( 30 );
```

### Initializers
An initializer must be declared and used to initialize all weights in each layer of the neural network.  Currently three initializers are implemented
- Gauss ( weight matrix elements initialized by sampling random normal N(0,1) variable )
```c++
template< typename NumericTraitsType >
class GaussInitializer
	: public BaseInitializer< NumericTraitsType >
```
- Glorot ( weight matrix elements by sampling a random normal N(0, 2/( inputs + outputs) ) variable )
```c++
template< typename NumericTraitsType >
	class GlorotInitializer
	: public BaseInitializer< NumericTraitsType >
```
- He ( weight matrix elements initialized by sampling a random normal N(0,2/outputs) variable )
```c++
template< typename NumericTraitsType >
class HeInitializer
	: public BaseInitializer< NumericTraitsType >
```
Initializers are applied to each trainable layer in a network to initialize weight matrices via different strategies.  Weights are initialized according by sampling various distributions usually parameterized by layer input or output size.

### Network
The neural network class is templated on data type and the initializer type e.g.
```c++
using InitializerType = GlorotInitializer< NumericTraitsType >;
using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
```
Layers may then be added to the network.  An efficient way to do this is via `std::make_shared` leveraging single heap-allocation of of the layer and shared pointer control block over the std::shared_ptr constructor.  For regression analysis a simple neural network may be constructed as,
```c++
NetworkType nnet( initializer );
nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 1, 30, LayerType::INPUT ) );
nnet.addLayer( std::make_shared< ActLayerType >( 30 ) );
nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 30, 20, LayerType::HIDDEN ) );
nnet.addLayer( std::make_shared< ActLayerType >( 20 ) );
nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 20, 10, LayerType::HIDDEN ) );
nnet.addLayer( std::make_shared< ActLayerType >( 10 ) );
nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 10, 1, LayerType::HIDDEN ) );
nnet.finalize( );
```
This example network takes a single input (e.g. a x-value), and feeds it into 30 different nodes in the input layer.  These 30 outputs are fed into a nonlinear activation layer and then a hidden layer which collapses the outputs to 20.  These 20 outputs are fed into a combination of activation layers and hidden layers until the last layer where the output is collapsed to 1 (e.g. the y-value prediction).  After the layers have been arranged, the network calls the member function `nnet.finalize()` which initializes the weight matrix elements using the supplied initializer and sets the bias weights to zero.

### Optimizers
Optimizers are classes which provide update rules for weight matrices.  This library implements the following optimizers,
- Stochastic Gradient Descent (SGD)
```c++
template< typename NetworkType >
class SGDOptimizer
	: public BaseOptimizer< NetworkType >
```
#### Optimizers with Momentum
- SGD with Momentum
```c++
template< typename NetworkType >
class MomentumOptimizer
	: public BaseOptimizer< NetworkType >
```
- Nesterov Accelerated Graident (NAG) Descent
```c++
template< typename NetworkType >
class NesterovMomentumOptimizer
	: public BaseOptimizer< NetworkType >
```
Optimizers with momentum typically apply an interim update
```c++
void applyInterimUpdate( ) override {
	auto v_iter = mWeightGradMatSaves.begin( );
		for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
			auto& weightMat = layerPtr -> getWeightMat( );
			weightMat = weightMat - mMomentum * ( *v_iter );
			++v_iter;
		}
}
```
that corrects a trainable layer's weight matrix with a previous gradient computation.

#### Optimizers with Adaptive Learning Rates
- AdaGrad (Adaptive Gradient)
```c++
template< typename NetworkType >
class AdaGradOptimizer
	: public BaseOptimizer< NetworkType >
```
- RMSProp (Root Mean Square Propagation)
```c++
template< typename NetworkType >
class RMSPropOptimizer
	: public BaseOptimizer< NetworkType >
```
#### Optimizers with Momentum and Adaptive Learning Rates
- RMSProp with NAG
```c++
template< typename NetworkType >
class RMSPropNestMomOptimizer
	: public BaseOptimizer< NetworkType >
```
All optimizers must implement `BaseOptimizer< NetworkType >::applyWeightUpdate( std::size_t batchSize )`.  This pure virtual function implements methods to update weights matrices that are stored in trainable layers.

### Network Trainer
The network trainer class may be used to train a neural network.  A network trainer combines the neural network with an optimizer, a loss function, and a data handler.  A neural network is trained by running training samples through the following three steps,
1. forward computation (matrix vector products)
2. loss computation
3. backward computation (error propagation)

These three steps are shown in the single sample compute function `runSingleSample` that operates on a single training sample with an associated input and target,
```c++
NumericType runSingleSample( VectorXType const& inputVec,
							 VectorXType const& targetVec ) {
	computeForward( inputVec );
	auto [loss, gradLoss] = computeLoss( getNetwork( ).getLastOutput( ), targetVec );
	// std::cout << "Single Sample Loss: " << loss << std::endl;
	computeBackward( gradLoss );
	return loss;
}
```
The functions `computeForward` and `computeBackward` operate sequentially on each layer in the network and return single sample loss metric.  The network trainer class provides methods to perform batch training
```c++
template< typename IterType, typename ActionType >
void for_each_batch( IterType begin, IterType end, std::size_t batchSize, ActionType&& action )
``` 
And training over an entire epoch (with randomly shuffled data),
```c++
NumericType trainEpoch( std::size_t batchSize )
```

## Network Training
After the training data, testing data, and network have been specified, the network may be trained by calling the member function over an epoch loop
```c++
NetworkTrainer< NetworkType, OptimizerType, MSELossFuction, DataHandlerType >::trainEpoch( batch_size )
```
Some of the supplied examples [tests/minst/minst-test.cpp](./tests/minst/minst-test.cpp) use `computeAccuracy( ... )` and `computePrediction( ... )` lambda functions to update the user with accuracy and prediction measurements after each epoch c.f.
```c++
// Train the network
std::ofstream OFS_LC( "learning-curves-minst.txt" );
OFS_LC << "#Epoch Training Validation Testing" << std::endl;
std::size_t num_epochs = 32, batch_size = 64;
for ( std::size_t epoch = 1; epoch <= num_epochs; ++epoch ) {
	// train a single epoch with given batch_size
	networkTrainer.trainEpoch( batch_size );
	// training accuracy
	double train_acc = computeAccuracy( networkTrainer, dataHandler.getTrainingData(), "Training accuracy = ", std::cout );
	// validation acuracy
	double valid_acc = 0;
	// testing accuracy
	double test_acc = computeAccuracy( networkTrainer, dataHandler.getTestingData(), "Testing accuracy = ", std::cout );
	// save learning curve data
	OFS_LC << epoch << " " << train_acc << " " << valid_acc << " " << test_acc << std::endl;
}
OFS_LC.close();
```

## Serialization, Saving, and Loading
For a neural network library to be useful, one must be able to save/serialize and load/de-serialize the network's state.  Obviously, if a network's trained state can not be saved then all future prediction power is lost after program execution completes.  A major requirement for me to release NNet was to have a working save/load serialization scheme.  I wanted the library to be useful in the sense that it could be used in the following real world ML work flow,

1. Train a network on real world data
2. Save the results to file
3. Load the data from file
4. Make predictions on newly acquired data by running a forward pass.

Serialization in NNet is currently implemented using `boost::serialization`.  A network may be serialized or de-serialized from a `NetworkTrainer` object using the member functions
```c++
bool NetworkTrainer< NetworkType, OptimizerType, MSELossFuction, DataHandlerType >::saveNetwork( std::string const& file_path )
bool NetworkTrainer< NetworkType, OptimizerType, MSELossFuction, DataHandlerType >::loadNetwork( std::string const& file_path )
```
Both text and binary archive types are currently supported.  Archive type is deduced based on the `file_path` extension e.g. .txt or .bin.

The implementation details of serialization are complex due to the inheritance hierarchy trees, different layer types, and need to serialize all data containers (e.g. Eigen vectors).  For more details see the examples found in the [tests](./tests) directory and the specific test `TEST( Serialization, NeuralNetwork )` in [tests/gtests/test.cpp](./tests/gtests/test.cpp).

# Todo List
- Implement different neural net architectures
	- CNN
	- GANs
   	- RNN
		- LSTM
		- autoencoder
- Implement weight regularization
- Review existing design and implement better design patterns
- Attempt time series prediction
- Saving and loading
- Implement exception handling with try catch error handling
- Develop training accuracy and testing accuracy outputs and plotting capabilities
	- confusion matrix for minst
- Format network info printer as table
	- integrate fmtlib as a submodule
- ~~Finish developing network serialization~~
	- ~~Serialize/savee~~
	- ~~Deserialize/load~~
- Proofread docs
- Generate nicer example docs
	- Produce training and testing accuracy plot
- Check library files and structure
	- Organize tests folder, which data to serialize for examples?
- ~~Release first project version~~
- Implement training with CUDA and or parallelize
- Qt window for live display outputs