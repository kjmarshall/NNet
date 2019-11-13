# Deep Neural Networks
Various formulas are derived in detail in a latex document found in the docs/ directory.  Specifically, backpropagation and gradient decent are derived using Einstein index notation.

## Table of Contents

## Introduction
This library implements a variety of building blocks for creating different types of neural networks using modern C++17 techniques.

## Requirements
- CMake 3.10
- Boost 1.56, serialization
- Eigen 3.3
- Google Test 1.10

## API
Basic usage of the library is shown in examples
- regression-test.cpp
- minst-test.cpp
The following subsections describe the relevant classes roughly in the order that they would be used in a typical ML problem.

### Data Handlers
Prior to training all nerual nets need some way to import training and testing data.  Data import is handled by a `BaseDataHandler` class templated on input data type and target data type c.f.
```c++
template< typename InputDataType, 
          typename TargetDataType >
class BaseDataHandler
```
For most use cases it is general enough to template on a pseudo-vector type e.g. `using InputDataType = Eigen::VectorXd`, that can adequately capture testing and training `(input,target)` pairs.  The `BaseDataHandler` internalizes this pair as an element of an `std::vector`.  Derived classes must implement a `loadData()` member function that populates testing and training data as a container `std::pair` of inputs and targets.

### Layer Declarations
This library currently only supports computions on fully connected layers and activation layers.  Layers are templated on numeric type.

#### Fully Connected Layers
Fully connected layers are the main currency by which information propagates throught net.  
```c++
template< typename NumericTraitsType >
class FullyConnectedLayer
	: public TrainableLayer< NumericTraitsType >
```
Fully connected layers are constructed by specifiying the number of inputs, outputs, and layer type
```c++
std::make_shared< FullyConnectedLayerType >( 30, 20, LayerType::HIDDEN );
```
Fully connected layers are trainable layers, notably different from activation layers.  This distintion is used during training via dynamic dispatch i.e. downcasting to the derived class.  Fully connected layers store input, output, and weight matrix data members as well as associated gradient information.

#### Activation Layers
Activation layers take inputs and produce a non-linear (usually) output.  They are templated on numeric type and activation function type, c.f.
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

Activation layers are constructed by specifying the number of inputs and outputs (note that constructor assumed |inputs| == |outputs|),
```c++
using ActLayerType = ActivationLayer< NumericTraitsType, ArcTanActivation >;
std::make_shared< ActLayerType >( 30 );
```

### Initializers
For efficiently an initializer must be decalared and used to initialize all weights in each layer of the neural network.  Currently three initializers are implemented
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

### Network
The neural network class is templated on data type and the intitializer type e.g.
```c++
using InitializerType = GlorotInitializer< NumericTraitsType >;
using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
```
Layers may then be added to the network.  An efficient way to do this is via `std::make_shared` leveraging the single heap-allocation over the std::shared_ptr constructor.  For regression analysis a simple neural network may be constructed as,
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
This example network takes a single input (e.g. an x-value), and feeds it into 30 different nodes in the input layer.  These 30 outputs are fed into a nonlinear activation layer and then a hidden layer which collapses the outputs to 20.  These 20 outputs are fed into a combination of activation layers and hidden layers until the last layer where the output is collapsed to 1 (e.g. the y-value prediction).  After the layers have been arranged, the network calls the member function `nnet.finalize()` which initializes the weight matrix elements using the supplied initializer and sets the bias weights to zero.

### Optimizer
Optimizers are classes which provide update rules for weight matrices.  This library implments the following optimizers,
- Stochastic Gradient Descent (SGD)
```c++
template< typename NetworkType >
class SGDOptimizer
	: public BaseOptimizer< NetworkType >
```
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
All optimizers must implement `BaseOptimizer< NetworkType >::applyWeightUpdate( std::size_t batchSize )`.  This virtual function implements methods to update weights matrices that are stored in trainable layers.
### Network Trainer
## Model Training
## Serialization, Saving, and Loading