// System includes --------------------
#include <iostream>
#include <map>
#include <string>

// Own includes --------------------
#include "nnet/activation-layer.hpp"
#include "nnet/fully-connected-layer.hpp"
#include "nnet/neural-network.hpp"
#include "nnet/weight-initializer.hpp"
#include "nnet/network-trainer.hpp"
#include "nnet/optimizers.hpp"

using namespace NNet;

int main(int argc, char *argv[]) {

	// declare numeric traits type
	using NumericTraitsType = NumericTraits< double >;

	// declare layer types
	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;
	using LogisticActLayerType = ActivationLayer< NumericTraitsType, LogisticActivation >;
	using SoftMaxActLayerType = ActivationLayer< NumericTraitsType, SoftMaxActivation >;

	// create activation functions
	using LogisticActFunType = LogisticActivation< NumericTraitsType >;
	LogisticActFunType logisticActFun;
	using SoftMaxActFunType = SoftMaxActivation< NumericTraitsType >;
	SoftMaxActFunType softMaxActFun;

	// create the neural network initializer
	using InitializerType = HeInitializer< NumericTraitsType >;
	InitializerType initializer;

	// create the neural network
	using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
	NetworkType nnet( initializer );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 784, 300, LayerType::INPUT ) );
	nnet.addLayer( std::make_shared< LogisticActLayerType >( 300, logisticActFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 300, 100, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< LogisticActLayerType >( 100, logisticActFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 100, 10, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< SoftMaxActLayerType >( 10, softMaxActFun ) );
	nnet.finalize( );
	nnet.printNetworkInfo( );

	// create the network trainer
	using OptimizerType = SGDOptimizer< NetworkType >;
	OptimizerType optimizer( nnet, 0.001 );
	using NetworkTrainerType = NetworkTrainer< NetworkType, OptimizerType, CrossEntropyLossFuction >;
	NetworkTrainerType networkTrainer( nnet, optimizer );
	return 0;
}
