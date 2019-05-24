// System includes --------------------
#include <iostream>
#include <map>
#include <string>

// Own includes --------------------
#include "nnet/activation-layer.hpp"
#include "nnet/fully-connected-layer.hpp"
#include "nnet/neural-network.hpp"

using namespace NNet;

int main(int argc, char *argv[]) {

	// declare numeric traits type
	using NumericTraitsType = NumericTraits< double >;

	// declare activation layer type
	using LogisticActFunType = LogisticActivation< NumericTraitsType >;
	using LogisticActLayerType = ActivationLayer< NumericTraitsType, LogisticActivation >;

	// create activation layer
	LogisticActFunType logisticActFun;
	auto actLayerPtr = std::make_shared< LogisticActLayerType >( 10, 10, logisticActFun );

	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;
	auto fcLayerPtr = std::make_shared< FullyConnectedLayerType >( 10, 10, TrainableLayerType::INPUT );

	// create the neural network
	NeuralNetwork< NumericTraitsType > nnet;
	nnet.addLayer( fcLayerPtr );
	nnet.printLayerInfo( );

	return 0;
}
