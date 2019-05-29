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
#include "nnet/data-handlers.hpp"

using namespace NNet;

int main(int argc, char *argv[]) {

	// declare numeric traits type
	using NumericTraitsType = NumericTraits< double >;

	// import data
	using DataHandlerType = RegressionDataHandler< NumericTraitsType::VectorXType, NumericTraitsType::VectorXType >;
	DataHandlerType dataHandler;
	dataHandler.loadData( "../tests/data/bessel_1.data", "", ' ' );
	// dataHandler.printData( );

	// declare layer types
	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;

	// create activation functions
	// using ActFunType = LogisticActivation< NumericTraitsType >;
	using ActFunType = TanHActivation< NumericTraitsType >;
	using ActLayerType = ActivationLayer< NumericTraitsType, TanHActivation >;
	ActFunType actFun;

	// create the neural network initializer
	using InitializerType = HeInitializer< NumericTraitsType >;
	InitializerType initializer;

	// create the neural network
	using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
	NetworkType nnet( initializer );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 1, 20, LayerType::INPUT ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 20, actFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 20, 20, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 20, actFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 20, 1, LayerType::HIDDEN ) );
	nnet.finalize( );
	nnet.printNetworkInfo( );

	// create the network trainer
	// using OptimizerType = SGDOptimizer< NetworkType >;
	using OptimizerType = NesterovMomentumOptimizer< NetworkType >;
	OptimizerType optimizer( nnet, 0.01 );
	using NetworkTrainerType = NetworkTrainer< NetworkType, OptimizerType, MSELossFuction, DataHandlerType >;
	NetworkTrainerType networkTrainer( nnet, optimizer, dataHandler );

	for ( std::size_t i = 0; i < 1000; ++i ) {
		auto epochLoss = networkTrainer.trainEpoch( 50 );
		std::cout << "Epoch Loss <" << i << ">: " << epochLoss << std::endl;
	}

	auto computePrediction = [&]( std::ostream& os, auto const& data ) {
		for ( auto const& dataPair : data ) {
			auto const& input = dataHandler.getInput( dataPair );
			networkTrainer.computeForward( input );
			os << input << " " << networkTrainer.getNetwork( ).getLastOutput( ) << std::endl;
		}
	};
	// computePrediction( std::cout, dataHandler.getTrainingData( ) );
	std::ofstream OFS_PREDICT( "prediction.data" );
	computePrediction( OFS_PREDICT, dataHandler.getTrainingData( ) );

	return 0;
}
