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
	std::string trainingDataFilePath = "../tests/data/bessel_1_noisy.data";
	std::string preditionFilePath = "../tests/data/bessel_1_noisy.prediction";
	dataHandler.loadData( trainingDataFilePath, "", ' ' );
	// dataHandler.printData( );

	// declare layer types
	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;

	// create activation functions
	// using ActFunType = LogisticActivation< NumericTraitsType >;
	// using ActFunType = TanHActivation< NumericTraitsType >;
	using ActFunType = ArcTanActivation< NumericTraitsType >;
	using ActLayerType = ActivationLayer< NumericTraitsType, ArcTanActivation >;
	ActFunType actFun;

	// create the neural network initializer
	using InitializerType = GlorotInitializer< NumericTraitsType >;
	InitializerType initializer;

	// create the neural network
	using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
	NetworkType nnet( initializer );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 1, 30, LayerType::INPUT ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 30, actFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 30, 20, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 20, actFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 20, 10, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 10, actFun ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 10, 1, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 1, actFun ) );
	nnet.finalize( );
	nnet.printNetworkInfo( );

	// create the network trainer
	// using OptimizerType = SGDOptimizer< NetworkType >;
	using OptimizerType = NesterovMomentumOptimizer< NetworkType >;
	OptimizerType optimizer( nnet, 0.005 );
	using NetworkTrainerType = NetworkTrainer< NetworkType, OptimizerType, MSELossFuction, DataHandlerType >;
	NetworkTrainerType networkTrainer( nnet, optimizer, dataHandler );

	auto computePrediction = [&]( std::ostream& os, auto const& data ) {
		for ( auto const& dataPair : data ) {
			auto const& input = dataHandler.getInput( dataPair );
			networkTrainer.computeForward( input );
			os << input << " " << networkTrainer.getNetwork( ).getLastOutput( ) << std::endl;
		}
	};

	std::stringstream ss;
	std::size_t predCtr = 0;
	for ( std::size_t i = 0; i < 1000; ++i ) {
		if ( predCtr % 10 == 0 ) {
			ss << "../tests/data/evo/pred_"  << predCtr << ".data";
			std::ofstream OFS( ss.str( ).c_str( ) );
			computePrediction( OFS, dataHandler.getTrainingData( ) );
			OFS.close( );
			std::stringstream( ).swap( ss );
		}
		predCtr++;

		auto epochLoss = networkTrainer.trainEpoch( 5 );
		std::cout << "Epoch Loss <" << i << ">: " << epochLoss << std::endl;
	}

	// computePrediction( std::cout, dataHandler.getTrainingData( ) );
	std::ofstream OFS_PREDICT( preditionFilePath );
	computePrediction( OFS_PREDICT, dataHandler.getTrainingData( ) );

	return 0;
}
