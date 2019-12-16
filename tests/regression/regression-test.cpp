// System includes --------------------
#include <iostream>
#include <map>
#include <string>

// Own includes --------------------
#include "layers/activation-layer.hpp"
#include "layers/fully-connected-layer.hpp"
#include "networks/neural-network.hpp"
#include "initializers/weight-initializer.hpp"
#include "networks/network-trainer.hpp"
#include "optimizers/optimizers.hpp"
#include "data-handlers/data-handlers.hpp"

using namespace NNet;

int main(int argc, char *argv[]) {

	// declare numeric traits type
	using NumericTraitsType = NumericTraits< double >;

	// import data
	using DataHandlerType = RegressionDataHandler< NumericTraitsType::VectorXType, NumericTraitsType::VectorXType >;
	DataHandlerType dataHandler;
	std::string trainingDataFilePath = "../tests/data/regression/regression_bessel/bessel_1_noisy.data";
	std::string predictionFilePath = "../tests/data/regression/regression_bessel/bessel_1_noisy.prediction";
	char delim = ' ';
	dataHandler.loadData( trainingDataFilePath, "", delim );
	// dataHandler.printData( );

	// declare layer types
	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;

	// define activation fuction and layer types
	// using ActFunType = LogisticActivation< NumericTraitsType >;
	// using ActFunType = TanHActivation< NumericTraitsType >;
	using ActFunType = ArcTanActivation< NumericTraitsType >;
	using ActLayerType = ActivationLayer< NumericTraitsType, ArcTanActivation >;

	// define network intitializer type
	using InitializerType = GlorotInitializer< NumericTraitsType >;

	// create the neural network
	using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
	NetworkType nnet;
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 1, 30, LayerType::INPUT ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 30 ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 30, 20, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 20 ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 20, 10, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 10 ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 10, 1, LayerType::HIDDEN ) );
	nnet.finalize( );
	nnet.printNetworkInfo( );

	// create the network trainer
	// using OptimizerType = SGDOptimizer< NetworkType >;
	// using OptimizerType = NesterovMomentumOptimizer< NetworkType >;
	// using OptimizerType = AdaGradOptimizer< NetworkType >;
	// using OptimizerType = RMSPropOptimizer< NetworkType >;
	using OptimizerType = RMSPropNestMomOptimizer< NetworkType >;
	OptimizerType optimizer( nnet );
	using NetworkTrainerType = NetworkTrainer< NetworkType, OptimizerType, MSELossFuction, DataHandlerType >;
	NetworkTrainerType networkTrainer( nnet, optimizer, dataHandler );

	auto computePrediction = [&]( auto const& data, std::ostream& os = std::cout ) {
		for ( auto const& dataPair : data ) {
			auto const& input = dataHandler.getInput( dataPair );
			networkTrainer.computeForward( input );
			os << input << " " << networkTrainer.getNetwork( ).getLastOutput( ) << std::endl;
		}
	};

	// std::stringstream ss;
	// std::size_t predCtr = 0;
	for ( std::size_t i = 0; i < 1000; ++i ) {
		// if ( predCtr % 10 == 0 ) {
		// 	ss << "../tests/data/evo/pred_"  << predCtr << ".data";
		// 	std::ofstream OFS( ss.str( ).c_str( ) );
		// 	computePrediction( OFS, dataHandler.getTrainingData( ) );
		// 	OFS.close( );
		// 	std::stringstream( ).swap( ss );
		// }
		// predCtr++;

		auto epochLoss = networkTrainer.trainEpoch( 32 );
		std::cout << "Epoch Loss <" << i << ">: " << epochLoss << std::endl;
	}

	// computePrediction( std::cout, dataHandler.getTrainingData( ) );
	std::ofstream OFS_PREDICT( predictionFilePath );
	computePrediction( dataHandler.getTrainingData( ), OFS_PREDICT );

	return 0;
}
