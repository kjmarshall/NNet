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
	using DataHandlerType = MINSTDataHandler< NumericTraitsType::VectorXType, NumericTraitsType::VectorXType >;
	std::string trainingImagesPath = "../tests/data/minst/train-images.idx3-ubyte";
	std::string trainingLabelsPath = "../tests/data/minst/train-labels.idx1-ubyte";
	std::string testImagesPath = "../tests/data/minst/t10k-images.idx3-ubyte";
	std::string testLabelsPath = "../tests/data/minst/t10k-labels.idx1-ubyte";
	DataHandlerType dataHandler( trainingImagesPath, trainingLabelsPath, testImagesPath, testLabelsPath );
	//dataHandler.printData( );

	// declare layer types
	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;

	// create activation functions
	// using ActFunType = LogisticActivation< NumericTraitsType >;
	// using ActFunType = TanHActivation< NumericTraitsType >;
	using ActFunType = ReLUActivation< NumericTraitsType >;
	using ActLayerType = ActivationLayer< NumericTraitsType, LogisticActivation >;

	using SoftMaxActFunType = SoftMaxActivation< NumericTraitsType >;
	using SoftMaxActLayerType = ActivationLayer< NumericTraitsType, SoftMaxActivation >;

	// create the neural network initializer
	using InitializerType = GlorotInitializer< NumericTraitsType >;
	InitializerType initializer;

	// create the neural network
	using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
	NetworkType nnet( initializer );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 784, 300, LayerType::INPUT ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 300 ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 300, 100, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< ActLayerType >( 100 ) );
	nnet.addLayer( std::make_shared< FullyConnectedLayerType >( 100, 10, LayerType::HIDDEN ) );
	nnet.addLayer( std::make_shared< SoftMaxActLayerType >( 10 ) );
	nnet.finalize( );
	nnet.printNetworkInfo( );

	// create the network trainer
	// using OptimizerType = SGDOptimizer< NetworkType >;
	using OptimizerType = NesterovMomentumOptimizer< NetworkType >;
	// using OptimizerType = AdaGradOptimizer< NetworkType >;
	// using OptimizerType = RMSPropOptimizer< NetworkType >;
	// using OptimizerType = RMSPropNestMomOptimizer< NetworkType >;
	OptimizerType optimizer( nnet );
	using NetworkTrainerType = NetworkTrainer< NetworkType, OptimizerType, CrossEntropyLossFuction, DataHandlerType >;
	NetworkTrainerType networkTrainer( nnet, optimizer, dataHandler );

	auto computePrediction = []( auto& network_trainer, auto const& data, std::size_t& correct, std::size_t& incorrect, double& totalLoss, std::ostream& os, std::optional< std::reference_wrapper< std::ostream > > pred_out = {} ) {
		for ( auto const& [input,target] : data ) {
			auto const& lastOutput = network_trainer.computePrediction( input );
			auto [loss,gradLoss] = network_trainer.computeLoss( lastOutput, target );
			totalLoss += loss;
			auto targLabel = std::distance( target.begin( ), std::max_element( target.begin( ), target.end( ) ) );
			auto predLabel = std::distance( lastOutput.begin( ), std::max_element( lastOutput.begin( ), lastOutput.end( ) ) );
			if ( pred_out )
				( *pred_out ).get() << predLabel << " " << targLabel << std::endl;
			if ( targLabel == predLabel )
				++correct;
			else
				++incorrect;
		}
	};

	auto computeAccuracy = [&computePrediction]( auto& network_trainer, auto const& data, std::string const& header, std::ostream& os, std::optional< std::reference_wrapper< std::ostream > > pred_out = {} ) {
		std::size_t correct = 0, incorrect = 0;
		std::size_t total = data.size( );
		double totalLoss = 0.0;
		computePrediction( network_trainer, data, correct, incorrect, totalLoss, os, pred_out );
		os << header << " " << double(correct)/double(total) * 100. << ", " << correct << "/" << total << ", Total Loss = " << totalLoss << std::endl;
		return double( correct ) / double( total ) * 100.;
	};

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

	// Output the final prediction after training
	std::ofstream OFS_PREDICT( "prediction-minst.txt" );
	computeAccuracy( networkTrainer, dataHandler.getTestingData( ), "Testing accuracy = ", std::cout, OFS_PREDICT );
	OFS_PREDICT.close();

	// Serialize the network to file
	networkTrainer.saveNetwork( "minst-network-ser.txt" );

	// Load the network
	NetworkType nnet_load; // dummy network
	NetworkTrainerType networkTrainerLoad( nnet_load, optimizer, dataHandler );
	networkTrainerLoad.loadNetwork( "minst-network-ser.txt" );

	// Test prediction again
	std::ofstream OFS_PREDICT_LOAD_TEST( "prediction-minst-load-test.txt" );
	computeAccuracy( networkTrainerLoad, dataHandler.getTestingData( ), "Testing accuracy = ", std::cout, OFS_PREDICT_LOAD_TEST );
	OFS_PREDICT_LOAD_TEST.close();

	return 0;
}
