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
	using ActFunType = LogisticActivation< NumericTraitsType >;
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
	// using OptimizerType = NesterovMomentumOptimizer< NetworkType >;
	using OptimizerType = AdaGradOptimizer< NetworkType >;
	OptimizerType optimizer( nnet, 0.05 );
	using NetworkTrainerType = NetworkTrainer< NetworkType, OptimizerType, CrossEntropyLossFuction, DataHandlerType >;
	NetworkTrainerType networkTrainer( nnet, optimizer, dataHandler );

	auto computePrediction = [&]( auto const& data, std::size_t& correct, std::size_t& incorrect, double& totalLoss, std::ostream& os ) {
		for ( auto const& dataPair : data ) {
			auto const& input = dataHandler.getInput( dataPair );
			auto const& target = dataHandler.getTarget( dataPair );
			networkTrainer.computeForward( input );
			auto const& lastOutput = networkTrainer.getNetwork( ).getLastOutput( );
			auto [loss,gradLoss] = networkTrainer.computeLoss( lastOutput, target );
			totalLoss += loss;
			auto targLabel = std::distance( target.begin( ), std::max_element( target.begin( ), target.end( ) ) );
			auto predLabel = std::distance( lastOutput.begin( ), std::max_element( lastOutput.begin( ), lastOutput.end( ) ) );
			if ( targLabel == predLabel )
				++correct;
			else
				++incorrect;
		}
	};

	auto computeAccuracy = []( auto const& data, std::string const& header, std::ostream& os ) {
		std::size_t correct = 0, incorrect = 0;
		std::size_t total = data.size( );
		double totalLoss = 0.0;
		computePrediction( data, correct, incorrect, totalLoss, os );
		os << header << " " << double(correct)/double(total) * 100. << ", " << correct << "/" << total << ", Loss = " << totalLoss << std::endl;
		return double( correct )/ double( incorrect );
	};

	auto computeTrainingAccuracy = []( auto const& data, std::ostream& os ) {
		std::string header = "Training accuracy =";
		computeAccuracy( data, header, os );
	};
	auto computeTestingAccuracy = []( auto const& data, std::ostream& os ) {
		std::string header = "Testing accuracy =";
		computeAccuracy( data, header, os );
	};

	// Train the network
	std::size_t num_epochs = 20, batch_size = 32;
	networkTrainer.trainNetwork( num_epochs, batch_size,computeTrainingAccuracy, computeTestingAccuracy );

	// Output the final prediction after training
	// std::ofstream OFS_PREDICT( preditionFilePath );
	// computePrediction( OFS_PREDICT, dataHandler.getTrainingData( ) );

	// Serialize the network to file

	return 0;
}
