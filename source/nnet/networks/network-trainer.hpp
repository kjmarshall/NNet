#ifndef NETWORK_TRAINER_HPP
#define NETWORK_TRAINER_HPP

// System includes --------------------
#include <filesystem>

// Own includes --------------------
#include "loss/loss-function.hpp"
#include "utils/progress-bar.hpp"

namespace NNet { // begin NNet

	/**
	 *NetworkTrainer.
	 */
	template< typename NetworkType,
			  typename OptimizerType,
			  template< typename > class LossFunType,
			  typename DataHandlerType >
	class NetworkTrainer {
	public: 	// public typedefs
		using NumericTraitsType = typename NetworkType::NumericTraitsType;
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;

	private: 	// private typedefs

	public: 	//public member functions
		NetworkTrainer( ) = default;
		explicit NetworkTrainer( NetworkType& network, OptimizerType& optimizer, DataHandlerType& dataHandler )
			: mNetwork( network ), mOptimizer( optimizer ), mDataHandler( dataHandler ) {
		}
		NetworkTrainer(const NetworkTrainer &c) = delete;
		~NetworkTrainer( ) = default;

		// accessors
		auto& getNetwork ( ) { return mNetwork; }
		auto const& getNetwork( ) const { return mNetwork; }
		auto& getOptimizer( ) const { return mOptimizer; }
		auto const& getLossFun( ) const { return mLossFun; }

		// training
		// compute forward
		void computeForward( VectorXType const& inputVec ) {
			// run forward compute
			VectorXType inputWorkVec, outputWorkVec;
			if ( auto firstLayer = getNetwork( ).getFirstLayer( ) ) {
				(*firstLayer) -> forwardCompute( inputVec, outputWorkVec );
				inputWorkVec = outputWorkVec;
			}
			else {
				throw std::runtime_error( "Can't forward compute on first layer..." );
			}
			for ( auto layerIter = getNetwork( ).begin( ) + 1; layerIter != getNetwork( ).end( ); ++layerIter ) {
				auto& layerPtr = (*layerIter);
				layerPtr -> forwardCompute( inputWorkVec, outputWorkVec );
				inputWorkVec = outputWorkVec;
			}
		}

		// compute loss
		auto
		computeLoss( VectorXType const& outputVec,
					 VectorXType const& targetVec ) {
			// compute loss
			NumericType loss = getLossFun( ).loss( outputVec, targetVec );
			auto gradLoss = getLossFun( ).gradLoss( outputVec, targetVec );
			return std::make_pair( loss, gradLoss );
		}

		// compute backward
		void computeBackward( VectorXType const& gradLoss ) {
			// run backward compute
			VectorXType dummyVec;
			VectorXType	inputDeltaWorkVec = gradLoss, outputDeltaWorkVec;
			if ( auto lastLayer = getNetwork( ).getLastLayer( ) ) {
				(*lastLayer) -> backwardCompute( dummyVec, dummyVec, inputDeltaWorkVec, outputDeltaWorkVec );
				inputDeltaWorkVec = outputDeltaWorkVec;
			}
			else {
				throw std::runtime_error( "Can't backward compute on last layer...");
			}
			for ( auto layerIter = getNetwork( ).rbegin( ) + 1; layerIter != getNetwork( ).rend( ); ++layerIter ) {
				auto& layerPtr = (*layerIter);
				layerPtr -> backwardCompute( dummyVec, dummyVec, inputDeltaWorkVec, outputDeltaWorkVec );
				inputDeltaWorkVec = outputDeltaWorkVec;
			}
		}

		//compute single prediction
		auto computePrediction( VectorXType const& inputVec ) {
			computeForward( inputVec );
			auto const& lastOutput = getNetwork( ).getLastOutput( );
			return lastOutput;
		}

		NumericType runSingleSample( VectorXType const& inputVec,
									 VectorXType const& targetVec ) {
			computeForward( inputVec );
			auto [loss, gradLoss] = computeLoss( getNetwork( ).getLastOutput( ), targetVec );
			// std::cout << "Single Sample Loss: " << loss << std::endl;
			computeBackward( gradLoss );
			return loss;
		}

		template< typename IterType, typename ActionType >
		void for_each_batch( IterType begin, IterType end, std::size_t batchSize, ActionType&& action ) {
			auto iterTo = begin;
			while ( iterTo != end ) {
				auto iterFrom = iterTo;
				auto ctr = batchSize;
				while ( ctr > 0 && iterTo != end ) {
					++iterTo;
					--ctr;
				}
				action( iterFrom, iterTo );
			}
		}

		NumericType trainEpoch( std::size_t batchSize ) {
			NumericType epochLoss = 0.0;
			std::size_t batchCtr = 1;
			std::size_t sampleCtr = 0;
			mDataHandler.shuffleTrainingData( getNetwork( ).getInitializer( ).getRandomEngine( ) );
			auto& data = mDataHandler.getTrainingData( );
			std::size_t num_batchs = data.size() / batchSize + 1;
			Utils::ProgressBar progress_bar( num_batchs, "" );
			for_each_batch( data.begin( ), data.end( ), batchSize,
							[&,this]( auto& iterFrom, auto& iterTo ) {
								progress_bar.updateLastPrintedMessage( "Training on batch " + std::to_string( batchCtr ) + "/" + std::to_string( num_batchs ) );
								getOptimizer( ).applyInterimUpdate( );
								NumericType batchLoss = 0.0;
								std::size_t realBatchSize = 0;
								for ( auto& iter = iterFrom; iter < iterTo; ++iter ) {
									auto const& inputVec = mDataHandler.getInput( *iter );
									auto const& targetVec = mDataHandler.getTarget( *iter );
									// std::cout << "inputVec, targetVec: "
									// 		  << inputVec << ", " << targetVec << std::endl;
									batchLoss += runSingleSample( inputVec, targetVec );
									// std::cout << "BatchLoss, batchCtr, sampleCtr: " << batchLoss << ", " << batchCtr << ", " << sampleCtr << std::endl;
									++sampleCtr;
									++realBatchSize;
								}
								// update weights
								batchLoss /= static_cast< NumericType >( realBatchSize );
								epochLoss += batchLoss;
								getOptimizer( ).applyWeightUpdate( realBatchSize );

								// reset gradients
								getOptimizer( ).resetGradients( );

								++batchCtr;
								++progress_bar;
							} );
			epochLoss /= static_cast< NumericType >( batchCtr );
			return epochLoss;
		}

		NumericType trainSingleSample( VectorXType const& inputVec,
									   VectorXType const& targetVec ) {
			getOptimizer( ).applyInterimUpdate( );
			NumericType loss = runSingleSample( inputVec, targetVec );
			// update weights
			std::size_t batchSize = 1;
			getOptimizer( ).applyWeightUpdate( batchSize );
			// reset gradients
			getOptimizer( ).resetGradients( );
			return loss;
		}

		bool saveNetwork( std::string const& file_path ) {
			auto path = std::filesystem::path( file_path );
			std::string ext = path.extension().string();
			if ( ext == ".txt" ) {
				using ArchiveOutType = boost::archive::text_oarchive;
				SerializationArchive< ArchiveOutType > ar( file_path );
				ar.OpenOutArchive();
				ar.Save( getNetwork() );
				return true;
			}
			else if ( ext == ".bin" ) {
				using ArchiveOutType = boost::archive::binary_oarchive;
				SerializationArchive< ArchiveOutType > ar( file_path );
				ar.OpenOutArchive();
				ar.Save( getNetwork() );
				return true;
			}
			else if ( ext == ".xml" ) {
				std::cout << "Saving to xml format is not supported." << std::endl;
				return false;
			}
			else {
				std::cout << "Unrecognized file extension." << std::endl;
				return false;
			}
		}

		bool loadNetwork( std::string const& file_path ) {
			auto path = std::filesystem::path( file_path );
			std::string ext = path.extension().string();
			if ( ext == ".txt" ) {
				using ArchiveOutType = boost::archive::text_oarchive;
				SerializationArchive< ArchiveOutType > ar( file_path );
				ar.OpenInArchive();
				ar.Load( getNetwork() );
				return true;
			}
			else if ( ext == ".bin" ) {
				using ArchiveOutType = boost::archive::binary_oarchive;
				SerializationArchive< ArchiveOutType > ar( file_path );
				ar.OpenInArchive();
				ar.Load( getNetwork() );
				return true;
			}
			else if ( ext == ".xml" ) {
				std::cout << "Saving to xml format is not supported." << std::endl;
				return false;
			}
			else {
				std::cout << "Unrecognized file extension." << std::endl;
				return false;
			}
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NetworkType& mNetwork;
		OptimizerType& mOptimizer;
		LossFunction< NumericTraitsType, LossFunType > mLossFun;
		DataHandlerType& mDataHandler;
	}; // end of class NetworkTrainer


} // end NNet

#endif // NETWORK_TRAINER_HPP
