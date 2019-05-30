#ifndef NETWORK_TRAINER_HPP
#define NETWORK_TRAINER_HPP

// Own includes --------------------
#include "nnet/loss-function.hpp"

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
			std::size_t batchCtr = 0;
			std::size_t sampleCtr = 0;
			mDataHandler.shuffleTrainingData( getNetwork( ).getInitializer( ).getRandomEngine( ) );
			auto& data = mDataHandler.getTrainingData( );
			getOptimizer( ).applyInterimUpdate( );

			for_each_batch( data.begin( ), data.end( ), batchSize,
							[&,this]( auto& iterFrom, auto& iterTo ) {
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

								batchCtr++;
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
