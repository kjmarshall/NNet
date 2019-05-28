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
			  template< typename > class LossFunType >
	class NetworkTrainer {
	public: 	// public typedefs
		using NumericTraitsType = typename NetworkType::NumericTraitsType;
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;

	private: 	// private typedefs

	public: 	//public member functions
		NetworkTrainer( ) = default;
		explicit NetworkTrainer( NetworkType& network, OptimizerType const& optimizer )
			: mNetwork( network ), mOptimizer( optimizer ) {
		}
		NetworkTrainer(const NetworkTrainer &c) = delete;
		~NetworkTrainer( ) = default;

		// accessors
		auto& getNetwork ( ) { return mNetwork; }
		auto const& getNetwork( ) const { return mNetwork; }
		auto const& getOptimizer( ) const { return mOptimizer; }
		auto const& getLossFun( ) const { return mLossFun; }

		// training
		// compute forward
		void computeForward( VectorXType const& inputVec ) {
			// run forward compute
			VectorXType inputWorkVec, outputWorkVec;
			if ( auto& firstLayer = getNetwork( ).getFirstLayer( ) ) {
				firstLayer -> forwardCompute( inputVec, outputWorkVec );
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
			if ( auto& lastLayer = getNetwork( ).getLastLayer( ) ) {
				lastLayer -> backwardCompute( dummyVec, dummyVec, inputDeltaWorkVec, outputDeltaWorkVec );
				inputDeltaWorkVec = outputDeltaWorkVec;
			}
			else {
				throw std::runtime_error( "Can't backward compute on last layer...");
			}
			for ( auto layerIter = getNetwork( ).rbegin( ) + 1; layerIter != getNetwork( ).rend( ); ++layerIter ) {
				auto& layerPtr = (*layerIter);
				layerPtr -> backwardCompute( dummyVec, dummyVec, inputDeltaWorkVec, outputDeltaWorkVec );
			}
		}
		void runSingleSample( VectorXType const& inputVec,
							  VectorXType const& targetVec ) {
			computeForward( inputVec );
			auto [loss, gradLoss] = computeLoss( getNetwork( ).getLastOutput( ), targetVec );
			computeBackward( gradLoss );
		}
		void trainSingleSample( VectorXType const& inputVec,
								VectorXType const& targetVec ) {
			getOptimizer( ).applyInterimUpdate( );
			runSingleSample( inputVec, targetVec );
			// update weights
			std::size_t batchSize = 1;
			getOptimizer( ).applyWeightUpdate( batchSize );
			// reset gradients
			getOptimizer( ).resetGradients( );
		}


	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NetworkType& mNetwork;
		OptimizerType const& mOptimizer;
		LossFunction< NumericTraitsType, LossFunType > mLossFun;
	}; // end of class NetworkTrainer


} // end NNet

#endif // NETWORK_TRAINER_HPP
