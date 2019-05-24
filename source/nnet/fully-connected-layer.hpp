#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

// System includes --------------------
#include <iostream>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "trainable-layer.hpp"

namespace NNet { // begin NNet

	/**
	 *FullyConnectedLayer.
	 */
	template< typename NumericTraitsType >
	class FullyConnectedLayer
		: public TrainableLayer< NumericTraitsType > {
	public: 	// public typedefs
		using LayerBaseType = TrainableLayer< NumericTraitsType >;
		using NumericType = typename LayerBaseType::NumericType;
		using VectorXType = typename LayerBaseType::VectorXType;
		using MatrixXType = typename LayerBaseType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		FullyConnectedLayer( ) = delete;
		explicit FullyConnectedLayer( std::size_t numInputs, std::size_t numOutputs, TrainableLayerType layerType )
			: TrainableLayer< NumericTraitsType >( numInputs, numOutputs, layerType ) {
		}
		FullyConnectedLayer( const FullyConnectedLayer &c ) = delete;
		~FullyConnectedLayer( ) = default;

		VectorXType& getOutputVec( ) override { return mOutputVec; }
		VectorXType const& getOutputVec( ) const override { return mOutputVec; }
		void setOutputVec( VectorXType const& outputVec ) { mOutputVec = outputVec; };
		VectorXType& getOutputDeltaVec( ) { return mOutputDeltaVec; }
		VectorXType const& getOutputDeltaVec( ) const { return mOutputDeltaVec; }
		MatrixXType& getWeightMat( ) { return mWeightMat; }
		MatrixXType const& getWeightMat( ) const { return mWeightMat; }

		// forward compute
		void forwardCompute( VectorXType const& inputVec, VectorXType &outputVec ) {
			outputVec = getWeightMat( ).transpose( ) * inputVec;
		}

		// backward compute
		void backwardCompute( VectorXType const& inputVec, VectorXType const& outputVec, VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) {
			mWeightGradMat = inputDeltaVec * inputVec.transpose( );
			outputDeltaVec = getWeightMat( ) * inputDeltaVec;
		};

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		MatrixXType mWeightMat, mWeightGradMat;
		VectorXType mOutputVec;
		VectorXType mOutputDeltaVec;
	}; // end of class FullyConnectedLayer

} // end NNet

#endif // FULLY_CONNECTED_LAYER_HPP
