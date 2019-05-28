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
		using BaseLayerType = TrainableLayer< NumericTraitsType >;
		using NumericType = typename BaseLayerType::NumericType;
		using VectorXType = typename BaseLayerType::VectorXType;
		using MatrixXType = typename BaseLayerType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		FullyConnectedLayer( ) = delete;
		explicit FullyConnectedLayer( std::size_t numInputs, std::size_t numOutputs, LayerType layerType )
			: TrainableLayer< NumericTraitsType >( numInputs, numOutputs, layerType ), mWeightMat( numInputs + 1, numOutputs ), mWeightGradMat( numInputs + 1, numOutputs ), mInputVec( numInputs), mOutputVec( numOutputs ), mOutputDeltaVec( numInputs + 1 ) {
			this -> resetWeightGradMat( );
		}
		FullyConnectedLayer( const FullyConnectedLayer &c ) = delete;
		~FullyConnectedLayer( ) = default;

		// get/set member functions
		VectorXType& getOutputVec( ) override { return mOutputVec; }
		VectorXType const& getOutputVec( ) const override { return mOutputVec; }
		void setOutputVec( VectorXType const& outputVec ) { mOutputVec = outputVec; };
		VectorXType& getOutputDeltaVec( ) override { return mOutputDeltaVec; }
		VectorXType const& getOutputDeltaVec( ) const override { return mOutputDeltaVec; }
		MatrixXType& getWeightMat( ) override { return mWeightMat; }
		MatrixXType const& getWeightMat( ) const override { return mWeightMat; }

		MatrixXType& getWeightGradMat( ) override { return mWeightGradMat; }
		MatrixXType const& getWeightGradMat( ) const override { return mWeightGradMat; }

		void resetWeightGradMat( ) override {
			this -> getWeightGradMat( ).setZero( );
		}

		// self interface
		void setNumNeurons( std::size_t n ) {
			this -> setNumOutputs( n );
		}

		// identifiers and information
		void printLayerInfo( std::ostream& os = std::cout ) const override {
			os << "Layer Type: " << this -> getLayerType( ) << std::endl;
			os << "Inputs, Outputs: " << this -> getNumInputs( ) << ", " << this -> getNumOutputs( ) << std::endl;
			os << "Outputs size: " << getOutputVec( ).size( ) << std::endl;
			os << "Output delta size: " << getOutputDeltaVec( ).size( ) << std::endl;
			os << "Weight Matrix <rows=#inputs, cols=#outputs>: " << getWeightMat( ).rows( ) << ", " << getWeightMat( ).cols( ) << std::endl;
			os << "Weight Grad Matrix <rows=#inputs, cols=#outputs>: " << getWeightMat( ).rows( ) << ", " << getWeightMat( ).cols( ) << std::endl;
		}

		// forward compute
		void forwardCompute( VectorXType const& inputVec, VectorXType &outputVec ) override {
			mInputVec << inputVec, 1.0;
			outputVec = getWeightMat( ).transpose( ) * mInputVec;
			mOutputVec = outputVec;
		}

		// backward compute
		void backwardCompute( VectorXType const& inputVec, VectorXType const& outputVec, VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) override {
			mWeightGradMat += mInputVec * inputDeltaVec.transpose( );
			outputDeltaVec = getWeightMat( ) * inputDeltaVec;
			mOutputDeltaVec = outputDeltaVec;
		};

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		MatrixXType mWeightMat, mWeightGradMat;
		VectorXType mInputVec, mOutputVec;
		VectorXType mOutputDeltaVec;
	}; // end of class FullyConnectedLayer

} // end NNet

#endif // FULLY_CONNECTED_LAYER_HPP
