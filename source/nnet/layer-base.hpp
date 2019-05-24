#ifndef LAYER_BASE_HPP
#define LAYER_BASE_HPP

// System includes --------------------
#include <cstddef>

namespace NNet { // begin NNet

	/**
	 *LayerBase.
	 */
	template< typename NumericTraitsType >
	class LayerBase {
	public: 	// public typedefs
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		using MatrixXType = typename NumericTraitsType::MatrixXType;
	private: 	// private typedefs

	public: 	//public member functions
		LayerBase( ) = delete;
		explicit LayerBase( std::size_t numInputs, std::size_t numOutputs )
			: mNumInputs( numInputs ), mNumOutputs( numOutputs ) {
		}
		LayerBase( LayerBase const& layerBase ) = delete;
		virtual ~LayerBase( ) = default;

		// get/set member functions
		std::size_t getNumInputs( ) const { return mNumInputs; }
		void setNumInputs( std::size_t numInputs ) { mNumInputs = numInputs; }
		std::size_t getNumOutputs( ) { return mNumOutputs; }
		void setNumOutputs( std::size_t numOutputs ) { mNumOutputs = numOutputs; }
		virtual VectorXType& getOutputVec( ) = 0;
		virtual VectorXType const& getOutputVec( ) const = 0;
		virtual void setOutputVec( VectorXType const& outputVec ) = 0;
		virtual VectorXType& getOutputDeltaVec( ) = 0;
		virtual VectorXType const& getOutputDeltaVec( ) const = 0;
		// forward compute
		virtual void forwardCompute( VectorXType const& inputVec, VectorXType &outputVec ) = 0;
		virtual void backwardCompute( VectorXType const& inputVec, VectorXType const& outputVec, VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) = 0;

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		std::size_t mNumInputs, mNumOutputs;
	}; // end of class LayerBase

} // end NNet

#endif // LAYER_BASE_HPP
