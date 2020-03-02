#ifndef FULLY_CONNECTED_LAYER_HPP
#define FULLY_CONNECTED_LAYER_HPP

// System includes --------------------
#include <iostream>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "trainable-layer.hpp"
#include "serialization/serialize.hpp"

namespace NNet { // begin NNet

	/**
	 *FullyConnectedLayer.
	 */
	template< typename NumericTraitsType >
	class FullyConnectedLayer
		: public TrainableLayer< NumericTraitsType > {
	public: 	// public typedefs
		using TrainableLayerType = TrainableLayer< NumericTraitsType >;
		using BaseLayerType = typename TrainableLayerType::BaseLayerType;
		using NumericType = typename BaseLayerType::NumericType;
		using VectorXType = typename BaseLayerType::VectorXType;
		using MatrixXType = typename BaseLayerType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		FullyConnectedLayer( ) = delete;
		explicit FullyConnectedLayer( std::size_t numInputs, std::size_t numOutputs, LayerType layerType )
			: TrainableLayer< NumericTraitsType >( numInputs, numOutputs, layerType ),
			mWeightMat( numInputs + 1, numOutputs ),
			mWeightGradMat( numInputs + 1, numOutputs ),
			mInputVec( numInputs + 1 ),
			mOutputVec( numOutputs ),
			mOutputDeltaVec( numInputs + 1 ) {
			this -> resetWeightGradMat( );
		}
		explicit FullyConnectedLayer( std::size_t numInputs, std::size_t numOutputs, LayerType layerType, MatrixXType const& weightMat, MatrixXType const& weightGradMat, VectorXType const& inputVec, VectorXType const& outputVec, VectorXType const& outputDeltaVec )
			: TrainableLayer< NumericTraitsType >( numInputs, numOutputs, layerType ),
			mWeightMat( weightMat ),
			mWeightGradMat( weightGradMat ),
			mInputVec( inputVec ),
			mOutputVec( outputVec ),
			mOutputDeltaVec( outputDeltaVec ) {
		}

		FullyConnectedLayer( const FullyConnectedLayer &c ) = delete;
		~FullyConnectedLayer( ) = default;

		// get/set member functions
		VectorXType& getInputVec( ) { return mInputVec; }
		VectorXType const& getInputVec( ) const { return mInputVec; }
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
		void backwardCompute( VectorXType const& /* inputVec */, VectorXType const& /* outputVec */, VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) override {
			// std::cout << "Backward Compute: " << std::endl;
			// std::cout << "mInputVec: " << std::endl
			// 		  << mInputVec << std::endl;
			// std::cout << "inputDeltaVec: " << std::endl
			// 		  << inputDeltaVec.transpose( ) << std::endl;
			mWeightGradMat += mInputVec * inputDeltaVec.transpose( );
			// std::cout << "WeightGradMat: " << std::endl
			// 		  << mWeightGradMat << std::endl;
			outputDeltaVec = getWeightMat( ) * inputDeltaVec;
			auto rows = outputDeltaVec.rows( );
			outputDeltaVec = outputDeltaVec.segment( 0, rows - 1 );
			mOutputDeltaVec = outputDeltaVec;
		};

		bool operator==( FullyConnectedLayer const& other ) const {
			return ( mWeightMat == other.getWeightMat() &&
					 mWeightGradMat == other.getWeightGradMat() &&
					 mInputVec == other.getInputVec() &&
					 mOutputVec == other.getOutputVec() &&
					 mOutputDeltaVec == other.getOutputDeltaVec() );
		}

		virtual bool equalTo( BaseLayerType const& other ) const override {
			bool equals = false;
			if ( FullyConnectedLayer const* fco = dynamic_cast< FullyConnectedLayer const * >( &other ) ) {
				equals = operator==( *fco );
			}
			return equals;
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		MatrixXType mWeightMat, mWeightGradMat;
		VectorXType mInputVec, mOutputVec;
		VectorXType mOutputDeltaVec;
	}; // end of class FullyConnectedLayer

} // end NNet

namespace boost::serialization { // begin boost::serialization
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::FullyConnectedLayer< NumericTraitsType >& obj, unsigned const /* version */ ) {
		ar & boost::serialization::base_object< NNet::TrainableLayer< NumericTraitsType > >( obj );
		ar & obj.getWeightMat();
		ar & obj.getWeightGradMat();
		ar & obj.getInputVec();
		ar & obj.getOutputVec();
		ar & obj.getOutputDeltaVec();
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void save_construct_data( ArchiveType &ar, NNet::FullyConnectedLayer< NumericTraitsType > const* obj, unsigned const /* version */ ) {
		std::size_t numInputs, numOutputs;
		NNet::LayerType layer_type;
		numInputs = obj->getNumInputs();
		numOutputs = obj->getNumOutputs();
		layer_type = obj->getLayerType();
		ar << numInputs;
		ar << numOutputs;
		ar << layer_type;

		auto const& weightMat = obj->getWeightMat();
		auto const& weightGradMat = obj->getWeightGradMat();
		auto const& inputVec = obj->getInputVec();
		auto const& outputVec = obj->getOutputVec();
		auto const& outputDeltaVec = obj->getOutputDeltaVec();
		ar << weightMat;
		ar << weightGradMat;
		ar << inputVec;
		ar << outputVec;
		ar << outputDeltaVec;
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void load_construct_data( ArchiveType &ar, NNet::FullyConnectedLayer< NumericTraitsType >* obj, unsigned const /* version */ ) {
		std::size_t numInputs, numOutputs;
		NNet::LayerType layer_type;
		ar >> numInputs;
		ar >> numOutputs;
		ar >> layer_type;

		typename NNet::FullyConnectedLayer< NumericTraitsType >::MatrixXType weightMat, weightGradMat;
		typename NNet::FullyConnectedLayer< NumericTraitsType >::VectorXType inputVec, outputVec, outputDeltaVec;
		ar >> weightMat;
		ar >> weightGradMat;
		ar >> inputVec;
		ar >> outputVec;
		ar >> outputDeltaVec;

		::new( obj )NNet::FullyConnectedLayer< NumericTraitsType >( numInputs, numOutputs, layer_type, weightMat, weightGradMat, inputVec, outputVec, outputDeltaVec );
	}

} // end boost::serialization

#endif // FULLY_CONNECTED_LAYER_HPP
