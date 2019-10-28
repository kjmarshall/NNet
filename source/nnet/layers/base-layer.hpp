#ifndef BASE_LAYER_HPP
#define BASE_LAYER_HPP

// System includes --------------------
#include <cstddef>
#include <iostream>

namespace NNet { // begin NNet

	enum class LayerType : unsigned char { INPUT, HIDDEN, ACTIVATION, OUTPUT, UNKNOWN };
	std::ostream& operator<<( std::ostream& os, LayerType layerType ) {
		switch ( layerType ) {
		case LayerType::INPUT: os << "input"; break;
		case LayerType::HIDDEN: os << "hidden"; break;
		case LayerType::ACTIVATION: os << "activation"; break;
		case LayerType::OUTPUT: os << "output"; break;
		case LayerType::UNKNOWN: os << "unknown"; break;
		default:
			os << "unknown";
		}
		return os;
	}

	/**
	 *BaseLayer.
	 */
	template< typename NumericTraitsType >
	class BaseLayer {
	public: 	// public typedefs
		using BaseLayerType = BaseLayer< NumericTraitsType >;
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		using MatrixXType = typename NumericTraitsType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		BaseLayer( ) = delete;
		explicit BaseLayer( std::size_t numInputs, std::size_t numOutputs, LayerType layerType )
			: mNumInputs( numInputs ), mNumOutputs( numOutputs ), mLayerType( layerType ) {
		}
		BaseLayer( BaseLayer const& layerBase ) = delete;
		virtual ~BaseLayer( ) = default;

		// get/set member functions
		std::size_t getNumInputs( ) const { return mNumInputs; }
		void setNumInputs( std::size_t numInputs ) { mNumInputs = numInputs; }
		std::size_t getNumOutputs( ) const { return mNumOutputs; }
		void setNumOutputs( std::size_t numOutputs ) { mNumOutputs = numOutputs; }
		virtual VectorXType& getOutputVec( ) = 0;
		virtual VectorXType const& getOutputVec( ) const = 0;
		virtual void setOutputVec( VectorXType const& outputVec ) = 0;
		virtual VectorXType& getOutputDeltaVec( ) = 0;
		virtual VectorXType const& getOutputDeltaVec( ) const = 0;
		LayerType getLayerType( ) const { return mLayerType; }

		// identifiers and information
		virtual void printLayerInfo( std::ostream& os = std::cout ) const = 0;
		virtual bool isTrainableLayer( ) const = 0;

		// forward compute
		virtual void forwardCompute( VectorXType const& inputVec, VectorXType &outputVec ) = 0;

		// backward compute
		virtual void backwardCompute( VectorXType const& inputVec, VectorXType const& outputVec,
									  VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) = 0;

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		std::size_t mNumInputs, mNumOutputs;
		LayerType mLayerType;
	}; // end of class BaseLayer

} // end NNet

namespace boost::serialization { // begin boost::serialization
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::BaseLayer< NumericTraitsType >& obj, unsigned const version ) {
		std::size_t numInputs = obj.getNumInputs();
		std::size_t numOutputs = obj.getNumOutputs();
		NNet::LayerType layerType = obj.getLayerType();
		ar & numInputs;
		ar & numOutputs;
		ar & layerType;
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void save_construct_data( ArchiveType &ar, NNet::BaseLayer< NumericTraitsType >* obj, unsigned const version ) {
		std::size_t numInputs = obj->getNumInputs();
		std::size_t numOutputs = obj->getNumOutputs();
		NNet::LayerType layerType = obj->getLayerType();
		ar << numInputs;
		ar << numOutputs;
		ar << layerType;
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void load_construct_data( ArchiveType &ar, NNet::BaseLayer< NumericTraitsType >* obj, unsigned const version ) {
		std::size_t numInputs;
		std::size_t numOutputs;
		NNet::LayerType layerType;
		ar >> numInputs;
		ar >> numOutputs;
		ar >> layerType;
		::new( obj )NNet::BaseLayer< NumericTraitsType >( numInputs, numOutputs, layerType );
	}
} // end boost::serialization

#endif // BASE_LAYER_HPP
