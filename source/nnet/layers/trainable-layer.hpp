#ifndef TRAINABLE_LAYER_HPP
#define TRAINABLE_LAYER_HPP

// System includes --------------------
#include <ostream>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "base-layer.hpp"
#include "serialization/serialize.hpp"

namespace NNet { // begin NNet

	/**
	 *TrainableLayer.
	 */
	template< typename NumericTraitsType >
	class TrainableLayer
		: public BaseLayer< NumericTraitsType > {
	public: 	// public typedefs
		using BaseLayerType = BaseLayer< NumericTraitsType >;
		using NumericType = typename BaseLayerType::NumericType;
		using VectorXType = typename BaseLayerType::VectorXType;
		using MatrixXType = typename BaseLayerType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		TrainableLayer( ) = delete;
		explicit TrainableLayer( std::size_t numInputs, std::size_t numOutputs, LayerType layerType )
			: BaseLayer< NumericTraitsType >( numInputs, numOutputs, layerType ) {
		}
		TrainableLayer( const TrainableLayer &c ) = delete;
		~TrainableLayer( ) = default;

		// get/set member functions
		virtual MatrixXType& getWeightMat( ) = 0;
		virtual MatrixXType const& getWeightMat( ) const = 0;
		virtual MatrixXType& getWeightGradMat( ) = 0;
		virtual MatrixXType const& getWeightGradMat( ) const = 0;
		virtual void resetWeightGradMat( ) = 0;

		bool isTrainableLayer( ) const override { return true; }

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members

	}; // end of class TrainableLayer

} // end NNet

namespace boost::serialization { // begin boost::serialization

	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::TrainableLayer< NumericTraitsType >& obj, unsigned const version ) {
		ar & boost::serialization::base_object< NNet::BaseLayer< NumericTraitsType > >( obj );
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void save_construct_data( ArchiveType &ar, NNet::TrainableLayer< NumericTraitsType > const* obj, unsigned const version ) {
		std::size_t numInputs = obj->getNumInputs();
		std::size_t numOutputs = obj->getNumOutputs();
		NNet::LayerType layerType = obj->getLayerType();
		ar << numInputs;
		ar << numOutputs;
		ar << layerType;
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void load_construct_data( ArchiveType &ar, NNet::TrainableLayer< NumericTraitsType >* obj, unsigned const version ) {
		std::size_t numInputs;
		std::size_t numOutputs;
		NNet::LayerType layerType;
		ar >> numInputs;
		ar >> numOutputs;
		ar >> layerType;
		::new( obj )NNet::TrainableLayer< NumericTraitsType >( numInputs, numOutputs, layerType );
	}

} // end boost::serialization

#endif // TRAINABLE_LAYER_HPP
