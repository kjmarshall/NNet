#ifndef TRAINABLE_LAYER_HPP
#define TRAINABLE_LAYER_HPP

// System includes --------------------
#include <ostream>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "layer-base.hpp"

namespace NNet { // begin NNet

	enum class TrainableLayerType : unsigned char { INPUT, HIDDEN, OUTPUT, UNKNOWN };
	std::ostream& operator>>( std::ostream& os, TrainableLayerType layerType ) {
		switch ( layerType ) {
		case TrainableLayerType::INPUT: os << "input"; break;
		case TrainableLayerType::HIDDEN: os << "hidden"; break;
		case TrainableLayerType::OUTPUT: os << "output"; break;
		case TrainableLayerType::UNKNOWN: os << "unknown"; break;
		default:
			os << "unknown";
		}
	}
	/**
	 *TrainableLayer.
	 */
	template< typename NumericTraitsType >
	class TrainableLayer
		: public LayerBase< NumericTraitsType > {
	public: 	// public typedefs
		using LayerBaseType = LayerBase< NumericTraitsType >;
		using NumericType = typename LayerBaseType::NumericType;
		using VectorXType = typename LayerBaseType::VectorXType;
		using MatrixXType = typename LayerBaseType::MatrixXType;
	private: 	// private typedefs

	public: 	//public member functions
		TrainableLayer( ) = delete;
		explicit TrainableLayer( std::size_t numInputs, std::size_t numOutputs, TrainableLayerType layerType )
			: LayerBase< NumericTraitsType >( numInputs, numOutputs ), mLayerType( layerType ) {
		}
		TrainableLayer( const TrainableLayer &c ) = delete;
		~TrainableLayer( ) = default;

		TrainableLayerType getLayerType( ) const { return mLayerType; }
		virtual MatrixXType& getWeightMat( ) = 0;
		virtual MatrixXType const& getWeightMat( ) const = 0;
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		TrainableLayerType mLayerType;
	}; // end of class TrainableLayer


} // end NNet

#endif // TRAINABLE_LAYER_HPP
