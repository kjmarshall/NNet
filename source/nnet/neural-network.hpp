#ifndef NNET_HPP
#define NNET_HPP

// System includes --------------------
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "layer-base.hpp"

namespace NNet { // begin NNet

	/**
	 *NeuralNetwork.
	 */
	template< typename NumericTraitsType >
	class NeuralNetwork {
	public: 	// public typedefs
		using LayerBaseType = LayerBase< NumericTraitsType >;
		using LayerBasePtrType = std::shared_ptr< LayerBaseType >;

	private: 	// private typedefs

	public: 	//public member functions
		NeuralNetwork( ) = default;
		NeuralNetwork( NeuralNetwork const& c ) = delete;
		~NeuralNetwork( ) = default;

		// get and set methods
		std::size_t getNumLayers( ) const { return mLayers.size( ); }
		auto& getLayers( ) { return mLayers; }
		auto const& getLayers( ) const { return mLayers; }
		auto& getLayer( std::size_t i ) { return mLayers[i]; };
		auto const& getLayer( std::size_t i ) const { return mLayers[i]; }

		void addLayer( LayerBasePtrType layerPtr ) {
			mLayers.emplace_back( layerPtr );
		}

		void printLayerInfo( std::ostream& os = std::cout ) {
			for ( auto const& layer : mLayers ) {
				layer -> printInfo( os );
			}
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		std::vector< LayerBasePtrType > mLayers;
	}; // end of class NeuralNetwork


} // end NNet

#endif // NNET_HPP
