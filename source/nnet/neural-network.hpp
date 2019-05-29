#ifndef NNET_HPP
#define NNET_HPP

// System includes --------------------
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include <optional>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "base-layer.hpp"

namespace NNet { // begin NNet

	/**
	 *NeuralNetwork.
	 */
	template< typename NumericTraitsType_,
			  typename InitializerType >
	class NeuralNetwork {
	public: 	// public typedefs
		using NumericTraitsType = NumericTraitsType_;
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		using MatrixXType = typename NumericTraitsType::MatrixXType;
		using BaseLayerType = BaseLayer< NumericTraitsType >;
		using TrainableLayerType = TrainableLayer< NumericTraitsType >;
		using BaseLayerPtrType = std::shared_ptr< BaseLayerType >;

	private: 	// private typedefs

	public: 	//public member functions
		NeuralNetwork( ) = delete;
		explicit NeuralNetwork( InitializerType& initializer )
			: mInitializer( initializer ) {
		}
		NeuralNetwork( NeuralNetwork const& c ) = delete;
		~NeuralNetwork( ) = default;

		// begin/ end iterators for range based loops
		auto begin( ) { return getLayers( ).begin( ); }
		auto const begin( ) const { return getLayers( ).begin( ); }
		auto rbegin( ) { return getLayers( ).rbegin( ); }
		auto const rbegin( ) const { return getLayers( ).rbegin( ); }
		auto end( ) { return getLayers( ).end( ); }
		auto const end( ) const { return getLayers.end( ); }
		auto rend( ) { return getLayers( ).rend( ); }
		auto const rend( ) const { return getLayers.rend( ); }

		// get and set methods
		auto& getInitializer( ) { return mInitializer; }
		std::size_t getNumLayers( ) const { return mLayers.size( ); }
		auto& getLayers( ) { return mLayers; }
		auto const& getLayers( ) const { return mLayers; }
		auto& getLayer( std::size_t i ) { return mLayers[i]; };
		auto const& getLayer( std::size_t i ) const { return mLayers[i]; }
		auto getFirstLayer( ) {
			return !getLayers( ).empty( ) ? std::make_optional( getLayers( ).front( ) )
				: std::nullopt; }
		auto getLastLayer( ) {
			return !getLayers( ).empty( ) ? std::make_optional( getLayers( ).back( ) )
				: std::nullopt; }
		auto const& getLastOutput( ) {
			if ( auto lastLayer = getLastLayer( ) )
				return (*lastLayer) -> getOutputVec( );
			else
				throw std::runtime_error( "Can't get last output vec..." );
		}

		void addLayer( BaseLayerPtrType layerPtr ) {
			if ( getLayers( ).empty( ) ) {
				mLayers.emplace_back( layerPtr );
			}
			else {
				if ( auto lastLayer = getLastLayer( ) ) {
					auto numOutputs = (*lastLayer) -> getNumOutputs( );
					if ( numOutputs == layerPtr -> getNumInputs( ) ) {
						mLayers.emplace_back( layerPtr );
					}
					else {
						std::cerr << "Warning: layer" << " has an incorrect number of inputs (" << layerPtr -> getNumInputs( ) <<  "), expected: " << numOutputs << std::endl;
						layerPtr -> setNumInputs( numOutputs );
						mLayers.emplace_back( layerPtr );
					}
				}
			}
		}

		void finalize( ) {
			for ( auto& layer : getLayers( ) ) {
				if ( layer -> isTrainableLayer( ) ) {
					auto trainbleLayerPtr = std::static_pointer_cast< TrainableLayerType >( layer );
					auto& weightMat = trainbleLayerPtr -> getWeightMat( );
					auto fanIns = layer -> getNumInputs( );
					auto fanOuts = layer -> getNumOutputs( );
					mInitializer.initWeightMat( weightMat, fanIns, fanOuts );
				}
			}
		}

		void printNetworkInfo( std::ostream& os = std::cout ) {
			for ( auto const& layer : mLayers ) {
				layer -> printLayerInfo( os );
			}
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		std::vector< BaseLayerPtrType > mLayers;
		InitializerType& mInitializer;
	}; // end of class NeuralNetwork


} // end NNet

#endif // NNET_HPP
