#ifndef NNET_HPP
#define NNET_HPP

// System includes --------------------
#include <vector>

// Own includes --------------------
#include "utils/numeric-traits.hpp"

namespace NNet { // begin NNet

	template< typename DataType >
	struct Neuron {
		using NeuronType = Neuron< DataType >;
		using NumericTraits = NumericTraits< DataType >;
		using NumericType = typename NumericTraits::NumericType;
		using VectorXType = typename NumericTraits::VectorXType;

		Neuron( )
			: mActivation( 0 ) {
		}

		template< typename ActivationFun >
		NumericType ComputeActivation( VectorXType const& inputVec, VectorXType const& weights, ActivationFun&& actFun ) {
			assert( inputVec.size( ) == weights.size( ) );
			NumericType sum;
			sum = inputVec.dot( weights );
			mActivation = actFun( sum );
			return mActivation;
		}

		NumericType GetActivation( ) const { return mActivation; }
		void SetActivation( NumericType activation ) { mActivation = activation; }

		NumericType mActivation;
	};

	enum class LayerType : unsigned char { INPUT, HIDDEN, OUTPUT, UNKNOWN };
	template< typename DataType >
	struct NNLayer {
		using NeuronType = Neuron< DataType >;
		using NumericType = typename NeuronType::NumericType;
		using MatrixXType = typename NeuronType::NumericTraits::MatrixXType;
		using VectorXType = typename NeuronType::NumericTraits::VectorXType;

		NNLayer( ) = default;
		explicit NNLayer( std::size_t numNeurons, LayerType layerType );

		template< typename ActivationFun >
		VectorXType ComputeActivation( VectorXType const& inputVec, ActivationFun&& actFun ) {
			assert( inputVec.size( ) == mWeightMat.cols( ) );
			mActVec = mWeightMat * input;
			std::transform( mActVec.begin( ), mActVec.end( ), mActVec.begin( ), actFun );
			return mActVec;
		}

		auto begin( ) { return mNeurons.end( ); }
		auto const begin( ) const { return mNeurons.begin( ); }
		auto end( ) { return mNeurons.end( ); }
		auto const end( ) const { return mNeurons.end( ); }

		MatrixXType const& GetWeightMat( ) { return mWeightMat; }
		VectorXType GetBiasVec( ) { return mWeightMat.col( mWeightMat.cols( ) - 1 ); }
		VectorXType GetWeightRow( std::size_t i ) { return mWeightMat.row( i ); }
		VectorXType const& GetActVec( ) const { return mActVec; }
		VectorXType& GetActVec( ) { return mActVec; }
		LayerType const GetLayerType( ) { return mLayerType; }

		std::vector< NeuronType > mNeurons = {};
		// mWeightMat is assumed to hold the bias term in the last column
		MatrixXType mWeightMat;
		VectorXType mActVec;
		LayerType mLayerType = LayerType::UNKNOWN;
	};

	/**
	 *NNet.
	 */
	template< typename DataType >
	class NNet {
	public: 	// public typedefs
		using NNLayerType = NNLayer< DataType >;
		using NumericType = typename NNLayerType::NumericType;

	private: 	// private typedefs

	public: 	//public member functions
		NNet( ) = default;
		NNet( NNet const& c ) = delete;
		~NNet( ) = default;

		void ComputeNetworkWeights( ) {
			auto IdentityFun = []( auto const& x ) {
								   return x;
							   };
			if ( mLayers.empty( ) ) { return; }
			else if ( GetNumLayers( ) == 1 ) {
				auto& outputLayer = mLayers.front( );
				ouputLayer.ComputeActivation( GetInputVec( ), IdentityFun );
			}
			else {
				auto LogisticFun = []( auto const& x ) {
									   return 1.0 / ( 1.0 + std::exp( x ) );
								   };
				// loop over all layers
				for ( std::size_t i = 0; i < GetNumLayers( ); ++i ) {
					auto& layer = GetLayer( i );
					if ( i == 0 ) {
						// first hidden layer
						layer.ComputeActivation( GetInputVec( ), LogisticFun );
					}
					else if ( i < GetNumLayers( ) - 1 ) {
						// second, third, fourth... hidden layers
						auto const& prevLayer = GetLayer( i - 1 );
						layer.ComputeActivation( prevLayer.GetActVec( ), LogisticFun );
					}
					else {
						// output layer
						auto const& prevLayer = GetLayer( i - 1 );
						layer.ComputeActivation( prevLayer.GetActVec( ), IdentityFun );
					}
				}
			}

			// classify with softmax
			auto& outputVec = GetOutputLayer( ).GetActVec( );
			NumericType sum = 0;
			std::transform( outputVec.begin( ), outputVec.end( ), outputVec.begin( ),
							[&sum]( auto const& ele ) {
								auto output = ele;
								if ( ele < 300.0 )
									output = std::exp( output );
								else
									output = std::exp( 300.0 );
								sum += output;
								return output;
							} );
			std::transform( outputVec.begin( ), outputVec.end( ), outputVec.begin( ),
							[&sum]( auto const& ele ) {
								return ele/sum;
							} );
		}

		// get and set methods
		std::size_t GetNumLayers( ) const { return mLayers.size( ); }
		auto& GetInputVec( ) { return mInputVec; }
		auto const& GetInputVec( ) const { return mInputVec; }
		auto& GetLayers( ) { return mLayers; }
		auto const& GetLayers( ) const { return mLayers; }
		auto& GetLayer( std::size_t i ) { return mLayers[i]; };
		auto const& GetLayer( std::size_t ) const { return mLayers[i]; }
		auto const& GetOutputLayer( ) const { return GetLayer( GetNumLayers( ) - 1 ); }

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		std::vector< NNLayerType > mLayers;
		std::vector< NumericType > mInputVec;
	}; // end of class NNet


} // end NNet

#endif // NNET_HPP
