#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

// System includes --------------------
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "layer-base.hpp"

namespace NNet { // begin NNet

	template< typename NumericTraitsType >
	struct IdentityActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			outputVec = inputVec;
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			deltaOutputVec = deltaInputVec;
		}
	};

	template< typename NumericTraitsType >
	struct LogisticActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			auto fun = []( auto const& input_x ) { return (1.0)/( 1.0 + std::exp( -input_x ) ); };
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& output_i, auto const& deltaInput_i ) {
				return deltaInput_i * output_i * ( 1.0 - output_i );
			};
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}
	};

	template< typename NumericTraitsType >
	struct TanHActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), std::tanh );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& output_i, auto const& deltaInput_i ) {
				return deltaInput_i ( 1.0 - output_i * output_i );
			};
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}
	};

	template< typename NumericTraitsType >
	struct ArcTanActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), std::atan );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& input_i, auto const& deltaInput_i ) {
				return deltaInput_i * ( 1.0 / ( input_i * input_i + 1.0 ) );
			};
			std::transform( inputVec.begin( ), inputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}
	};

	template< typename NumericTraitsType >
	struct ReLUActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			auto fun = []( auto const& input_i ) { return ( input_i < 0.0 ) ? 0.0 : input_i; };
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& output_i, auto const& deltaInput_i ) { return ( output_i < 0.0 ) ? 0.0 : deltaInput_i; };
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}
	};

	template< typename NumericTraitsType >
	struct PReLUActivation {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		PReLUActivation( ) = default;
		explicit PReLUActivation( NumericType alpha ) : mAlpha( alpha ) {
			if ( mAlpha < 0.0 )
				throw std::runtime_error( "alpha parameter is less then zero, alpha = " << alpha );
		}
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			auto fun = [this]( auto const& input_i ) { return ( input_i < 0.0 ) ? mAlpha * input_i : input_i; };
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = [this]( auto const& output_i, auto const& deltaInput_i ) { return ( output_i < 0.0 ) ? mAlpha * deltaInput_i : deltaInput_i; };
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		NumericType mAlpha = 0.0;
	};

	template< typename NumericTraitsType >
	struct ELUActivation {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		ELUActivation( ) = default;
		explicit ELUActivation( NumericType alpha ) : mAlpha( alpha ) {
			if ( mAlpha < 0.0 )
				throw std::runtime_error( "alpha parameter is less then zero, alpha = " << alpha );
		}
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			auto fun = [this]( auto const& input_i ) { return ( input_i < 0.0 ) ? mAlpha * ( std::exp( input_i ) - 1.0 ) : input_i; };
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = [this]( auto const& output_i, auto const& deltaInput_i ) { return ( output_i < 0.0 ) ? deltaInput_i * ( output_i + mAlpha ) : deltaInput_i; };
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		NumericType mAlpha = 0.0;
	};

	template< typename NumericTraitsType >
	struct SoftMaxActivation {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		using MatrixXType = typename NumericTraitsType::MatrixXType;
		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			NumericType max = inputVec.maxCoeff( );
			NumericType sum = 0.0;
			auto fun = [&sum,max]( auto const& input_i ) {
				NumericType output_i = std::exp( input_i - max );
				sum += output_i;
			};
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
			std::transform( outputVec.begin( ), outputVec.end( ), outputVec.begin( ), [sum]( auto const& ele ) { return ele/sum; } );
		}
		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			deltaOutputVec = ( outputVec.matrix( ).asDiagonal( )  - outputVec * outputVec.transpose( ) ) * deltaInputVec;
		}
	};

	/**
	 *ActivationLayer.
	 */
	template< typename NumericTraitsType,
			  template < typename > class ActFun >
	class ActivationLayer
		: public LayerBase< NumericTraitsType > {
	public: 	// public typedefs
		using ActFunType = ActFun< NumericTraitsType >;
		using LayerBaseType = LayerBase< NumericTraitsType >;
		using NumericType = typename LayerBaseType::NumericType;
		using VectorXType = typename LayerBaseType::VectorXType;

	private: 	// private typedefs

	public: 	//public member functions
		ActivationLayer( ) = delete;
		explicit ActivationLayer( std::size_t numInputs, std::size_t numOutputs, ActFunType const& actFun )
			: LayerBase< NumericTraitsType >( numInputs, numOutputs ), mActFun( actFun ), mOutputVec( numOutputs ) {
		}
		ActivationLayer( const ActivationLayer &c ) = delete;
		~ActivationLayer( ) = default;

		// get/set member functions
		VectorXType& getOutputVec( ) override { return mOutputVec; }
		VectorXType const& getOutputVec( ) const override { return mOutputVec; }
		void setOutputVec( VectorXType const& outputVec ) { mOutputVec = outputVec; }
		VectorXType& getOutputDeltaVec( ) override { return mOutputDeltaVec; }
		VectorXType const& getOutputDeltaVec( ) const override { return mOutputDeltaVec; }


		// forward compute
		void forwardCompute( VectorXType const& inputVec, VectorXType& /* outputVec */ ) override {
			mActFun.forwardActivate( inputVec, getOutputVec( ) );
		}

		// backward compute
		void backwardCompute( VectorXType const& inputVec, VectorXType const& outputVec, VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) override {
			mActFun.backwardActivate( inputVec, outputVec, inputDeltaVec, outputDeltaVec );
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		ActFunType const& mActFun;
		VectorXType mOutputVec;
		VectorXType mOutputDeltaVec;
	}; // end of class ActivationLayer

} // end NNet

#endif // ACTIVATION_LAYER_HPP
