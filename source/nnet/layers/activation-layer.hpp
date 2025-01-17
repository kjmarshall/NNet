#ifndef ACTIVATION_LAYER_HPP
#define ACTIVATION_LAYER_HPP

// System includes --------------------
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "base-layer.hpp"
#include "serialization/serialize.hpp"

namespace NNet { // begin NNet

	template< typename NumericTraitsType >
	struct IdentityActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;

		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			outputVec = inputVec;
		}

		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& /* outputVec */,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			deltaOutputVec = deltaInputVec;
		}

		bool operator==( IdentityActivation const& /* other */ ) const {
			return true;
		}
	};

	template< typename NumericTraitsType >
	struct LogisticActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;

		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			auto fun = []( auto const& input_x ) { return (1.0)/( 1.0 + std::exp( -input_x ) ); };
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
		}

		void backwardActivate( VectorXType const& /* inputVec */,
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

		bool operator==( LogisticActivation const& /* other */ ) const {
			return true;
		}
	};

	template< typename NumericTraitsType >
	struct TanHActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;

		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ),
							[]( auto const& x ) { return std::tanh( x ); } );
		}

		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& output_i, auto const& deltaInput_i ) {
				return deltaInput_i * ( 1.0 - output_i * output_i );
			};
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		bool operator==( TanHActivation const& /* other */ ) const {
			return true;
		}
	};

	template< typename NumericTraitsType >
	struct ArcTanActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;

		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), []( auto const& x ) {
					return std::atan( x );
				} );
		}

		void backwardActivate( VectorXType const& inputVec,
							   VectorXType const& /* outputVec */,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& input_i, auto const& deltaInput_i ) {
				return deltaInput_i * ( 1.0 / ( input_i * input_i + 1.0 ) );
			};
			std::transform( inputVec.begin( ), inputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		bool operator==( ArcTanActivation const& /* other */ ) const {
			return true;
		}
	};

	template< typename NumericTraitsType >
	struct ReLUActivation {
		using VectorXType = typename NumericTraitsType::VectorXType;

		void forwardActivate( VectorXType const& inputVec, VectorXType& outputVec ) const {
			auto fun = []( auto const& input_i ) { return ( input_i < 0.0 ) ? 0.0 : input_i; };
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
		}

		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& output_i, auto const& deltaInput_i ) { return ( output_i < 0.0 ) ? 0.0 : deltaInput_i; };
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		bool operator==( ReLUActivation const& /* other */ ) const {
			return true;
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
		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = [this]( auto const& output_i, auto const& deltaInput_i ) { return ( output_i < 0.0 ) ? mAlpha * deltaInput_i : deltaInput_i; };
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		bool operator==( PReLUActivation const& other ) const {
			return ( mAlpha == other.mAlpha );
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

		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = [this]( auto const& output_i, auto const& deltaInput_i ) { return ( output_i < 0.0 ) ? deltaInput_i * ( output_i + mAlpha ) : deltaInput_i; };
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		bool operator==( ELUActivation const& other ) const {
			return ( mAlpha == other.mAlpha );
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
				return output_i;
			};
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), fun );
			std::transform( outputVec.begin( ), outputVec.end( ), outputVec.begin( ), [sum]( auto const& ele ) { return ele/sum; } );
		}

		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			MatrixXType diagMat = outputVec.asDiagonal( );
			deltaOutputVec = ( diagMat  - outputVec * outputVec.transpose( ) ) * deltaInputVec;
		}

		bool operator==( SoftMaxActivation const& /* other */ ) const {
			return true;
		}
	};

	template< typename NumericTraitsType >
	struct LogSoftMaxActivation {
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
			std::for_each( inputVec.begin( ), inputVec.end( ), fun );
			std::transform( inputVec.begin( ), inputVec.end( ), outputVec.begin( ), [sum,max]( auto const& ele ) { return ele - max - std::log(sum); } );
		}

		void backwardActivate( VectorXType const& /* inputVec */,
							   VectorXType const& outputVec,
							   VectorXType const& deltaInputVec,
							   VectorXType& deltaOutputVec ) const {
			auto derFun = []( auto const& output_i, auto const& deltaInput_i ) {
				return std::exp( output_i ) + deltaInput_i;
			};
			std::transform( outputVec.begin( ), outputVec.end( ),
							deltaInputVec.begin( ), deltaOutputVec.begin( ),
							derFun );
		}

		bool operator==( LogSoftMaxActivation const& /* other */ ) const {
			return true;
		}
	};

	/**
	 *ActivationLayer.
	 */
	template< typename NumericTraitsType,
			  template < typename > class ActFun >
	class ActivationLayer
		: public BaseLayer< NumericTraitsType > {
	public: 	// public typedefs
		using ActFunType = ActFun< NumericTraitsType >;
		using ActivationLayerType = ActivationLayer< NumericTraitsType, ActFun >;
		using BaseLayerType = BaseLayer< NumericTraitsType >;
		using NumericType = typename BaseLayerType::NumericType;
		using VectorXType = typename BaseLayerType::VectorXType;

	private: 	// private typedefs

	public: 	//public member functions
		ActivationLayer( ) = default;
		explicit ActivationLayer( std::size_t numInputs, std::size_t numOutputs )
			: BaseLayer< NumericTraitsType >( numInputs, numOutputs, LayerType::ACTIVATION ), mActFun(), mInputVec( numInputs ), mOutputVec( numOutputs ), mOutputDeltaVec( numInputs ) {
		}
		explicit ActivationLayer( std::size_t numInputs )
			: BaseLayer< NumericTraitsType >( numInputs, numInputs, LayerType::ACTIVATION ), mActFun(), mInputVec( numInputs ), mOutputVec( numInputs ), mOutputDeltaVec( numInputs ) {
		}
		explicit ActivationLayer( std::size_t numInputs, std::size_t numOutputs, VectorXType const& inputVec, VectorXType const& outputVec, VectorXType const& outputDeltaVec )
			: BaseLayer< NumericTraitsType >( numInputs, numOutputs, LayerType::ACTIVATION ), mActFun(), mInputVec( inputVec ), mOutputVec( outputVec ), mOutputDeltaVec( outputDeltaVec ) {
		}
		ActivationLayer( ActivationLayer const& other ) = default;
		ActivationLayer( ActivationLayer&& other ) = default;
		ActivationLayer& operator=( ActivationLayer const& rhs ) = default;
		ActivationLayer& operator=( ActivationLayer&& rhs ) = default;
		~ActivationLayer( ) = default;

		// get/set member functions
		VectorXType& getInputVec( )	{ return mInputVec; }
		VectorXType const& getInputVec( ) const { return mInputVec; }
		VectorXType& getOutputVec( ) override { return mOutputVec; }
		VectorXType const& getOutputVec( ) const override { return mOutputVec; }
		void setOutputVec( VectorXType const& outputVec ) { mOutputVec = outputVec; }
		VectorXType& getOutputDeltaVec( ) override { return mOutputDeltaVec; }
		VectorXType const& getOutputDeltaVec( ) const override { return mOutputDeltaVec; }

		// identifiers and information
		ActFunType& getActFun( ) { return mActFun; }
		ActFunType const& getActFun( ) const { return mActFun; }
		void printLayerInfo( std::ostream& os = std::cout ) const override {
			std::cout << "Layer Type: " << this -> getLayerType( ) << std::endl;
			os << "Inputs, Outputs: " << this -> getNumInputs( ) << ", " << this -> getNumOutputs( ) << std::endl;
			std::cout << "Output size: " << getOutputVec( ).size( ) << std::endl;
			std::cout << "Output delta size: " << getOutputDeltaVec( ).size( ) << std::endl;
		}
		bool isTrainableLayer( ) const override { return false; }

		// forward compute
		void forwardCompute( VectorXType const& inputVec, VectorXType& outputVec ) override {
			mInputVec = inputVec;
			mActFun.forwardActivate( inputVec, outputVec );
			mOutputVec = outputVec;
		}

		// backward compute
		void backwardCompute( VectorXType const& /* inputVec */, VectorXType const& /* outputVec */, VectorXType const& inputDeltaVec, VectorXType& outputDeltaVec ) override {
			mActFun.backwardActivate( mInputVec, mOutputVec, inputDeltaVec, mOutputDeltaVec );
			outputDeltaVec = mOutputDeltaVec;
		}

		bool operator==( ActivationLayer const& other ) const {
			return ( mActFun == other.getActFun() &&
					 mInputVec == other.getInputVec() &&
					 mOutputVec == other.getOutputVec() &&
					 mOutputDeltaVec == other.getOutputDeltaVec() );
		}

		virtual bool equalTo( BaseLayerType const& other ) const override {
			bool equals = false;
			if ( ActivationLayerType const* alo = dynamic_cast< ActivationLayerType const * >( &other ) ) {
				equals = operator==( *alo );
			}
			return equals;
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		ActFunType mActFun;
		VectorXType mInputVec, mOutputVec;
		VectorXType mOutputDeltaVec;
	}; // end of class ActivationLayer

} // end NNet

namespace boost::serialization { // begin boost::serialization
	template< typename ArchiveType, typename NumericTraitsType, template < typename > class ActFun >
	void serialize( ArchiveType &ar, NNet::ActivationLayer< NumericTraitsType, ActFun >& obj, unsigned const /* version */ ) {
		ar & boost::serialization::base_object< NNet::BaseLayer< NumericTraitsType > >( obj );
		ar & obj.getActFun();
		ar & obj.getInputVec();
		ar & obj.getOutputVec();
		ar & obj.getOutputDeltaVec();
	}

	template< typename ArchiveType, typename NumericTraitsType, template < typename > class ActFun >
	void save_construct_data( ArchiveType &ar, NNet::ActivationLayer< NumericTraitsType, ActFun > const* obj, unsigned const /* version */ ) {
		std::size_t numInputs, numOutputs;
		numInputs = obj->getNumInputs();
		numOutputs = obj->getNumOutputs();
		ar << numInputs;
		ar << numOutputs;

		auto const& inputVec = obj->getInputVec();
		auto const& outputVec = obj->getOutputVec();
		auto const& outputDeltaVec = obj->getOutputDeltaVec();
		ar << inputVec;
		ar << outputVec;
		ar << outputDeltaVec;
	}

	template< typename ArchiveType, typename NumericTraitsType, template < typename > class ActFun >
	void load_construct_data( ArchiveType &ar, NNet::ActivationLayer< NumericTraitsType, ActFun >* obj, unsigned const /* version */ ) {
		std::size_t numInputs, numOutputs;
		ar >> numInputs;
		ar >> numOutputs;

		typename NNet::ActivationLayer< NumericTraitsType, ActFun >::VectorXType inputVec, outputVec, outputDeltaVec;
		ar >> inputVec;
		ar >> outputVec;
		ar >> outputDeltaVec;

		::new( obj )NNet::ActivationLayer< NumericTraitsType, ActFun >( numInputs, numOutputs, inputVec, outputVec, outputDeltaVec );
	}

	// IdentityActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::IdentityActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// LogisticActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::LogisticActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// TanHActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::TanHActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// ArcTanActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::ArcTanActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// ReLUActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::ReLUActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// PReLUActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::PReLUActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// ELUActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::ELUActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// SoftMaxActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::SoftMaxActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

	// LogSoftMaxActivation< NumericTraitsType >
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType & /* ar */, NNet::LogSoftMaxActivation< NumericTraitsType >& /* obj */, unsigned const /* version */ ) {
	}

} // end boost::serialization

#endif // ACTIVATION_LAYER_HPP
