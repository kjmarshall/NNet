#ifndef BASE_INITIALIZER_HPP
#define BASE_INITIALIZER_HPP

// System includes --------------------
#include <random>
#include <algorithm>
#include <functional>

// Own includes --------------------
#include "utils/numeric-traits.hpp"

namespace NNet { // begin NNet
	/**
	 *BaseInitializer.
	 */
	template< typename NumericTraitsType >
	class BaseInitializer {
	public: 	// public typedefs
		using BaseInitializerType = BaseInitializer< NumericTraitsType >;
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;
		using MatrixXType = typename NumericTraitsType::MatrixXType;
		using RandomEngineType = std::mt19937;

	private: 	// private typedefs

	public: 	//public member functions
		BaseInitializer( )
			: mRandUniformDist( 0.0, 1.0 ), mRandNormalDist( 0.0, 1.0 ) {
			std::random_device rdev;
			std::seed_seq::result_type data[ RandomEngineType::state_size ];
			std::generate_n( data, RandomEngineType::state_size, std::ref( rdev ) );

			std::seed_seq prng_seed ( data, data + RandomEngineType::state_size );
			mRandGen.seed(prng_seed);
		}
		BaseInitializer( BaseInitializer const& other ) = default ;
		BaseInitializer( BaseInitializer&& other ) = default;
		BaseInitializer& operator=( BaseInitializer const& rhs ) = default;
		BaseInitializer& operator=( BaseInitializer&& rhs ) = default;
		virtual ~BaseInitializer( ) = default;

		// get/set member functions
		RandomEngineType& getRandomEngine( ) { return mRandGen; }

		NumericType random( ) {
			return mRandUniformDist( mRandGen );
		}
		NumericType randBernoulli( NumericType const input ) {
			if ( random( ) < input )
				return 1.0;
			else
				return 0.0;
		}

		int randomInt( int const hiEx ) {
			return int( std::floor( NumericType( hiEx ) * random( ) ) );
		}
		int randomInt( int const low, int const hiEx ) {
			return low + int( std::floor( double( hiEx - low ) * random( ) ) );
		}
		NumericType randNormal( ) {
			return mRandNormalDist( mRandGen );
		}
		NumericType randNormal(  NumericType const mean, NumericType const stdDev ) {
			return stdDev * randNormal( ) + mean;
		}
		virtual void initWeightMat( MatrixXType& weightMat, std::size_t fanIns, std::size_t fanOuts ) = 0;

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		RandomEngineType mRandGen;
		mutable std::uniform_real_distribution< > mRandUniformDist;
		mutable std::normal_distribution< > mRandNormalDist;
	}; // end of class BaseInitializer

} // end NNet

namespace boost::serialization { // begin boost::serialization
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::BaseInitializer< NumericTraitsType > &obj, unsigned const version ) {
	}
} // end boost::serialization

#endif // BASE_INITIALIZER_HPP
