#ifndef WEIGHT_INITIALIZER_HPP
#define WEIGHT_INITIALIZER_HPP

// Own includes --------------------
#include "base-initializer.hpp"

namespace NNet { // begin NNet

	/**
	 *GaussInitializer.
	 */
	template< typename NumericTraitsType >
	class GaussInitializer
		: public BaseInitializer< NumericTraitsType > {
	public: 	// public typedefs
		using BaseInitializerType = BaseInitializer< NumericTraitsType >;
		using NumericType = typename BaseInitializerType::NumericType;
		using VectorXType = typename BaseInitializerType::VectorXType;
		using MatrixXType = typename BaseInitializerType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		GaussInitializer( )
			: BaseInitializer< NumericTraitsType >( ) {
		}
		GaussInitializer( GaussInitializer const& c ) = default;
		~GaussInitializer( ) = default;

		void initWeightMat( MatrixXType& weightMat, std::size_t /* fanIns */, std::size_t /* fanOuts */ ) override {
			weightMat = weightMat.unaryExpr( [this]( auto const& dummy ) {
					return this -> randNormal( );
				} );
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members

	}; // end of class GaussInitializer

	/**
	 *GlorotInitializer.
	 */
	template< typename NumericTraitsType >
	class GlorotInitializer
		: public BaseInitializer< NumericTraitsType >{
	public: 	// public typedefs
		using BaseInitializerType = BaseInitializer< NumericTraitsType >;
		using NumericType = typename BaseInitializerType::NumericType;
		using VectorXType = typename BaseInitializerType::VectorXType;
		using MatrixXType = typename BaseInitializerType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		GlorotInitializer( )
			: BaseInitializer< NumericTraitsType >( ) {
		}
		GlorotInitializer( GlorotInitializer const& c ) = default;
		~GlorotInitializer( ) = default;

		void initWeightMat( MatrixXType& weightMat, std::size_t fanIns, std::size_t fanOuts ) override {
			NumericType stdDev = 2.0 / ( fanIns + fanOuts );
			weightMat = weightMat.unaryExpr( [&,this]( auto const& dummy ) {
					return this -> randNormal( 0.0, stdDev );
				} );
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members

	}; // end of class GlorotInitializer

	/**
	*HeInitializer.
	*/
	template< typename NumericTraitsType >
	class HeInitializer
		: public BaseInitializer< NumericTraitsType >{
	public: 	// public typedefs
		using BaseInitializerType = BaseInitializer< NumericTraitsType >;
		using NumericType = typename BaseInitializerType::NumericType;
		using VectorXType = typename BaseInitializerType::VectorXType;
		using MatrixXType = typename BaseInitializerType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		HeInitializer( )
			: BaseInitializer< NumericTraitsType >( ) {
		}
		HeInitializer( HeInitializer const& c ) = default;
		~HeInitializer( ) = default;

		void initWeightMat( MatrixXType& weightMat, std::size_t fanIns, std::size_t /* fanOuts */ ) override {
			NumericType stdDev = 2.0 / ( fanIns );
			weightMat = weightMat.unaryExpr( [&,this]( auto const& dummy ) {
					return this -> randNormal( 0.0, stdDev );
				} );
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members

	}; // end of class HeInitializer

} // end NNet

#endif // WEIGHT_INITIALIZER_HPP
