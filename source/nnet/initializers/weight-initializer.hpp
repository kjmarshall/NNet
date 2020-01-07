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
		GaussInitializer( GaussInitializer const& other ) = default;
		GaussInitializer( GaussInitializer&& other ) = default;
		GaussInitializer& operator=( GaussInitializer const& rhs ) = default;
		GaussInitializer& operator=( GaussInitializer&& rhs ) = default;
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
		GlorotInitializer( GlorotInitializer const& other ) = default;
		GlorotInitializer( GlorotInitializer&& other ) = default;
		GlorotInitializer& operator=( GlorotInitializer const& rhs ) = default;
		GlorotInitializer& operator=( GlorotInitializer&& rhs ) = default;
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
		HeInitializer( HeInitializer&& other ) = default;
		HeInitializer& operator=( HeInitializer const& rhs ) = default;
		HeInitializer& operator=( HeInitializer&& rhs ) = default;
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

namespace boost::serialization { // begin boost::serialization
	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::GaussInitializer< NumericTraitsType > &obj, unsigned const version ) {
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::GlorotInitializer< NumericTraitsType > &obj, unsigned const version ) {
	}

	template< typename ArchiveType, typename NumericTraitsType >
	void serialize( ArchiveType &ar, NNet::HeInitializer< NumericTraitsType > &obj, unsigned const version ) {
	}
} // end boost::serialization

#endif // WEIGHT_INITIALIZER_HPP
