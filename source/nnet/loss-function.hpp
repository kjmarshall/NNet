#ifndef LOSS_FUNCTION_HPP
#define LOSS_FUNCTION_HPP

// Own includes --------------------
#include "utils/numeric-traits.hpp"

namespace NNet { // begin NNet

	template< typename NumericTraitsType >
	struct MSELossFuction {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;

		// cost
		NumericType loss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return ( outputVec - targetVec ).squaredNorm( );
		}

		// gradLoss
		VectorXType gradLoss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return ( outputVec - targetVec );
		}

	};

	template< typename NumericTraitsType >
	struct BinaryCrossEntropyLossFunction {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;

		// cost
		NumericType loss( NumericType output, NumericType target ) {
			return -target * std::log( output ) - ( 1.0 - output ) * std::log( 1.0 - target );
		}

		// gradLoss
		VectorXType gradLoss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			VectorXType onesVec = VectorXType::Ones( outputVec.size( ) );
			return ( outputVec - targetVec ).cwiseQuotient( outputVec.cwiseProduct( onesVec - outputVec ) );
		}

	};

	template< typename NumericTraitsType >
	struct CrossEntropyLossFuction {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;

		// cost
		NumericType loss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return -targetVec.dot( outputVec.array( ).log( ) );
		}

		// gradLoss
		VectorXType gradLoss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return -targetVec.cwiseQuotient( outputVec );
		}

	};

	template< typename NumericTraitsType >
	struct LogCrossEntropyLossFuction {
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;

		// cost
		NumericType loss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return -targetVec.dot( outputVec );
		}

		// gradLoss
		VectorXType gradLoss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return -targetVec;
		}

	};

	/**
	 *LossFunction.
	 */
	template< typename NumericTraitsType,
			  template< typename > class LossFunType >
	class LossFunction {
	public: 	// public typedefs
		using NumericType = typename NumericTraitsType::NumericType;
		using VectorXType = typename NumericTraitsType::VectorXType;

	private: 	// private typedefs

	public: 	//public member functions
		LossFunction( ) = default;
		LossFunction(const LossFunction &c) = delete;
		~LossFunction( ) = default;

		// loss
		NumericType loss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return mLossFun.loss( outputVec, targetVec );
		}

		// gradLoss = \pderiv{ loss }{ \hat{y}_{i}^{p}}
		VectorXType gradLoss( VectorXType const& outputVec, VectorXType const& targetVec ) {
			return mLossFun.gradLoss( outputVec, targetVec );
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		LossFunType< NumericTraitsType > mLossFun;
	}; // end of class LossFunction


} // end NNet

#endif // LOSS_FUNCTION_HPP
