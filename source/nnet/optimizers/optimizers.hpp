#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

// Eigen includes --------------------
#include <Eigen/Dense>

// Own includes --------------------
#include "optimizers/base-optimizer.hpp"

namespace NNet { // begin NNet

	/**
	 *SGDOptimizer.
	 */
	template< typename NetworkType >
	class SGDOptimizer
		: public BaseOptimizer< NetworkType > {
	public: 	// public typedefs
		using NumericType = typename NetworkType::NumericType;

	private: 	// private typedefs

	public: 	//public member functions
		SGDOptimizer( ) = delete;
		explicit SGDOptimizer( NetworkType& network, NumericType learningRate = 0.001 )
			: BaseOptimizer< NetworkType >( network ), mLearningRate( learningRate ) {
		}
		SGDOptimizer( const SGDOptimizer &c ) = delete;
		~SGDOptimizer( ) = default;

		//get/set member functions
		NumericType getLearningRate( ) { return mLearningRate; }

		// interface
		void applyWeightUpdate( std::size_t batchSize ) override {
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto const& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				// std::cout << "LearningRate, coeff: " << mLearningRate << ", " << coeff << std::endl;
				// std::cout << "WeightMat Before: " << weightMat.rows( ) << ", " << weightMat.cols( ) << std::endl
				// 		  << weightMat << std::endl;
				// std::cout << "WeightGradMat Before: " << std::endl
				// 		  << weightGradMat << std::endl;
				weightMat = weightMat - mLearningRate * coeff * weightGradMat;
			}
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate;

	}; // end of class SGDOptimizer

    /**
	 *MomentumOptimizer.
	 */
	template< typename NetworkType >
	class MomentumOptimizer
		: public BaseOptimizer< NetworkType > {
	public: 	// public typedefs
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;
		using MatrixXType = typename NetworkType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		MomentumOptimizer( ) = delete;
		explicit MomentumOptimizer( NetworkType& network, NumericType learningRate = 0.001, NumericType momentum = 0.9 )
			: BaseOptimizer< NetworkType >( network ), mLearningRate( learningRate ), mMomentum( momentum ) {
			for ( auto const& layer : this -> getTrainableLayers( ) ) {
				auto const& weightGradMat = layer -> getWeightGradMat( );
				auto numRows = weightGradMat.rows( );
				auto numCols = weightGradMat.cols( );
				MatrixXType mat( numRows, numCols );
				mat.setZero( );
				mWeightGradMatSaves.emplace_back( mat );
			}
		}
		MomentumOptimizer( const MomentumOptimizer &c ) = delete;
		~MomentumOptimizer( ) = default;

		//get/set member functions
		NumericType getLearningRate( ) const { return mLearningRate; }
		void setLearningRate( NumericType learningRate ) { mLearningRate = learningRate; }
		NumericType getMomentum( ) const { return mMomentum; }
		void setMomentum( NumericType momentum ) { mMomentum = momentum; }

		// interface
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto v_iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto const& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				auto v = mMomentum * (*v_iter) + mLearningRate * coeff * weightGradMat;
				*v_iter = v;
				weightMat = weightMat - v;
				++v_iter;
			}
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate, mMomentum;
		std::vector< MatrixXType > mWeightGradMatSaves = { };
	}; // end of class MomentumOptimizer

	/**
	 *NesterovMomentumOptimizer.
	 */
	template< typename NetworkType >
	class NesterovMomentumOptimizer
		: public BaseOptimizer< NetworkType > {
	public: 	// public typedefs
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;
		using MatrixXType = typename NetworkType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		NesterovMomentumOptimizer( ) = delete;
		explicit NesterovMomentumOptimizer( NetworkType& network, NumericType learningRate = 0.001, NumericType momentum = 0.9 )
			: BaseOptimizer< NetworkType >( network ), mLearningRate( learningRate ), mMomentum( momentum ) {
			for ( auto const& layer : this -> getTrainableLayers( ) ) {
				auto const& weightGradMat = layer -> getWeightGradMat( );
				auto numRows = weightGradMat.rows( );
				auto numCols = weightGradMat.cols( );
				MatrixXType mat( numRows, numCols );
				mat.setZero( );
				mWeightGradMatSaves.emplace_back( mat );
			}
		}
		NesterovMomentumOptimizer( const NesterovMomentumOptimizer &c ) = delete;
		~NesterovMomentumOptimizer( ) = default;

		//get/set member functions
		NumericType getLearningRate( ) { return mLearningRate; }
		void setLearningRate( NumericType learningRate ) { mLearningRate = learningRate; }
		NumericType getMomentum( ) const { return mMomentum; }
		void setMomentum( NumericType momentum ) { mMomentum = momentum; }

		// interface
		void applyInterimUpdate( ) override {
			auto v_iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				weightMat = weightMat - mMomentum * ( *v_iter );
				++v_iter;
			}
		}
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto v_iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto const& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				auto v = mMomentum * (*v_iter) + mLearningRate * coeff * weightGradMat;
				*v_iter = v;
				weightMat = weightMat - v;
				++v_iter;
			}
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate, mMomentum;
		std::vector< MatrixXType > mWeightGradMatSaves = { };
	}; // end of class NesterovMomentumOptimizer

	/**
	 *AdaGradOptimizer.
	 */
	template< typename NetworkType >
	class AdaGradOptimizer
		: public BaseOptimizer< NetworkType > {
	public: 	// public typedefs
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;
		using MatrixXType = typename NetworkType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		AdaGradOptimizer() = default;
		explicit AdaGradOptimizer( NetworkType& network, NumericType learningRate = 0.001 )
			: BaseOptimizer< NetworkType >( network ), mLearningRate( learningRate ) {
			for ( auto const& layer : this -> getTrainableLayers( ) ) {
				auto const& weightGradMat = layer -> getWeightGradMat( );
				auto numRows = weightGradMat.rows( );
				auto numCols = weightGradMat.cols( );
				MatrixXType mat( numRows, numCols );
				mat.setZero( );
				mGradMatAccumSaves.emplace_back( mat );
			}
		}
		AdaGradOptimizer(const AdaGradOptimizer &c) = delete;
		~AdaGradOptimizer() = default;

		//get/set member functions
		NumericType getLearningRate( ) const { return mLearningRate; }
		void setLearningRate( NumericType learningRate ) { mLearningRate = learningRate; }

		// interface
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto r_iter = mGradMatAccumSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto const& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				// accumulate squared gradient
				auto grad_sq = coeff * coeff * weightGradMat.cwiseProduct( weightGradMat );
				*r_iter = (*r_iter) + grad_sq;
				auto r = *r_iter;
				r = r.unaryExpr( [this]( auto const& ele ) {
					return ( mLearningRate / ( 1.0e-7 + std::sqrt( ele ) ) );
				} );
				weightMat = weightMat - r.cwiseProduct( coeff * weightGradMat );
				++r_iter;
			}
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate;
		std::vector< MatrixXType > mGradMatAccumSaves = { };
	}; // end of class AdaGradOptimizer

	/**
	 *RMSPropOptimizer.
	 */
	template< typename NetworkType >
	class RMSPropOptimizer
		: public BaseOptimizer< NetworkType > {
	public: 	// public typedefs
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;
		using MatrixXType = typename NetworkType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		RMSPropOptimizer() = default;
		explicit RMSPropOptimizer( NetworkType& network, NumericType learningRate = 0.001, NumericType decayRate = 0.9 )
			: BaseOptimizer< NetworkType >( network ), mLearningRate( learningRate ), mDecayRate( decayRate ) {
			for ( auto const& layer : this -> getTrainableLayers( ) ) {
				auto const& weightGradMat = layer -> getWeightGradMat( );
				auto numRows = weightGradMat.rows( );
				auto numCols = weightGradMat.cols( );
				MatrixXType mat( numRows, numCols );
				mat.setZero( );
				mGradMatAccumSaves.emplace_back( mat );
			}
		}
		RMSPropOptimizer(const RMSPropOptimizer &c) = delete;
		~RMSPropOptimizer() = default;

		//get/set member functions
		NumericType getLearningRate( ) const { return mLearningRate; }
		void setLearningRate( NumericType learningRate ) { mLearningRate = learningRate; }
		NumericType getDecayRate( ) const { return mDecayRate; }
		void setDecayRate( NumericType decayRate ) { mDecayRate = decayRate; }

		// interface
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto r_iter = mGradMatAccumSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto const& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				// accumulate squared gradient
				auto grad_sq = coeff * coeff * weightGradMat.cwiseProduct( weightGradMat );
				*r_iter = mDecayRate * (*r_iter) + (1.0 - mDecayRate ) * grad_sq;
				auto r = *r_iter;
				r = r.unaryExpr( [this]( auto const& ele ) {
					return ( mLearningRate / ( 1.0e-7 + std::sqrt( ele ) ) );
				} );
				weightMat = weightMat - r.cwiseProduct( coeff * weightGradMat );
				++r_iter;
			}
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate, mDecayRate;
		std::vector< MatrixXType > mGradMatAccumSaves = { };
	}; // end of class RMSPropOptimizer

	/**
	 *RMSPropNestMomOptimizer.
	 */
	template< typename NetworkType >
	class RMSPropNestMomOptimizer
		: public BaseOptimizer< NetworkType > {
	public: 	// public typedefs
		using NumericType = typename NetworkType::NumericType;
		using VectorXType = typename NetworkType::VectorXType;
		using MatrixXType = typename NetworkType::MatrixXType;

	private: 	// private typedefs

	public: 	//public member functions
		RMSPropNestMomOptimizer() = default;
		explicit RMSPropNestMomOptimizer( NetworkType& network, NumericType learningRate = 0.001, NumericType momentum = 0.9, NumericType decayRate = 0.9 )
			: BaseOptimizer< NetworkType >( network ), mLearningRate( learningRate ), mMomentum( momentum ), mDecayRate( decayRate ) {
			for ( auto const& layer : this -> getTrainableLayers( ) ) {
				auto const& weightGradMat = layer -> getWeightGradMat( );
				auto numRows = weightGradMat.rows( );
				auto numCols = weightGradMat.cols( );
				MatrixXType mat( numRows, numCols );
				mat.setZero( );
				mWeightGradMatSaves.emplace_back( mat );
				mGradMatAccumSaves.emplace_back( mat );
			}
		}
		RMSPropNestMomOptimizer(const RMSPropNestMomOptimizer &c) = delete;
		~RMSPropNestMomOptimizer() = default;

		//get/set member functions
		NumericType getLearningRate( ) const { return mLearningRate; }
		void setLearningRate( NumericType learningRate ) { mLearningRate = learningRate; }
		NumericType getMomentum( ) const { return mMomentum; }
		void setMomentum( NumericType momentum ) { mMomentum = momentum; }
		NumericType getDecayRate( ) const { return mDecayRate; }
		void setDecayRate( NumericType decayRate ) { mDecayRate = decayRate; }

		// interface
		void applyInterimUpdate( ) override {
			auto v_iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				weightMat = weightMat - mMomentum * ( *v_iter );
				++v_iter;
			}
		}
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto v_iter = mWeightGradMatSaves.begin( );
			auto r_iter = mGradMatAccumSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto const& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				// accumulate squared gradient
				auto grad_sq = coeff * coeff * weightGradMat.cwiseProduct( weightGradMat );
				*r_iter = mDecayRate * (*r_iter) + (1.0 - mDecayRate ) * grad_sq;
				auto r = *r_iter;
				r = r.unaryExpr( [this]( auto const& ele ) {
					return ( mLearningRate / ( 1.0e-7 + std::sqrt( ele ) ) );
				} );
				auto v = mMomentum * (*v_iter) + r.cwiseProduct( coeff * weightGradMat );
				*v_iter = v;
				weightMat = weightMat - v;
				++v_iter;
				++r_iter;
			}
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate, mMomentum, mDecayRate;
		std::vector< MatrixXType > mWeightGradMatSaves = { }, mGradMatAccumSaves = { };
	}; // end of class RMSPropNestMomOptimizer

} // end NNet

#endif // OPTIMIZERS_HPP
