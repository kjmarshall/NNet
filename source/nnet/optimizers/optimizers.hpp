#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

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
				auto& weightGradMat = layerPtr -> getWeightGradMat( );
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
		NumericType getLearningRate( ) { return mLearningRate; }

		// interface
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				auto v = mMomentum * (*iter) + mLearningRate * coeff * weightGradMat;
				*iter = v;
				weightMat = weightMat - v;
				++iter;
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

		// interface
		void applyInterimUpdate( ) override {
			auto iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				weightMat = weightMat - mMomentum * ( *iter );
				++iter;
			}
		}
		void applyWeightUpdate( std::size_t batchSize ) override {
			auto iter = mWeightGradMatSaves.begin( );
			for ( auto& layerPtr : this -> getTrainableLayers( ) ) {
				auto& weightMat = layerPtr -> getWeightMat( );
				auto& weightGradMat = layerPtr -> getWeightGradMat( );
				auto coeff = 1.0 / static_cast< NumericType >( batchSize );
				auto v = mMomentum * (*iter) + mLearningRate * coeff * weightGradMat;
				*iter = v;
				weightMat = weightMat - v;
				++iter;
			}
		}
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NumericType mLearningRate, mMomentum;
		std::vector< MatrixXType > mWeightGradMatSaves = { };
	}; // end of class NesterovMomentumOptimizer

} // end NNet

#endif // OPTIMIZERS_HPP
