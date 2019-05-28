#ifndef BASE_OPTIMIZER_HPP
#define BASE_OPTIMIZER_HPP

namespace NNet { // begin NNet

	/**
	 *BaseOptimizer.
	 */
	template< typename NetworkType >
	class BaseOptimizer {
	public: 	// public typedefs
		using TrainableLayerType = typename NetworkType::TrainableLayerType;
		using TrainableLayerVecType = std::vector< std::shared_ptr< TrainableLayerType > >;

	private: 	// private typedefs

	public: 	//public member functions
		BaseOptimizer( ) = delete;
		BaseOptimizer( NetworkType& network )
			: mNetwork( network ) {
			for ( auto const& layerPtr : this -> getNetwork( ) ) {
				if ( layerPtr -> isTrainableLayer( ) ) {
					auto trainableLayerPtr = std::static_pointer_cast< TrainableLayerType >( layerPtr );
					mTrainableLayers.emplace_back( trainableLayerPtr );
				}
			}
		}
		BaseOptimizer( const BaseOptimizer &c ) = delete;
		~BaseOptimizer( ) = default;

		// get/set member functions
		NetworkType& getNetwork( ) { return mNetwork; }
		TrainableLayerVecType& getTrainableLayers( );

		// interface
		void resetGradients( ) {
			for ( auto& trainableLayerPtr : getTrainableLayers( ) ) {
				trainableLayerPtr -> resetWeightGradMat( );
			}
		}
		virtual void applyInterimUpdate( ) { }
		virtual void applyWeightUpdate( std::size_t bachSize ) = 0;
	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		NetworkType& mNetwork;
		TrainableLayerVecType mTrainableLayers = {};

	}; // end of class BaseOptimizer

} // end NNet

#endif // BASE_OPTIMIZER_HPP
