#ifndef BASE_DATA_HANDLER_HPP
#define BASE_DATA_HANDLER_HPP

// System includes --------------------
#include <random>
#include <iostream>
#include <vector>
#include <utility>
#include <string>

namespace NNet { // begin NNet

	/**
	 *BaseDataHandler.
	 */
	template< typename InputDataType,
			  typename TargetDataType >
	class BaseDataHandler {
	public: 	// public typedefs
		using DataPairType = std::pair< InputDataType, TargetDataType >;
		using VectorInputDataType = std::vector< InputDataType >;
		using VectorTargetDataType = std::vector< TargetDataType >;
		using VectorDataPairType = std::vector< DataPairType >;

	private: 	// private typedefs

	public: 	//public member functions
		BaseDataHandler( ) = default;
		BaseDataHandler( const BaseDataHandler &c ) = delete;
		~BaseDataHandler( ) = default;

		// get/set member functions
		VectorDataPairType& getTrainingData( ) { return mTrainingData; }
		VectorDataPairType const& getTrainingData( ) const { return mTrainingData; }
		VectorDataPairType& getTestingData( ) { return mTestingData; }
		VectorDataPairType const& getTestingData( ) const { return mTestingData; }

		auto const& getInput( DataPairType const& dataPair ) { return dataPair.first; }
		auto const& getTarget( DataPairType const& dataPair ) { return dataPair.second; }

		// interface functions
		virtual void loadData( std::string const& trainingDataPath, std::string const& testingDataPath, char delimiter ) { };

		void printData( std::ostream& os = std::cout ) {
			auto dataPrinter = [&]( auto const& data, std::string const& title ) {
				os << title << std::endl;
				os << "#Input, Output" << std::endl;
				for ( auto const& dataPair : data ) {
					os << getInput( dataPair ) << " "
					<< getTarget( dataPair ) << std::endl;
				}
			};
			dataPrinter( getTrainingData( ), "#Training Data: size = " + std::to_string( getTrainingData( ).size( ) ) );
			dataPrinter( getTestingData( ), "#Testing Data: size = " + std::to_string( getTestingData( ).size( ) ) );
		}

		// random shuffle
		template< typename IterType, typename RandomEngineType >
		void shuffleRange( IterType first, IterType last, RandomEngineType&& g ) {
			for( auto i = ( last - first ) - 1; i > 0; --i ) {
				std::uniform_int_distribution< decltype( i ) > d( 0, i );
				std::swap( first[i], first[ d( g ) ] );
			}
		}

		// shuffle data
		template< typename RandomEngineType >
		void shuffleTrainingData( RandomEngineType&& g ) {
			shuffle( getTrainingData( ).begin( ), getTrainingData( ).end( ), g );
		}

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		VectorDataPairType mTrainingData = { }, mTestingData = { };
	}; // end of class BaseDataHandler

} // end NNet

#endif // BASE_DATA_HANDLER_HPP
