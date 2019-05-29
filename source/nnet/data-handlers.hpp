#ifndef DATA_HANDLERS_HPP
#define DATA_HANDLERS_HPP

// System includes --------------------
#include <iostream>
#include <fstream>

// Own includes --------------------
#include "nnet/base-data-handler.hpp"
#include "utils/utility-functions.hpp"

namespace NNet { // begin NNet

	/**
	 *RegressionDataHandler.
	 */
	template< typename InputDataType,
			  typename TargetDataType >
	class RegressionDataHandler
		: public BaseDataHandler< InputDataType, TargetDataType > {
	public: 	// public typedefs

	private: 	// private typedefs

	public: 	//public member functions
		RegressionDataHandler( ) = default;
		RegressionDataHandler( const RegressionDataHandler &c ) = delete;
		~RegressionDataHandler( ) = default;

		// interface functions
		void loadData( std::string const& trainingDataPath, std::string const& testingDataPath, char delimiter ) override {
			loadDataFile( trainingDataPath, delimiter, this->getTrainingData( ) );
		}

	private: 	//private member functions
		template< typename DataVectorType >
		void loadDataFile( std::string const& dataFilePath, char delimiter, DataVectorType& dataVec ) {
			std::ifstream IFS( dataFilePath.c_str( ) );
			std::string line;
			if ( IFS.is_open( ) ) {
				while( getline( IFS, line ) ) {
					auto splits = Utils::split( line, delimiter );
					InputDataType input(1);
					input << std::stod( splits[0] );
					TargetDataType target(1);
					target << std::stod( splits[1] );
					dataVec.emplace_back( std::make_pair( input, target) );
				}
			}
			else {
				throw std::runtime_error( "Unable to open input file..." );
			}
			IFS.close( );
		}

	public: 	//public data members

	private: 	//private data members

	}; // end of class RegressionDataHandler


} // end NNet

#endif // DATA_HANDLERS_HPP
