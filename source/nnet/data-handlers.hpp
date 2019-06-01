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

	/**
	 *MINSTDataHandler.
	 */
	template< typename InputDataType,
			  typename TargetDataType >
	class MINSTDataHandler
		: public BaseDataHandler< InputDataType, TargetDataType > {
	public: 	// public typedefs
		using BaseDataHandlerType = BaseDataHandler< InputDataType, TargetDataType >;
		using VectorInputDataType = typename BaseDataHandlerType::VectorInputDataType;
		using VectorTargetDataType = typename BaseDataHandlerType::VectorTargetDataType;

	private: 	// private typedefs

	public: 	//public member functions
		MINSTDataHandler( ) = delete;
		explicit MINSTDataHandler( std::string const& trainingImagesPath,
								   std::string const& trainingLabelsPath,
								   std::string const& testImagesPath,
								   std::string const& testLabelsPath ) {
			VectorInputDataType trainingImageData, testImageData;
			VectorTargetDataType trainingLabelData, testLabelData;
			int numImages, imageSize, numLabels;

			trainingImageData = readMINSTImages( trainingImagesPath, numImages, imageSize );
			std::cout << "Finished reading " << numImages << " training images with image size = " << imageSize << std::endl;

			testImageData = readMINSTImages( testImagesPath, numImages, imageSize );
			std::cout << "Finished reading " << numImages << " testing images with image size = " << imageSize << std::endl;

			trainingLabelData = readMINSTLabels( trainingLabelsPath, numLabels );
			std::cout << "Finished reading " << numLabels << " training labels" << std::endl;

			testLabelData = readMINSTLabels( testLabelsPath, numLabels );
			std::cout << "Finished reading " << numLabels << " testing labels" << std::endl;

			std::cout << "Finished reading minst data... packing into data containers..." << std::endl;
			auto& trainingData = this -> getTrainingData( );
			for ( std::size_t i = 0; i < trainingImageData.size( ); ++i ) {
				auto& image = trainingImageData[i];
				auto& label = trainingLabelData[i];
				// std::cout << "Label: " << i << std::endl;
				// std::cout << label << std::endl;
				// Eigen::Map< Eigen::MatrixXd > imageSquare( image.data( ), 28, 28 );
				// std::cout << "Image:" << std::endl;
				// std::cout << imageSquare << std::endl;
				trainingData.emplace_back( std::make_pair( image, label ) );
			}
			auto& testingData = this -> getTestingData( );
			for ( std::size_t i = 0; i < testImageData.size( ); ++i ) {
				auto& image = testImageData[i];
				auto& label = testLabelData[i];
				testingData.emplace_back( std::make_pair( image, label ) );
			}
		}
		MINSTDataHandler(const MINSTDataHandler &c) = delete;
		~MINSTDataHandler( ) = default;

	private: 	//private member functions
		VectorInputDataType readMINSTImages( std::string full_path,
											 int& number_of_images,
											 int& image_size ) {
			auto reverseInt = []( int i ) {
				unsigned char c1, c2, c3, c4;
				c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
				return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
			};

			typedef unsigned char uchar;

			std::ifstream file( full_path, std::ios::binary );

			if( file.is_open( ) ) {
				int magic_number = 0, n_rows = 0, n_cols = 0;

				file.read((char *)&magic_number, sizeof(magic_number));
				magic_number = reverseInt(magic_number);

				if(magic_number != 2051) throw std::runtime_error( "Invalid MNIST image file!" );

				file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
				file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
				file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

				image_size = n_rows * n_cols;

				VectorInputDataType dataSet;
				dataSet.reserve( number_of_images );
				for ( int i = 0; i < number_of_images; i++ ) {
					InputDataType imageData( image_size );
					unsigned char buffer[image_size];
					file.read( (char*)&buffer[0], image_size );
					for ( std::size_t j = 0; j < image_size; ++j ) {
						imageData[j] = static_cast<double>(buffer[j]);
					}
					dataSet.emplace_back( imageData );
				}
				return dataSet;
			}
			else {
				throw std::runtime_error("Cannot open file `" + full_path + "`!");
			}
		}

		VectorTargetDataType readMINSTLabels( std::string full_path,
											  int& number_of_labels ) {
			auto reverseInt = [](int i) {
				unsigned char c1, c2, c3, c4;
				c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
				return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
			};

			typedef unsigned char uchar;

			std::ifstream file(full_path, std::ios::binary);

			if(file.is_open()) {
				int magic_number = 0;
				file.read((char *)&magic_number, sizeof(magic_number));
				magic_number = reverseInt(magic_number);

				if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

				file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

				VectorTargetDataType dataSet;
				dataSet.reserve( number_of_labels );
				for(int i = 0; i < number_of_labels; i++) {
					TargetDataType labelData( 10 );
					labelData.setZero( );
					unsigned char buffer[1];
					file.read( (char*)&buffer[0], 1 );
					std::size_t idx = static_cast< std::size_t >( buffer[0] );
					labelData[idx] = 1.0;
					dataSet.emplace_back( labelData );
				}
				return dataSet;
			}
			else {
				throw std::runtime_error( "Unable to open file `" + full_path + "`!" );
			}
		}
	public: 	//public data members

	private: 	//private data members

	}; // end of class MINSTDataHandler


} // end NNet

#endif // DATA_HANDLERS_HPP
