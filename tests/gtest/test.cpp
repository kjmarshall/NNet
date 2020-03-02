// System includes --------------------
#include <numeric>

// GTest includes --------------------
#include "gtest/gtest.h"

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "nnet/serialization/serialize.hpp"
#include "nnet/layers/activation-layer.hpp"
#include "nnet/layers/fully-connected-layer.hpp"
#include "nnet/initializers/weight-initializer.hpp"
#include "nnet/networks/neural-network.hpp"

using namespace NNet;

TEST( Eigen, UnaryExpr ) {
	Eigen::MatrixXd mat( 2, 2 );
	mat << 2, 2, 2, 2;

	Eigen::MatrixXd matsq = mat.cwiseProduct( mat );
	Eigen::MatrixXd matsq_base( 2, 2 );
	matsq_base << 4, 4, 4, 4;
	ASSERT_EQ( matsq_base, matsq );

	Eigen::MatrixXd mat_inv;
	mat_inv = mat.unaryExpr([]( auto const& ele ) {
		return 1. / ele;
	} );
	Eigen::MatrixXd mat_inv_base( 2, 2 );
	mat_inv_base << 0.5, 0.5, 0.5, 0.5;
	ASSERT_EQ( mat_inv_base, mat_inv );
}

TEST( Serialization, ArchiveStreams ) {
	using ArchiveOutType = boost::archive::text_oarchive;
	std::vector< std::size_t > data( 10 );
	std::iota( data.begin(), data.end(), 0 );
	{
		SerializationArchive< ArchiveOutType > ar( "archive_test.txt" );
		ar.OpenOutArchive();
		ar.Save( data );
	}
	{
		SerializationArchive< ArchiveOutType > ar( "archive_test.txt" );
		ar.OpenInArchive();
		std::vector< std::size_t > cmp;
		ar.Load( cmp );
		ASSERT_EQ( data, cmp );
	}
}

TEST( Serialization, ActivationLayer ) {
	// define types
	using ArchiveType = boost::archive::text_oarchive;

	using NumericTraitsType = NumericTraits< double >;
	// using ActFunType = IdentityActivation< NumericTraitsType >;
	using ActLayerType = ActivationLayer< NumericTraitsType, IdentityActivation >;
	// create an ActivationLayer
	// auto act_layer_in = std::make_shared< ActLayerType >( 10, 100 );
	auto act_layer_in = ActLayerType( 10, 100 );
	{
	// archive out
	SerializationArchive< ArchiveType > ar( "act_layer_ser.txt" );
	ar.OpenOutArchive( );

	// save layer
	ar.Save( act_layer_in );
	}

	// create a	ActivationLayer
	// std::shared_ptr< ActLayerType > act_layer_out;
	ActLayerType act_layer_out;
	{
	// archive in
	SerializationArchive< ArchiveType > ar( "act_layer_ser.txt" );
	ar.OpenInArchive( );

	// load layer
	ar.Load( act_layer_out );
	}
	// ASSERT_EQ( *act_layer_in, *act_layer_out );
	ASSERT_EQ( act_layer_in, act_layer_out );
}

TEST( Serialization, NeuralNetwork ) {
	// define types
	using ArchiveType = boost::archive::text_oarchive;

	using NumericTraitsType = NumericTraits< double >;

	// declare layer types
	using FullyConnectedLayerType = FullyConnectedLayer< NumericTraitsType >;
	// using ActFunType = ArcTanActivation< NumericTraitsType >;
	using ActLayerType = ActivationLayer< NumericTraitsType, ArcTanActivation >;

	// define network initializer type
	using InitializerType = GlorotInitializer< NumericTraitsType >;
	// create the neural network
	using NetworkType = NeuralNetwork< NumericTraitsType, InitializerType >;
	NetworkType nnet_in;
	nnet_in.addLayer( std::make_shared< FullyConnectedLayerType >( 1, 30, LayerType::INPUT ) );
	nnet_in.addLayer( std::make_shared< ActLayerType >( 30 ) );
	nnet_in.addLayer( std::make_shared< FullyConnectedLayerType >( 30, 20, LayerType::HIDDEN ) );
	nnet_in.addLayer( std::make_shared< ActLayerType >( 20 ) );
	nnet_in.addLayer( std::make_shared< FullyConnectedLayerType >( 20, 10, LayerType::HIDDEN ) );
	nnet_in.addLayer( std::make_shared< ActLayerType >( 10 ) );
	nnet_in.addLayer( std::make_shared< FullyConnectedLayerType >( 10, 1, LayerType::HIDDEN ) );
	nnet_in.finalize( );
	nnet_in.printNetworkInfo( );

	{
	// archive out
	SerializationArchive< ArchiveType > ar( "nnet_ser.txt" );
	ar.OpenOutArchive( );

	// save layer
	ar.Save( nnet_in );
	}

	NetworkType nnet_out;
	{
	// archive in
	SerializationArchive< ArchiveType > ar( "nnet_ser.txt" );
	ar.OpenInArchive( );

	// load layer
	ar.Load( nnet_out );
	}
	ASSERT_EQ( nnet_in, nnet_out );
}
