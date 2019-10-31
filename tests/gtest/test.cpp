// System includes --------------------
#include <numeric>

// GTest includes --------------------
#include "gtest/gtest.h"

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "nnet/serialization/serialize.hpp"
#include "nnet/layers/activation-layer.hpp"

using namespace NNet;

TEST( Serialization, ArchiveStreams ) {
	using ArchiveOutType = boost::archive::text_oarchive;
	std::vector< std::size_t > data(10);
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
		ar.Load(cmp);
		ASSERT_EQ( data, cmp );
	}
}

TEST( Serialization, ActivationLayer ) {
	try {
		// define types
		using ArchiveType = boost::archive::text_oarchive;

		using NumericTraitsType = NumericTraits< double >;
		using ActFunType = IdentityActivation< NumericTraitsType >;
		using ActLayerType = ActivationLayer< NumericTraitsType, IdentityActivation >;
		// create an ActivationLayer
		auto act_layer_in = std::make_shared< ActLayerType >( 10, 100 );
		{
		// archive out
		SerializationArchive< ArchiveType > ar( "base_layer.txt" );
		ar.OpenOutArchive( );

		// save layer
		ar.Save( act_layer_in );
		}

		// create a	ActivationLayer
		std::shared_ptr< ActLayerType > act_layer_out;
		{
		// archive in
		SerializationArchive< ArchiveType > ar( "base_layer.txt" );
		ar.OpenInArchive( );

		// load layer
		ar.Load( act_layer_out );
		}
		ASSERT_EQ( *act_layer_in, *act_layer_out );

	} catch ( std::exception const& e ) {
		std::cout << "std::exception caught: ";
		std::cout << e.what() << std::endl;
	} catch ( ... ) {
	}
}
