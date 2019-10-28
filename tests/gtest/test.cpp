// System includes --------------------
#include <numeric>

// GTest includes --------------------
#include "gtest/gtest.h"

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "nnet/serialization/serialize.hpp"
#include "nnet/layers/activation-layer.hpp"

using namespace NNet;

TEST( test, t1 ) {
	EXPECT_EQ( 0, 0 );
}

TEST( serialization, ArchiveStreams ) {
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

TEST( serialization, ActivationLayer ) {
	try {
		// define types
		using ArchiveOutType = boost::archive::text_oarchive;

		using NumericTraitsType = NumericTraits< double >;
		using ActFunType = IdentityActivation< NumericTraitsType >;
		using ActLayerType = ActivationLayer< NumericTraitsType, IdentityActivation >;
		{
		// archive out
		SerializationArchive< ArchiveOutType > ar( "base_layer.txt" );
		ar.OpenOutArchive( );

		// create an ActivationLayer
		auto act_layer = std::make_shared< ActLayerType >( 10, 100 );
		// ActLayerType *act_layer = new ActLayerType( 10, 100 );
		std::cout << "NUM INPUTS: " << act_layer->getNumInputs() << std::endl;
		std::cout << "NUM OUTPUTS: " << act_layer->getNumOutputs() << std::endl;

		// ActLayerType act_layer( 10, 100 );
		// std::cout << "NUM INPUTS: " << act_layer.getNumInputs() << std::endl;
		// std::cout << "NUM OUTPUTS: " << act_layer.getNumOutputs() << std::endl;

		// save layer
		ar.Save( act_layer.get() );
		std::cout << "Saving finished" << std::endl;
		}

		{
		// archive in
		SerializationArchive< ArchiveOutType > ar( "base_layer.txt" );
		ar.OpenInArchive( );

		// create a	ActivationLayer
		ActLayerType *act_layer;

		// load layer
		ar.Load( act_layer );
		}

	} catch ( std::exception const& e ) {
		std::cout << "std::exception caught: ";
		std::cout << e.what() << std::endl;
	} catch ( ... ) {
	}
}
