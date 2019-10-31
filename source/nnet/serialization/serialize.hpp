#ifndef SERIALIZE_HPP
#define SERIALIZE_HPP

// Boost includes --------------------
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/shared_ptr.hpp>

// System includes --------------------
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <memory>

// Own includes --------------------

namespace NNet { // begin NNet

	template< typename ArchiveOutType >
	struct ArchiveTraits {
	};

	template<>
	struct ArchiveTraits< boost::archive::binary_oarchive > {
		using ArchiveOutType = boost::archive::binary_oarchive;
		using ArchiveInType = boost::archive::binary_iarchive;

		static void OpenInStream( std::ifstream& ifs, std::string const& filename ) {
			ifs.open( filename.c_str(), std::ios_base::binary );
		}
		static void CloseInStream( std::ifstream& ifs ) {
			ifs.close();
		}

		static void OpenOutStream( std::ofstream& ofs, std::string const& filename ) {
			ofs.open( filename.c_str(), std::ios_base::binary );
		}
		static void CloseOutStream( std::ofstream& ofs ) {
			ofs.close();
		}
	};

	template<>
	struct ArchiveTraits< boost::archive::text_oarchive > {
		using ArchiveOutType = boost::archive::text_oarchive;
		using ArchiveInType = boost::archive::text_iarchive;

		static void OpenInStream( std::ifstream& ifs, std::string const& filename ) {
			ifs.open( filename.c_str(), std::ios_base::in );
		}
		static void CloseInStream( std::ifstream& ifs ) {
			ifs.close();
		}

		static void OpenOutStream( std::ofstream& ofs, std::string const& filename ) {
			ofs.open( filename.c_str(), std::ios_base::out );
		}
		static void CloseOutStream( std::ofstream& ofs ) {
			ofs.flush();
			ofs.close();
		}
	};
	/**
	 *SerializationArchive.
	 */
	template< typename ArchiveOutType >
	class SerializationArchive {
	public: 	// public typedefs
		using ArchiveTraitsType = ArchiveTraits< ArchiveOutType >;
		using ArchiveOut = typename ArchiveTraitsType::ArchiveOutType;
		using ArchiveIn = typename ArchiveTraitsType::ArchiveInType;
		using ArchiveOutPtr = std::shared_ptr< ArchiveOut >;
		using ArchiveInPtr = std::shared_ptr< ArchiveIn >;

	private: 	// private typedefs

	public: 	//public member functions
		SerializationArchive() = default;
		explicit SerializationArchive( std::string const& filename)
			: mFileName( filename ) { }
		SerializationArchive( SerializationArchive const& c ) = delete;
		~SerializationArchive() = default;

		void OpenOutArchive() {
			ArchiveTraitsType::OpenOutStream( mOFS, mFileName );
			if ( !mArchiveOut )
				mArchiveOut = std::make_shared< ArchiveOut >( mOFS );
		}
		void CloseOutArchive() {
			ArchiveTraitsType::CloseOutStream( mOFS );
		}
		void OpenInArchive() {
			ArchiveTraitsType::OpenInStream( mIFS, mFileName );
			if ( !mArchiveIn ) {
				mArchiveIn = std::make_shared< ArchiveIn >( mIFS );
			}
		}

		template< typename T >
		void Save( T&& element ) {
			if ( mArchiveOut )
				*mArchiveOut << element;
			else
				throw std::runtime_error( "mArchiveOut is null.");
		}

		template< typename T, typename... Args >
		void Save( T&& first, Args&&... args ) {
			Save( std::forward< T >( first ) );
			Save( std::forward< Args >( args )... );
		}

		template< typename T >
		void Load( T&& element ) {
			if ( mArchiveIn )
				*mArchiveIn >> element;
			else
				throw std::runtime_error( "mArchiveIn is null.");
		}

		template< typename T, typename... Args >
		void Load( T&& first, Args&&... args ) {
			Load( std::forward< T >( first ) );
		    Load( std::forward< Args >( args )... );
		}

		ArchiveOutPtr getOutArchive( ) { return mArchiveOut; }
		ArchiveInPtr getInArchive() { return mArchiveIn; }

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		std::ofstream mOFS;
		std::ifstream mIFS;

		ArchiveOutPtr mArchiveOut = nullptr;
		ArchiveInPtr mArchiveIn = nullptr;

		std::string mFileName;
	}; // end of class SerializationArchive


} // end NNet

// namespace boost::serialization { // begin boost::serialization
// 	template< typename ArchiveType >
// 	void serialize( ArchiveType &ar, unsigned const version ) {
// 	}
// } // end boost::serialization


#endif // SERIALIZE_HPP
