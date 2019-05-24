#ifndef CRTP_HELPER_HPP
#define CRTP_HELPER_HPP

template< typename DerivedType >
struct CRTPHelper {
	DerivedType& underlying( ) { return static_cast< DerivedType& >( *this ); }
	DerivedType const& underlying( ) const { return static_cast< DerivedType const& >( *this ); }
};

#endif // CRTP_HELPER_HPP
