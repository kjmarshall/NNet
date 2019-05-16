#ifndef NUMERIC_TRAITS_HPP
#define NUMERIC_TRAITS_HPP

// Eigen includes --------------------
#include <Eigen/Core>

namespace NNet { // begin NNet

	template< typename DataType >
	struct NumericTraits {
		using NumericType = DataType;
		using MatrixXType = Eigen::Matrix< DataType,	Eigen::Dynamic, Eigen::Dynamic >;
		using VectorXType = Eigen::Matrix< DataType, Eigen::Dynamic, 1 >;
		using RowVectorXType = Eigen::Matrix< DataType, 1, Eigen::Dynamic >;

		using ArrayXXType = Eigen::Array< DataType,	Eigen::Dynamic, Eigen::Dynamic >;
		using ArrayXType = Eigen::Array< DataType, Eigen::Dynamic, 1 >;
		using RowArrayXType = Eigen::Array< DataType, 1, Eigen::Dynamic >;
	};

} // end NNet

#endif // NUMERIC_TRAITS_HPP
