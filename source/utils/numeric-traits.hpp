#ifndef NUMERIC_TRAITS_HPP
#define NUMERIC_TRAITS_HPP

// Boost includes --------------------
#include <boost/serialization/array.hpp>

// Eigen includes --------------------
#include <Eigen/Core>

namespace NNet { // begin NNet

	template< typename DataType >
	struct NumericTraits {
		using NumericType = DataType;
		using MatrixXType = Eigen::Matrix< DataType, Eigen::Dynamic, Eigen::Dynamic >;
		using VectorXType = Eigen::Matrix< DataType, Eigen::Dynamic, 1 >;
		using RowVectorXType = Eigen::Matrix< DataType, 1, Eigen::Dynamic >;

		using ArrayXXType = Eigen::Array< DataType,	Eigen::Dynamic, Eigen::Dynamic >;
		using ArrayXType = Eigen::Array< DataType, Eigen::Dynamic, 1 >;
		using RowArrayXType = Eigen::Array< DataType, 1, Eigen::Dynamic >;
	};

} // end NNet

namespace boost::serialization { // begin boost::serialization

    template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline void save( Archive& ar,
                      const Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M,
                      const unsigned int /* file_version */ ) {
      typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows = M.rows();
      typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index cols = M.cols();

      ar << rows;
      ar << cols;

      ar << make_array( M.data(), M.size() );
    }

    template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline void load( Archive& ar,
                      Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M,
                      const unsigned int /* file_version */ ) {
      typename Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;

      ar >> rows;
      ar >> cols;

      //if (rows=!_Rows) throw std::exception(/*"Unexpected number of rows"*/);
      //if (cols=!_Cols) throw std::exception(/*"Unexpected number of cols"*/);

      ar >> make_array( M.data(), M.size() );
    }

    template<class Archive, typename _Scalar, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline void load( Archive& ar,
                      Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>& M,
                      const unsigned int /* file_version */ ) {
      typename Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols, _Options, _MaxRows, _MaxCols>::Index rows, cols;

      ar >> rows;
      ar >> cols;

      //if (cols=!_Cols) throw std::exception(/*"Unexpected number of cols"*/);

      M.resize(rows, Eigen::NoChange);

      ar >> make_array( M.data(), M.size() );
    }

    template<class Archive, typename _Scalar, int _Rows, int _Options, int _MaxRows, int _MaxCols>
    inline void load( Archive& ar,
                      Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& M,
                      const unsigned int /* file_version */ ) {
      typename Eigen::Matrix<_Scalar, _Rows, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>::Index rows, cols;

      ar >> rows;
      ar >> cols;

      //if (rows=!_Rows) throw std::exception(/*"Unexpected number of rows"*/);

      M.resize(Eigen::NoChange, cols);

      ar >> make_array( M.data(), M.size() );
    }

    template<class Archive, typename _Scalar, int _Options, int _MaxRows, int _MaxCols>
    inline void load( Archive& ar,
                      Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>& M,
                      const unsigned int /* file_version */ ) {
      typename Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic, _Options, _MaxRows, _MaxCols>::Index rows, cols;

      ar >> rows;
      ar >> cols;

      M.resize(rows, cols);

      ar >> make_array( M.data(), M.size() );
    }

    template<class Archive, typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    inline void serialize(Archive & ar,
                          Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& M,
                          const unsigned int file_version ) {
      split_free(ar, M, file_version);
    }
} // end boost::serialization

#endif // NUMERIC_TRAITS_HPP
