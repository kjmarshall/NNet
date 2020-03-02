// System includes --------------------
#include <iostream>
#include <iomanip>
#include <sstream>

// Own includes --------------------
#include "progress-bar.hpp"

constexpr int progress_bar_length = 55;
constexpr float bin_size_percentage = ( 100.0 / progress_bar_length );

namespace NNet::Utils { // begin NNet::Utils

    static std::string generateProgressBar( unsigned int percentage ) {
        const int progress = static_cast<int>( percentage / bin_size_percentage );
        std::ostringstream ss;
        ss << " " << std::setw( 3 ) << std::right << percentage << "% ";
        std::string bar( "[" + std::string( progress_bar_length - 2, ' ' ) + "]" );

        unsigned int numberOfSymbols = std::min(
			std::max( 0, progress - 1 ),
			progress_bar_length - 2 );

        bar.replace( 1, numberOfSymbols, std::string( numberOfSymbols, '|' ) );

        ss << bar;
        return ss.str();
    }

	ProgressBar::ProgressBar( unsigned int numIterations, const std::string& initialMessage)
		: mTotalIterations( numIterations ), mNumberOfTicks( 0 ), mEnded( false ) {
		std::cout << initialMessage << "\n";
		mLengthOfLastPrintedMessage = initialMessage.size();
		std::cout << generateProgressBar(0) << "\r" << std::flush;
	}

	ProgressBar::~ProgressBar() {
		endProgressBar();
	}

	void ProgressBar::operator++() {
		if (mEnded) {
			throw std::runtime_error(
				"Attempted to use progress bar after having terminated it" );
		}

		mNumberOfTicks = std::min( mTotalIterations, mNumberOfTicks + 1 );
		const unsigned int percentage = static_cast<unsigned int>(
            mNumberOfTicks * 100.0 / mTotalIterations );

		std::cout << generateProgressBar( percentage ) << "\r" << std::flush;
	}

	void ProgressBar::printNewMessage(const std::string& message) {
		if ( mEnded ) {
			throw std::runtime_error(
				"Attempted to use progress bar after having terminated it" );
		}

		std::cout << "\r"
				  << std::left
				  << std::setw( progress_bar_length + 6 )
				  << message << "\n";
		mLengthOfLastPrintedMessage = message.size();
		const unsigned int percentage = static_cast<unsigned int>(
			mNumberOfTicks * 100.0 /mTotalIterations );

		std::cout << generateProgressBar( percentage ) << "\r" << std::flush;
	}

	void ProgressBar::updateLastPrintedMessage(const std::string& message) {
		if ( mEnded ) {
			throw std::runtime_error(
                "Attempted to use progress bar after having terminated it");
		}

		std::cout << "\r\033[F"
				  << std::left
				  << std::setw( mLengthOfLastPrintedMessage )
				  << message << "\n";
		mLengthOfLastPrintedMessage = message.size();
	}

	void ProgressBar::endProgressBar() {
		if (!mEnded) {
			std::cout << std::string( 2, '\n' );
		}
		mEnded = true;
	}
} // end NNet::Utils
