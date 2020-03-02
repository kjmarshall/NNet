#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

// System includes --------------------
#include <string>

namespace NNet::Utils { // begin NNet::Utils
	/**
	 *ProgressBar is a RAII class that implements console indicators/graphics
	 *for monitoring the progress during exection of a loop.
	 */
	class ProgressBar {
	public: 	// public typedefs

	private: 	// private typedefs

	public: 	//public member functions
		ProgressBar() = default;
		explicit ProgressBar( unsigned int numIterations, std::string const& initialMessage = "" );

		// Explicitly make non copyable and non moveable even though the
		// move constructor and move assignment would not be generated
		ProgressBar( ProgressBar const& other ) = delete;
		ProgressBar( ProgressBar && other ) = delete;
		ProgressBar& operator=( ProgressBar const& rhs ) = delete;
		ProgressBar& operator=( ProgressBar&& rhs ) = delete;
		~ProgressBar();

		/// Restores the positoin of the cursor to the end of the output
		/// Invoked when the on destruction.
		void endProgressBar();

		/// Print new message
		void printNewMessage( std::string const& message );

		/// Print a message on top of the last printed message
		void updateLastPrintedMessage( std::string const& message );

		/// Overloaded prefix operator, indicates new iteration
		void operator++();

	private: 	//private member functions

	public: 	//public data members

	private: 	//private data members
		unsigned int mTotalIterations;
		unsigned int mNumberOfTicks;
		bool mEnded;
		size_t mLengthOfLastPrintedMessage;
	}; // end of class ProgressBar

} // end NNet::Utils

#endif // PROGRESS_BAR_HPP
