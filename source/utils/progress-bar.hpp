#ifndef PROGRESS_BAR_HPP
#define PROGRESS_BAR_HPP

// System includes --------------------
#include <string>

namespace NNet::Utils { // begin NNet::Utils
   /**
    * ProgressBar.
    */
	class ProgressBar {
	public:
		/**
		 * Constructor.
		 * It takes two values: the expected number of iterations whose progress we
		 * want to monitor and an initial message to be displayed on top of the bar
		 * (which can be updated with updateLastPrintedMessage()).
		 */
		ProgressBar( uint32_t expectedIterations, std::string const& initialMessage = "" );

		/**
		 * Destructor to guarantee RAII.
		 */
		~ProgressBar();

		// Make the object non-copyable
		ProgressBar( ProgressBar const& o ) = delete;
		ProgressBar& operator=( ProgressBar const& o ) = delete;

		/**
		 * Must be invoked when the progress bar is no longer needed to restore the
		 * position of the cursor to the end of the output.
		 * It is automatically invoked when the object is destroyed.
		 */
		void endProgressBar();

		/**
		 * Prints a new message under the last printed message, without overwriting
		 * it. This moves the progress bar down to be placed under the newly
		 * written message.
		 */
		void printNewMessage( std::string const& message );

		/**
		 * Prints a message while the progress bar is on the screen on top on the
		 * last printed message. Since the cursor is right at the beginning of the
		 * progress bar, it moves the cursor up by one line before printing, and
		 * then returns it to its original position.
		 */
		void updateLastPrintedMessage( std::string const& message );

		/**
		 * Overloaded prefix operator, used to indicate that the has been a new
		 * iteration.
		 */
		void operator++();

	private:
		unsigned int mTotalIterations;
		unsigned int mNumberOfTicks;
		bool mEnded;
		size_t mLengthOfLastPrintedMessage;
	};
} // end NNet::Utils

#endif // PROGRESS_BAR_HPP
