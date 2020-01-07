#ifndef UTILITY_FUNCTIONS_HPP
#define UTILITY_FUNCTIONS_HPP

// System includes --------------------
#include <vector>
#include <string>
#include <iostream>

// Own includes --------------------
#include "numeric-traits.hpp"

namespace NNet::Utils { // begin NNet::Utils

	std::vector< std::string > split( std::string const& s, char delimiter ) {
		std::vector<std::string> tokens;
		std::string token;
		std::istringstream tokenStream( s );
		while( std::getline( tokenStream, token, delimiter ) ) {
			tokens.emplace_back( token );
		}
		return tokens;
	}

} // end NNet::Utils

#endif // UTILITY_FUNCTIONS_HPP
