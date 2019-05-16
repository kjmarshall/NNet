// System includes --------------------
#include <iostream>
#include <map>
#include <string>

// Own includes --------------------
#include "utils/numeric-traits.hpp"
#include "utils/utility-functions.hpp"
#include "nnet/nnet.hpp"

int main(int argc, char *argv[]) {
	std::cout << "Project1: Hello World!" << std::endl;

	std::map< std::string, std::string > strToStr;
	for ( std::size_t i = 0; i < 100; ++i )
		strToStr.emplace( "key" + std::to_string(i), "value" + std::to_string(i) );

	for( auto [key, value] : strToStr )
		std::cout << "Key: " << key << ", " << "Value: " << value << std::endl;

	return 0;
}
