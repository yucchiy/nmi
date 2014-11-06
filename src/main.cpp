#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>

#include "nmi.hpp"

const cv::String keys =
"{h help usage          | | print this message }"
"{@truth_input_path     | | input file path    }"
"{@estimated_input_path | | input file path    }";


int main(const int argc, const char* const argv[]) {

    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("h")) {
        parser.printMessage();
        return 0;
    }

    std::vector<yucchiy::Cluster> truthClusters, estimatedClusters;
    if (
            !yucchiy::nmi::ReadInput(parser.get<cv::String>(0), truthClusters) || 
            !yucchiy::nmi::ReadInput(parser.get<cv::String>(1), estimatedClusters)
       ) {
        return 1;
    }

    std::cerr << "Grund Truth:" << std::endl;
    std::cerr << yucchiy::nmi::GetClustersStr(truthClusters) << std::endl;

    std::cerr << "Estimated:" << std::endl;
    std::cerr << yucchiy::nmi::GetClustersStr(estimatedClusters) << std::endl;

    std::cerr << "NMI: " << yucchiy::nmi::NMI(estimatedClusters, truthClusters) << std::endl;
    std::cout << yucchiy::nmi::NMI(truthClusters, estimatedClusters) << std::endl;

    return 0;
}
