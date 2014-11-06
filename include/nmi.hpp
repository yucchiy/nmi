#ifndef __YUCCHIY_NMI_HPP__
#define __YUCCHIY_NMI_HPP__

#include <vector>
#include <cmath>

namespace yucchiy {

typedef std::pair<int, std::vector<int> > Cluster;

namespace nmi {

    static bool ReadInput(const cv::String& inputPath, std::vector<Cluster>& clusters) {

        std::ifstream ifs(inputPath.c_str());
        if (ifs.fail()) {
            std::cerr << "Failed reading input path: " << inputPath << std::endl;
            return false;
        }

        int numCluster;
        ifs >> numCluster;
        if (numCluster <= 0) {
            std::cerr << inputPath << " has no cluster" << std::endl;
            return false;
        }

        clusters.clear();
        for (int i = 0; i < numCluster; i++) {
            int numElement;
            ifs >> numElement;
            if (numElement <= 0) {
                std::cerr << "Cluster " << i << " has no element" << std::endl;
            }

            std::vector<int> elements;
            for (int j = 0; j < numElement; j++) {
                int element;
                ifs >> element;
                elements.push_back(element);
            }

            std::sort(elements.begin(), elements.end());
            clusters.push_back(Cluster(elements[0], elements));
        }

        std::sort(clusters.begin(), clusters.end());
        return true;
    }

    static cv::String GetClusterStr(const Cluster& cluster) {
        cv::String str = "{";
        for (int i = 0; i < cluster.second.size(); i++) {
            str += cv::format("%d, ", cluster.second[i]);
        }

        str += "}";
        return str;
    }

    static cv::String GetClustersStr(const std::vector<Cluster>& clusters) {
        cv::String str = "";
        for (int i = 0; i < clusters.size(); i++) {
            str += cv::format("%d: ", i + 1) + GetClusterStr(clusters[i]) + "\n";
        }

        return str;
    }

    static double H(const std::vector<Cluster>& X) {
        int numElement = 0;
        for (int i = 0; i < X.size(); i++) {
            numElement += X[i].second.size();
        }

        double h = 0.0;
        for (int i = 0; i < X.size(); i++) {
            double pr = (double)X[i].second.size() / numElement;
            h += -1 * pr * log2(pr);
        }

        return h;
    }

    static double H(const std::vector<Cluster>& X, const std::vector<Cluster>& Y) {
        int numElement = 0;
        for (int i = 0; i < X.size(); i++) {
            numElement += X[i].second.size();
        }

        cv::Mat_<double> pr(X.size(), Y.size(), 0.0);
        for (int i = 0; i < X.size(); i++) {
            for (int j = 0; j < X[i].second.size(); j++) {
                for (int k = 0; k < Y.size(); k++) {
                    for (int l = 0; l < Y[k].second.size(); l++) {
                        if (X[i].second.at(j) == Y[k].second.at(l)) {
                            pr(i, k) += 1.0;
                        }
                    }
                }
            }
        }

        pr /= numElement;
        double h = 0.0;
        for (int i = 0; i < pr.rows; i++) {
            for (int j = 0; j < pr.cols; j++) {
                if (pr(i, j) > 1e-20) {
                    h += -1 * pr(i, j) * log2(pr(i, j));
                }
            }
        }

        return h;
    }

    static double MI(const std::vector<Cluster>& X, const std::vector<Cluster>& Y) {
        return H(X) + H(Y) - H(X, Y);
    }

    static double NMI(const std::vector<Cluster>& X, const std::vector<Cluster>& Y) {
        return 2.0 * MI(X, Y) / ( H(X) + H(Y) );
    }

} // namespace nmi

} // namespace yucchiy

#endif // #define __YUCCHIY_NMI_HPP__
