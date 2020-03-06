#include<vector>
#include<numeric>
#include <iostream>
#include <iostream>
using namespace std;

double mymin(double a, double b) {
  return a > b ? b : a;
}

/**
 * Assumes that scores are sorted in descending order
 */
void getMixMaxCounts(const vector<pair<double, bool> >& combined,
    std::vector<double>& h_w_le_z, std::vector<double>& h_z_le_z) {
  int cnt_z = 0, cnt_w = 0, queue = 0;
  std::vector<pair<double, bool> >::const_reverse_iterator myPairRev = combined.rbegin();
  for ( ; myPairRev != combined.rend(); ++myPairRev) {
    if (myPairRev->second) {
      ++cnt_w; // target PSM
    } else {
      ++cnt_z; // decoy PSM
      ++queue;
    }
    
    // handles ties
    if (myPairRev+1 == combined.rend() || myPairRev->first != (myPairRev+1)->first) {
      for (int i = 0; i < queue; ++i) {
        h_w_le_z.push_back(static_cast<double>(cnt_w));
        h_z_le_z.push_back(static_cast<double>(cnt_z));
      }
      queue = 0;
    }
  }
}

/**
 * This is a reimplementation of 
 *   Crux/src/app/AssignConfidenceApplication.cpp::compute_decoy_qvalues_mixmax 
 * Which itself was a reimplementation of Uri Keich's code written in R.
 *
 * Assumes that scores are sorted in descending order
 * 
 * If pi0 == 1.0 this is equal to the "traditional" q-value calculation
 */
void getQValues(double pi0, 
    const vector<pair<double, bool> >& combined, vector<double>& q,
    bool skipDecoysPlusOne) {
  std::vector<double> h_w_le_z, h_z_le_z; // N_{w<=z} and N_{z<=z}
  if (pi0 < 1.0) {
    getMixMaxCounts(combined, h_w_le_z, h_z_le_z);
  }

  double estPx_lt_zj = 0.0;
  double E_f1_mod_run_tot = 0.0;
  double fdr = 0.0;

  int n_z_ge_w = 1, n_w_ge_w = 0; // N_{z>=w} and N_{w>=w}
  if (skipDecoysPlusOne) n_z_ge_w = 0;
  
  std::vector<pair<double, bool> >::const_iterator myPair = combined.begin();
  int decoyQueue = 0, targetQueue = 0; // handles ties
  for ( ; myPair != combined.end(); ++myPair) {
    if (myPair->second) { 
      ++n_w_ge_w; // target PSM
      ++targetQueue;
    } else {
      ++n_z_ge_w; // decoy PSM
      ++decoyQueue;
    }
    
    // handles ties
    if (myPair+1 == combined.end() || myPair->first != (myPair+1)->first) {
      if (pi0 < 1.0 && decoyQueue > 0) {
        int j = h_w_le_z.size() - (n_z_ge_w - 1);
        int cnt_w = h_w_le_z.at(j);
        int cnt_z = h_z_le_z.at(j);
        estPx_lt_zj = (double)(cnt_w - pi0*cnt_z) / ((1.0 - pi0)*cnt_z);
        estPx_lt_zj = estPx_lt_zj > 1 ? 1 : estPx_lt_zj;
        estPx_lt_zj = estPx_lt_zj < 0 ? 0 : estPx_lt_zj;
        E_f1_mod_run_tot += decoyQueue * estPx_lt_zj * (1.0 - pi0);
        // if (VERB > 4) {
        //   std::cerr << "Mix-max num negatives correction: "
        //     << (1.0-pi0) * n_z_ge_w << " vs. " << E_f1_mod_run_tot << std::endl;
        // }
      }
      
      // if (includeNegativesInResult) {
      if (true) {
        targetQueue += decoyQueue;
      }
      fdr = (n_z_ge_w * pi0 + E_f1_mod_run_tot) / (double)((std::max)(1, n_w_ge_w));
      for (int i = 0; i < targetQueue; ++i) {
        q.push_back((std::min)(fdr, 1.0));
      }
      decoyQueue = 0;
      targetQueue = 0;
    }
  }
  // Convert the FDRs into q-values.
  partial_sum(q.rbegin(), q.rend(), q.rbegin(), mymin);
}

int main(){
  double scores[] = {0.78086,0.10307,0.32862,0.30650,0.73567,0.55191,0.33581,0.84533,0.93682,0.72977};
  bool labels[] = {true, false,true, false,true, false,true, false,true, false};
  vector<pair<double, bool> > combined (10);
  vector<pair<double, bool> >::iterator myPair = combined.begin();
  int i = 0;
  double pi0 = 1;
  vector<double> qvals;

  for ( ; myPair != combined.end(); ++myPair) {
    myPair->first = scores[i];
    myPair->second = labels[i];
    i++;
  }
  // Run first test
  bool skipDecoysPlusOne = true;
  getQValues(pi0, combined, qvals, skipDecoysPlusOne);
  vector<double>::iterator q = qvals.begin();
  for ( ; q != qvals.end(); ++q) {
    cout << *q << " ";
  }
  cout << "\n";
  // Next test
  skipDecoysPlusOne = false;
  qvals.clear();
  getQValues(pi0, combined, qvals, skipDecoysPlusOne);
  q = qvals.begin();
  for ( ; q != qvals.end(); ++q) {
    cout << *q << " ";
  }
  cout << "\n";

  // Flip the labels and test
  myPair = combined.begin();
  i = 0;
  for ( ; myPair != combined.end(); ++myPair) {
    myPair->first = scores[i];
    myPair->second = !labels[i];
    i++;
  }
  // Run first test
  skipDecoysPlusOne = true;
  qvals.clear();
  getQValues(pi0, combined, qvals, skipDecoysPlusOne);
  q = qvals.begin();
  for ( ; q != qvals.end(); ++q) {
    cout << *q << " ";
  }
  cout << "\n";
  // Next test
  skipDecoysPlusOne = false;
  qvals.clear();
  getQValues(pi0, combined, qvals, skipDecoysPlusOne);
  q = qvals.begin();
  for ( ; q != qvals.end(); ++q) {
    cout << *q << " ";
  }
  cout << "\n";
  
  return 0;
}
