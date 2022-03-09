/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cmath>
#include <random>
#include "graphlearn/core/operator/sampler/sampler.h"
#include "graphlearn/include/config.h"
#include <vector>
// #include <map>
#include <stack>

using std::vector;
// using std::map;
using std::stack;

namespace graphlearn {
namespace op {

class RandomPqSampler : public Sampler {
public:
  RandomPqSampler(){
    _p = 1.0f;
    _q = 0.5f;
  }

  virtual ~RandomPqSampler() {}

  Status Sample(const SamplingRequest* req,
                SamplingResponse* res) override {
    int32_t count = req->NeighborCount();
    int32_t batch_size = req->BatchSize();

    res->SetBatchSize(batch_size);
    res->SetNeighborCount(count);
    res->InitNeighborIds(batch_size * count);
    res->InitEdgeIds(batch_size * count);

    const std::string& edge_type = req->Type();
    Graph* graph = graph_store_->GetGraph(edge_type);
    auto storage = graph->GetLocalStorage();

    thread_local static std::random_device rd;
    thread_local static std::mt19937 engine(rd());

    const int64_t* src_ids = req->GetSrcIds();
    const int64_t* prev_ids = req->GetPrevIds();

    int32_t max_nbr_size = 0;
    for (int i = 0; i < batch_size; ++i){
      int32_t nbr_szie = storage->GetNeighbors(src_ids[i]).Size();
      if (nbr_szie > max_nbr_size)
        max_nbr_size == nbr_szie;
    }

    vector<float> probs(max_nbr_size);
    vector<int> J(max_nbr_size);
    vector<float> q(max_nbr_size);

    for (int32_t i = 0; i < batch_size; ++i) {
      int64_t src_id = src_ids[i];
      int64_t prev_id =  prev_ids[i];
      auto neighbor_ids = storage->GetNeighbors(src_id);
      int32_t nbr_size = neighbor_ids.Size();
      
      if (nbr_size == 0) {
        res->FillWith(GLOBAL_FLAG(DefaultNeighborId), -1);
      } 
      else {
        auto edge_ids = storage->GetOutEdges(src_id);
        if (prev_id == -1){
          float sumw = 0.f;
          for (int i = 0; i < nbr_size; ++i){
            probs[i] = storage->GetEdgeWeight(edge_ids[i]);
            sumw += probs[i];
          }
          for (int i = 0; i < nbr_size; ++i)
            probs[i] /= sumw;
          
          alias_setup(probs, J, q, nbr_size);

          std::uniform_real_distribution<double> dist(0.0,1.0);
          
          for (int j = 0; j < count; ++j){
            int32_t idx = (int32_t)(dist(engine)*nbr_size);
            if (dist(rng) >= q[idx])
              idx = J[idx];
            res->AppendNeighborId(neighbor_ids[idx]);
            res->AppendEdgeId(edge_ids[idx]);
            res->AppendNeighborIdPrev(src_id);
          }
        }
        else {
          float sumw = 0.f;
          for (int i = 0; i < nbr_size; ++i){
            int nbr_id = neighbor_ids[i];
            float w, ew = storage->GetEdgeWeight(edge_ids[i]);
            if (nbr_id == prev_id)
              w = ew / _p;
            else if (_hasEdge(nbr_id, prev_id))
              w = ew;
            else
              w = ew / _q;
            sumw += w;
          }
          for (int i = 0; i < nbr_size; ++i)
            probs[i] /= sumw;

          alias_setup(probs, J, q, nbr_size);

          std::uniform_real_distribution<double> dist(0.0,1.0);
          
          for (int j = 0; j < count; ++j){
            int32_t idx = (int32_t)(dist(engine)*nbr_size);
            if (dist(rng) >= q[idx])
              idx = J[idx];
            res->AppendNeighborId(neighbor_ids[idx]);
            res->AppendEdgeId(edge_ids[idx]);
            res->AppendNeighborIdPrev(src_id);
          }
    }
    return Status::OK();
  }
  
  private:
  bool _hasEdge(graphlearn::io::GraphStorage* storage, int src_id, int dst_id){
    auto neighbor_ids = storage->GetNeighbors(src_id);
    for (int nbr_id: neighbor_ids){
      if (nbr_id == dst_id)
        return true;
    }
    return false;
  }

  void alias_setup(vector<float> &probs, vector<int> &J, vector<float> &q, int K){
    stack<int> smaller, larger;
    
    for (int i = 0; i < K; i++){
        q[i] = K * probs[i];
        if (q[i] < 1.0f)
            smaller.push(i);
        else
            larger.push(i);
    }

    while (!smaller.empty() && !larger.empty()) {
        int small = smaller.top();
        smaller.pop();
        int large = larger.top();
        larger.pop();

        J[small] = large;
        q[large] = q[large] + q[small] - 1.0;
        if (q[large] < 1.0f)
            smaller.push(large);
        else
            larger.push(large);
    }
  }

  private:
  float _p, _q;
};

REGISTER_OPERATOR("RandomPqSampler", RandomPqSampler);

}  // namespace op
}  // namespace graphlearn
