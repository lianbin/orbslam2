// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "optimization_algorithm_levenberg.h"

#include <iostream>

#include "g2o/stuff/timeutil.h"

#include "sparse_optimizer.h"
#include "solver.h"
#include "batch_stats.h"
using namespace std;

namespace g2o {

  OptimizationAlgorithmLevenberg::OptimizationAlgorithmLevenberg(Solver* solver) :
    OptimizationAlgorithmWithHessian(solver)
  {
    _currentLambda = -1.;
    _tau = 1e-5;
    _goodStepUpperScale = 2./3.;
    _goodStepLowerScale = 1./3.;
    _userLambdaInit = _properties.makeProperty<Property<double> >("initialLambda", 0.);
    _maxTrialsAfterFailure = _properties.makeProperty<Property<int> >("maxTrialsAfterFailure", 10);
    _ni=2.;
    _levenbergIterations = 0;
  }

  OptimizationAlgorithmLevenberg::~OptimizationAlgorithmLevenberg()
  {
  }

  OptimizationAlgorithm::SolverResult OptimizationAlgorithmLevenberg::solve(int iteration, bool online)
  {
    assert(_optimizer && "_optimizer not set");
    assert(_solver->optimizer() == _optimizer && "underlying linear solver operates on different graph");

    if (iteration == 0 && !online) { // built up the CCS structure, here due to easy time measure
      bool ok = _solver->buildStructure();
      if (! ok) {
        cerr << __PRETTY_FUNCTION__ << ": Failure while building CCS structure" << endl;
        return OptimizationAlgorithm::Fail;
      }
    }

    double t=get_monotonic_time();
	//计算所有误差项的误差
    _optimizer->computeActiveErrors();
    G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
    if (globalStats) {
      globalStats->timeResiduals = get_monotonic_time()-t;
      t=get_monotonic_time();
    }
	//所有误差项的平方的加和
    double currentChi = _optimizer->activeRobustChi2();
    double tempChi=currentChi;

    _solver->buildSystem();
    if (globalStats) {
      globalStats->timeQuadraticForm = get_monotonic_time()-t;
    }

    // core part of the Levenbarg algorithm
    if (iteration == 0) {
      _currentLambda = computeLambdaInit();//初始化Lm算法的lambda
      _ni = 2;
    }

    double rho=0;
    int& qmax = _levenbergIterations;
    qmax = 0;
    do {
      _optimizer->push();
      if (globalStats) {
        globalStats->levenbergIterations++;
        t=get_monotonic_time();
      }
      // update the diagonal of the system matrix
      _solver->setLambda(_currentLambda, true);
      bool ok2 = _solver->solve();//求解 Hx=-b
      if (globalStats) {
        globalStats->timeLinearSolution+=get_monotonic_time()-t;
        t=get_monotonic_time();
      }
	  //更新参数
      _optimizer->update(_solver->x());
      if (globalStats) {
        globalStats->timeUpdate = get_monotonic_time()-t;
      }

      // restore the diagonal
      _solver->restoreDiagonal();
      //计算误差函数
      _optimizer->computeActiveErrors();
	  //计算误差的平方加和
      tempChi = _optimizer->activeRobustChi2();

      if (! ok2)
        tempChi=std::numeric_limits<double>::max();
	 //由于在优化函数中，计算rho和scale 前面都有一个½的系数
	 //直接可以约掉，所以我们看到，在程序中实际计算rho和scale
	 //的时候，没有乘以½这个系数。
     // F(x) - F(x+hlm)
      rho = (currentChi-tempChi);
	  //L(0) - L(hlm)
      double scale = computeScale();
      scale += 1e-3; // make sure it's non-zero :)
      rho /=  scale;

      if (rho>0 && g2o_isfinite(tempChi)){ // last step was good
        double alpha = 1.-pow((2*rho-1),3);
        // crop lambda between minimum and maximum factors
        alpha = (std::min)(alpha, _goodStepUpperScale);
        double scaleFactor = (std::max)(_goodStepLowerScale, alpha);
        _currentLambda *= scaleFactor;
        _ni = 2;
        currentChi=tempChi;
        _optimizer->discardTop();//没进去看，感觉上应该是抛弃存储的上一次的参数的栈
      } else {
        _currentLambda*=_ni;
        _ni*=2;
		//恢复之前的参数，不进行更新
        _optimizer->pop(); // restore the last state before trying to optimize
      }
      qmax++;
    } while (rho<0 && qmax < _maxTrialsAfterFailure->value() && ! _optimizer->terminate());

    if (qmax == _maxTrialsAfterFailure->value() || rho==0)
      return Terminate;
    return OK;
  }

  //初始化lm算法中(H+λI)δx=-b,中的λ
  double OptimizationAlgorithmLevenberg::computeLambdaInit() const
  {
    if (_userLambdaInit->value() > 0)
      return _userLambdaInit->value();
    double maxDiagonal=0.;
    for (size_t k = 0; k < _optimizer->indexMapping().size(); k++) {
      OptimizableGraph::Vertex* v = _optimizer->indexMapping()[k];
      assert(v);
      int dim = v->dimension();
      for (int j = 0; j < dim; ++j){
        maxDiagonal = std::max(fabs(v->hessian(j,j)),maxDiagonal);
      }
    }
    return _tau*maxDiagonal;
  }

  double OptimizationAlgorithmLevenberg::computeScale() const
  {
    double scale = 0.;
    for (size_t j=0; j < _solver->vectorSize(); j++){
      scale += _solver->x()[j] * (_currentLambda * _solver->x()[j] + _solver->b()[j]);
    }
    return scale;
  }

  void OptimizationAlgorithmLevenberg::setMaxTrialsAfterFailure(int max_trials)
  {
    _maxTrialsAfterFailure->setValue(max_trials);
  }

  void OptimizationAlgorithmLevenberg::setUserLambdaInit(double lambda)
  {
    _userLambdaInit->setValue(lambda);
  }

  void OptimizationAlgorithmLevenberg::printVerbose(std::ostream& os) const
  {
    os
      << "\t schur= " << _solver->schur()
      << "\t lambda= " << FIXED(_currentLambda)
      << "\t levenbergIter= " << _levenbergIterations;
  }

} // end namespace