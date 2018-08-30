/*
Copyright (c) 2010-2016, Mathieu Labbe - IntRoLab - Universite de Sherbrooke
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Universite de Sherbrooke nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef PARTICLEFILTER_H_
#define PARTICLEFILTER_H_

//#include <rtabmap/utilite/UMath.h>
//#include <rtabmap/utilite/ULogger.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <vector>
#include <cmath>

//namespace ORB_SLAM2 {


class ParticleFilter
{
public:
	ParticleFilter(unsigned int nParticles = 200,
				   double noise = 0.1,
				   double lambda = 10.0,
				   double initValue = 0.0) :
		noise_(noise),
		lambda_(lambda)
	{
		particles_.resize(nParticles, initValue);
	}

	void init(double initValue = 0.0f)
	{
		particles_ = std::vector<double>(particles_.size(), initValue);
	}
	
	// taken from http://www.developpez.net/forums/d544518/c-cpp/c/equivalent-randn-matlab-c/
#define TWOPI (6.2831853071795864769252867665590057683943387987502) /* 2 * pi */
	
	/*
	   RAND is a macro which returns a pseudo-random numbers from a uniform
	   distribution on the interval [0 1]
	*/
	//rand返回从0到最大随机数之间的任意一个数
#define RAND (rand())/((double) RAND_MAX) //返回0~1至今的任意数
	
	/*
	   RANDN is a macro which returns a pseudo-random numbers from a normal
	   distribution with mean zero and standard deviation one. This macro uses Box
	   Muller's algorithm
	*/
#define RANDN (sqrt(-2.0*log(RAND))*cos(TWOPI*RAND))
	
	std::vector<double> cumSum(const std::vector<double> & v)
	{
		std::vector<double> cum(v.size());
		double sum = 0;
		for(unsigned int i=0; i<v.size(); ++i)
		{
			cum[i] = v[i] + sum;
			sum += v[i];
		}
		return cum;
	}
	
	
	/**
	 * Return true if the number is finite.
	 */
	template<class T>
	inline bool uIsFinite(const T & value)
	{
#if _MSC_VER
		return _finite(value) != 0;
#else
		return std::isfinite(value);
#endif
	}
	
	
	/**
	 * Get the sum of all values contained in an array: sum(x).
	 * @param v the array
	 * @param size the size of the array
	 * @return the sum of values of the array
	 */
	template<class T>
	inline T uSum(const T * v, unsigned int size)
	{
		T sum = 0;
		if(v && size)
		{
			for(unsigned int i=0; i<size; ++i)
			{
				sum += v[i];
			}
		}
		return sum;
	}
	
	/**
	 * Get the sum of all values contained in a vector. Provided for convenience.
	 * @param v the vector
	 * @return the sum of values of the vector
	 */
	template<class T>
	inline T uSum(const std::vector<T> & v)
	{
		return uSum(v.data(), (int)v.size());
	}
	
	std::vector<double> resample(const std::vector<double> & p, // particles
								 const std::vector<double> & w, // weights
								 bool normalizeWeights = false)
	{
		std::vector<double> np; //new particles
		if(p.size() != w.size() || p.size() == 0)
		{
			//UERROR("particles (%d) and weights (%d) are not the same size", p.size(), w.size());
			return np;
		}
	
		std::vector<double> cs;
		if(normalizeWeights)
		{
			double wSum = uSum(w);
			std::vector<double> wNorm(w.size());
			for(unsigned int i=0; i<w.size(); ++i)
			{
				wNorm[i] = w[i]/wSum;
			}
			cs = cumSum(wNorm); // cumulative sum
		}
		else
		{
			cs = cumSum(w); // cumulative sum
		}
		
		for(unsigned int j=0; j<cs.size(); ++j)
		{
			cs[j]/=cs.back();
		}
	
		np.resize(p.size());
		for(unsigned int i=0; i<np.size(); ++i)
		{
			unsigned int index = 0;
			double randnum = RAND;
			for(unsigned int j=0; j<cs.size(); ++j)
			{
				if(randnum < cs[j])
				{
					index = j;
					break;
				}
			}
			np[i] = p[index];
		}
		return np;
	}

	double filter(double val)
	{
		std::vector<double> weights(particles_.size(), 1);
		double sumWeights = 0;
		for(unsigned int i=0; i<particles_.size(); ++i)
		{
			// add noise to particle
			particles_[i] += noise_ * RANDN;//假设运动返程是xt = xt-1 + v（噪声）

			// compute weight
			double dist = fabs(particles_[i] - val);
			//dist = sqrt(dist*dist);
			double w = exp(-lambda_*dist);//在高斯分布的假设下
			if(uIsFinite(w) && w > 0)
			{
				weights[i] = w;
			}
			sumWeights += weights[i];
		}


		//normalize and compute estimated value
		double value =0.0;
		for(unsigned int i=0; i<weights.size(); ++i)
		{
			weights[i] /= sumWeights;//归一化权重
			value += weights[i] * particles_[i];//得到估计值
		}

		//resample the particles
		particles_ = resample(particles_, weights, false);

		return value;
	}

private:
	std::vector<double> particles_;
	double noise_;
	double lambda_;
};

//}
#endif /* PARTICLEFILTER_H_ */
