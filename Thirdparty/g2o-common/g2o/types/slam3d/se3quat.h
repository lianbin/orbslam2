// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
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

#ifndef G2O_SE3QUAT_H_
#define G2O_SE3QUAT_H_

#include "se3_ops.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {

  typedef Eigen::Matrix<double, 6, 1, Eigen::ColMajor> Vector6d;
  typedef Eigen::Matrix<double, 7, 1, Eigen::ColMajor> Vector7d;

  class G2O_TYPES_SLAM3D_API SE3Quat {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    protected:

      Eigen::Quaterniond _r;
      Vector3D _t;

    public:
      SE3Quat(){
        _r.setIdentity();
        _t.setZero();
      }
      //使用旋转矩阵+平移初始化位姿
      SE3Quat(const Matrix3D& R, const Vector3D& t):_r(Eigen::Quaterniond(R)),_t(t){ 
        normalizeRotation();
      }

      SE3Quat(const Eigen::Quaterniond& q, const Vector3D& t):_r(q),_t(t){
        normalizeRotation();
      }

      /**
       * templaized constructor which allows v to be an arbitrary Eigen Vector type, e.g., Vector6d or Map<Vector6d>
       */
      template <typename Derived>
        explicit SE3Quat(const Eigen::MatrixBase<Derived>& v)
        {
          assert((v.size() == 6 || v.size() == 7) && "Vector dimension does not match");
          if (v.size() == 6) {
            for (int i=0; i<3; i++){
              _t[i]=v[i];
              _r.coeffs()(i)=v[i+3];
            }
            _r.w() = 0.; // recover the positive w
            if (_r.norm()>1.){
              _r.normalize();
            } else {
              double w2=1.-_r.squaredNorm();
              _r.w()= (w2<0.) ? 0. : sqrt(w2);
            }
          }
          else if (v.size() == 7) {
            int idx = 0;
            for (int i=0; i<3; ++i, ++idx)
              _t(i) = v(idx);
            for (int i=0; i<4; ++i, ++idx)
              _r.coeffs()(i) = v(idx);
            normalizeRotation();
          }
        }

      inline const Vector3D& translation() const {return _t;}

      inline void setTranslation(const Vector3D& t_) {_t = t_;}

      inline const Eigen::Quaterniond& rotation() const {return _r;}

      void setRotation(const Eigen::Quaterniond& r_) {_r=r_;}
      /*
	  |R1 t1||R2 t2|  
      |0   1||0   1|=

	  =|R1R2 R1t2+t1|
	   | 0      1   |
     */
      inline SE3Quat operator* (const SE3Quat& tr2) const{
        SE3Quat result(*this);
		//eigen中重载了*号运算，用于使用四元数旋转一个向量
		//，直接相乘就可以。相当于李群中的Rp.
		//在数学中，p'=qpq^-1 ，表示四元数对向量进行旋转
        result._t += _r*tr2._t;
		//两个四元素直接相乘，结果还是一个四元数。
		//等同于两个旋转矩阵直接相乘
        result._r*=tr2._r;
		//归一化
        result.normalizeRotation();
        return result;
      }

      inline SE3Quat& operator*= (const SE3Quat& tr2){
        _t+=_r*tr2._t;
        _r*=tr2._r;
        normalizeRotation();
        return *this;
      }

      inline Vector3D operator* (const Vector3D& v) const {
        return _t+_r*v;
      }

      inline SE3Quat inverse() const{
        SE3Quat ret;
        ret._r=_r.conjugate();
        ret._t=ret._r*(_t*-1.);
        return ret;
      }

      inline double operator [](int i) const {
        assert(i<7);
        if (i<3)
          return _t[i];
        return _r.coeffs()[i-3];
      }


      inline Vector7d toVector() const{
        Vector7d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        v[6]=_r.w();
        return v;
      }

      inline void fromVector(const Vector7d& v){
        _r=Eigen::Quaterniond(v[6], v[3], v[4], v[5]);
        _t=Vector3D(v[0], v[1], v[2]);
      }

      inline Vector6d toMinimalVector() const{
        Vector6d v;
        v[0]=_t(0);
        v[1]=_t(1);
        v[2]=_t(2);
        v[3]=_r.x();
        v[4]=_r.y();
        v[5]=_r.z();
        return v;
      }

      inline void fromMinimalVector(const Vector6d& v){
        double w = 1.-v[3]*v[3]-v[4]*v[4]-v[5]*v[5];
        if (w>0){
          _r=Eigen::Quaterniond(sqrt(w), v[3], v[4], v[5]);
        } else {
          _r=Eigen::Quaterniond(0, -v[3], -v[4], -v[5]);
        }
        _t=Vector3D(v[0], v[1], v[2]);
      }


	  //SE3到的se3对数映射
      Vector6d log() const {
        Vector6d res;
		//四元素->旋转矩阵
        Matrix3D _R = _r.toRotationMatrix();
		//θ=arccos(tr(R)-1)/2) 
		//这里求tr(R)-1)/2
        double d =  0.5*(_R(0,0)+_R(1,1)+_R(2,2)-1);
        Vector3D omega;
        Vector3D upsilon;


        Vector3D dR = deltaR(_R);
        Matrix3D V_inv;

        if (d>0.99999)
        {

          omega=0.5*dR;
          Matrix3D Omega = skew(omega);
          V_inv = Matrix3D::Identity()- 0.5*Omega + (1./12.)*(Omega*Omega);
        }
        else
        {
          //得到旋转角θ
          double theta = acos(d);
          //求解旋转向量
          omega = theta/(2*sqrt(1-d*d))*dR;
          Matrix3D Omega = skew(omega);
		  //求解J^-1
          V_inv = ( Matrix3D::Identity() - 0.5*Omega
              + ( 1-theta/(2*tan(theta/2)))/(theta*theta)*(Omega*Omega) );
        }

        upsilon = V_inv*_t;
        for (int i=0; i<3;i++){
          res[i]=omega[i];
        }
        for (int i=0; i<3;i++){
          res[i+3]=upsilon[i];
        }

        return res;

      }
      //世界坐标->相机坐标系坐标
      //P'=RP+t
      Vector3D map(const Vector3D & xyz) const
      {
        return _r*xyz + _t;
      }

      //se3到SE3的指数映射
      static SE3Quat exp(const Vector6d & update)
      {
      
	  //旋转在前 平移在后
        Vector3D omega;
        for (int i=0; i<3; i++)
          omega[i]=update[i];//旋转向量
        Vector3D upsilon;
        for (int i=0; i<3; i++)
          upsilon[i]=update[i+3];//平移量(不是真的平移量，是李代数中的平移量
                                 //  ρ，与实际的平移量相差一个乘子J.)

        double theta = omega.norm(); //旋转角
        Matrix3D Omega = skew(omega);//旋转向量的反对称矩阵

        Matrix3D R;
        Matrix3D V;
        if (theta<0.00001)
        {
          //TODO: CHECK WHETHER THIS IS CORRECT!!!
          R = (Matrix3D::Identity() + Omega + Omega*Omega);

          V = R;
        }
        else
        {
          Matrix3D Omega2 = Omega*Omega;
          //so3部分的指数映射->SO3
          R = (Matrix3D::Identity()
              + sin(theta)/theta *Omega
              + (1-cos(theta))/(theta*theta)*Omega2);
          //se3->SE3映射，t=Jρ，这里计算的是J。
          V = (Matrix3D::Identity()
              + (1-cos(theta))/(theta*theta)*Omega
              + (theta-sin(theta))/(pow(theta,3))*Omega2);
        }
        return SE3Quat(Eigen::Quaterniond(R),V*upsilon);
      }

      Eigen::Matrix<double, 6, 6, Eigen::ColMajor> adj() const
      {
        Matrix3D R = _r.toRotationMatrix();
        Eigen::Matrix<double, 6, 6, Eigen::ColMajor> res;
        res.block(0,0,3,3) = R;
        res.block(3,3,3,3) = R;
        res.block(3,0,3,3) = skew(_t)*R;
        res.block(0,3,3,3) = Matrix3D::Zero(3,3);
        return res;
      }

      Eigen::Matrix<double,4,4,Eigen::ColMajor> to_homogeneous_matrix() const
      {
        Eigen::Matrix<double,4,4,Eigen::ColMajor> homogeneous_matrix;
        homogeneous_matrix.setIdentity();
        homogeneous_matrix.block(0,0,3,3) = _r.toRotationMatrix();
        homogeneous_matrix.col(3).head(3) = translation();

        return homogeneous_matrix;
      }
      //q=[cosθ/2,x,y,z],w是实部，如果w<0,说明角度大于180
      //所以直接取负号，相当于θ+2π，而不改变四元素
      void normalizeRotation(){
        if (_r.w()<0){
          _r.coeffs() *= -1;
        }
		//单位四元素才能表示旋转
        _r.normalize();
      }

      /**
       * cast SE3Quat into an Isometry3D
       */
      operator Isometry3D() const
      {
        Isometry3D result = (Isometry3D) rotation();
        result.translation() = translation();
        return result;
      }
  };

  inline std::ostream& operator <<(std::ostream& out_str, const SE3Quat& se3)
  {
    out_str << se3.to_homogeneous_matrix()  << std::endl;
    return out_str;
  }

  //G2O_TYPES_SLAM3D_API Eigen::Quaterniond euler_to_quat(double yaw, double pitch, double roll);
  //G2O_TYPES_SLAM3D_API void quat_to_euler(const Eigen::Quaterniond& q, double& yaw, double& pitch, double& roll);
  //G2O_TYPES_SLAM3D_API void jac_quat3_euler3(Eigen::Matrix<double, 6, 6, Eigen::ColMajor>& J, const SE3Quat& t);

} // end namespace

#endif
