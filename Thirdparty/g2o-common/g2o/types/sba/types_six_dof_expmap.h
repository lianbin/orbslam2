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

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/se3_ops.h"
#include "types_sba.h"
#include <Eigen/Geometry>

namespace g2o {
namespace types_six_dof_expmap {
void init();
}

typedef Eigen::Matrix<double, 6, 6, Eigen::ColMajor> Matrix6d;

class G2O_TYPES_SBA_API CameraParameters : public g2o::Parameter
{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    CameraParameters();

    CameraParameters(double focal_length,
        const Vector2D & principle_point,
        double baseline)
      : focal_length(focal_length),
      principle_point(principle_point),
      baseline(baseline){}

    Vector2D cam_map (const Vector3D & trans_xyz) const;

    Vector3D stereocam_uvu_map (const Vector3D & trans_xyz) const;

    virtual bool read (std::istream& is){
      is >> focal_length;
      is >> principle_point[0];
      is >> principle_point[1];
      is >> baseline;
      return true;
    }

    virtual bool write (std::ostream& os) const {
      os << focal_length << " ";
      os << principle_point.x() << " ";
      os << principle_point.y() << " ";
      os << baseline << " ";
      return true;
    }

    double focal_length;
    Vector2D principle_point;
    double baseline;
};

/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */

//李代数位姿，6维表达。内部采用四元数的数据结构
class G2O_TYPES_SBA_API VertexSE3Expmap : public BaseVertex<6, SE3Quat>{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }
  //在基类Vertex中，是纯虚函数，在重新定义继承类顶点时
  //必须重新定义此函数，用以更新参数
  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);
	//使用扰动(左乘)，得到微小变化之后的位姿
    setEstimate(SE3Quat::exp(update)*estimate());
  }
};


/**
 * \brief 6D edge between two Vertex6
 */
class G2O_TYPES_SBA_API EdgeSE3Expmap : public BaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      EdgeSE3Expmap();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError()  {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
      const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[1]);

      SE3Quat C(_measurement);
      SE3Quat error_= v2->estimate().inverse()*C*v1->estimate();
      _error = error_.log();
    }

    virtual void linearizeOplus();
};

//继承的是两元边BaseBinaryEdge，每个边连接两个顶点
//3D->2D投影方程边。2维 使用Vector2D类型表示，
//VertexSBAPointXYZ为边对应的0编号的顶点数据类型，VertexSE3Expmap对应编号为1的顶点数据类型
class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZ2UV();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;
    //计算本条边的误差
    void computeError()  {
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
      Vector2D obs(_measurement);//观测
      //实际测量-投影                        //P'=RP+t
      _error = obs-cam->cam_map(v1->estimate().map(v2->estimate()));
    }
    //求误差函数的雅克比矩阵
    //Jacobin函数可以不定义，g2o会采用数值导数方法计算雅克比
    //在这里进行重新定义，用以计算解析导数。(如果能够计算解析导数的时候，应该优先考虑解析导数)
    virtual void linearizeOplus();

    CameraParameters * _cam;
};


class G2O_TYPES_SBA_API EdgeProjectPSI2UV : public  g2o::BaseMultiEdge<2, Vector2D>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectPSI2UV()  {
    resizeParameters(1);
    installParameter(_cam, 0);
  }

  virtual bool read  (std::istream& is);
  virtual bool write (std::ostream& os) const;
  void computeError  ();
  virtual void linearizeOplus ();
  CameraParameters * _cam;
};



//Stereo Observations
// U: left u
// V: left v
// U: right u
class G2O_TYPES_SBA_API EdgeProjectXYZ2UVU : public  BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZ2UVU();

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError(){
      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
      const CameraParameters * cam
        = static_cast<const CameraParameters *>(parameter(0));
      Vector3D obs(_measurement);
      _error = obs-cam->stereocam_uvu_map(v1->estimate().map(v2->estimate()));
    }
    //  virtual void linearizeOplus();
    CameraParameters * _cam;
};

} // end namespace

#endif
