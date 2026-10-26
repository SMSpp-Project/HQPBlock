/*--------------------------------------------------------------------------*/
/*--------------------------- File test_common.h ---------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * What the testers of HQPBlock share: the dense description of a
 * hierarchical QP, its loading into an HQPBlock, the independent reference
 * solution computed level by level in the null space of the equality tasks
 * of the higher levels (with a small dense primal active-set method for the
 * QP of each level), and the run of the HQPSolver of a BlockSolverConfig.
 *
 * The reference follows the classical approach: level p is solved in the
 * variables \f$ ( y , w_p ) \f$, \f$ x = \bar{x}_{p-1} + Z_p y \f$, where
 * the columns of \f$ Z_p \f$ are an orthonormal basis of the null space of
 * the equality tasks of the levels \f$ q < p \f$, as
 *
 * \f[
 *   \min \Big\{ \tfrac{1}{2} \| A_p ( \bar{x}_{p-1} + Z_p y ) - b_p \|^2 +
 *     \tfrac{1}{2} \| w_p \|^2 + \tfrac{\varepsilon}{2} \| y \|^2 +
 *     \tfrac{\rho}{2} \| w_p \|^2 \;:\;
 *     C_q ( \bar{x}_{p-1} + Z_p y ) - d_q \leq w^*_q \;\; q < p \,,\;
 *     C_p ( \bar{x}_{p-1} + Z_p y ) - d_p \leq w_p \,,\; w_p \geq 0
 *   \Big\} \;\;,
 * \f]
 *
 * the weights being already in \f$ A_p \f$, \f$ b_p \f$, \f$ C_p \f$ and
 * \f$ d_p \f$ (or with \f$ \| \bar{x}_{p-1} + Z_p y \|^2 \f$ in place of
 * \f$ \| y \|^2 \f$, i.e., the regularization centerd at 0); \f$ \rho =
 * \varepsilon \f$ gives the problem of the
 * implementation of the robotics literature this approach comes from, \f$
 * \rho = 0 \f$ exactly the problem that an HQPSolver with int_HQP_Prox == 1
 * solves on an HQPBlock with regularization \f$ \varepsilon \f$ (with
 * int_HQP_Prox == 0 for the center at 0).
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 */
/*--------------------------------------------------------------------------*/

#ifndef __HQPBLOCK_TEST_COMMON_H
 #define __HQPBLOCK_TEST_COMMON_H

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "BlockSolverConfig.h"

#include "HQPBlock.h"

#include "HQPSolver.h"

/*--------------------------------------------------------------------------*/

namespace hqp_test {

using namespace SMSpp_di_unipi_it;

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

/*--------------------------------------------------------------------------*/
/// the dense data of a hierarchical QP (empty weights mean all ones)

struct Instance {
 std::string name;
 int n = 0;
 std::vector< Mat > A , C;
 std::vector< Vec > b , d , we , wi;
 double eps = 0;

 [[nodiscard]] int levels( void ) const { return( int( A.size() ) ); }
 };

/*--------------------------------------------------------------------------*/
/// the solution of each level: x after it and its value

struct RefSolution {
 std::vector< Vec > x;
 std::vector< double > value;
 bool ok = true;
 };

/*--------------------------------------------------------------------------*/
/// loads the Instance into the HQPBlock

inline void load( HQPBlock & block , const Instance & in )
{
 std::vector< HQPBlock::SparseMatrix > A , C;
 std::vector< std::vector< double > > b , d , we , wi;
 for( int p = 0 ; p < in.levels() ; ++p ) {
  A.push_back( in.A[ p ].sparseView() );
  C.push_back( in.C[ p ].sparseView() );
  b.emplace_back( in.b[ p ].data() , in.b[ p ].data() + in.b[ p ].size() );
  d.emplace_back( in.d[ p ].data() , in.d[ p ].data() + in.d[ p ].size() );
  we.emplace_back( in.we.empty() ? std::vector< double >() :
                   std::vector< double >( in.we[ p ].data() ,
                          in.we[ p ].data() + in.we[ p ].size() ) );
  wi.emplace_back( in.wi.empty() ? std::vector< double >() :
                   std::vector< double >( in.wi[ p ].data() ,
                          in.wi[ p ].data() + in.wi[ p ].size() ) );
  }
 block.load( in.n , std::move( A ) , std::move( b ) , std::move( C ) ,
             std::move( d ) , std::move( we ) , std::move( wi ) , in.eps );
 }

/*--------------------------------------------------------------------------*/
/// orthonormal basis of the null space of M (the columns of the result)

inline Mat null_space( const Mat & M )
{
 if( M.cols() == 0 )
  return( Mat( 0 , 0 ) );
 if( M.rows() == 0 )
  return( Mat::Identity( M.cols() , M.cols() ) );
 Eigen::JacobiSVD< Mat > svd( M , Eigen::ComputeFullV );
 const auto & s = svd.singularValues();
 const double tol = std::max( M.rows() , M.cols() ) *
  ( s.size() ? s( 0 ) : 0.0 ) * 1e-12;
 int r = 0;
 for( int i = 0 ; i < s.size() ; ++i )
  if( s( i ) > tol )
   ++r;
 return( svd.matrixV().rightCols( M.cols() - r ) );
 }

/*--------------------------------------------------------------------------*/
/// min 1/2 z'Hz + g'z s.t. G z <= h by a primal active-set method
/** H must be positive definite and z feasible on entry; returns false if
 * the iterations run out. */

inline bool active_set_qp( const Mat & H , const Vec & g , const Mat & G ,
                           const Vec & h , Vec & z )
{
 const int nz = z.size();
 std::vector< int > W;
 std::vector< char > inW( G.rows() , 0 );
 for( int it = 0 ; it < 100000 ; ++it ) {
  const int k = W.size();
  Mat K = Mat::Zero( nz + k , nz + k );
  K.topLeftCorner( nz , nz ) = H;
  for( int a = 0 ; a < k ; ++a ) {
   K.block( 0 , nz + a , nz , 1 ) = G.row( W[ a ] ).transpose();
   K.block( nz + a , 0 , 1 , nz ) = G.row( W[ a ] );
   }
  Vec rhs = Vec::Zero( nz + k );
  rhs.head( nz ) = -( H * z + g );
  Vec sol = K.fullPivLu().solve( rhs );
  Vec s = sol.head( nz );
  if( s.norm() <= 1e-13 * ( 1 + z.norm() ) ) {
   int worst = -1;
   double wl = -1e-11;
   for( int a = 0 ; a < k ; ++a )
    if( sol( nz + a ) < wl ) {
     wl = sol( nz + a );
     worst = a;
     }
   if( worst < 0 )
    return( true );
   inW[ W[ worst ] ] = 0;
   W.erase( W.begin() + worst );
   continue;
   }
  double alpha = 1;
  int block = -1;
  for( int i = 0 ; i < G.rows() ; ++i ) {
   if( inW[ i ] )
    continue;
   const double gs = G.row( i ).dot( s );
   if( gs <= 1e-12 * G.row( i ).norm() * s.norm() )
    continue;
   const double a = std::max( 0.0 , ( h( i ) - G.row( i ).dot( z ) ) / gs );
   if( a < alpha ) {
    alpha = a;
    block = i;
    }
   }
  z += alpha * s;
  if( block >= 0 ) {
   W.push_back( block );
   inW[ block ] = 1;
   }
  }
 return( false );
 }

/*--------------------------------------------------------------------------*/
/// the reference solution, level by level in the null space
/** With rho the weight of the regularization of the slacks [see the file
 * comment]; if prox == false the regularization is \f$ \varepsilon \| x
 * \|^2 / 2 \f$ rather than \f$ \varepsilon \| y \|^2 / 2 \f$, i.e., its
 * center is 0 rather than the solution of the previous level. The value of
 * level p is that of its slacks at x after it. */

inline RefSolution reference( const Instance & in , bool prox = true ,
                              double rho = 0 )
{
 RefSolution sol;
 const int n = in.n;
 Vec xbar = Vec::Zero( n );
 Mat Z = Mat::Identity( n , n );
 Mat Cst( 0 , n );   // the C_q of the higher levels, weighted
 Vec dst( 0 );       // d_q + w*_q of the higher levels, weighted

 for( int p = 0 ; p < in.levels() ; ++p ) {
  Mat A = in.A[ p ] , C = in.C[ p ];
  Vec b = in.b[ p ] , d = in.d[ p ];
  if( ! in.we.empty() ) {
   A = in.we[ p ].asDiagonal() * A;
   b = in.we[ p ].cwiseProduct( b );
   }
  if( ! in.wi.empty() ) {
   C = in.wi[ p ].asDiagonal() * C;
   d = in.wi[ p ].cwiseProduct( d );
   }
  const int k = Z.cols() , l = C.rows();

  // variables ( y , w ): objective and constraints
  Mat H = Mat::Zero( k + l , k + l );
  Vec g = Vec::Zero( k + l );
  H.topLeftCorner( k , k ) = ( A * Z ).transpose() * ( A * Z ) +
   in.eps * Mat::Identity( k , k );
  g.head( k ) = ( A * Z ).transpose() * ( A * xbar - b );
  if( ! prox )   // Z' Z = I, the center is 0
   g.head( k ) += in.eps * Z.transpose() * xbar;
  H.bottomRightCorner( l , l ) = ( 1 + rho ) * Mat::Identity( l , l );

  const int mh = Cst.rows();
  Mat G = Mat::Zero( l + mh + l , k + l );
  Vec h = Vec::Zero( l + mh + l );
  G.block( 0 , k , l , l ) = - Mat::Identity( l , l );
  G.block( l , 0 , mh , k ) = Cst * Z;
  h.segment( l , mh ) = dst - Cst * xbar;
  G.block( l + mh , 0 , l , k ) = C * Z;
  G.block( l + mh , k , l , l ) = - Mat::Identity( l , l );
  h.segment( l + mh , l ) = d - C * xbar;

  // a feasible start: y = 0, w at the violation of the level
  Vec z = Vec::Zero( k + l );
  z.tail( l ) = ( C * xbar - d ).cwiseMax( 0.0 );
  h = h.cwiseMax( G * z );   // round-off of the higher levels only

  if( ! active_set_qp( H , g , G , h , z ) )
   sol.ok = false;

  xbar += Z * z.head( k );
  const Vec wst = ( C * xbar - d ).cwiseMax( 0.0 );
  const Vec v = A * xbar - b;
  sol.x.push_back( xbar );
  sol.value.push_back( ( v.squaredNorm() + wst.squaredNorm() ) / 2 );

  Mat Cn( mh + l , n );
  Cn << Cst , C;
  Vec dn( mh + l );
  dn << dst , d + z.tail( l );
  Cst = Cn;
  dst = dn;
  if( A.rows() )
   Z = Z * null_space( A * Z );
  }
 return( sol );
 }

/*--------------------------------------------------------------------------*/
/// relative distance of two values

inline double rel( double a , double b , double floor = 1 )
{
 return( std::abs( a - b ) / std::max( floor , std::abs( b ) ) );
 }

/*--------------------------------------------------------------------------*/
/// the name of the inner Solver of an HQPSolver, with its parameters

inline std::string label( HQPSolver * s )
{
 return( ( s->get_inner_Solver() ? s->get_inner_Solver()->classname()
                                 : std::string( "none" ) ) +
         " prox " + std::to_string( s->get_int_par(
                                      HQPSolver::int_HQP_Prox ) ) +
         " relax " + std::to_string( s->get_int_par(
                                      HQPSolver::int_HQP_RelaxLower ) ) );
 }

/*--------------------------------------------------------------------------*/
/// applies the BlockSolverConfig in file to block, returns its HQPSolver

inline std::vector< HQPSolver * > attach( Block & block ,
                                          const std::string & file )
{
 auto c = Configuration::deserialize( file );
 auto bsc = dynamic_cast< BlockSolverConfig * >( c );
 if( ! bsc ) {
  delete c;
  throw( std::invalid_argument( "attach: " + file + " is not a "
                                "BlockSolverConfig" ) );
  }
 bsc->apply( & block );
 delete bsc;

 std::vector< HQPSolver * > res;
 for( auto s : block.get_registered_solvers() )
  if( auto h = dynamic_cast< HQPSolver * >( s ) )
   res.push_back( h );
 return( res );
 }

/*--------------------------------------------------------------------------*/
/// compares the levels solved by s with ref; returns the number of errors
/** The values of the levels are compared with relative tolerance vtol
 * (absolute below 1) and, if xtol > 0, x after each level with absolute
 * tolerance xtol, relative to max( 1 , | x | ). */

inline int compare( const std::string & what , HQPSolver * s ,
                    const RefSolution & ref , double vtol , double xtol )
{
 int errors = 0;
 const int P = ref.value.size();
 if( int( s->get_num_solved_levels() ) != P ) {
  std::cout << what << " [" << label( s ) << "]: "
            << s->get_num_solved_levels() << " levels solved out of " << P
            << " -> KO" << std::endl;
  return( 1 );
  }
 double dv = 0 , dx = 0;
 for( int p = 0 ; p < P ; ++p ) {
  dv = std::max( dv , rel( s->get_level_value( p ) , ref.value[ p ] ) );
  const auto & x = s->get_level_x( p );
  const Vec xs = Eigen::Map< const Vec >( x.data() , x.size() );
  dx = std::max( dx , ( xs - ref.x[ p ] ).lpNorm< Eigen::Infinity >() /
                      std::max( 1.0 , ref.x[ p ].lpNorm< Eigen::Infinity >()
                                ) );
  }
 const bool ok = ( dv <= vtol ) && ( ( xtol <= 0 ) || ( dx <= xtol ) );
 if( ! ok )
  ++errors;
 std::cout << what << " [" << label( s ) << "]: max value error " << dv
           << ", max x error " << dx << ( ok ? " -> OK" : " -> KO" )
           << std::endl;
 if( ! ok )
  for( int p = 0 ; p < P ; ++p )
   std::cout << "   level " << p << ": " << s->get_level_value( p )
             << " vs " << ref.value[ p ] << std::endl;
 return( errors );
 }

/*--------------------------------------------------------------------------*/

}  // end( namespace hqp_test )

#endif  /* test_common.h included */

/*--------------------------------------------------------------------------*/
/*------------------------ End File test_common.h --------------------------*/
/*--------------------------------------------------------------------------*/
