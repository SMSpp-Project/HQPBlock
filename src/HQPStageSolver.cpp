/*--------------------------------------------------------------------------*/
/*------------------------ File HQPStageSolver.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the HQPStageSolver class, a Solver for one level of an
 * HQPBlock with a stage structure.
 *
 * \author Antonio Frangioni \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 *
 * \copyright &copy; by Antonio Frangioni, Donato Meoli
 */
/*--------------------------------------------------------------------------*/
/*---------------------------- IMPLEMENTATION ------------------------------*/
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <algorithm>

#include <functional>

#include <iomanip>

#include <limits>

#include "HQPStageSolver.h"

#include "LinearFunction.h"

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

using Mat = HQPStageSolver::Mat;
using Vec = HQPStageSolver::Vec;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register HQPStageSolver in the Solver factory

SMSpp_insert_in_factory_cpp_0( HQPStageSolver );

/*--------------------------------------------------------------------------*/
/*--------------------------- LOCAL FUNCTIONS ------------------------------*/
/*--------------------------------------------------------------------------*/

namespace {

using Index = Block::Index;

// the rank-revealing decomposition M = U diag( s ) V^T of an m x n matrix,
// the first rank singular values being those above tol times the largest
// of the largest one and scale (the size of the matrices M comes from)

struct RankRevealed {
 Mat U , V;
 Vec s;
 Index rank = 0;
 };

RankRevealed rank_reveal( const Mat & M , double tol , double scale )
{
 RankRevealed d;
 const Index m = M.rows() , n = M.cols();
 if( ( ! m ) || ( ! n ) ) {
  d.U = Mat::Identity( m , m );
  d.V = Mat::Identity( n , n );
  return( d );
  }

 Eigen::JacobiSVD< Mat > svd( M , Eigen::ComputeFullU | Eigen::ComputeFullV );
 d.U = svd.matrixU();
 d.V = svd.matrixV();
 const Vec & s = svd.singularValues();
 const double t = std::max( tol , std::numeric_limits< double >::epsilon() *
                            double( std::max( m , n ) ) );
 const double ref = t * std::max( s( 0 ) , scale );
 if( s( 0 ) > 0 )
  while( ( d.rank < s.size() ) && ( s( d.rank ) > ref ) )
   ++d.rank;
 d.s = s.head( d.rank );
 return( d );
 }

/*--------------------------------------------------------------------------*/
// the solution X of Z X = Y, Z symmetric positive semidefinite: by Cholesky
// if positive definite, the one of minimum norm otherwise

Mat psd_solve( const Mat & Z , const Mat & Y )
{
 Eigen::LLT< Mat > llt( Z );
 if( llt.info() == Eigen::Success )
  return( llt.solve( Y ) );
 Eigen::CompleteOrthogonalDecomposition< Mat > cod( Z );
 return( cod.solve( Y ) );
 }

/*--------------------------------------------------------------------------*/
// the product of the row h of M with z

double row_dot( const HQPBlock::SparseMatrix & M , Index h , const Vec & z )
{
 double v = 0;
 for( HQPBlock::SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
  v += it.value() * z( it.col() );
 return( v );
 }

/*--------------------------------------------------------------------------*/
// the minimizer on [ 0 , T ] of the convex piecewise quadratic function
// whose derivative is b + a t + sum_i w_i max( 0 , rho_i + sigma_i t )
// sigma_i

double piecewise_min( double a , double b , const std::vector< double > & w ,
                      const std::vector< double > & rho ,
                      const std::vector< double > & sigma , double T )
{
 std::vector< double > bp;
 for( Index i = 0 ; i < rho.size() ; ++i )
  if( sigma[ i ] != 0 ) {
   const double t = - rho[ i ] / sigma[ i ];
   if( ( t > 0 ) && ( t < T ) )
    bp.push_back( t );
   }
 std::sort( bp.begin() , bp.end() );
 bp.push_back( T );

 double ta = 0;
 for( double tb : bp ) {
  if( tb <= ta )
   continue;
  // on ( ta , tb ) the derivative is alpha + beta t
  const double tm = ( ta + tb ) / 2;
  double alpha = b , beta = a;
  for( Index i = 0 ; i < rho.size() ; ++i )
   if( rho[ i ] + sigma[ i ] * tm > 0 ) {
    alpha += w[ i ] * rho[ i ] * sigma[ i ];
    beta += w[ i ] * sigma[ i ] * sigma[ i ];
    }
  if( alpha + beta * tb >= 0 ) {
   if( alpha + beta * ta >= 0 )
    return( ta );
   return( std::clamp( - alpha / beta , ta , tb ) );
   }
  ta = tb;
  }
 return( T );
 }

}  // end( unnamed namespace )

/*--------------------------------------------------------------------------*/
/*---------------------------- StageEQP ------------------------------------*/
/*--------------------------------------------------------------------------*/

void HQPStageSolver::StageEQP::resize( Index N )
{
 A.assign( N , Mat() );
 B.assign( N , Mat() );
 c.assign( N , Vec() );
 S.assign( N , Mat() );
 R.assign( N , Mat() );
 r.assign( N , Vec() );
 Q.assign( N + 1 , Mat() );
 q.assign( N + 1 , Vec() );
 G.assign( N + 1 , Mat() );
 H.assign( N + 1 , Mat() );
 g.assign( N + 1 , Vec() );
 }

/*--------------------------------------------------------------------------*/

double HQPStageSolver::StageEQP::solve( double tol )
{
 const Index N = num_stages();
 v_st.assign( N + 2 , Stage() );
 dropped = to_go = 0;
 double incons = 0;

 Mat P , Gam;
 Vec p , gam;

 // one step of the recursion: the stage with the given blocks, the rows
 // C_x x + C_u u = e; P, p, Gam and gam become those of the stage
 auto step = [ & ]( Stage & st , const Mat & Qxx , const Mat & Quu ,
                    const Mat & Qux , const Vec & qx , const Vec & qu ,
                    const Mat & Cx , const Mat & Cu , const Vec & e ) {
  const Index m = Cu.rows() , nu = Cu.cols();
  // the rows that are 0 on u, or 0 altogether, are so relative to the
  // whole of them
  double scale = 0;
  if( Cx.size() )
   scale = Cx.lpNorm< Eigen::Infinity >();
  if( Cu.size() )
   scale = std::max( scale , Cu.lpNorm< Eigen::Infinity >() );
  auto d = rank_reveal( Cu , tol , scale );
  const Index rk = d.rank;
  const auto U1 = d.U.leftCols( rk );
  const auto U2 = d.U.rightCols( m - rk );
  const auto V1 = d.V.leftCols( rk );
  const auto V2 = d.V.rightCols( nu - rk );
  const Vec sinv = d.s.cwiseInverse();

  // u = Kc x + kc + V2 zeta
  Mat Kc = - V1 * ( sinv.asDiagonal() * ( U1.transpose() * Cx ) );
  Vec kc = V1 * ( sinv.asDiagonal() * ( U1.transpose() * e ) );

  // the rows not involving u, reduced to independent ones
  const Mat M2 = U2.transpose() * Cx;
  const Vec e2 = U2.transpose() * e;
  auto d2 = rank_reveal( M2 , tol , scale );
  const Index r2 = d2.rank , m2 = M2.rows();
  st.T = d2.U.leftCols( r2 ).transpose();
  Gam = st.T * M2;
  gam = st.T * e2;
  if( r2 < m2 )
   incons = std::max( incons , ( d2.U.rightCols( m2 - r2 ).transpose() *
                                 e2 ).cwiseAbs().maxCoeff() );
  dropped += m2 - r2;
  to_go += r2;

  // minimization over zeta
  if( nu > rk ) {
   const Mat Z = V2.transpose() * Quu * V2;
   Mat Y( nu - rk , Kc.cols() + 1 );
   Y.leftCols( Kc.cols() ) = V2.transpose() * ( Quu * Kc + Qux );
   Y.col( Kc.cols() ) = V2.transpose() * ( Quu * kc + qu );
   const Mat X = psd_solve( Z , Y );
   st.K = Kc - V2 * X.leftCols( Kc.cols() );
   st.k = kc - V2 * X.col( Kc.cols() );
   }
  else {
   st.K = std::move( Kc );
   st.k = std::move( kc );
   }

  const Mat QuuK = Quu * st.K;
  P = Qxx + st.K.transpose() * QuuK + st.K.transpose() * Qux +
      Qux.transpose() * st.K;
  P = ( P + P.transpose() ) / 2;
  p = qx + st.K.transpose() * ( Quu * st.k ) + st.K.transpose() * qu +
      Qux.transpose() * st.k;

  st.Quu = Quu;
  st.Qux = Qux;
  st.qu = qu;
  st.U = std::move( d.U );
  st.V = std::move( d.V );
  st.sigma = std::move( d.s );
  };

 // stage N: no input
 {
  const Index nx = Q[ N ].rows();
  v_st[ N ].m = G[ N ].rows();
  step( v_st[ N ] , Q[ N ] , Mat( 0 , 0 ) , Mat( 0 , nx ) , q[ N ] ,
        Vec( 0 ) , G[ N ] , Mat( G[ N ].rows() , 0 ) , g[ N ] );
  }

 for( Index k = N ; k-- > 0 ; ) {
  const Vec pb = p + P * c[ k ];
  const Mat PA = P * A[ k ] , PB = P * B[ k ];
  const Mat Qxx = Q[ k ] + A[ k ].transpose() * PA;
  const Mat Quu = R[ k ] + B[ k ].transpose() * PB;
  const Mat Qux = S[ k ] + B[ k ].transpose() * PA;
  const Vec qx = q[ k ] + A[ k ].transpose() * pb;
  const Vec qu = r[ k ] + B[ k ].transpose() * pb;

  const Index mk = G[ k ].rows() , rg = Gam.rows();
  Mat Cx( mk + rg , A[ k ].cols() ) , Cu( mk + rg , B[ k ].cols() );
  Vec e( mk + rg );
  Cx.topRows( mk ) = G[ k ];
  Cu.topRows( mk ) = H[ k ];
  e.head( mk ) = g[ k ];
  if( rg ) {
   Cx.bottomRows( rg ) = Gam * A[ k ];
   Cu.bottomRows( rg ) = Gam * B[ k ];
   e.tail( rg ) = gam - Gam * c[ k ];
   }
  v_st[ k ].m = mk;
  step( v_st[ k ] , Qxx , Quu , Qux , qx , qu , Cx , Cu , e );
  }

 // x_0 minimizes the cost-to-go on the constraint-to-go
 {
  const Mat P0 = P , G0 = Gam;
  const Vec p0 = p , g0 = gam;
  v_st[ N + 1 ].m = 0;
  step( v_st[ N + 1 ] , Mat( 0 , 0 ) , P0 , Mat( P0.rows() , 0 ) ,
        Vec( 0 ) , p0 , Mat( G0.rows() , 0 ) , G0 , g0 );
  }

 // the multipliers pi of the rows C_x x + C_u u = e of a stage
 auto mult = []( const Stage & st , const Vec & xk , const Vec & uk ,
                 const Vec & nu ) {
  const Index rk = st.sigma.size() , m = st.U.rows();
  Vec pi = st.U.rightCols( m - rk ) * ( st.T.transpose() * nu );
  if( rk ) {
   const Vec gu = st.Quu * uk + st.Qux * xk + st.qu;
   pi -= st.U.leftCols( rk ) * ( ( st.V.leftCols( rk ).transpose() * gu )
                                 .cwiseQuotient( st.sigma ) );
   }
  return( pi );
  };

 // the forward sweep
 x.assign( N + 1 , Vec() );
 u.assign( N , Vec() );
 mu.assign( N + 1 , Vec() );
 x[ 0 ] = v_st[ N + 1 ].k;
 Vec nu = mult( v_st[ N + 1 ] , Vec( 0 ) , x[ 0 ] , Vec( 0 ) );
 for( Index k = 0 ; k < N ; ++k ) {
  const Stage & st = v_st[ k ];
  u[ k ] = st.K * x[ k ] + st.k;
  x[ k + 1 ] = A[ k ] * x[ k ] + B[ k ] * u[ k ] + c[ k ];
  const Vec pi = mult( st , x[ k ] , u[ k ] , nu );
  mu[ k ] = pi.head( st.m );
  nu = pi.tail( pi.size() - st.m );
  }
 mu[ N ] = mult( v_st[ N ] , x[ N ] , Vec( 0 ) , nu );

 return( incons );

 }  // end( HQPStageSolver::StageEQP::solve )

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void HQPStageSolver::set_Block( Block * block )
{
 if( block == f_Block )
  return;

 Solver::set_Block( block );
 f_rebuild = true;
 f_has_sol = false;
 f_status = kUnEval;
 f_ws_level = Inf< Index >();
 v_W.clear();
 v_V.clear();
 }

/*--------------------------------------------------------------------------*/

void HQPStageSolver::set_par( idx_type par , int value )
{
 switch( par ) {
  case( intWarmStart ):
   if( ( value < 0 ) || ( value > 1 ) )
    throw( std::invalid_argument( "HQPStageSolver::set_par: intWarmStart "
                                  "is not 0 or 1" ) );
   f_warm_start = value;
   break;
  case( intMaxIter ):
   f_max_iter = std::max( value , 1 );
   break;
  case( intLogVerb ):
   f_verb = value;
   break;
  default:
   Solver::set_par( par , value );
  }
 }

/*--------------------------------------------------------------------------*/

void HQPStageSolver::set_par( idx_type par , double value )
{
 if( ( par != dblRankTol ) && ( par != dblFeasTol ) && ( par != dblDualTol ) )
 {
  Solver::set_par( par , value );
  return;
  }
 if( ! ( value >= 0 ) )
  throw( std::invalid_argument( "HQPStageSolver::set_par: negative " +
                                dbl_par_idx2str( par ) ) );
 if( par == dblRankTol )
  f_rank_tol = value;
 else
  if( par == dblFeasTol )
   f_feas_tol = value;
  else
   f_dual_tol = value;
 }

/*--------------------------------------------------------------------------*/

void HQPStageSolver::add_Modification( sp_Mod & mod )
{
 if( f_no_Mod )
  return;

 std::function< void( const Modification * ) > scan =
  [ & ]( const Modification * m ) {
  if( dynamic_cast< const NBModification * >( m ) ) {
   f_rebuild = true;
   f_ws_level = Inf< Index >();
   v_W.clear();
   v_V.clear();
   return;
   }
  if( auto g = dynamic_cast< const GroupModification * >( m ) ) {
   for( const auto & s : g->sub_Modifications() )
    scan( s.get() );
   return;
   }
  // only the coefficients of the rows, not those of the Objective
  Function * f = nullptr;
  if( auto fm = dynamic_cast< const FunctionMod * >( m ) )
   f = fm->function();
  else
   if( auto fv = dynamic_cast< const FunctionModVars * >( m ) )
    f = fv->function();
  if( dynamic_cast< LinearFunction * >( f ) )
   f_rebuild = true;
  };

 scan( mod.get() );
 }

/*--------------------------------------------------------------------------*/
/*------------------------- THE STAGE STRUCTURE ----------------------------*/
/*--------------------------------------------------------------------------*/

void HQPStageSolver::build_stages( const HQPBlock & hb )
{
 f_N = hb.get_num_stages();
 f_dyn = hb.get_dynamics_level();
 const Index n = hb.get_num_var() , N = f_N;
 const auto & vs = hb.get_var_stage();
 const auto & vt = hb.get_var_state();

 v_ns.assign( N + 1 , 0 );
 v_nu.assign( N , 0 );
 v_pos.resize( n );
 for( Index j = 0 ; j < n ; ++j )
  v_pos[ j ] = vt[ j ] ? v_ns[ vs[ j ] ]++ : v_nu[ vs[ j ] ]++;

 const std::string where = "HQPStageSolver::compute: ";
 const std::string dl = "the level " + std::to_string( f_dyn ) +
                        " of the dynamics";
 if( hb.get_num_ineq_rows( f_dyn ) )
  throw( std::invalid_argument( where + dl + " has inequality rows" ) );

 // the dynamics: each row of the level is s_{k+1,i} out of s_k and u_k
 v_A.resize( N );
 v_B.resize( N );
 v_dyn_row.assign( N , {} );
 v_dyn_piv.assign( N , {} );
 for( Index k = 0 ; k < N ; ++k ) {
  v_A[ k ] = Mat::Zero( v_ns[ k + 1 ] , v_ns[ k ] );
  v_B[ k ] = Mat::Zero( v_ns[ k + 1 ] , v_nu[ k ] );
  v_dyn_row[ k ].assign( v_ns[ k + 1 ] , Inf< Index >() );
  v_dyn_piv[ k ].assign( v_ns[ k + 1 ] , 0 );
  }

 const auto & AD = hb.get_A( f_dyn );
 for( Index h = 0 ; h < Index( AD.rows() ) ; ++h ) {
  const std::string bad = where + "row " + std::to_string( h ) + " of " +
                          dl;
  Index top = 0 , cnt = 0 , piv = 0;
  for( HQPBlock::SparseMatrix::InnerIterator it( AD , h ) ; it ; ++it ) {
   const Index j = it.col();
   if( ( it.value() == 0 ) || ( ! vt[ j ] ) )
    continue;
   if( vs[ j ] > top ) {
    top = vs[ j ];
    cnt = 1;
    piv = j;
    }
   else
    if( vs[ j ] == top )
     ++cnt;
   }
  if( ( ! top ) || ( cnt != 1 ) )
   throw( std::invalid_argument( bad + " has not one entry on the state of "
                                 "its stage" ) );
  const Index k = top - 1 , i = v_pos[ piv ];
  if( v_dyn_row[ k ][ i ] != Inf< Index >() )
   throw( std::invalid_argument( bad + " gives a state that another row "
                                 "gives" ) );
  double a = 0;
  for( HQPBlock::SparseMatrix::InnerIterator it( AD , h ) ; it ; ++it )
   if( Index( it.col() ) == piv )
    a += it.value();
  for( HQPBlock::SparseMatrix::InnerIterator it( AD , h ) ; it ; ++it ) {
   const Index j = it.col();
   if( ( j == piv ) || ( it.value() == 0 ) )
    continue;
   if( vs[ j ] != k )
    throw( std::invalid_argument( bad + " has an entry out of its "
                                  "stages" ) );
   if( vt[ j ] )
    v_A[ k ]( i , v_pos[ j ] ) -= it.value() / a;
   else
    v_B[ k ]( i , v_pos[ j ] ) -= it.value() / a;
   }
  v_dyn_row[ k ][ i ] = h;
  v_dyn_piv[ k ][ i ] = a;
  }
 for( Index k = 0 ; k < N ; ++k )
  for( auto r : v_dyn_row[ k ] )
   if( r == Inf< Index >() )
    throw( std::invalid_argument( where + "a state of stage " +
                                  std::to_string( k + 1 ) + " has no row "
                                  "in " + dl ) );

 // the dynamics of the augmented state x_k = ( s_k , u_{k-1} )
 f_eqp.resize( N );
 for( Index k = 0 ; k < N ; ++k ) {
  const Index nx = v_ns[ k ] + ( k ? v_nu[ k - 1 ] : 0 );
  const Index nx1 = v_ns[ k + 1 ] + v_nu[ k ];
  f_eqp.A[ k ] = Mat::Zero( nx1 , nx );
  f_eqp.A[ k ].topLeftCorner( v_ns[ k + 1 ] , v_ns[ k ] ) = v_A[ k ];
  f_eqp.B[ k ] = Mat::Zero( nx1 , v_nu[ k ] );
  f_eqp.B[ k ].topRows( v_ns[ k + 1 ] ) = v_B[ k ];
  f_eqp.B[ k ].bottomRows( v_nu[ k ] ).setIdentity();
  }
 v_Bhat = f_eqp.B;

 // all the rows but those of the dynamics, on their stages
 const auto & es = hb.get_eq_stage();
 const auto & is = hb.get_ineq_stage();
 v_eq_rows.assign( hb.get_eq_start( hb.get_num_levels() ) , StageRow() );
 v_in_rows.assign( hb.get_ineq_start( hb.get_num_levels() ) , StageRow() );
 for( Index q = 0 ; q < hb.get_num_levels() ; ++q ) {
  if( q != f_dyn )
   for( Index h = 0 ; h < hb.get_num_eq_rows( q ) ; ++h ) {
    const Index i = hb.get_eq_start( q ) + h;
    v_eq_rows[ i ] = stage_row( hb.get_A( q ) , h ,
                                es.empty() ? N : es[ i ] );
    }
  for( Index h = 0 ; h < hb.get_num_ineq_rows( q ) ; ++h ) {
   const Index i = hb.get_ineq_start( q ) + h;
   v_in_rows[ i ] = stage_row( hb.get_C( q ) , h ,
                               is.empty() ? N : is[ i ] );
   }
  }
 }  // end( HQPStageSolver::build_stages )

/*--------------------------------------------------------------------------*/

HQPStageSolver::StageRow HQPStageSolver::stage_row(
                         const HQPBlock::SparseMatrix & M , Index h ,
                         Index given_stage ) const
{
 const auto & hb = *f_hb;
 const auto & vs = hb.get_var_stage();
 const auto & vt = hb.get_var_state();

 StageRow sr;
 Index k = 0;
 if( given_stage < f_N )
  k = given_stage;
 else
  for( HQPBlock::SparseMatrix::InnerIterator it( M , h ) ; it ; ++it ) {
   if( it.value() == 0 )
    continue;
   const Index j = it.col();
   const Index cand = vt[ j ] ? ( vs[ j ] ? vs[ j ] - 1 : 0 ) : vs[ j ];
   k = std::max( k , cand );
   }

 sr.stage = k;
 const Index ns = v_ns[ k ] , nup = k ? v_nu[ k - 1 ] : 0;
 sr.delta = Eigen::RowVectorXd::Zero( ns + nup );
 sr.eta = Eigen::RowVectorXd::Zero( v_nu[ k ] );
 sr.alpha = Eigen::RowVectorXd::Zero( v_ns[ k + 1 ] );
 for( HQPBlock::SparseMatrix::InnerIterator it( M , h ) ; it ; ++it ) {
  if( it.value() == 0 )
   continue;
  sr.empty = false;
  const Index j = it.col() , st = vs[ j ] , ps = v_pos[ j ];
  const double v = it.value();
  if( ! vt[ j ] ) {
   if( st == k )
    sr.eta( ps ) += v;
   else
    if( k && ( st + 1 == k ) )
     sr.delta( ns + ps ) += v;
    else
     sr.fits = false;
   }
  else
   if( st == k + 1 )
    sr.alpha( ps ) += v;
   else
    if( st == k )
     sr.delta( ps ) += v;
    else
     sr.fits = false;
  }
 sr.delta.head( ns ) += sr.alpha * v_A[ k ];
 sr.eta += sr.alpha * v_B[ k ];
 return( sr );
 }

/*--------------------------------------------------------------------------*/

void HQPStageSolver::to_columns( Vec & z ) const
{
 const auto & vs = f_hb->get_var_stage();
 const auto & vt = f_hb->get_var_state();
 z.resize( v_pos.size() );
 for( Index j = 0 ; j < v_pos.size() ; ++j )
  z( j ) = vt[ j ] ? f_eqp.x[ vs[ j ] ]( v_pos[ j ] ) :
                     f_eqp.u[ vs[ j ] ]( v_pos[ j ] );
 }

/*--------------------------------------------------------------------------*/
/*------------------------- THE SUBPROBLEMS --------------------------------*/
/*--------------------------------------------------------------------------*/

void HQPStageSolver::solve_eqp( const std::vector< char > & work ,
                                const std::vector< char > & viol ,
                                Vec & zbar )
{
 const auto & hb = *f_hb;
 const Index N = f_N , n = hb.get_num_var();
 auto & E = f_eqp;

 // at the level of the dynamics their slacks v are inputs too, s_{k+1} =
 // A_k s_k + B_k u_k + c_k + diag( v_dyn_inv[ k ] ) v_k, with cost |v|^2 / 2
 const bool slk = ( f_p == f_dyn );
 for( Index k = 0 ; k <= N ; ++k ) {
  const Index nx = v_ns[ k ] + ( k ? v_nu[ k - 1 ] : 0 );
  E.Q[ k ] = Mat::Zero( nx , nx );
  E.q[ k ] = Vec::Zero( nx );
  if( k < N ) {
   const Index nv = slk ? v_ns[ k + 1 ] : 0 , nu = v_nu[ k ] + nv;
   E.R[ k ] = Mat::Zero( nu , nu );
   E.R[ k ].bottomRightCorner( nv , nv ).setIdentity();
   E.S[ k ] = Mat::Zero( nu , nx );
   E.r[ k ] = Vec::Zero( nu );
   E.c[ k ] = Vec::Zero( v_ns[ k + 1 ] + v_nu[ k ] );
   E.c[ k ].head( v_ns[ k + 1 ] ) = v_c[ k ];
   E.B[ k ] = Mat::Zero( v_Bhat[ k ].rows() , nu );
   E.B[ k ].leftCols( v_nu[ k ] ) = v_Bhat[ k ];
   if( nv )
    E.B[ k ].block( 0 , v_nu[ k ] , nv , nv ) =
                                             v_dyn_inv[ k ].asDiagonal();
   }
  }

 // the regularization
 const double eps = hb.get_regularization();
 const auto & rw = hb.get_regularization_weights();
 const auto & ctr = hb.get_center();
 const auto & vs = hb.get_var_stage();
 const auto & vt = hb.get_var_state();
 for( Index j = 0 ; j < n ; ++j ) {
  const double w = eps * ( rw.empty() ? 1.0 : rw[ j ] );
  const double cj = ctr.empty() ? 0.0 : ctr[ j ];
  const Index k = vs[ j ] , ps = v_pos[ j ];
  if( vt[ j ] ) {
   E.Q[ k ]( ps , ps ) += w;
   E.q[ k ]( ps ) -= w * cj;
   }
  else {
   E.R[ k ]( ps , ps ) += w;
   E.r[ k ]( ps ) -= w * cj;
   }
  }

 // the tasks of the level: the equality ones and the violated inequality
 // ones, 1/2 w^2 ( delta x + eta u + alpha c - rhs )^2
 auto add_cost = [ & ]( const StageRow & sr , double w , double rhs ) {
  if( sr.empty || ( w == 0 ) )
   return;
  const Index k = sr.stage;
  const double w2 = w * w;
  const double res = ( sr.alpha.size() ?
                       sr.alpha.transpose().dot( v_c[ k ] ) : 0.0 ) - rhs;
  E.Q[ k ].noalias() += w2 * sr.delta.transpose() * sr.delta;
  const Index nu = v_nu[ k ];
  E.S[ k ].topRows( nu ).noalias() += w2 * sr.eta.transpose() * sr.delta;
  E.R[ k ].topLeftCorner( nu , nu ).noalias() +=
                                        w2 * sr.eta.transpose() * sr.eta;
  E.q[ k ] += ( w2 * res ) * sr.delta.transpose();
  E.r[ k ].head( nu ) += ( w2 * res ) * sr.eta.transpose();
  };

 if( f_p != f_dyn ) {
  const Index e0 = hb.get_eq_start( f_p );
  for( Index h = 0 ; h < hb.get_num_eq_rows( f_p ) ; ++h )
   add_cost( v_eq_rows[ e0 + h ] , hb.get_we()[ e0 + h ] ,
             hb.get_b()[ e0 + h ] );
  for( Index t = 0 ; t < v_task_in.size() ; ++t )
   if( viol[ t ] ) {
    const Index i = v_task_in[ t ];
    add_cost( v_in_rows[ i ] , hb.get_wi()[ i ] , hb.get_d()[ i ] );
    }
  }

 // the hard rows of the working set, stage by stage
 v_eqp_rows.assign( N + 1 , {} );
 v_eqp_scale.assign( N + 1 , {} );
 for( Index t = 0 ; t < v_hard.size() ; ++t ) {
  if( ! work[ t ] )
   continue;
  const auto & sr = v_hard[ t ].eq ? v_eq_rows[ v_hard[ t ].i ] :
                                      v_in_rows[ v_hard[ t ].i ];
  if( ! sr.empty )
   v_eqp_rows[ sr.stage ].push_back( t );
  }
 for( Index k = 0 ; k <= N ; ++k ) {
  const Index m = v_eqp_rows[ k ].size();
  const Index nx = E.Q[ k ].rows() , nu = k < N ? E.R[ k ].rows() : 0;
  E.G[ k ].resize( m , nx );
  E.H[ k ].resize( m , nu );
  E.g[ k ].resize( m );
  v_eqp_scale[ k ].resize( m );
  for( Index h = 0 ; h < m ; ++h ) {
   const auto & hr = v_hard[ v_eqp_rows[ k ][ h ] ];
   const auto & sr = hr.eq ? v_eq_rows[ hr.i ] : v_in_rows[ hr.i ];
   // each row of largest coefficient 1
   double sc = 0;
   if( sr.delta.size() )
    sc = sr.delta.lpNorm< Eigen::Infinity >();
   if( sr.eta.size() )
    sc = std::max( sc , sr.eta.lpNorm< Eigen::Infinity >() );
   if( sc == 0 )
    sc = 1;
   v_eqp_scale[ k ][ h ] = sc;
   E.G[ k ].row( h ) = sr.delta / sc;
   if( k < N ) {
    E.H[ k ].row( h ).head( v_nu[ k ] ) = sr.eta / sc;
    if( slk )
     E.H[ k ].row( h ).tail( v_ns[ k + 1 ] ) =
             sr.alpha.cwiseProduct( v_dyn_inv[ k ].transpose() ) / sc;
    }
   E.g[ k ]( h ) = ( hr.rhs - ( sr.alpha.size() ?
                     sr.alpha.transpose().dot( v_c[ k ] ) : 0.0 ) ) / sc;
   }
  }

 // the smallest rank tolerance, from dblRankTol up by factors of 100 to
 // 1e-6, whose solution satisfies the rows of the working set to
 // dblFeasTol: rows that are dependent but for rounding give a subproblem
 // so ill-conditioned that its solution does not
 double tol = f_rank_tol , incons = 0;
 for( ; ; ) {
  incons = E.solve( tol );
  to_columns( zbar );
  bool ok = true;
  double worst = 0;
  for( Index t = 0 ; t < v_hard.size() ; ++t )
   if( work[ t ] ) {
    const double r = std::abs( hard_slack( t , zbar ) ) /
                     ( 1 + std::abs( v_hard[ t ].rhs ) );
    worst = std::max( worst , r );
    if( r > f_feas_tol )
     ok = false;
    }
  if( f_log && ( f_verb > 2 ) )
   *f_log << "HQPStageSolver: rank tolerance " << tol << " residual "
          << worst << " objective " << level_objective( zbar ) << std::endl;
  if( ok || ( tol >= 1e-6 ) )
   break;
  tol = std::max( 100 * tol , 1e-14 );
  }
 f_incons = std::max( f_incons , incons );
 ++f_iter;
 }

/*--------------------------------------------------------------------------*/

double HQPStageSolver::hard_slack( Index t , const Vec & z ) const
{
 const auto & hr = v_hard[ t ];
 const auto & M = hr.eq ? f_hb->get_A( hr.q ) : f_hb->get_C( hr.q );
 return( hr.rhs - row_dot( M , hr.h , z ) );
 }

/*--------------------------------------------------------------------------*/

double HQPStageSolver::level_objective( const Vec & z ) const
{
 const auto & hb = *f_hb;
 double val = 0;

 const auto & A = hb.get_A( f_p );
 const Index e0 = hb.get_eq_start( f_p );
 for( Index h = 0 ; h < Index( A.rows() ) ; ++h ) {
  const double v = hb.get_we()[ e0 + h ] *
                   ( row_dot( A , h , z ) - hb.get_b()[ e0 + h ] );
  val += v * v;
  }
 const auto & C = hb.get_C( f_p );
 const Index i0 = hb.get_ineq_start( f_p );
 for( Index h = 0 ; h < Index( C.rows() ) ; ++h ) {
  const double w = std::max( 0.0 , hb.get_wi()[ i0 + h ] *
                             ( row_dot( C , h , z ) - hb.get_d()[ i0 + h ] ) );
  val += w * w;
  }

 const double eps = hb.get_regularization();
 const auto & rw = hb.get_regularization_weights();
 const auto & ctr = hb.get_center();
 double reg = 0;
 for( Index j = 0 ; j < z.size() ; ++j ) {
  const double dj = z( j ) - ( ctr.empty() ? 0.0 : ctr[ j ] );
  reg += ( rw.empty() ? 1.0 : rw[ j ] ) * dj * dj;
  }

 return( ( val + eps * reg ) / 2 );
 }

/*--------------------------------------------------------------------------*/
/*--------------------- METHODS FOR SOLVING THE MODEL ----------------------*/
/*--------------------------------------------------------------------------*/

int HQPStageSolver::compute( bool changedvars )
{
 lock();

 f_has_sol = false;
 f_warm = false;
 f_iter = 0;
 f_incons = 0;
 f_status = kError;

 auto hb = dynamic_cast< HQPBlock * >( f_Block );
 if( ! hb ) {
  unlock();
  throw( std::invalid_argument( "HQPStageSolver::compute: the Block is not "
                                "an HQPBlock" ) );
  }

 const bool owned = hb->is_owned_by( f_id );
 if( ( ! owned ) && ( ! hb->read_lock() ) ) {
  unlock();
  return( f_status = kBlockLocked );
  }

 // the locks are released also when the HQPBlock does not fit
 struct Release {
  HQPStageSolver * s;
  HQPBlock * b;
  bool owned;
  ~Release() {
   if( ! owned )
    b->read_unlock();
   s->unlock();
   }
  } release{ this , hb , owned };

 const std::string where = "HQPStageSolver::compute: ";
 if( ! hb->get_num_stages() )
  throw( std::invalid_argument( where + "the HQPBlock has no stage data" ) );
 f_hb = hb;
 f_p = hb->get_level();
 if( f_p < hb->get_dynamics_level() )
  throw( std::invalid_argument( where + "level " + std::to_string( f_p ) +
                                " is above the level " +
                                std::to_string( hb->get_dynamics_level() ) +
                                " of the dynamics" ) );
 if( hb->get_x().empty() )
  throw( std::logic_error( where + "the Variable are not there" ) );

 if( f_rebuild ) {
  build_stages( *hb );
  f_rebuild = false;
  }

 for( Index q = 0 ; q <= f_p ; ++q ) {
  auto check = [ & ]( const std::vector< StageRow > & rows , Index first ,
                      Index m , const char * kind ) {
   for( Index h = 0 ; h < m ; ++h )
    if( ! rows[ first + h ].fits )
     throw( std::invalid_argument( where + kind + " row " +
                                   std::to_string( h ) + " of level " +
                                   std::to_string( q ) + " has an entry "
                                   "out of its stage" ) );
   };
  if( q != f_dyn )
   check( v_eq_rows , hb->get_eq_start( q ) , hb->get_num_eq_rows( q ) ,
          "equality" );
  check( v_in_rows , hb->get_ineq_start( q ) , hb->get_num_ineq_rows( q ) ,
         "inequality" );
  }

 const auto & b = hb->get_b();
 const auto & d = hb->get_d();
 const auto & we = hb->get_we();
 const auto & wi = hb->get_wi();
 const auto & vv = hb->get_v();
 const auto & ww = hb->get_w();

 // the c_k of the dynamics, whose slacks are fixed below their level
 const Index N = f_N , e0 = hb->get_eq_start( f_dyn );
 v_c.resize( N );
 v_dyn_inv.resize( N );
 for( Index k = 0 ; k < N ; ++k ) {
  v_c[ k ].resize( v_ns[ k + 1 ] );
  v_dyn_inv[ k ].resize( v_ns[ k + 1 ] );
  for( Index i = 0 ; i < v_ns[ k + 1 ] ; ++i ) {
   const Index row = e0 + v_dyn_row[ k ][ i ];
   if( we[ row ] == 0 )
    throw( std::invalid_argument( where + "a row of the dynamics has "
                                  "weight 0" ) );
   double rhs = b[ row ];
   if( f_p > f_dyn )
    rhs += vv[ row ].get_value() / we[ row ];
   v_c[ k ]( i ) = rhs / v_dyn_piv[ k ][ i ];
   v_dyn_inv[ k ]( i ) = 1 / ( we[ row ] * v_dyn_piv[ k ][ i ] );
   }
  }

 // the hard rows: those of the levels above, moved by their fixed slacks
 v_hard.clear();
 const Index nin = hb->get_ineq_start( hb->get_num_levels() );
 std::vector< Index > hard_of( nin , Inf< Index >() );
 for( Index q = 0 ; q < f_p ; ++q ) {
  if( q != f_dyn )
   for( Index h = 0 ; h < hb->get_num_eq_rows( q ) ; ++h ) {
    const Index i = hb->get_eq_start( q ) + h;
    if( we[ i ] != 0 )
     v_hard.push_back( { true , q , h , i ,
                         b[ i ] + vv[ i ].get_value() / we[ i ] } );
    }
  for( Index h = 0 ; h < hb->get_num_ineq_rows( q ) ; ++h ) {
   const Index i = hb->get_ineq_start( q ) + h;
   if( wi[ i ] != 0 ) {
    hard_of[ i ] = v_hard.size();
    v_hard.push_back( { false , q , h , i ,
                        d[ i ] + ww[ i ].get_value() / wi[ i ] } );
    }
   }
  }

 // the inequality tasks of the level
 v_task_in.clear();
 std::vector< Index > task_of( nin , Inf< Index >() );
 if( f_p != f_dyn )
  for( Index h = 0 ; h < hb->get_num_ineq_rows( f_p ) ; ++h ) {
   const Index i = hb->get_ineq_start( f_p ) + h;
   task_of[ i ] = v_task_in.size();
   v_task_in.push_back( i );
   }

 // the starting point and the slacks of the hard rows there
 Vec z( hb->get_num_var() );
 for( Index j = 0 ; j < z.size() ; ++j )
  z( j ) = hb->get_x()[ j ].get_value();

 auto hard_row = [ & ]( Index t ) -> const HQPBlock::SparseMatrix & {
  return( v_hard[ t ].eq ? hb->get_A( v_hard[ t ].q ) :
                           hb->get_C( v_hard[ t ].q ) );
  };
 auto slack = [ & ]( Index t , const Vec & x ) {
  return( hard_slack( t , x ) );
  };
 auto tol_of = [ & ]( Index t ) {
  return( f_feas_tol * ( 1 + std::abs( v_hard[ t ].rhs ) ) );
  };
 auto violated_tasks = [ & ]( const Vec & x ) {
  std::vector< char > viol( v_task_in.size() , 0 );
  for( Index t = 0 ; t < v_task_in.size() ; ++t ) {
   const Index i = v_task_in[ t ];
   const double r = row_dot( hb->get_C( f_p ) ,
                             i - hb->get_ineq_start( f_p ) , x ) - d[ i ];
   viol[ t ] = ( r > 1e-15 * ( 1 + std::abs( d[ i ] ) ) );
   }
  return( viol );
  };
 // a hard inequality row not in the working set that is violated at x
 auto infeasible = [ & ]( const Vec & x , const std::vector< char > & work ) {
  for( Index t = 0 ; t < v_hard.size() ; ++t )
   if( ( ! v_hard[ t ].eq ) && ( ! work[ t ] ) &&
       ( slack( t , x ) < - tol_of( t ) ) )
    return( true );
  return( false );
  };

 std::vector< char > work( v_hard.size() , 0 );
 bool start_violated = false;
 for( Index t = 0 ; t < v_hard.size() ; ++t )
  if( v_hard[ t ].eq )
   work[ t ] = 1;
  else {
   const double s = slack( t , z );
   if( s <= tol_of( t ) )
    work[ t ] = 1;
   if( s < - tol_of( t ) )
    start_violated = true;
   }

 Vec zbar;
 double incons_tol = 0;
 for( const auto & hr : v_hard )
  incons_tol = std::max( incons_tol , std::abs( hr.rhs ) );
 incons_tol = f_feas_tol * ( 1 + incons_tol );

 // the working set and the violated tasks of the previous compute(); if
 // the subproblem they give is solved by a feasible point that violates the
 // same tasks, the first iteration starts there with its solution
 bool reuse = false;
 if( ( f_warm_start == 1 ) && ( f_ws_level == f_p ) ) {
  auto wtry = work;
  for( auto i : v_W )
   if( ( i < nin ) && ( hard_of[ i ] != Inf< Index >() ) )
    wtry[ hard_of[ i ] ] = 1;
  std::vector< char > vtry( v_task_in.size() , 0 );
  for( auto i : v_V )
   if( ( i < nin ) && ( task_of[ i ] != Inf< Index >() ) )
    vtry[ task_of[ i ] ] = 1;
  const double inc0 = f_incons;
  f_incons = 0;
  solve_eqp( wtry , vtry , zbar );
  if( ( f_incons <= incons_tol ) && ( ! infeasible( zbar , wtry ) ) ) {
   z = zbar;
   work = std::move( wtry );
   f_warm = true;
   start_violated = false;
   reuse = ( violated_tasks( z ) == vtry );
   }
  f_incons = std::max( inc0 , f_incons );
  }

 // the minimizer along d = zbar - z on [ 0 , T ] of the objective of the
 // level, and its derivative at 0
 const auto & Ap = hb->get_A( f_p );
 const auto & Cp = hb->get_C( f_p );
 const Index pe = hb->get_eq_start( f_p ) , pi = hb->get_ineq_start( f_p );
 const double eps = hb->get_regularization();
 const auto & rw = hb->get_regularization_weights();
 const auto & ctr = hb->get_center();
 auto line = [ & ]( const Vec & dz , double T , double & slope0 ) {
  double a = 0 , bb = 0;
  for( Index h = 0 ; h < Index( Ap.rows() ) ; ++h ) {
   const double w2 = we[ pe + h ] * we[ pe + h ];
   const double rho = row_dot( Ap , h , z ) - b[ pe + h ];
   const double sig = row_dot( Ap , h , dz );
   a += w2 * sig * sig;
   bb += w2 * rho * sig;
   }
  for( Index j = 0 ; j < z.size() ; ++j ) {
   const double r = eps * ( rw.empty() ? 1.0 : rw[ j ] );
   a += r * dz( j ) * dz( j );
   bb += r * ( z( j ) - ( ctr.empty() ? 0.0 : ctr[ j ] ) ) * dz( j );
   }
  Vec_double w( Cp.rows() ) , rho( Cp.rows() ) , sig( Cp.rows() );
  slope0 = bb;
  for( Index h = 0 ; h < Index( Cp.rows() ) ; ++h ) {
   w[ h ] = wi[ pi + h ] * wi[ pi + h ];
   rho[ h ] = row_dot( Cp , h , z ) - d[ pi + h ];
   sig[ h ] = row_dot( Cp , h , dz );
   if( rho[ h ] > 0 )
    slope0 += w[ h ] * rho[ h ] * sig[ h ];
   }
  return( piecewise_min( a , bb , w , rho , sig , T ) );
  };

 // the projection steps: the rows violated at the start are in the working
 // set, and the steps are taken in full, up to the first blocking row
 int status = kOK;
 for( Index it = 0 ; start_violated ; ++it ) {
  if( ( f_iter >= f_max_iter ) || ( it > v_hard.size() + 1 ) ) {
   status = kError;
   break;
   }
  solve_eqp( work , violated_tasks( z ) , zbar );
  const Vec dz = zbar - z;
  double T = 1;
  Index blk = Inf< Index >();
  for( Index t = 0 ; t < v_hard.size() ; ++t ) {
   if( work[ t ] )
    continue;
   // a row blocks if the whole step would violate it by more than half
   // its tolerance, so that a row dependent on the working set, which the
   // step moves only by rounding, does not
   const double s = row_dot( hard_row( t ) , v_hard[ t ].h , dz );
   const double sl = slack( t , z );
   if( ( s > 0 ) && ( sl - s < - tol_of( t ) / 2 ) ) {
    const double tt = std::max( 0.0 , sl ) / s;
    if( tt < T ) {
     T = tt;
     blk = t;
     }
    }
   }
  z += T * dz;
  if( blk != Inf< Index >() )
   work[ blk ] = 1;
  else {
   start_violated = false;
   for( Index t = 0 ; t < v_hard.size() ; ++t )
    if( ( ! v_hard[ t ].eq ) && work[ t ] &&
        ( slack( t , z ) < - tol_of( t ) ) )
     start_violated = true;
   if( start_violated && ( f_incons > incons_tol ) ) {
    status = kError;
    break;
    }
   }
  }

 // the primal active-set iterations
 std::vector< char > viol;
 while( status == kOK ) {
  if( f_iter >= f_max_iter ) {
   status = kStopIter;
   break;
   }

  viol = violated_tasks( z );
  if( reuse )
   zbar = z;
  else
   solve_eqp( work , viol , zbar );
  reuse = false;
  const Vec dz = zbar - z;

  double T = 1;
  Index blk = Inf< Index >();
  for( Index t = 0 ; t < v_hard.size() ; ++t ) {
   if( work[ t ] )
    continue;
   // a row blocks if the whole step would violate it by more than half
   // its tolerance, so that a row dependent on the working set, which the
   // step moves only by rounding, does not
   const double s = row_dot( hard_row( t ) , v_hard[ t ].h , dz );
   const double sl = slack( t , z );
   if( ( s > 0 ) && ( sl - s < - tol_of( t ) / 2 ) ) {
    const double tt = std::max( 0.0 , sl ) / s;
    if( tt < T ) {
     T = tt;
     blk = t;
     }
    }
   }

  double slope0 = 0;
  const double tstar = line( dz , T , slope0 );
  const double phi = level_objective( z );
  // z solves the subproblem if the step is 0; a step that is not a descent
  // one, which only rounding in an ill-conditioned subproblem gives (z is on
  // its rows), is taken to its end, so that the multipliers are read at the
  // point they belong to
  const bool still = ( dz.lpNorm< Eigen::Infinity >() <=
                       1e-12 * ( 1 + z.lpNorm< Eigen::Infinity >() ) );
  const bool flat = ( slope0 >= - 1e-14 * ( std::abs( phi ) + 1e-30 ) );

  if( f_log && ( f_verb > 1 ) ) {
   double wres = 0 , wz = 0 , vz = 0;
   for( Index t = 0 ; t < v_hard.size() ; ++t )
    if( work[ t ] ) {
     wres = std::max( wres , std::abs( slack( t , zbar ) ) );
     wz = std::max( wz , std::abs( slack( t , z ) ) );
     }
    else
     if( ! v_hard[ t ].eq )
      vz = std::max( vz , - slack( t , z ) );
   *f_log << "HQPStageSolver: level " << f_p << " it " << f_iter
          << " |W| " << std::count( work.begin() , work.end() , 1 )
          << " |V| " << std::count( viol.begin() , viol.end() , 1 )
          << " |d| " << dz.lpNorm< Eigen::Infinity >() << " step " << tstar
          << " slope " << slope0 << " dropped " << f_eqp.dropped
          << " to go " << f_eqp.to_go << " W residual " << wres
          << " at z " << wz << " violation " << vz;
   if( blk != Inf< Index >() )
    *f_log << " blocked by " << v_hard[ blk ].i << " at " << T;
   *f_log << std::endl;
   }

  if( still ) {
   // the most negative multiplier of an inequality row of the working set
   double mmin = - f_dual_tol;
   Index out = Inf< Index >();
   for( Index k = 0 ; k <= N ; ++k )
    for( Index h = 0 ; h < v_eqp_rows[ k ].size() ; ++h ) {
     const Index t = v_eqp_rows[ k ][ h ];
     const double mu = f_eqp.mu[ k ]( h ) / v_eqp_scale[ k ][ h ];
     if( ( ! v_hard[ t ].eq ) && ( mu < mmin ) ) {
      mmin = mu;
      out = t;
      }
     }
   if( out == Inf< Index >() )
    break;
   if( f_log && ( f_verb > 1 ) )
    *f_log << "HQPStageSolver: " << v_hard[ out ].i << " leaves, multiplier "
           << mmin << std::endl;
   work[ out ] = 0;
   continue;
   }

  const double t = flat ? T : tstar;
  z += t * dz;
  if( ( blk != Inf< Index >() ) && ( t >= T ) )
   work[ blk ] = 1;
  }

 f_status = status;
 f_has_sol = ( status == kOK ) || ( status == kStopIter );
 if( f_has_sol ) {
  f_z = z;
  f_sol_level = f_p;
  double fixed = 0;
  for( Index q = 0 ; q < f_p ; ++q ) {
   for( Index h = 0 ; h < hb->get_num_eq_rows( q ) ; ++h ) {
    const double v = vv[ hb->get_eq_start( q ) + h ].get_value();
    fixed += v * v;
    }
   for( Index h = 0 ; h < hb->get_num_ineq_rows( q ) ; ++h ) {
    const double w = ww[ hb->get_ineq_start( q ) + h ].get_value();
    fixed += w * w;
    }
   }
  f_value = hb->get_objective_scale() * ( fixed / 2 + level_objective( z ) );

  v_W.clear();
  for( Index t = 0 ; t < v_hard.size() ; ++t )
   if( ( ! v_hard[ t ].eq ) && work[ t ] )
    v_W.push_back( v_hard[ t ].i );
  v_V.clear();
  viol = violated_tasks( z );
  for( Index t = 0 ; t < v_task_in.size() ; ++t )
   if( viol[ t ] )
    v_V.push_back( v_task_in[ t ] );
  f_ws_level = f_p;
  }

 if( f_log && f_verb )
  *f_log << "HQPStageSolver: level " << f_p << " status " << f_status
         << " iterations " << f_iter << ( f_warm ? " (warm)" : "" )
         << " |W| " << v_W.size() << " |V| " << v_V.size()
         << " inconsistency " << std::setprecision( 3 ) << f_incons
         << " value " << std::setprecision( 12 ) << f_value << std::endl;

 return( f_status );

 }  // end( HQPStageSolver::compute )

/*--------------------------------------------------------------------------*/
/*---------------------- METHODS FOR READING RESULTS -----------------------*/
/*--------------------------------------------------------------------------*/

void HQPStageSolver::get_var_solution( Configuration * solc )
{
 auto hb = dynamic_cast< HQPBlock * >( f_Block );
 if( ( ! hb ) || ( ! f_has_sol ) || ( f_z.size() != hb->get_num_var() ) )
  return;

 hb->set_x_values( Vec_double( f_z.data() , f_z.data() + f_z.size() ) );

 const Index p = f_sol_level;
 const auto & A = hb->get_A( p );
 const auto & C = hb->get_C( p );
 const Index e0 = hb->get_eq_start( p ) , i0 = hb->get_ineq_start( p );
 Vec_double v( A.rows() ) , w( C.rows() );
 for( Index h = 0 ; h < v.size() ; ++h )
  v[ h ] = hb->get_we()[ e0 + h ] *
           ( row_dot( A , h , f_z ) - hb->get_b()[ e0 + h ] );
 for( Index h = 0 ; h < w.size() ; ++h )
  w[ h ] = std::max( 0.0 , hb->get_wi()[ i0 + h ] *
                     ( row_dot( C , h , f_z ) - hb->get_d()[ i0 + h ] ) );
 hb->set_slack_values( p , v , w );
 }

/*--------------------------------------------------------------------------*/
/*---------------------- End File HQPStageSolver.cpp -----------------------*/
/*--------------------------------------------------------------------------*/
