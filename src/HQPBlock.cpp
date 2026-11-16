/*--------------------------------------------------------------------------*/
/*--------------------------- File HQPBlock.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the HQPBlock class, which implements the Block concept
 * [see Block.h] for a hierarchical (lexicographic) Quadratic Program.
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

#include "HQPBlock.h"

#include <algorithm>

#include <cmath>

#include <limits>

#include <numeric>

#include <set>

#include "ColVariableSolution.h"

#include "SMSnetCDF.h"

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register HQPBlock in the Block factory

SMSpp_insert_in_factory_cpp_1( HQPBlock );

/*--------------------------------------------------------------------------*/
/*--------------------------- LOCAL FUNCTIONS ------------------------------*/
/*--------------------------------------------------------------------------*/

namespace {

// reads the rows of all the levels of one kind (equality or inequality) out
// of the netCDF group, as described in HQPBlock::deserialize(); if !
// levels, the hard rows of one kind, as the only level P = 1 with neither
// the variable of the rows of each level nor the weights

void read_rows( const netCDF::NcGroup & group , const std::string & pfx ,
                std::size_t n , std::size_t P ,
                std::vector< HQPBlock::SparseMatrix > & M ,
                std::vector< std::vector< double > > & rhs ,
                std::vector< std::vector< double > > & wgh ,
                bool levels = true )
{
 M.assign( P , HQPBlock::SparseMatrix() );
 rhs.assign( P , {} );
 wgh.assign( P , {} );

 std::size_t m = 0;
 ::deserialize_dim( group , "Num" + pfx + "Rows" , m );
 if( ! m ) {
  for( auto & Mp : M )
   Mp.resize( 0 , n );
  return;
  }

 std::vector< unsigned int > rows( 1 , m );
 if( levels )
  ::deserialize( group , pfx + "Rows" , P , rows , false );
 std::size_t nnz = 0;
 ::deserialize_dim( group , "Num" + pfx + "Elements" , nnz );
 std::vector< unsigned int > start , index;
 std::vector< double > value , r , w;
 ::deserialize( group , pfx + "Start" , m , start , false );
 if( nnz ) {
  ::deserialize( group , pfx + "Index" , nnz , index , false );
  ::deserialize( group , pfx + "Value" , nnz , value , false );
  }
 ::deserialize( group , pfx + "RHS" , m , r , false );
 if( levels )
  ::deserialize( group , pfx + "Weight" , m , w , true );

 if( std::accumulate( rows.begin() , rows.end() , std::size_t( 0 ) ) != m )
  throw( std::invalid_argument( "HQPBlock::deserialize: the sum of " + pfx +
                                "Rows is not Num" + pfx + "Rows" ) );

 for( std::size_t i = 0 , p = 0 , first = 0 ; p < P ; ++p ) {
  std::vector< Eigen::Triplet< double > > trp;
  for( std::size_t h = 0 ; h < rows[ p ] ; ++h , ++i ) {
   const std::size_t end = ( i + 1 < m ) ? start[ i + 1 ] : nnz;
   if( ( start[ i ] > end ) || ( end > nnz ) )
    throw( std::invalid_argument( "HQPBlock::deserialize: " + pfx +
                                  "Start is not non-decreasing" ) );
   for( std::size_t k = start[ i ] ; k < end ; ++k ) {
    if( index[ k ] >= n )
     throw( std::invalid_argument( "HQPBlock::deserialize: " + pfx +
                                   "Index out of NumVar" ) );
    trp.emplace_back( h , index[ k ] , value[ k ] );
    }
   }
  M[ p ].resize( rows[ p ] , n );
  M[ p ].setFromTriplets( trp.begin() , trp.end() );
  rhs[ p ].assign( r.begin() + first , r.begin() + first + rows[ p ] );
  if( ! w.empty() )
   wgh[ p ].assign( w.begin() + first , w.begin() + first + rows[ p ] );
  first += rows[ p ];
  }
 }

/*--------------------------------------------------------------------------*/
// writes the rows of all the levels of one kind (equality or inequality)
// into the netCDF group, as described in HQPBlock::deserialize(); with no
// levels, the hard rows of one kind (M of one element), with no variable of
// the rows of each level

void write_rows( netCDF::NcGroup & group , const std::string & pfx ,
                 const netCDF::NcDim * levels ,
                 const std::vector< HQPBlock::SparseMatrix > & M ,
                 const std::vector< double > & rhs ,
                 const std::vector< double > & wgh )
{
 if( rhs.empty() )
  return;

 std::vector< unsigned int > rows , start , index;
 std::vector< double > value;
 for( const auto & Mp : M ) {
  rows.push_back( Mp.rows() );
  for( Eigen::Index i = 0 ; i < Mp.rows() ; ++i ) {
   start.push_back( index.size() );
   for( HQPBlock::SparseMatrix::InnerIterator it( Mp , i ) ; it ; ++it ) {
    index.push_back( it.col() );
    value.push_back( it.value() );
    }
   }
  }

 auto dm = group.addDim( "Num" + pfx + "Rows" , rhs.size() );
 if( levels )
  group.addVar( pfx + "Rows" , netCDF::NcUint() , *levels ).putVar(
                                                               rows.data() );
 group.addVar( pfx + "Start" , netCDF::NcUint() , dm ).putVar(
                                                              start.data() );
 if( ! index.empty() ) {
  auto de = group.addDim( "Num" + pfx + "Elements" , index.size() );
  group.addVar( pfx + "Index" , netCDF::NcUint() , de ).putVar(
                                                              index.data() );
  group.addVar( pfx + "Value" , netCDF::NcDouble() , de ).putVar(
                                                              value.data() );
  }
 group.addVar( pfx + "RHS" , netCDF::NcDouble() , dm ).putVar( rhs.data() );
 if( std::any_of( wgh.begin() , wgh.end() ,
                  []( double x ) { return( x != 1 ); } ) )
  group.addVar( pfx + "Weight" , netCDF::NcDouble() , dm ).putVar(
                                                               wgh.data() );
 }

/*--------------------------------------------------------------------------*/
// checks the stage structure of a hierarchy of n variables, P levels, me
// equality and mi inequality rows, as described in HQPBlock::StageData

void check_stages( const HQPBlock::StageData & st , std::size_t n ,
		   std::size_t P , std::size_t me , std::size_t mi )
{
 auto fail = []( const std::string & what ) {
  throw( std::invalid_argument( "HQPBlock::guts_of_load: " + what ) );
  };

 const auto N = st.num_stages;
 if( ! N ) {
  if( ! ( st.var_stage.empty() && st.var_state.empty() &&
	  st.eq_stage.empty() && st.ineq_stage.empty() &&
	  ( st.dynamics_level == 0 ) ) )
   fail( "stage data with no stages" );
  return;
  }

 if( ( st.var_stage.size() != n ) || ( st.var_state.size() != n ) )
  fail( "VarStage or VarState has not n entries" );
 for( std::size_t j = 0 ; j < n ; ++j ) {
  if( st.var_state[ j ] > 1 )
   fail( "VarState of column " + std::to_string( j ) + " is not 0 or 1" );
  if( st.var_stage[ j ] + ( st.var_state[ j ] ? 0 : 1 ) > N )
   fail( "column " + std::to_string( j ) + " out of the stages" );
  }

 for( int k = 0 ; k < 2 ; ++k ) {
  const auto & rs = k ? st.ineq_stage : st.eq_stage;
  if( rs.empty() )
   continue;
  if( rs.size() != ( k ? mi : me ) )
   fail( std::string( k ? "IneqStage" : "EqStage" ) + " has not one entry "
						      "for each row" );
  if( std::any_of( rs.begin() , rs.end() ,
		   [ N ]( auto r ) { return( r >= N ); } ) )
   fail( std::string( k ? "IneqStage" : "EqStage" ) + " out of the "
						      "stages" );
  }

 if( st.dynamics_level >= P )
  fail( "DynamicsLevel is not a level" );
 }

/*--------------------------------------------------------------------------*/
// the nonlinear hard rows, on the matrices of HQPBlock::HardRows

using SpMat = Eigen::SparseMatrix< double , Eigen::RowMajor >;
using Vec_double = std::vector< double >;

// the product of the row i of M with x

double row_dot( const SpMat & M , Eigen::Index i , const Vec_double & x )
{
 double v = 0;
 for( SpMat::InnerIterator it( M , i ) ; it ; ++it )
  v += it.value() * x[ it.col() ];
 return( v );
 }

/*--------------------------------------------------------------------------*/
// g += a times the row i of M

void add_row( const SpMat & M , Eigen::Index i , double a , Vec_double & g )
{
 for( SpMat::InnerIterator it( M , i ) ; it ; ++it )
  g[ it.col() ] += a * it.value();
 }

/*--------------------------------------------------------------------------*/

double dot( const Vec_double & a , const Vec_double & b )
{
 return( std::inner_product( a.begin() , a.end() , b.begin() , 0.0 ) );
 }

/*--------------------------------------------------------------------------*/
// 1/2 x' Q x + c' x, c the row i of Cq and Q its lower triangle, and in *g
// (if not nullptr) its gradient Q x + c

double quad_value( const SpMat & Q , const SpMat & Cq , Eigen::Index i ,
		   const Vec_double & x , Vec_double * g )
{
 double v = row_dot( Cq , i , x );
 if( g ) {
  g->assign( x.size() , 0 );
  add_row( Cq , i , 1 , *g );
  }
 for( Eigen::Index j = 0 ; j < Q.rows() ; ++j )
  for( SpMat::InnerIterator it( Q , j ) ; it ; ++it ) {
   const Eigen::Index k = it.col();
   if( k == j ) {
    v += it.value() * x[ j ] * x[ j ] / 2;
    if( g )
     ( *g )[ j ] += it.value() * x[ j ];
    continue;
    }
   v += it.value() * x[ j ] * x[ k ];
   if( g ) {
    ( *g )[ j ] += it.value() * x[ k ];
    ( *g )[ k ] += it.value() * x[ j ];
    }
   }
 return( v );
 }

/*--------------------------------------------------------------------------*/
// log sum_k c_k exp( a_k' x + b_k ), a_k the rows of A, computed shifted by
// the largest term, and in *g (if not nullptr) its gradient A' sigma, sigma_k
// being the term k divided by the sum; - infinity if A has no rows

double lse_value( const SpMat & A , const Vec_double & b ,
                  const Vec_double & c , const Vec_double & x ,
                  Vec_double * g )
{
 if( g )
  g->assign( x.size() , 0 );
 const Eigen::Index K = A.rows();
 Vec_double t( K );
 double m = - std::numeric_limits< double >::infinity();
 for( Eigen::Index k = 0 ; k < K ; ++k ) {
  t[ k ] = row_dot( A , k , x ) + b[ k ] + std::log( c[ k ] );
  if( std::isnan( t[ k ] ) )
   return( t[ k ] );
  m = std::max( m , t[ k ] );
  }
 if( ! std::isfinite( m ) )
  return( m );

 double s = 0;
 for( auto & tk : t )
  s += ( tk = std::exp( tk - m ) );
 if( g )
  for( Eigen::Index k = 0 ; k < K ; ++k )
   add_row( A , k , t[ k ] / s , *g );
 return( m + std::log( s ) );
 }

/*--------------------------------------------------------------------------*/
// || z ||_p, z = A x + b and p >= 1 finite, computed scaled by the largest
// | z_k |, and in *g (if not nullptr) A' w and in *w (if not nullptr) w,
// the vector of the dual norm with w' z = || z ||_p, i.e., w_k = sign( z_k )
// ( | z_k | / || z ||_p )^( p - 1 ), and w = 0 if z = 0

double pnorm_value( const SpMat & A , const Vec_double & b , double p ,
		    const Vec_double & x , Vec_double * g , Vec_double * w )
{
 const Eigen::Index K = A.rows();
 if( g )
  g->assign( x.size() , 0 );
 if( w )
  w->assign( K , 0 );
 Vec_double z( K );
 double mz = 0;
 for( Eigen::Index k = 0 ; k < K ; ++k ) {
  z[ k ] = row_dot( A , k , x ) + b[ k ];
  if( std::isnan( z[ k ] ) )
   return( z[ k ] );
  mz = std::max( mz , std::abs( z[ k ] ) );
  }
 if( mz == 0 )
  return( 0 );

 double nrm = 0;
 if( p == 1 )
  for( auto zk : z )
   nrm += std::abs( zk );
 else {
  for( auto zk : z )
   nrm += std::pow( std::abs( zk ) / mz , p );
  nrm = mz * std::pow( nrm , 1 / p );
  }

 if( g || w )
  for( Eigen::Index k = 0 ; k < K ; ++k ) {
   if( z[ k ] == 0 )
    continue;
   const double wk = std::copysign( ( p == 1 ) ? 1.0 :
                                    std::pow( std::abs( z[ k ] ) / nrm ,
                                              p - 1 ) , z[ k ] );
   if( g )
    add_row( A , k , wk , *g );
   if( w )
    ( *w )[ k ] = wk;
   }
 return( nrm );
 }

/*--------------------------------------------------------------------------*/
// the end hi of a bracket [ lo , hi ] of the root in ( 0 , 1 ) of the convex
// phi, phi( 0 ) = f0 < 0 < phi( 1 ) = f1, with 0 <= phi( hi ) <= ftol, by
// regula falsi with the Illinois correction, bisecting when the step falls
// out of the bracket or phi is NaN (taken as positive); at most 200 steps

template< class F >
double boundary( const F & phi , double f0 , double f1 , double ftol )
{
 const double inf = std::numeric_limits< double >::infinity();
 double lo = 0 , hi = 1 , wlo = f0 , whi = f1 , fhi = f1;
 int side = 0;
 for( int it = 0 ; ( it < 200 ) && ( fhi > ftol ) && ( hi - lo > 1e-15 ) ;
      ++it ) {
  double t = ( lo * whi - hi * wlo ) / ( whi - wlo );
  if( ! ( ( t > lo ) && ( t < hi ) ) )
   t = ( lo + hi ) / 2;
  const double ft = phi( t );
  if( ! ( ft < 0 ) ) {
   hi = t;
   fhi = whi = std::isnan( ft ) ? inf : ft;
   if( side > 0 )
    wlo /= 2;
   side = 1;
   }
  else {
   lo = t;
   wlo = ft;
   if( side < 0 )
    whi /= 2;
   side = -1;
   }
  }
 return( hi );
 }

/*--------------------------------------------------------------------------*/
// the tolerance in the Configuration c, a SimpleConfiguration< double > or
// a SimpleConfiguration< std::pair< double , int > >, which also gives the
// int; false if c is neither

bool extract_pair( Configuration * c , double & tol , int & bits )
{
 if( auto tc = dynamic_cast< SimpleConfiguration< double > * >( c ) ) {
  tol = tc->f_value;
  return( true );
  }
 if( auto tc = dynamic_cast< SimpleConfiguration< std::pair< double ,
                                                  int > > * >( c ) ) {
  tol = tc->f_value.first;
  bits = tc->f_value.second;
  return( true );
  }
 return( false );
 }

/*--------------------------------------------------------------------------*/

}  // end( unnamed namespace )

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void HQPBlock::load( Index n , std::vector< SparseMatrix > && A ,
		     std::vector< Vec_double > && b ,
		     std::vector< SparseMatrix > && C ,
		     std::vector< Vec_double > && d ,
		     std::vector< Vec_double > && we ,
		     std::vector< Vec_double > && wi , double eps ,
		     Vec_double && rw , StageData && stages ,
		     HardRows && hard )
{
 guts_of_load( n , std::move( A ) , std::move( b ) , std::move( C ) ,
	       std::move( d ) , std::move( we ) , std::move( wi ) , eps ,
	       std::move( rw ) , std::move( stages ) , std::move( hard ) );

 // whoever is listening has to find the abstract representation there
 if( anyone_there() ) {
  generate_abstract_variables();
  generate_abstract_constraints();
  generate_objective();
  add_Modification( std::make_shared< NBModification >( this ) );
  }
 }  // end( HQPBlock::load( memory ) )

/*--------------------------------------------------------------------------*/

void HQPBlock::guts_of_load( Index n , std::vector< SparseMatrix > && A ,
			     std::vector< Vec_double > && b ,
			     std::vector< SparseMatrix > && C ,
			     std::vector< Vec_double > && d ,
			     std::vector< Vec_double > && we ,
			     std::vector< Vec_double > && wi , double eps ,
			     Vec_double && rw , StageData && stages ,
			     HardRows && hard )
{
 const Index P = A.size();
 if( ( ! n ) || ( ! P ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: no variables or "
                                "no levels" ) );
 if( ( b.size() != P ) || ( C.size() != P ) || ( d.size() != P ) ||
     ( ( ! we.empty() ) && ( we.size() != P ) ) ||
     ( ( ! wi.empty() ) && ( wi.size() != P ) ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: the vectors of "
                                "the levels have different sizes" ) );
 if( ! ( eps >= 0 ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: negative "
                                "regularization" ) );
 if( ( ! rw.empty() ) && ( rw.size() != n ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: wrong size of "
                                "the weights of the regularization" ) );
 if( std::any_of( rw.begin() , rw.end() ,
                  []( double r ) { return( ! ( r > 0 ) ); } ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: a weight of the "
                                "regularization is not positive" ) );

 for( Index p = 0 ; p < P ; ++p ) {
  if( ( Index( A[ p ].cols() ) != n ) || ( Index( C[ p ].cols() ) != n ) )
   throw( std::invalid_argument( "HQPBlock::guts_of_load: A or C of level " +
                                 std::to_string( p ) + " has not n "
                                 "columns" ) );
  if( ( Index( A[ p ].rows() ) != b[ p ].size() ) ||
      ( Index( C[ p ].rows() ) != d[ p ].size() ) )
   throw( std::invalid_argument( "HQPBlock::guts_of_load: A and b, or C "
                                 "and d, of level " + std::to_string( p ) +
                                 " have different sizes" ) );
  if( ( ! we.empty() ) && ( ! we[ p ].empty() ) &&
      ( we[ p ].size() != b[ p ].size() ) )
   throw( std::invalid_argument( "HQPBlock::guts_of_load: wrong size of "
                                 "we of level " + std::to_string( p ) ) );
  if( ( ! wi.empty() ) && ( ! wi[ p ].empty() ) &&
      ( wi[ p ].size() != d[ p ].size() ) )
   throw( std::invalid_argument( "HQPBlock::guts_of_load: wrong size of "
                                 "wi of level " + std::to_string( p ) ) );
  }

 // the hard rows: a matrix of no rows is none, whatever its columns
 for( int k = 0 ; k < 2 ; ++k ) {
  auto & M = k ? hard.C : hard.A;
  const auto & r = k ? hard.d : hard.b;
  if( ! M.rows() )
   M.resize( 0 , n );
  if( ( Index( M.cols() ) != n ) || ( Index( M.rows() ) != r.size() ) )
   throw( std::invalid_argument( std::string( "HQPBlock::guts_of_load: "
                                 "the hard " ) + ( k ? "C and d" : "A and "
                                 "b" ) + " have wrong sizes" ) );
  M.makeCompressed();
  }

 // the quadratic hard rows: Q[ i ] is n x n, of which the lower triangle
 // is kept
 if( ! hard.Cq.rows() )
  hard.Cq.resize( 0 , n );
 if( ( Index( hard.Cq.cols() ) != n ) ||
     ( Index( hard.Cq.rows() ) != hard.eq.size() ) ||
     ( hard.Q.size() != hard.eq.size() ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: the hard Q, Cq "
                                "and eq have wrong sizes" ) );
 hard.Cq.makeCompressed();
 for( auto & Qi : hard.Q ) {
  if( ( Index( Qi.rows() ) != n ) || ( Index( Qi.cols() ) != n ) )
   throw( std::invalid_argument( "HQPBlock::guts_of_load: a hard Q is not "
                                 "n x n" ) );
  Qi = SparseMatrix( Qi.triangularView< Eigen::Lower >() );
  Qi.makeCompressed();
  }

 // the sum-exp and the p-norm hard rows: each matrix has n columns, its b
 // (empty meaning all 0) and its c (empty meaning all ones) one entry per
 // row of it, and the linear terms are the rows of an n-column matrix
 const auto conic = [ n ]( std::vector< SparseMatrix > & M ,
			   std::vector< Vec_double > & B ,
			   std::vector< Vec_double > * W , SparseMatrix & L ,
			   const Vec_double & u , const char * nm ) {
  const std::string what = std::string( "HQPBlock::guts_of_load: the hard " )
                           + nm;
  const Index m = u.size();
  if( ! L.rows() )
   L.resize( 0 , n );
  if( B.empty() )
   B.resize( m );
  if( W && W->empty() )
   W->resize( m );
  if( ( Index( L.cols() ) != n ) || ( Index( L.rows() ) != m ) ||
      ( M.size() != m ) || ( B.size() != m ) || ( W && ( W->size() != m ) ) )
   throw( std::invalid_argument( what + " rows have wrong sizes" ) );
  L.makeCompressed();
  for( Index i = 0 ; i < m ; ++i ) {
   const Index r = M[ i ].rows();
   if( ! r )
    M[ i ].resize( 0 , n );
   if( Index( M[ i ].cols() ) != n )
    throw( std::invalid_argument( what + " matrix " + std::to_string( i ) +
                                  " has not n columns" ) );
   M[ i ].makeCompressed();
   if( B[ i ].empty() )
    B[ i ].assign( r , 0 );
   if( B[ i ].size() != r )
    throw( std::invalid_argument( what + " b " + std::to_string( i ) +
                                  " has wrong size" ) );
   if( ! W )
    continue;
   auto & w = ( *W )[ i ];
   if( w.empty() )
    w.assign( r , 1 );
   if( w.size() != r )
    throw( std::invalid_argument( what + " c " + std::to_string( i ) +
                                  " has wrong size" ) );
   for( auto c : w )
    if( ! ( c > 0 ) )
     throw( std::invalid_argument( what + " c " + std::to_string( i ) +
                                   " is not positive" ) );
   }
  };
 conic( hard.SE , hard.SEb , & hard.SEc , hard.SEl , hard.se , "sum-exp" );
 conic( hard.PN , hard.PNb , nullptr , hard.PNl , hard.pn , "p-norm" );
 if( hard.PNs.empty() )
  hard.PNs.assign( hard.pn.size() , 1 );
 if( ( hard.PNp.size() != hard.pn.size() ) ||
     ( hard.PNs.size() != hard.pn.size() ) )
  throw( std::invalid_argument( "HQPBlock::guts_of_load: the hard PNp or "
                                "PNs have wrong sizes" ) );
 for( Index i = 0 ; i < hard.pn.size() ; ++i )
  if( ( ! ( hard.PNp[ i ] >= 1 ) ) || ( ! std::isfinite( hard.PNp[ i ] ) ) ||
      ( ! ( hard.PNs[ i ] > 0 ) ) || ( ! std::isfinite( hard.PNs[ i ] ) ) )
   throw( std::invalid_argument( "HQPBlock::guts_of_load: the hard p-norm "
                                 "row " + std::to_string( i ) + " has not "
                                 "p >= 1 and s > 0" ) );

 Index me = 0 , mi = 0;
 for( Index p = 0 ; p < P ; ++p ) {
  me += b[ p ].size();
  mi += d[ p ].size();
  }
 check_stages( stages , n , P , me , mi );

 guts_of_destructor();

 f_n = n;
 f_stages = std::move( stages );
 f_hard = std::move( hard );
 f_eps = eps;
 v_rw = std::move( rw );
 v_center.clear();

 v_eq_start.assign( P + 1 , 0 );
 v_in_start.assign( P + 1 , 0 );
 v_b.clear();
 v_d.clear();
 v_we.clear();
 v_wi.clear();
 for( Index p = 0 ; p < P ; ++p ) {
  A[ p ].makeCompressed();
  C[ p ].makeCompressed();
  v_eq_start[ p + 1 ] = v_eq_start[ p ] + b[ p ].size();
  v_in_start[ p + 1 ] = v_in_start[ p ] + d[ p ].size();
  v_b.insert( v_b.end() , b[ p ].begin() , b[ p ].end() );
  v_d.insert( v_d.end() , d[ p ].begin() , d[ p ].end() );
  if( we.empty() || we[ p ].empty() )
   v_we.insert( v_we.end() , b[ p ].size() , 1.0 );
  else
   v_we.insert( v_we.end() , we[ p ].begin() , we[ p ].end() );
  if( wi.empty() || wi[ p ].empty() )
   v_wi.insert( v_wi.end() , d[ p ].size() , 1.0 );
  else
   v_wi.insert( v_wi.end() , wi[ p ].begin() , wi[ p ].end() );
  }
 v_A = std::move( A );
 v_C = std::move( C );

 // the abstract representation is that of level 0
 f_level = 0;
 f_relax_lower = false;
 v_lvl_state.assign( P , char( kFree ) );
 v_lvl_obj.assign( P , 0 );
 v_lvl_obj[ 0 ] = 1;

 }  // end( HQPBlock::guts_of_load )

/*--------------------------------------------------------------------------*/

void HQPBlock::deserialize( const netCDF::NcGroup & group )
{
 std::size_t n = 0 , P = 0;
 ::deserialize_dim( group , "NumVar" , n , false );
 ::deserialize_dim( group , "NumLevels" , P , false );

 std::vector< SparseMatrix > A , C;
 std::vector< Vec_double > b , d , we , wi;
 read_rows( group , "Eq" , n , P , A , b , we );
 read_rows( group , "Ineq" , n , P , C , d , wi );

 double eps = 0;
 ::deserialize( group , eps , "Regularization" , true );
 Vec_double rw;
 ::deserialize( group , "RegularizationWeight" , n , rw , true );

 StageData st;
 std::size_t N = 0;
 ::deserialize_dim( group , "NumStages" , N );
 if( N ) {
  std::size_t me = 0 , mi = 0;
  ::deserialize_dim( group , "NumEqRows" , me );
  ::deserialize_dim( group , "NumIneqRows" , mi );
  std::vector< unsigned int > vs , vt , es , is;
  ::deserialize( group , "VarStage" , n , vs , false );
  ::deserialize( group , "VarState" , n , vt , false );
  if( me )
   ::deserialize( group , "EqStage" , me , es , true );
  if( mi )
   ::deserialize( group , "IneqStage" , mi , is , true );
  unsigned int dl = 0;
  ::deserialize( group , dl , "DynamicsLevel" , true );
  st.num_stages = N;
  st.var_stage.assign( vs.begin() , vs.end() );
  st.var_state.assign( vt.begin() , vt.end() );
  st.eq_stage.assign( es.begin() , es.end() );
  st.ineq_stage.assign( is.begin() , is.end() );
  st.dynamics_level = dl;
  }

 HardRows hard;
 for( int k = 0 ; k < 2 ; ++k ) {
  std::vector< SparseMatrix > M;
  std::vector< Vec_double > r , w;
  read_rows( group , k ? "HardIneq" : "HardEq" , n , 1 , M , r , w ,
             false );
  ( k ? hard.C : hard.A ) = std::move( M[ 0 ] );
  ( k ? hard.d : hard.b ) = std::move( r[ 0 ] );
  }
 {
  std::vector< SparseMatrix > M;
  std::vector< Vec_double > r , w;
  read_rows( group , "HardQuad" , n , 1 , M , r , w , false );
  hard.Cq = std::move( M[ 0 ] );
  hard.eq = std::move( r[ 0 ] );
  const Index mq = hard.eq.size();
  std::size_t nq = 0;
  ::deserialize_dim( group , "NumHardQuadQElements" , nq );
  std::vector< unsigned int > qr , qi , qj;
  std::vector< double > qv;
  if( nq ) {
   ::deserialize( group , "HardQuadQRow" , nq , qr , false );
   ::deserialize( group , "HardQuadQI" , nq , qi , false );
   ::deserialize( group , "HardQuadQJ" , nq , qj , false );
   ::deserialize( group , "HardQuadQValue" , nq , qv , false );
   }
  std::vector< std::vector< Eigen::Triplet< double > > > trp( mq );
  for( std::size_t k = 0 ; k < nq ; ++k ) {
   if( ( qr[ k ] >= mq ) || ( qi[ k ] >= n ) || ( qj[ k ] > qi[ k ] ) )
    throw( std::invalid_argument( "HQPBlock::deserialize: an element of "
                                  "a hard Q out of range, or above the "
                                  "diagonal" ) );
   trp[ qr[ k ] ].emplace_back( qi[ k ] , qj[ k ] , qv[ k ] );
   }
  hard.Q.assign( mq , SparseMatrix( n , n ) );
  for( Index i = 0 ; i < mq ; ++i )
   hard.Q[ i ].setFromTriplets( trp[ i ].begin() , trp[ i ].end() );
  }
 {
  // the sum-exp and the p-norm hard rows: their linear terms, then their
  // terms, the hard rows taking the place of the levels
  std::vector< SparseMatrix > M;
  std::vector< Vec_double > r , w;
  read_rows( group , "HardSumExp" , n , 1 , M , r , w , false );
  hard.SEl = std::move( M[ 0 ] );
  hard.se = std::move( r[ 0 ] );
  if( ! hard.se.empty() )
   read_rows( group , "HardSumExpTerm" , n , hard.se.size() , hard.SE ,
              hard.SEb , hard.SEc );
  read_rows( group , "HardPNorm" , n , 1 , M , r , w , false );
  hard.PNl = std::move( M[ 0 ] );
  hard.pn = std::move( r[ 0 ] );
  if( const std::size_t np = hard.pn.size() ) {
   read_rows( group , "HardPNormTerm" , n , np , hard.PN , hard.PNb , w );
   ::deserialize( group , "HardPNormP" , np , hard.PNp , false );
   ::deserialize( group , "HardPNormScale" , np , hard.PNs , true );
   }
  }

 // the NBModification is issued by Block::deserialize()
 guts_of_load( n , std::move( A ) , std::move( b ) , std::move( C ) ,
	       std::move( d ) , std::move( we ) , std::move( wi ) , eps ,
	       std::move( rw ) , std::move( st ) , std::move( hard ) );

 // whoever is listening has to find the abstract representation there
 if( anyone_there() ) {
  generate_abstract_variables();
  generate_abstract_constraints();
  generate_objective();
  }

 Block::deserialize( group );

 }  // end( HQPBlock::deserialize )

/*--------------------------------------------------------------------------*/

void HQPBlock::serialize( netCDF::NcGroup & group ) const
{
 Block::serialize( group );

 auto dn = group.addDim( "NumVar" , f_n );
 auto dl = group.addDim( "NumLevels" , get_num_levels() );

 write_rows( group , "Eq" , & dl , v_A , v_b , v_we );
 write_rows( group , "Ineq" , & dl , v_C , v_d , v_wi );
 write_rows( group , "HardEq" , nullptr , { f_hard.A } , f_hard.b , {} );
 write_rows( group , "HardIneq" , nullptr , { f_hard.C } , f_hard.d , {} );
 write_rows( group , "HardQuad" , nullptr , { f_hard.Cq } , f_hard.eq , {} );
 {
  std::vector< unsigned int > qr , qi , qj;
  std::vector< double > qv;
  for( Index i = 0 ; i < f_hard.Q.size() ; ++i )
   for( Index j = 0 ; j < Index( f_hard.Q[ i ].rows() ) ; ++j )
    for( SparseMatrix::InnerIterator it( f_hard.Q[ i ] , j ) ; it ; ++it ) {
     qr.push_back( i );
     qi.push_back( j );
     qj.push_back( it.col() );
     qv.push_back( it.value() );
     }
  if( ! qv.empty() ) {
   auto dq = group.addDim( "NumHardQuadQElements" , qv.size() );
   group.addVar( "HardQuadQRow" , netCDF::NcUint() , dq ).putVar(
                                                               qr.data() );
   group.addVar( "HardQuadQI" , netCDF::NcUint() , dq ).putVar( qi.data() );
   group.addVar( "HardQuadQJ" , netCDF::NcUint() , dq ).putVar( qj.data() );
   group.addVar( "HardQuadQValue" , netCDF::NcDouble() , dq ).putVar(
                                                               qv.data() );
   }
  }
 write_rows( group , "HardSumExp" , nullptr , { f_hard.SEl } , f_hard.se ,
             {} );
 if( ! f_hard.se.empty() ) {
  const auto dr = group.getDim( "NumHardSumExpRows" );
  Vec_double b , c;
  for( Index i = 0 ; i < f_hard.se.size() ; ++i ) {
   b.insert( b.end() , f_hard.SEb[ i ].begin() , f_hard.SEb[ i ].end() );
   c.insert( c.end() , f_hard.SEc[ i ].begin() , f_hard.SEc[ i ].end() );
   }
  write_rows( group , "HardSumExpTerm" , & dr , f_hard.SE , b , c );
  }
 write_rows( group , "HardPNorm" , nullptr , { f_hard.PNl } , f_hard.pn ,
             {} );
 if( ! f_hard.pn.empty() ) {
  const auto dr = group.getDim( "NumHardPNormRows" );
  Vec_double b;
  for( const auto & bi : f_hard.PNb )
   b.insert( b.end() , bi.begin() , bi.end() );
  write_rows( group , "HardPNormTerm" , & dr , f_hard.PN , b , {} );
  group.addVar( "HardPNormP" , netCDF::NcDouble() , dr ).putVar(
                                                        f_hard.PNp.data() );
  if( std::any_of( f_hard.PNs.begin() , f_hard.PNs.end() ,
                   []( double x ) { return( x != 1 ); } ) )
   group.addVar( "HardPNormScale" , netCDF::NcDouble() , dr ).putVar(
                                                        f_hard.PNs.data() );
  }

 if( f_eps != 0 )
  group.addVar( "Regularization" , netCDF::NcDouble() ).putVar( &f_eps );

 if( std::any_of( v_rw.begin() , v_rw.end() ,
                  []( double r ) { return( r != 1 ); } ) )
  group.addVar( "RegularizationWeight" , netCDF::NcDouble() , dn ).putVar(
                                                               v_rw.data() );

 if( ! f_stages.num_stages )
  return;

 group.addDim( "NumStages" , f_stages.num_stages );
 auto put = [ & ]( const std::string & name , const netCDF::NcDim & dim ,
		   const std::vector< Index > & data ) {
  std::vector< unsigned int > u( data.begin() , data.end() );
  group.addVar( name , netCDF::NcUint() , dim ).putVar( u.data() );
  };
 put( "VarStage" , dn , f_stages.var_stage );
 put( "VarState" , dn , f_stages.var_state );
 if( ! f_stages.eq_stage.empty() )
  put( "EqStage" , group.getDim( "NumEqRows" ) , f_stages.eq_stage );
 if( ! f_stages.ineq_stage.empty() )
  put( "IneqStage" , group.getDim( "NumIneqRows" ) , f_stages.ineq_stage );
 const unsigned int dyn = f_stages.dynamics_level;
 group.addVar( "DynamicsLevel" , netCDF::NcUint() ).putVar( &dyn );

 }  // end( HQPBlock::serialize )

/*--------------------------------------------------------------------------*/
/*-------------------- Methods for the abstract representation -------------*/
/*--------------------------------------------------------------------------*/

void HQPBlock::generate_abstract_variables( Configuration * stvv )
{
 if( AR & HasVar )  // the Variable are there already
  return;           // nothing to do

 if( ! f_n )
  return;

 v_x = std::vector< ColVariable >( f_n );
 v_v = std::vector< ColVariable >( v_b.size() );
 v_w = std::vector< ColVariable >( v_d.size() );
 for( auto & w : v_w )
  w.set_type( ColVariable::kNonNegative , eNoMod );

 // the slacks of the levels that are not free are fixed (to 0, since there
 // is no solution yet to take their optimal value from)
 for( Index p = 0 ; p < get_num_levels() ; ++p )
  if( v_lvl_state[ p ] != kFree ) {
   for( Index i = v_eq_start[ p ] ; i < v_eq_start[ p + 1 ] ; ++i ) {
    v_v[ i ].set_value( 0 );
    v_v[ i ].is_fixed( true , eNoMod );
    }
   for( Index i = v_in_start[ p ] ; i < v_in_start[ p + 1 ] ; ++i ) {
    v_w[ i ].set_value( 0 );
    v_w[ i ].is_fixed( true , eNoMod );
    }
   }

 add_static_variable( v_x , "x" );
 add_static_variable( v_v , "v" );
 add_static_variable( v_w , "w" );

 AR |= HasVar;

 }  // end( HQPBlock::generate_abstract_variables )

/*--------------------------------------------------------------------------*/

void HQPBlock::generate_abstract_constraints( Configuration * stcc )
{
 if( AR & HasCns )  // the Constraint are there already
  return;           // nothing to do

 if( ! f_n )
  return;

 generate_abstract_variables();

 // bit 1 of the static_constraints: the quadratic hard rows are cuts
 if( ( ! stcc ) && f_BlockConfig )
  stcc = f_BlockConfig->f_static_constraints_Configuration;
 f_quad_cuts = false;
 if( auto sci = dynamic_cast< SimpleConfiguration< int > * >( stcc ) )
  f_quad_cuts = sci->f_value & 1;

 const Index ne = v_b.size();
 v_eq = std::vector< FRowConstraint >( ne );
 v_in = std::vector< FRowConstraint >( v_d.size() );
 v_cols.assign( ne + v_d.size() , {} );

 for( Index p = 0 ; p < get_num_levels() ; ++p ) {
  for( int k = 0 ; k < 2 ; ++k ) {
   const bool eq = ( k == 0 );
   const auto & M = eq ? v_A[ p ] : v_C[ p ];
   const Index first = eq ? v_eq_start[ p ] : v_in_start[ p ];
   for( Index h = 0 ; h < Index( M.rows() ) ; ++h ) {
    const Index i = first + h;
    const double wg = eq ? v_we[ i ] : v_wi[ i ];
    auto & cols = v_cols[ eq ? i : ne + i ];
    LinearFunction::v_coeff_pair cf;
    cf.reserve( M.row( h ).nonZeros() + 1 );
    cf.emplace_back( eq ? & v_v[ i ] : & v_w[ i ] , -1.0 );
    for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it ) {
     cf.emplace_back( & v_x[ it.col() ] , wg * it.value() );
     cols.push_back( it.col() );
     }
    auto & row = eq ? v_eq[ i ] : v_in[ i ];
    row.set_function( new LinearFunction( std::move( cf ) ) , eNoMod );
    if( eq )
     row.set_both( wg * v_b[ i ] , eNoMod );
    else {
     row.set_lhs( -Inf< double >() , eNoMod );
     row.set_rhs( wg * v_d[ i ] , eNoMod );
     }
    if( v_lvl_state[ p ] == kOff )
     row.relax( true , eNoMod );
    }
   }
  }

 // the hard rows, with no slack
 const Index nhe = f_hard.b.size();
 v_heq = std::vector< FRowConstraint >( nhe );
 v_hin = std::vector< FRowConstraint >( f_hard.d.size() );
 v_hcols.assign( nhe + f_hard.d.size() , {} );
 for( int k = 0 ; k < 2 ; ++k ) {
  const bool eq = ( k == 0 );
  const auto & M = eq ? f_hard.A : f_hard.C;
  for( Index h = 0 ; h < Index( M.rows() ) ; ++h ) {
   auto & cols = v_hcols[ eq ? h : nhe + h ];
   LinearFunction::v_coeff_pair cf;
   cf.reserve( M.row( h ).nonZeros() );
   for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it ) {
    cf.emplace_back( & v_x[ it.col() ] , it.value() );
    cols.push_back( it.col() );
    }
   auto & row = eq ? v_heq[ h ] : v_hin[ h ];
   row.set_function( new LinearFunction( std::move( cf ) ) , eNoMod );
   if( eq )
    row.set_both( f_hard.b[ h ] , eNoMod );
   else {
    row.set_lhs( -Inf< double >() , eNoMod );
    row.set_rhs( f_hard.d[ h ] , eNoMod );
    }
   }
  }

 add_static_constraint( v_eq , "eq" );
 add_static_constraint( v_in , "ineq" );
 // the quadratic hard rows, each with a QuadFunction on the x it touches:
 // 1/2 x' Q x has x_j^2 with Q_jj / 2 and x_j x_k (j > k) with Q_jk; none
 // if they are cuts
 v_hqd = std::vector< FRowConstraint >( f_quad_cuts ? 0 : f_hard.eq.size() );
 {
  std::vector< int > pos( f_n , -1 );
  for( Index i = 0 ; i < v_hqd.size() ; ++i ) {
   DQuadFunction::v_coeff_triple vt;
   std::vector< Index > used;
   auto act = [ & ]( Index j ) -> Index {
    if( pos[ j ] < 0 ) {
     pos[ j ] = vt.size();
     vt.emplace_back( & v_x[ j ] , 0.0 , 0.0 );
     used.push_back( j );
     }
    return( pos[ j ] );
    };
   for( SparseMatrix::InnerIterator it( f_hard.Cq , i ) ; it ; ++it ) {
    const Index a = act( it.col() );
    std::get< 1 >( vt[ a ] ) += it.value();
    }
   QuadFunction::v_off_diag_term od;
   const auto & Qi = f_hard.Q[ i ];
   for( Index j = 0 ; j < Index( Qi.rows() ) ; ++j )
    for( SparseMatrix::InnerIterator it( Qi , j ) ; it ; ++it ) {
     const Index a = act( j );
     const Index c = act( it.col() );
     if( Index( it.col() ) == j )
      std::get< 2 >( vt[ a ] ) += it.value() / 2;
     else
      od.emplace_back( std::max( a , c ) , std::min( a , c ) , it.value() );
     }
   for( auto j : used )
    pos[ j ] = -1;
   auto & row = v_hqd[ i ];
   row.set_function( new QuadFunction( std::move( vt ) , std::move( od ) ) ,
                     eNoMod );
   row.set_lhs( -Inf< double >() , eNoMod );
   row.set_rhs( f_hard.eq[ i ] , eNoMod );
   }
  }

 // the cuts of the nonlinear hard rows, none yet
 const Index nnl = f_hard.eq.size() + f_hard.se.size() + f_hard.pn.size();
 v_row_cuts.assign( nnl , {} );
 v_inner_pt.assign( nnl , nullptr );

 add_static_constraint( v_heq , "hard_eq" );
 add_static_constraint( v_hin , "hard_ineq" );
 add_static_constraint( v_hqd , "hard_quad" );
 add_dynamic_constraint( v_link , "link" );
 add_dynamic_constraint( v_cuts , "hard_cuts" );

 AR |= HasCns;

 }  // end( HQPBlock::generate_abstract_constraints )

/*--------------------------------------------------------------------------*/

void HQPBlock::generate_dynamic_constraints( Configuration * dycc )
{
 if( ! ( AR & HasCns ) )
  return;

 double tol = 1e-9;
 int bits = 0;
 if( ( ! extract_pair( dycc , tol , bits ) ) && f_BlockConfig )
  extract_pair( f_BlockConfig->f_dynamic_constraints_Configuration , tol ,
                bits );

 sync_cuts();
 Vec_double x;
 get_x_values( x );

 // the rows strictly satisfied at x share x as their point inside
 std::shared_ptr< const Vec_double > here;
 std::list< FRowConstraint > newcuts;
 std::vector< std::pair< Index , CutRef > > newrefs;
 Vec_double xf , y( f_n );
 for( int h = f_quad_cuts ? kQuadRow : kSumExpRow ; h <= kPNormRow ; ++h ) {
  const auto k = hard_row_kind( h );
  for( Index i = 0 ; i < num_nl_rows( k ) ; ++i ) {
   const Index r = nl_row( k , i );
   const double u = nl_rhs( k , i );
   const double val = hard_row_value( k , i , x );
   if( val < u ) {
    if( ! here )
     here = std::make_shared< const Vec_double >( x );
    v_inner_pt[ r ] = here;
    }
   const double ftol = tol * std::max( 1.0 , std::abs( u ) );
   if( ! ( val - u > ftol ) )
    continue;

   // Kelley at x, or Veinott between the point inside and x
   xf = x;
   if( ( bits & 1 ) && v_inner_pt[ r ] ) {
    const auto & x0 = *v_inner_pt[ r ];
    const double f0 = hard_row_value( k , i , x0 ) - u;
    if( f0 < 0 ) {
     const auto at = [ & ]( double t ) {
      for( Index j = 0 ; j < f_n ; ++j )
       y[ j ] = x0[ j ] + t * ( x[ j ] - x0[ j ] );
      };
     const double t = boundary( [ & ]( double lam ) {
                                 at( lam );
                                 return( hard_row_value( k , i , y ) - u );
                                 } , f0 , val - u , ftol );
     at( t );
     xf = y;
     }
    }

   LinearFunction::v_coeff_pair cf;
   double rhs = 0 , scale = 1;
   bool lg = false;
   Vec_double mb;
   if( ( ! row_cut( k , i , xf , bits & 2 , cf , rhs , scale , lg , mb ) )
       || known_cut( r , cf , rhs ) )
    continue;
   newcuts.emplace_back();
   auto & cut = newcuts.back();
   cut.set_lhs( -Inf< double >() , eNoMod );
   cut.set_rhs( rhs , eNoMod );
   cut.set_function( new LinearFunction( std::move( cf ) ) , eNoMod );
   newrefs.push_back( { r , { std::prev( newcuts.end() ) , & cut , scale ,
                              lg , std::move( mb ) } } );
   }
  }

 if( newcuts.empty() )
  return;

 // the iterators stay valid through the splice into v_cuts
 add_dynamic_constraints( v_cuts , newcuts , eNoBlck );
 for( auto & [ r , ref ] : newrefs )
  v_row_cuts[ r ].push_back( ref );

 }  // end( HQPBlock::generate_dynamic_constraints )

/*--------------------------------------------------------------------------*/

double HQPBlock::hard_row_value( hard_row_kind k , Index i ,
                                 const Vec_double & x , Vec_double * g ) const
{
 if( i >= num_nl_rows( k ) )
  throw( std::invalid_argument( "HQPBlock::hard_row_value: row " +
                                std::to_string( i ) + " out of range" ) );
 if( x.size() != f_n )
  throw( std::invalid_argument( "HQPBlock::hard_row_value: wrong size" ) );

 if( k == kQuadRow )
  return( quad_value( f_hard.Q[ i ] , f_hard.Cq , i , x , g ) );

 if( k == kSumExpRow ) {
  const double S = std::exp( lse_value( f_hard.SE[ i ] , f_hard.SEb[ i ] ,
                                        f_hard.SEc[ i ] , x , g ) );
  if( g ) {
   for( auto & gj : *g )
    gj *= S;
   add_row( f_hard.SEl , i , 1 , *g );
   }
  return( S + row_dot( f_hard.SEl , i , x ) );
  }

 const double s = f_hard.PNs[ i ];
 const double nrm = pnorm_value( f_hard.PN[ i ] , f_hard.PNb[ i ] ,
                                 f_hard.PNp[ i ] , x , g , nullptr );
 if( g ) {
  for( auto & gj : *g )
   gj *= s;
  add_row( f_hard.PNl , i , 1 , *g );
  }
 return( s * nrm + row_dot( f_hard.PNl , i , x ) );

 }  // end( HQPBlock::hard_row_value )

/*--------------------------------------------------------------------------*/

bool HQPBlock::is_feasible( bool useabstract , Configuration * fsbc )
{
 double tol = DefaultFeasTol;
 int rel = 1;
 if( ( ! extract_pair( fsbc , tol , rel ) ) && f_BlockConfig )
  extract_pair( f_BlockConfig->f_is_feasible_Configuration , tol , rel );

 if( useabstract && ( AR & HasCns ) )
  return( ColVariable::is_feasible( v_x , tol ) &&
          ColVariable::is_feasible( v_v , tol ) &&
          ColVariable::is_feasible( v_w , tol ) &&
          RowConstraint::is_feasible( v_eq , tol , rel ) &&
          RowConstraint::is_feasible( v_in , tol , rel ) &&
          RowConstraint::is_feasible( v_heq , tol , rel ) &&
          RowConstraint::is_feasible( v_hin , tol , rel ) &&
          RowConstraint::is_feasible( v_hqd , tol , rel ) &&
          RowConstraint::is_feasible( v_link , tol , rel ) &&
          RowConstraint::is_feasible( v_cuts , tol , rel ) );

 if( ! ( AR & HasVar ) )
  return( Block::is_feasible( useabstract , fsbc ) );

 Vec_double x;
 get_x_values( x );
 // the violation of a row of right-hand side r is within the tolerance
 const auto within = [ tol , rel ]( double viol , double r ) {
  return( viol <= tol * ( rel ? std::max( 1.0 , std::abs( r ) ) : 1.0 ) );
  };

 for( Index h = 0 ; h < f_hard.b.size() ; ++h )
  if( ! within( std::abs( row_dot( f_hard.A , h , x ) - f_hard.b[ h ] ) ,
                f_hard.b[ h ] ) )
   return( false );
 for( Index h = 0 ; h < f_hard.d.size() ; ++h )
  if( ! within( row_dot( f_hard.C , h , x ) - f_hard.d[ h ] ,
                f_hard.d[ h ] ) )
   return( false );
 for( int h = kQuadRow ; h <= kPNormRow ; ++h ) {
  const auto k = hard_row_kind( h );
  for( Index i = 0 ; i < num_nl_rows( k ) ; ++i )
   if( ! within( hard_row_value( k , i , x ) - nl_rhs( k , i ) ,
                 nl_rhs( k , i ) ) )
    return( false );
  }

 return( Block::is_feasible( useabstract , fsbc ) );

 }  // end( HQPBlock::is_feasible )

/*--------------------------------------------------------------------------*/

void HQPBlock::generate_objective( Configuration * objc )
{
 if( AR & HasObj )  // the Objective is there already
  return;           // nothing to do

 if( ! f_n )
  return;

 generate_abstract_variables();

 DQuadFunction::v_coeff_triple t;
 t.reserve( f_n + v_v.size() + v_w.size() );
 for( Index j = 0 ; j < f_n ; ++j )
  t.emplace_back( & v_x[ j ] , v_center.empty() ? 0.0 :
                  - f_scale * f_eps * reg_weight( j ) * v_center[ j ] ,
                  f_scale * f_eps * reg_weight( j ) / 2 );
 for( Index p = 0 ; p < get_num_levels() ; ++p )
  for( Index i = v_eq_start[ p ] ; i < v_eq_start[ p + 1 ] ; ++i )
   t.emplace_back( & v_v[ i ] , 0.0 , v_lvl_obj[ p ] ? f_scale / 2 : 0.0 );
 for( Index p = 0 ; p < get_num_levels() ; ++p )
  for( Index i = v_in_start[ p ] ; i < v_in_start[ p + 1 ] ; ++i )
   t.emplace_back( & v_w[ i ] , 0.0 , v_lvl_obj[ p ] ? f_scale / 2 : 0.0 );

 f_obj.set_function( new DQuadFunction( std::move( t ) , reg_constant() ) ,
                     eNoMod );
 f_obj.set_sense( Objective::eMin , eNoMod );
 set_objective( & f_obj , eNoMod );

 AR |= HasObj;

 }  // end( HQPBlock::generate_objective )

/*--------------------------------------------------------------------------*/
/*----------------------- Methods for handling the levels ------------------*/
/*--------------------------------------------------------------------------*/

void HQPBlock::reset_levels( bool relax_lower , ModParam issueMod )
{
 const Index P = get_num_levels();
 if( ! P )
  return;

 f_relax_lower = relax_lower;
 for( Index q = 0 ; q < P ; ++q ) {
  const bool first = ( q == 0 );
  set_level_state( q , ( first || ! relax_lower ) ? kFree : kOff , 0 ,
                   issueMod );
  set_level_in_objective( q , first || relax_lower , issueMod );
  }
 f_level = 0;

 }  // end( HQPBlock::reset_levels )

/*--------------------------------------------------------------------------*/

void HQPBlock::next_level( double tol , ModParam issueMod )
{
 if( f_level + 1 >= get_num_levels() )
  throw( std::logic_error( "HQPBlock::next_level: the current level is the "
                           "last one" ) );
 if( ! ( AR & HasVar ) )
  throw( std::logic_error( "HQPBlock::next_level: the Variable are not "
                           "there" ) );
 if( ! ( tol >= 0 ) )
  throw( std::invalid_argument( "HQPBlock::next_level: negative "
                                "tolerance" ) );

 set_level_state( f_level , kFixed , tol , issueMod );
 ++f_level;
 set_level_state( f_level , kFree , 0 , issueMod );
 set_level_in_objective( f_level , true , issueMod );

 }  // end( HQPBlock::next_level )

/*--------------------------------------------------------------------------*/

double HQPBlock::level_slacks( Index p , Vec_double & v ,
                               Vec_double & w ) const
{
 check_level( p , "level_slacks" );
 if( ! ( AR & HasVar ) )
  throw( std::logic_error( "HQPBlock::level_slacks: the Variable are not "
                           "there" ) );

 double val = 0;
 const auto & A = v_A[ p ];
 v.resize( A.rows() );
 for( Eigen::Index h = 0 ; h < A.rows() ; ++h ) {
  const Index i = v_eq_start[ p ] + h;
  double ax = 0;
  for( SparseMatrix::InnerIterator it( A , h ) ; it ; ++it )
   ax += it.value() * v_x[ it.col() ].get_value();
  v[ h ] = v_we[ i ] * ( ax - v_b[ i ] );
  val += v[ h ] * v[ h ];
  }

 const auto & C = v_C[ p ];
 w.resize( C.rows() );
 for( Eigen::Index h = 0 ; h < C.rows() ; ++h ) {
  const Index i = v_in_start[ p ] + h;
  double cx = 0;
  for( SparseMatrix::InnerIterator it( C , h ) ; it ; ++it )
   cx += it.value() * v_x[ it.col() ].get_value();
  w[ h ] = std::max( 0.0 , v_wi[ i ] * ( cx - v_d[ i ] ) );
  val += w[ h ] * w[ h ];
  }

 return( val / 2 );

 }  // end( HQPBlock::level_slacks )

/*--------------------------------------------------------------------------*/

void HQPBlock::refix_level( Index p , ModParam issueMod )
{
 check_level( p , "refix_level" );
 if( v_lvl_state[ p ] != kFixed )
  throw( std::logic_error( "HQPBlock::refix_level: level " +
                           std::to_string( p ) + " is not fixed" ) );

 Vec_double v , w;
 level_slacks( p , v , w );

 auto refix = [ & ]( ColVariable & var , double val ) {
  if( var.get_value() == val )
   return;
  var.is_fixed( false , issueMod );
  var.set_value( val );
  var.is_fixed( true , issueMod );
  };

 for( Index h = 0 ; h < get_num_eq_rows( p ) ; ++h )
  refix( v_v[ v_eq_start[ p ] + h ] , v[ h ] );
 for( Index h = 0 ; h < get_num_ineq_rows( p ) ; ++h ) {
  auto & var = v_w[ v_in_start[ p ] + h ];
  refix( var , std::max( var.get_value() , w[ h ] ) );
  }
 }  // end( HQPBlock::refix_level )

/*--------------------------------------------------------------------------*/

void HQPBlock::get_x_values( Vec_double & x ) const
{
 x.resize( v_x.size() );
 for( Index j = 0 ; j < v_x.size() ; ++j )
  x[ j ] = v_x[ j ].get_value();
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_x_values( const Vec_double & x )
{
 if( x.size() != v_x.size() )
  throw( std::invalid_argument( "HQPBlock::set_x_values: wrong size" ) );
 for( Index j = 0 ; j < v_x.size() ; ++j )
  v_x[ j ].set_value( x[ j ] );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_slack_values( Index p , const Vec_double & v ,
                                 const Vec_double & w )
{
 check_level( p , "set_slack_values" );
 if( ( v.size() != get_num_eq_rows( p ) ) ||
     ( w.size() != get_num_ineq_rows( p ) ) )
  throw( std::invalid_argument( "HQPBlock::set_slack_values: wrong "
                                "size" ) );
 if( ! ( AR & HasVar ) )
  return;
 // a fixed slack keeps its value, w + tol for a fixed level
 const auto put = []( ColVariable & var , double val ) {
  if( ! var.is_fixed() )
   var.set_value( val );
  };
 for( Index h = 0 ; h < v.size() ; ++h )
  put( v_v[ v_eq_start[ p ] + h ] , v[ h ] );
 for( Index h = 0 ; h < w.size() ; ++h )
  put( v_w[ v_in_start[ p ] + h ] , w[ h ] );
 }

/*--------------------------------------------------------------------------*/
/*----------------------- Methods for handling Solution --------------------*/
/*--------------------------------------------------------------------------*/

Solution * HQPBlock::get_Solution( Configuration * solc , bool emptys )
{
 auto sol = new ColVariableSolution;
 if( ! emptys )
  sol->read( this );
 return( sol );
 }

/*--------------------------------------------------------------------------*/
/*-------------------------- Methods for changing data ---------------------*/
/*--------------------------------------------------------------------------*/

void HQPBlock::set_b( Index p , const Vec_double & b , ModParam issueMod )
{
 check_level( p , "set_b" );
 if( b.size() != get_num_eq_rows( p ) )
  throw( std::invalid_argument( "HQPBlock::set_b: wrong size" ) );

 std::copy( b.begin() , b.end() , v_b.begin() + v_eq_start[ p ] );
 set_rows( true , p , nullptr , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_d( Index p , const Vec_double & d , ModParam issueMod )
{
 check_level( p , "set_d" );
 if( d.size() != get_num_ineq_rows( p ) )
  throw( std::invalid_argument( "HQPBlock::set_d: wrong size" ) );

 std::copy( d.begin() , d.end() , v_d.begin() + v_in_start[ p ] );
 set_rows( false , p , nullptr , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_A( Index p , const SparseMatrix & A , ModParam issueMod )
{
 check_level( p , "set_A" );
 if( ( A.rows() != v_A[ p ].rows() ) || ( A.cols() != v_A[ p ].cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_A: wrong size" ) );

 set_rows( true , p , & A , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_C( Index p , const SparseMatrix & C , ModParam issueMod )
{
 check_level( p , "set_C" );
 if( ( C.rows() != v_C[ p ].rows() ) || ( C.cols() != v_C[ p ].cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_C: wrong size" ) );

 set_rows( false , p , & C , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_we( Index p , const Vec_double & we , ModParam issueMod )
{
 check_level( p , "set_we" );
 if( we.size() != get_num_eq_rows( p ) )
  throw( std::invalid_argument( "HQPBlock::set_we: wrong size" ) );

 std::copy( we.begin() , we.end() , v_we.begin() + v_eq_start[ p ] );
 set_rows( true , p , nullptr , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_wi( Index p , const Vec_double & wi , ModParam issueMod )
{
 check_level( p , "set_wi" );
 if( wi.size() != get_num_ineq_rows( p ) )
  throw( std::invalid_argument( "HQPBlock::set_wi: wrong size" ) );

 std::copy( wi.begin() , wi.end() , v_wi.begin() + v_in_start[ p ] );
 set_rows( false , p , nullptr , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_b( const Vec_double & b , ModParam issueMod )
{
 if( b.size() != f_hard.b.size() )
  throw( std::invalid_argument( "HQPBlock::set_hard_b: wrong size" ) );

 f_hard.b = b;
 set_hard_rows( true , nullptr , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_d( const Vec_double & d , ModParam issueMod )
{
 if( d.size() != f_hard.d.size() )
  throw( std::invalid_argument( "HQPBlock::set_hard_d: wrong size" ) );

 f_hard.d = d;
 set_hard_rows( false , nullptr , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_A( const SparseMatrix & A , ModParam issueMod )
{
 if( ( A.rows() != f_hard.A.rows() ) || ( A.cols() != f_hard.A.cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_hard_A: wrong size" ) );

 set_hard_rows( true , & A , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_C( const SparseMatrix & C , ModParam issueMod )
{
 if( ( C.rows() != f_hard.C.rows() ) || ( C.cols() != f_hard.C.cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_hard_C: wrong size" ) );

 set_hard_rows( false , & C , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_eq( const Vec_double & e , ModParam issueMod )
{
 if( e.size() != f_hard.eq.size() )
  throw( std::invalid_argument( "HQPBlock::set_hard_eq: wrong size" ) );

 const Vec_double old = f_hard.eq;
 f_hard.eq = e;
 if( AR & HasCns )
  for( Index i = 0 ; i < e.size() ; ++i ) {
   if( ( i < v_hqd.size() ) && ( v_hqd[ i ].get_rhs() != e[ i ] ) )
    v_hqd[ i ].set_rhs( e[ i ] , issueMod );
   shift_cuts( kQuadRow , i , old[ i ] , e[ i ] , issueMod );
   }
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_se( const Vec_double & u , ModParam issueMod )
{
 if( u.size() != f_hard.se.size() )
  throw( std::invalid_argument( "HQPBlock::set_hard_se: wrong size" ) );

 const Vec_double old = f_hard.se;
 f_hard.se = u;
 if( AR & HasCns )
  for( Index i = 0 ; i < u.size() ; ++i )
   shift_cuts( kSumExpRow , i , old[ i ] , u[ i ] , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_pn( const Vec_double & u , ModParam issueMod )
{
 if( u.size() != f_hard.pn.size() )
  throw( std::invalid_argument( "HQPBlock::set_hard_pn: wrong size" ) );

 const Vec_double old = f_hard.pn;
 f_hard.pn = u;
 if( AR & HasCns )
  for( Index i = 0 ; i < u.size() ; ++i )
   shift_cuts( kPNormRow , i , old[ i ] , u[ i ] , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_Cq( const SparseMatrix & Cq , ModParam issueMod )
{
 if( ( Cq.rows() != f_hard.Cq.rows() ) || ( Cq.cols() != f_hard.Cq.cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_hard_Cq: wrong size" ) );
 set_linear_terms( kQuadRow , Cq , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_SEl( const SparseMatrix & L , ModParam issueMod )
{
 if( ( L.rows() != f_hard.SEl.rows() ) || ( L.cols() != f_hard.SEl.cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_hard_SEl: wrong size" ) );
 set_linear_terms( kSumExpRow , L , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_PNl( const SparseMatrix & L , ModParam issueMod )
{
 if( ( L.rows() != f_hard.PNl.rows() ) || ( L.cols() != f_hard.PNl.cols() ) )
  throw( std::invalid_argument( "HQPBlock::set_hard_PNl: wrong size" ) );
 set_linear_terms( kPNormRow , L , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_SEb( const std::vector< Vec_double > & b ,
                             ModParam issueMod )
{
 auto nb = shifts_of( f_hard.SE , b , "set_hard_SEb" );
 for( Index i = 0 ; i < nb.size() ; ++i ) {
  if( nb[ i ] == f_hard.SEb[ i ] )
   continue;
  f_hard.SEb[ i ] = std::move( nb[ i ] );
  // the cuts are not linear in b: they go
  if( AR & HasCns )
   edit_cuts( kSumExpRow , i , []( CutRef & ) { return( false ); } ,
              issueMod );
  }
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_hard_PNb( const std::vector< Vec_double > & b ,
                             ModParam issueMod )
{
 auto nb = shifts_of( f_hard.PN , b , "set_hard_PNb" );
 for( Index i = 0 ; i < nb.size() ; ++i ) {
  if( nb[ i ] == f_hard.PNb[ i ] )
   continue;
  // s w'( A x + b ) + l' x <= u becomes s w'( A x + b' ) + l' x <= u
  Vec_double db( nb[ i ].size() );
  for( Index k = 0 ; k < db.size() ; ++k )
   db[ k ] = f_hard.PNb[ i ][ k ] - nb[ i ][ k ];
  f_hard.PNb[ i ] = std::move( nb[ i ] );
  if( AR & HasCns )
   edit_cuts( kPNormRow , i , [ & ]( CutRef & c ) {
               const double d = dot( c.mb , db );
               if( d != 0 )
                c.cut->set_rhs( c.cut->get_rhs() + d / c.scale , issueMod );
               return( true );
               } , issueMod );
  }
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_regularization( double eps , ModParam issueMod )
{
 if( ! ( eps >= 0 ) )
  throw( std::invalid_argument( "HQPBlock::set_regularization: negative "
                                "regularization" ) );
 if( eps == f_eps )
  return;

 f_eps = eps;
 if( ! ( AR & HasObj ) )
  return;

 DQuadFunction::v_coeff quad( f_n ) , lin( f_n , 0 );
 for( Index j = 0 ; j < f_n ; ++j ) {
  quad[ j ] = f_scale * f_eps * reg_weight( j ) / 2;
  if( ! v_center.empty() )
   lin[ j ] = - f_scale * f_eps * reg_weight( j ) * v_center[ j ];
  }
 static_cast< DQuadFunction * >( f_obj.get_function() )->modify_terms(
       quad.cbegin() , lin.cbegin() , DQuadFunction::Range( 0 , f_n ) ,
       issueMod );
 auto f = static_cast< DQuadFunction * >( f_obj.get_function() );
 if( f->get_constant_term() != reg_constant() )
  f->set_constant_term( reg_constant() , issueMod );

 }  // end( HQPBlock::set_regularization )

/*--------------------------------------------------------------------------*/

void HQPBlock::set_center( const Vec_double & c , ModParam issueMod )
{
 if( ( ! c.empty() ) && ( c.size() != f_n ) )
  throw( std::invalid_argument( "HQPBlock::set_center: wrong size" ) );

 v_center = c;
 if( ( ! ( AR & HasObj ) ) || ( f_eps == 0 ) )
  return;

 DQuadFunction::v_coeff lin( f_n , 0 );
 if( ! v_center.empty() )
  for( Index j = 0 ; j < f_n ; ++j )
   lin[ j ] = - f_scale * f_eps * reg_weight( j ) * v_center[ j ];
 auto f = static_cast< DQuadFunction * >( f_obj.get_function() );
 f->modify_linear_coefficients( std::move( lin ) ,
                                DQuadFunction::Range( 0 , f_n ) , issueMod );
 if( f->get_constant_term() != reg_constant() )
  f->set_constant_term( reg_constant() , issueMod );

 }  // end( HQPBlock::set_center )

/*--------------------------------------------------------------------------*/

double HQPBlock::reg_constant( void ) const
{
 double c2 = 0;
 for( Index j = 0 ; j < v_center.size() ; ++j )
  c2 += reg_weight( j ) * v_center[ j ] * v_center[ j ];
 return( f_scale * f_eps * c2 / 2 );
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::set_objective_scale( double s , ModParam issueMod )
{
 if( ! ( s > 0 ) || ( s == Inf< double >() ) )
  throw( std::invalid_argument( "HQPBlock::set_objective_scale: the scale "
                                "is not finite and positive" ) );
 if( s == f_scale )
  return;

 f_scale = s;
 if( ! ( AR & HasObj ) )
  return;

 // all the terms at once: x, then the slacks of each kind by level
 const Index ne = v_b.size() , ni = v_d.size();
 DQuadFunction::v_coeff quad( f_n + ne + ni , 0 ) , lin( f_n + ne + ni , 0 );
 for( Index j = 0 ; j < f_n ; ++j ) {
  quad[ j ] = f_scale * f_eps * reg_weight( j ) / 2;
  if( ! v_center.empty() )
   lin[ j ] = - f_scale * f_eps * reg_weight( j ) * v_center[ j ];
  }
 for( Index p = 0 ; p < get_num_levels() ; ++p ) {
  const double q = v_lvl_obj[ p ] ? f_scale / 2 : 0.0;
  for( Index i = v_eq_start[ p ] ; i < v_eq_start[ p + 1 ] ; ++i )
   quad[ f_n + i ] = q;
  for( Index i = v_in_start[ p ] ; i < v_in_start[ p + 1 ] ; ++i )
   quad[ f_n + ne + i ] = q;
  }
 auto f = static_cast< DQuadFunction * >( f_obj.get_function() );
 f->modify_terms( quad.cbegin() , lin.cbegin() ,
                  DQuadFunction::Range( 0 , f_n + ne + ni ) , issueMod );
 if( f->get_constant_term() != reg_constant() )
  f->set_constant_term( reg_constant() , issueMod );

 }  // end( HQPBlock::set_objective_scale )

/*--------------------------------------------------------------------------*/
/*---------------------------- Methods for printing ------------------------*/
/*--------------------------------------------------------------------------*/

void HQPBlock::print( std::ostream & output , char vlvl ) const
{
 output << "HQPBlock with " << f_n << " variables and " << get_num_levels()
        << " levels, regularization " << f_eps << std::endl;
 if( has_hard_rows() ) {
  output << " hard rows: " << f_hard.b.size() << " equality, "
         << f_hard.d.size() << " inequality, " << f_hard.eq.size()
         << " quadratic" << std::endl;
  if( ! ( f_hard.se.empty() && f_hard.pn.empty() ) )
   output << " conic hard rows: " << f_hard.se.size() << " sum-exp, "
          << f_hard.pn.size() << " p-norm" << std::endl;
  if( ! v_cuts.empty() )
   output << " " << v_cuts.size() << " cuts of the nonlinear hard rows"
          << std::endl;
  for( int k = 0 ; ( vlvl > 0 ) && ( k < 2 ) ; ++k ) {
   const bool eq = ( k == 0 );
   const auto & M = eq ? f_hard.A : f_hard.C;
   for( Index h = 0 ; h < Index( M.rows() ) ; ++h ) {
    output << "  ";
    for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
     output << " " << it.value() << " x_" << it.col();
    output << ( eq ? " = " : " <= " )
           << ( eq ? f_hard.b[ h ] : f_hard.d[ h ] ) << std::endl;
    }
   }
  }
 if( f_stages.num_stages ) {
  output << " " << f_stages.num_stages << " stages, the dynamics at level "
	 << f_stages.dynamics_level << std::endl;
  if( vlvl > 0 ) {
   output << "  stage of each column (s state, u input):";
   for( Index j = 0 ; j < f_n ; ++j )
    output << " " << ( f_stages.var_state[ j ] ? "s" : "u" )
	   << f_stages.var_stage[ j ];
   output << std::endl;
   }
  }
 for( Index p = 0 ; p < get_num_levels() ; ++p ) {
  output << " level " << p << ": " << get_num_eq_rows( p )
         << " equality rows, " << get_num_ineq_rows( p )
         << " inequality rows" << std::endl;
  if( vlvl <= 0 )
   continue;
  for( int k = 0 ; k < 2 ; ++k ) {
   const bool eq = ( k == 0 );
   const auto & M = eq ? v_A[ p ] : v_C[ p ];
   const Index first = eq ? v_eq_start[ p ] : v_in_start[ p ];
   for( Index h = 0 ; h < Index( M.rows() ) ; ++h ) {
    const Index i = first + h;
    output << "  " << ( eq ? v_we[ i ] : v_wi[ i ] ) << " * (";
    for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
     output << " " << it.value() << " x_" << it.col();
    output << ( eq ? " - v = " : " - w <= " )
           << ( eq ? v_b[ i ] : v_d[ i ] ) << " )" << std::endl;
    }
   }
  }
 }  // end( HQPBlock::print )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

void HQPBlock::guts_of_destructor( void )
{
 for( auto & row : v_eq )
  row.clear();
 for( auto & row : v_in )
  row.clear();
 for( auto & row : v_heq )
  row.clear();
 for( auto & row : v_hin )
  row.clear();
 for( auto & row : v_hqd )
  row.clear();
 for( auto & row : v_link )
  row.clear();
 for( auto & row : v_cuts )
  row.clear();
 f_obj.clear();

 reset_static_constraints();
 reset_dynamic_constraints();
 reset_static_variables();
 reset_objective();

 v_eq.clear();
 v_in.clear();
 v_heq.clear();
 v_hin.clear();
 v_hqd.clear();
 v_link.clear();
 v_cuts.clear();
 v_row_cuts.clear();
 v_inner_pt.clear();
 v_cols.clear();
 v_hcols.clear();
 v_x.clear();
 v_v.clear();
 v_w.clear();

 AR = 0;
 }

/*--------------------------------------------------------------------------*/

void HQPBlock::check_level( Index p , const char * method ) const
{
 if( p >= get_num_levels() )
  throw( std::invalid_argument( std::string( "HQPBlock::" ) + method +
                                ": level " + std::to_string( p ) +
                                " out of range" ) );
 }

/*--------------------------------------------------------------------------*/
// the coefficients and the right-hand sides of the rows of level p (the
// equality ones if eq) are brought to the data, M (if not nullptr) becoming
// the new matrix of the level

void HQPBlock::set_rows( bool eq , Index p , const SparseMatrix * M ,
                         ModParam issueMod )
{
 auto & Mp = eq ? v_A[ p ] : v_C[ p ];
 if( M ) {
  Mp = *M;
  Mp.makeCompressed();
  }

 if( ! ( AR & HasCns ) )
  return;

 const Index first = eq ? v_eq_start[ p ] : v_in_start[ p ];
 update_rows( eq , Mp , ( eq ? v_eq.data() : v_in.data() ) + first ,
              v_cols.data() + ( eq ? 0 : v_b.size() ) + first ,
              ( eq ? v_b.data() : v_d.data() ) + first ,
              ( eq ? v_we.data() : v_wi.data() ) + first , 1 , issueMod );

 }  // end( HQPBlock::set_rows )

/*--------------------------------------------------------------------------*/
// the coefficients and the right-hand sides of the hard rows (the equality
// ones if eq) are brought to the data, M (if not nullptr) becoming their
// new matrix

void HQPBlock::set_hard_rows( bool eq , const SparseMatrix * M ,
                              ModParam issueMod )
{
 auto & Mh = eq ? f_hard.A : f_hard.C;
 if( M ) {
  Mh = *M;
  Mh.makeCompressed();
  }

 if( ! ( AR & HasCns ) )
  return;

 update_rows( eq , Mh , eq ? v_heq.data() : v_hin.data() ,
              v_hcols.data() + ( eq ? 0 : f_hard.b.size() ) ,
              eq ? f_hard.b.data() : f_hard.d.data() , nullptr , 0 ,
              issueMod );

 }  // end( HQPBlock::set_hard_rows )

/*--------------------------------------------------------------------------*/

void HQPBlock::update_rows( bool eq , const SparseMatrix & M ,
                            FRowConstraint * rows ,
                            std::vector< Index > * cols , const double * rhs ,
                            const double * wgh , Index off ,
                            ModParam issueMod )
{
 std::vector< int > pos( f_n , -1 );

 for( Index h = 0 ; h < Index( M.rows() ) ; ++h ) {
  const double wg = wgh ? wgh[ h ] : 1.0;
  auto & row = rows[ h ];
  auto lf = static_cast< LinearFunction * >( row.get_function() );
  auto & cl = cols[ h ];

  // the new coefficient of each active Variable, the slack excluded
  Vec_double nc( cl.size() , 0 );
  for( Index k = 0 ; k < cl.size() ; ++k )
   pos[ cl[ k ] ] = k;
  LinearFunction::v_coeff_pair added;
  for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
   if( pos[ it.col() ] >= 0 )
    nc[ pos[ it.col() ] ] = wg * it.value();
   else
    added.emplace_back( & v_x[ it.col() ] , wg * it.value() );
  for( auto c : cl )
   pos[ c ] = -1;

  // only the coefficients that change are sent
  Vec_double chg;
  LinearFunction::Subset nms;
  for( Index k = 0 ; k < cl.size() ; ++k )
   if( lf->get_coefficient( k + off ) != nc[ k ] ) {
    chg.push_back( nc[ k ] );
    nms.push_back( k + off );
    }
  if( ! nms.empty() )
   lf->modify_coefficients( std::move( chg ) , std::move( nms ) , true ,
                            issueMod );

  if( ! added.empty() ) {
   for( const auto & a : added )
    cl.push_back( std::distance( v_x.data() , a.first ) );
   lf->add_variables( std::move( added ) , issueMod );
   }

  if( eq ) {
   if( row.get_rhs() != wg * rhs[ h ] )
    row.set_both( wg * rhs[ h ] , issueMod );
   }
  else
   if( row.get_rhs() != wg * rhs[ h ] )
    row.set_rhs( wg * rhs[ h ] , issueMod );
  }
 }  // end( HQPBlock::update_rows )

/*--------------------------------------------------------------------------*/

void HQPBlock::set_level_state( Index p , level_state s , double tol ,
                                ModParam issueMod )
{
 if( ( v_lvl_state[ p ] == s ) && ( s != kFixed ) )
  return;

 if( AR & HasVar ) {
  Vec_double v , w;
  if( s == kFixed ) {
   level_slacks( p , v , w );
   for( auto & wi : w )
    wi += tol;
   }

  auto fix = [ & ]( ColVariable & var , double val ) {
   if( s == kFree ) {
    if( var.is_fixed() )
     var.is_fixed( false , issueMod );
    return;
    }
   if( var.is_fixed() )
    var.is_fixed( false , issueMod );
   var.set_value( val );
   var.is_fixed( true , issueMod );
   };

  for( Index h = 0 ; h < get_num_eq_rows( p ) ; ++h )
   fix( v_v[ v_eq_start[ p ] + h ] , s == kFixed ? v[ h ] : 0.0 );
  for( Index h = 0 ; h < get_num_ineq_rows( p ) ; ++h )
   fix( v_w[ v_in_start[ p ] + h ] , s == kFixed ? w[ h ] : 0.0 );
  }

 if( AR & HasCns ) {
  const bool relax = ( s == kOff );
  for( Index i = v_eq_start[ p ] ; i < v_eq_start[ p + 1 ] ; ++i )
   if( v_eq[ i ].is_relaxed() != relax )
    v_eq[ i ].relax( relax , issueMod );
  for( Index i = v_in_start[ p ] ; i < v_in_start[ p + 1 ] ; ++i )
   if( v_in[ i ].is_relaxed() != relax )
    v_in[ i ].relax( relax , issueMod );
  }

 v_lvl_state[ p ] = char( s );

 }  // end( HQPBlock::set_level_state )

/*--------------------------------------------------------------------------*/

void HQPBlock::set_level_in_objective( Index p , bool in , ModParam issueMod )
{
 if( bool( v_lvl_obj[ p ] ) == in )
  return;

 v_lvl_obj[ p ] = in;
 if( ! ( AR & HasObj ) )
  return;

 auto f = static_cast< DQuadFunction * >( f_obj.get_function() );
 const double q = in ? f_scale / 2 : 0.0;
 const Index ne = v_b.size();
 for( int k = 0 ; k < 2 ; ++k ) {
  const Index start = f_n + ( k ? ne + v_in_start[ p ] : v_eq_start[ p ] );
  const Index m = k ? get_num_ineq_rows( p ) : get_num_eq_rows( p );
  if( ! m )
   continue;
  DQuadFunction::v_coeff quad( m , q ) , lin( m , 0 );
  f->modify_terms( quad.cbegin() , lin.cbegin() ,
                   DQuadFunction::Range( start , start + m ) , issueMod );
  }
 }  // end( HQPBlock::set_level_in_objective )

/*--------------------------------------------------------------------------*/

HQPBlock::Index HQPBlock::nl_row( hard_row_kind k , Index i ) const
{
 if( k == kQuadRow )
  return( i );
 if( k == kSumExpRow )
  return( f_hard.eq.size() + i );
 return( f_hard.eq.size() + f_hard.se.size() + i );
 }

/*--------------------------------------------------------------------------*/

HQPBlock::Index HQPBlock::num_nl_rows( hard_row_kind k ) const
{
 if( k == kQuadRow )
  return( f_hard.eq.size() );
 return( k == kSumExpRow ? f_hard.se.size() : f_hard.pn.size() );
 }

/*--------------------------------------------------------------------------*/

double HQPBlock::nl_rhs( hard_row_kind k , Index i ) const
{
 if( k == kQuadRow )
  return( f_hard.eq[ i ] );
 return( k == kSumExpRow ? f_hard.se[ i ] : f_hard.pn[ i ] );
 }

/*--------------------------------------------------------------------------*/

bool HQPBlock::row_cut( hard_row_kind k , Index i , const Vec_double & xf ,
                        bool want_log , LinearFunction::v_coeff_pair & cf ,
                        double & rhs , double & scale , bool & is_log ,
                        Vec_double & mb )
{
 const double u = nl_rhs( k , i );
 Vec_double g;
 is_log = false;
 mb.clear();
 if( k == kPNormRow ) {
  // s w'( A x + b ) + l' x <= u, w that of the dual norm at xf
  const double s = f_hard.PNs[ i ];
  pnorm_value( f_hard.PN[ i ] , f_hard.PNb[ i ] , f_hard.PNp[ i ] , xf ,
               & g , & mb );
  for( auto & gj : g )
   gj *= s;
  for( auto & wk : mb )
   wk *= s;
  add_row( f_hard.PNl , i , 1 , g );
  rhs = u - dot( mb , f_hard.PNb[ i ] );
  }
 else
  if( k == kSumExpRow ) {
   const double lse = lse_value( f_hard.SE[ i ] , f_hard.SEb[ i ] ,
                                 f_hard.SEc[ i ] , xf , & g );
   const double lx = row_dot( f_hard.SEl , i , xf );
   if( want_log && ( u - lx > 0 ) && std::isfinite( lse ) ) {
    // log sum_k c_k exp( z_k ) - log( u - l' x ) <= 0
    is_log = true;
    add_row( f_hard.SEl , i , 1 / ( u - lx ) , g );
    rhs = dot( g , xf ) - lse + std::log( u - lx );
    }
   else {
    const double S = std::exp( lse );
    for( auto & gj : g )
     gj *= S;
    add_row( f_hard.SEl , i , 1 , g );
    rhs = u - S - lx + dot( g , xf );
    }
   }
  else
   rhs = u - quad_value( f_hard.Q[ i ] , f_hard.Cq , i , xf , & g ) +
         dot( g , xf );

 if( ! std::isfinite( rhs ) )
  return( false );
 scale = 1;
 for( auto gj : g ) {
  if( ! std::isfinite( gj ) )
   return( false );
  scale = std::max( scale , std::abs( gj ) );
  }

 cf.clear();
 for( Index j = 0 ; j < f_n ; ++j )
  if( g[ j ] != 0 )
   cf.emplace_back( & v_x[ j ] , g[ j ] / scale );
 rhs /= scale;
 return( true );

 }  // end( HQPBlock::row_cut )

/*--------------------------------------------------------------------------*/

bool HQPBlock::known_cut( Index r , const LinearFunction::v_coeff_pair & cf ,
                          double rhs ) const
{
 const auto near = []( double a , double b ) {
  return( std::abs( a - b ) <= 1e-9 * std::max( 1.0 , std::abs( b ) ) );
  };

 for( const auto & c : v_row_cuts[ r ] ) {
  if( ! near( c.row->get_rhs() , rhs ) )
   continue;
  const auto lf = static_cast< const LinearFunction * >(
                                                     c.row->get_function() );
  if( lf->get_v_var().size() != cf.size() )
   continue;
  // cf is ordered as x, a cut changed by set_linear_terms() may be not
  const auto by_var = []( const auto & a , const auto & b ) {
   return( a.first < b.first );
   };
  LinearFunction::v_coeff_pair sorted;
  const auto * vp = & lf->get_v_var();
  if( ! std::is_sorted( vp->begin() , vp->end() , by_var ) ) {
   sorted = *vp;
   std::sort( sorted.begin() , sorted.end() , by_var );
   vp = & sorted;
   }
  const auto & v = *vp;
  bool same = true;
  for( Index h = 0 ; same && ( h < cf.size() ) ; ++h )
   same = ( v[ h ].first == cf[ h ].first ) &&
          near( v[ h ].second , cf[ h ].second );
  if( same )
   return( true );
  }
 return( false );

 }  // end( HQPBlock::known_cut )

/*--------------------------------------------------------------------------*/

void HQPBlock::shift_cuts( hard_row_kind k , Index i , double u0 , double u1 ,
                           ModParam issueMod )
{
 if( u0 == u1 )
  return;

 // the logarithmic form shifts by log( u1 / u0 ) only if l_i = 0
 bool l0 = true;
 if( k == kSumExpRow )
  for( SparseMatrix::InnerIterator it( f_hard.SEl , i ) ; it ; ++it )
   if( it.value() != 0 )
    l0 = false;

 edit_cuts( k , i , [ & ]( CutRef & c ) {
             double d = u1 - u0;
             if( c.log ) {
              if( ! ( l0 && ( u0 > 0 ) && ( u1 > 0 ) ) )
               return( false );
              d = std::log( u1 / u0 );
              }
             c.cut->set_rhs( c.cut->get_rhs() + d / c.scale , issueMod );
             return( true );
             } , issueMod );

 }  // end( HQPBlock::shift_cuts )

/*--------------------------------------------------------------------------*/

void HQPBlock::edit_cuts( hard_row_kind k , Index i ,
                          const std::function< bool( CutRef & ) > & f ,
                          ModParam issueMod )
{
 sync_cuts();
 auto & rc = v_row_cuts[ nl_row( k , i ) ];
 std::vector< std::list< FRowConstraint >::iterator > rmvd;
 for( auto it = rc.begin() ; it != rc.end() ; )
  if( f( *it ) )
   ++it;
  else {
   rmvd.push_back( it->cut );
   it = rc.erase( it );
   }

 if( ! rmvd.empty() )
  remove_dynamic_constraints( v_cuts , rmvd , issueMod );

 }  // end( HQPBlock::edit_cuts )

/*--------------------------------------------------------------------------*/

void HQPBlock::set_linear_terms( hard_row_kind k , const SparseMatrix & L ,
                                 ModParam issueMod )
{
 auto & Lk = ( k == kQuadRow ) ? f_hard.Cq :
             ( k == kSumExpRow ) ? f_hard.SEl : f_hard.PNl;
 SparseMatrix L1 = L;  // L may be Lk itself
 L1.makeCompressed();
 SparseMatrix L0 = std::move( Lk );
 Lk = std::move( L1 );
 if( ! ( AR & HasCns ) )
  return;

 std::vector< int > pos( f_n , -1 );
 std::vector< std::pair< Index , double > > dl;
 for( Index i = 0 ; i < Index( Lk.rows() ) ; ++i ) {
  // the change l' - l of the row, by merging the two sorted rows
  dl.clear();
  SparseMatrix::InnerIterator a( L0 , i ) , b( Lk , i );
  while( a || b )
   if( b && ( ( ! a ) || ( b.col() < a.col() ) ) ) {
    dl.emplace_back( b.col() , b.value() );
    ++b;
    }
   else
    if( a && ( ( ! b ) || ( a.col() < b.col() ) ) ) {
     dl.emplace_back( a.col() , - a.value() );
     ++a;
     }
    else {
     dl.emplace_back( a.col() , b.value() - a.value() );
     ++a;
     ++b;
     }
  dl.erase( std::remove_if( dl.begin() , dl.end() , []( const auto & d ) {
             return( d.second == 0 ); } ) , dl.end() );
  if( dl.empty() )
   continue;

  // the linear part of the QuadFunction
  if( ( k == kQuadRow ) && ( i < v_hqd.size() ) ) {
   auto qf = static_cast< QuadFunction * >( v_hqd[ i ].get_function() );
   const auto & vt = qf->get_v_var();
   for( Index h = 0 ; h < vt.size() ; ++h )
    pos[ std::get< 0 >( vt[ h ] ) - v_x.data() ] = h;
   Vec_double chg;
   Subset nms;
   DQuadFunction::v_coeff_triple added;
   for( const auto & [ j , d ] : dl )
    if( pos[ j ] >= 0 ) {
     nms.push_back( pos[ j ] );
     chg.push_back( qf->get_linear_coefficient( pos[ j ] ) + d );
     }
    else
     added.emplace_back( & v_x[ j ] , d , 0.0 );
   for( const auto & t : vt )
    pos[ std::get< 0 >( t ) - v_x.data() ] = -1;
   if( ! nms.empty() )
    qf->modify_linear_coefficients( std::move( chg ) , std::move( nms ) ,
                                    false , issueMod );
   if( ! added.empty() )
    qf->add_variables( std::move( added ) , {} , issueMod );
   }

  // the cuts take the change on their coefficients, their right-hand
  // side being independent of l, and are scaled again; the logarithmic
  // form of a sum-exp row is not linear in l: it goes
  edit_cuts( k , i , [ & ]( CutRef & c ) {
   if( c.log )
    return( false );
   auto lf = static_cast< LinearFunction * >( c.cut->get_function() );
   const auto & vv = lf->get_v_var();
   Vec_double g( vv.size() );
   for( Index h = 0 ; h < vv.size() ; ++h ) {
    pos[ vv[ h ].first - v_x.data() ] = h;
    g[ h ] = vv[ h ].second * c.scale;
    }
   LinearFunction::v_coeff_pair added;
   for( const auto & [ j , d ] : dl )
    if( pos[ j ] >= 0 )
     g[ pos[ j ] ] += d;
    else
     added.emplace_back( & v_x[ j ] , d );
   for( const auto & v : vv )
    pos[ v.first - v_x.data() ] = -1;
   double sc = 1;
   for( auto gj : g )
    sc = std::max( sc , std::abs( gj ) );
   for( const auto & a : added )
    sc = std::max( sc , std::abs( a.second ) );

   Vec_double chg;
   Subset nms;
   for( Index h = 0 ; h < g.size() ; ++h )
    if( lf->get_coefficient( h ) != g[ h ] / sc ) {
     chg.push_back( g[ h ] / sc );
     nms.push_back( h );
     }
   if( ! nms.empty() )
    lf->modify_coefficients( std::move( chg ) , std::move( nms ) , true ,
                             issueMod );
   if( ! added.empty() ) {
    for( auto & a : added )
     a.second /= sc;
    lf->add_variables( std::move( added ) , issueMod );
    }
   if( sc != c.scale ) {
    c.cut->set_rhs( c.cut->get_rhs() * c.scale / sc , issueMod );
    c.scale = sc;
    }
   return( true );
   } , issueMod );
  }
 }  // end( HQPBlock::set_linear_terms )

/*--------------------------------------------------------------------------*/

std::vector< HQPBlock::Vec_double > HQPBlock::shifts_of(
                                    const std::vector< SparseMatrix > & M ,
                                    const std::vector< Vec_double > & b ,
                                    const char * method ) const
{
 if( b.size() != M.size() )
  throw( std::invalid_argument( std::string( "HQPBlock::" ) + method +
                                ": wrong size" ) );
 auto nb = b;
 for( Index i = 0 ; i < nb.size() ; ++i ) {
  if( nb[ i ].empty() )
   nb[ i ].assign( M[ i ].rows() , 0 );
  if( nb[ i ].size() != Index( M[ i ].rows() ) )
   throw( std::invalid_argument( std::string( "HQPBlock::" ) + method +
                                 ": b " + std::to_string( i ) +
                                 " has wrong size" ) );
  }
 return( nb );

 }  // end( HQPBlock::shifts_of )

/*--------------------------------------------------------------------------*/

void HQPBlock::sync_cuts( void )
{
 std::size_t nc = 0;
 for( const auto & rc : v_row_cuts )
  nc += rc.size();
 if( nc == v_cuts.size() )
  return;

 std::set< const FRowConstraint * > there;
 for( const auto & c : v_cuts )
  there.insert( & c );
 for( auto & rc : v_row_cuts )
  rc.erase( std::remove_if( rc.begin() , rc.end() ,
                            [ & there ]( const CutRef & c ) {
                             return( ! there.count( c.row ) );
                             } ) , rc.end() );

 }  // end( HQPBlock::sync_cuts )

/*--------------------------------------------------------------------------*/
/*------------------------ End File HQPBlock.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
