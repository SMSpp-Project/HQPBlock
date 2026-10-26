/*--------------------------------------------------------------------------*/
/*------------------------------ File test.cpp -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Tester of HQPBlock and HQPSolver on small hierarchies built here: each
 * one is solved by every HQPSolver of the BlockSolverConfig BSPar.txt (one
 * for each inner Solver and setting) and compared level by level with the
 * reference solution computed in the null space of the equality tasks of
 * the higher levels [see test_common.h]. The cases are:
 *
 * - random hierarchies of mixed levels;
 *
 * - a hierarchy with a level of equalities only, one of inequalities only,
 *   one with both and an empty one;
 *
 * - an inequality task that cannot be satisfied at its level (positive
 *   slack, of known value) and one already satisfied (zero slack);
 *
 * - dependent rows: a repeated row, a linear combination of rows and two
 *   equal rows asking for different values, in both kinds of tasks;
 *
 * - weights different from 1;
 *
 * - a change of b, d, A (also out of the pattern) and C of the same
 *   HQPBlock between two compute(), as a step of a Model Predictive Control
 *   scheme, checked against the reference and against a new HQPBlock with
 *   the new data solved from scratch;
 *
 * - the round trip of an HQPBlock through serialize() and deserialize().
 *
 * The HQPSolver with int_HQP_Prox == 1 solve exactly the problem of the
 * reference, hence also x is compared; for the others only the values of
 * the levels, which are unique, are. A run with no HQPSolver able to work
 * (no inner Solver in the build) checks only the reference and the netCDF
 * round trip.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 */
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <filesystem>
#include <random>

#include "test_common.h"

/*--------------------------------------------------------------------------*/

using namespace hqp_test;

static const char * const config = "BSPar.txt";

/*--------------------------------------------------------------------------*/
// compares s with the reference with its center of the regularization: the
// values of the levels are well conditioned, x is not (the regularization
// is all that fixes it in some directions) and an interior-point inner
// Solver gives it to about 1e-4; the rows of the lower levels left in the
// model with free slacks make the Hessian singular, which the inner Solver
// deals with less accurately, so the tolerances are then wider

static int check( const std::string & what , HQPSolver * s ,
                  const RefSolution & ref , const RefSolution & ref0 )
{
 const bool lower = ! s->get_int_par( HQPSolver::int_HQP_RelaxLower );
 return( compare( what , s ,
                  s->get_int_par( HQPSolver::int_HQP_Prox ) ? ref : ref0 ,
                  lower ? 1e-5 : 1e-6 , lower ? 1e-2 : 1e-3 ) );
 }

/*--------------------------------------------------------------------------*/
// a random matrix with about density * size non-zero entries

static Mat random_matrix( std::mt19937 & g , int r , int c ,
                          double density )
{
 std::uniform_real_distribution< double > u( -1 , 1 ) , z( 0 , 1 );
 Mat M = Mat::Zero( r , c );
 for( int i = 0 ; i < r ; ++i )
  for( int j = 0 ; j < c ; ++j )
   if( z( g ) < density )
    M( i , j ) = u( g );
 return( M );
 }

/*--------------------------------------------------------------------------*/

static Vec random_vector( std::mt19937 & g , int r )
{
 std::uniform_real_distribution< double > u( -1 , 1 );
 Vec v( r );
 for( int i = 0 ; i < r ; ++i )
  v( i ) = u( g );
 return( v );
 }

/*--------------------------------------------------------------------------*/

static void add_level( Instance & in , Mat A , Vec b , Mat C , Vec d )
{
 in.A.push_back( std::move( A ) );
 in.b.push_back( std::move( b ) );
 in.C.push_back( std::move( C ) );
 in.d.push_back( std::move( d ) );
 }

/*--------------------------------------------------------------------------*/

static Instance random_instance( unsigned seed )
{
 std::mt19937 g( seed );
 Instance in;
 in.name = "random " + std::to_string( seed );
 in.n = 10;
 in.eps = 1e-3;
 std::uniform_int_distribution< int > rows( 0 , 4 );
 for( int p = 0 ; p < 5 ; ++p ) {
  const int m = rows( g ) , l = rows( g );
  // small right-hand sides of the inequalities, so that some are violated
  // at the solution of the higher levels
  add_level( in , random_matrix( g , m , in.n , 0.5 ) ,
             random_vector( g , m ) , random_matrix( g , l , in.n , 0.5 ) ,
             0.3 * random_vector( g , l ) );
  }
 return( in );
 }

/*--------------------------------------------------------------------------*/
// equalities only, inequalities only, both, nothing, equalities only

static Instance kinds_instance( void )
{
 std::mt19937 g( 77 );
 Instance in;
 in.name = "kinds of levels";
 in.n = 8;
 in.eps = 1e-3;
 add_level( in , random_matrix( g , 3 , 8 , 0.6 ) , random_vector( g , 3 ) ,
            Mat( 0 , 8 ) , Vec( 0 ) );
 add_level( in , Mat( 0 , 8 ) , Vec( 0 ) , random_matrix( g , 6 , 8 , 0.6 ) ,
            -0.5 * Vec::Ones( 6 ) );
 add_level( in , random_matrix( g , 2 , 8 , 0.6 ) , random_vector( g , 2 ) ,
            random_matrix( g , 3 , 8 , 0.6 ) , 0.1 * random_vector( g , 3 ) );
 add_level( in , Mat( 0 , 8 ) , Vec( 0 ) , Mat( 0 , 8 ) , Vec( 0 ) );
 add_level( in , Mat::Identity( 8 , 8 ) , random_vector( g , 8 ) ,
            Mat( 0 , 8 ) , Vec( 0 ) );
 return( in );
 }

/*--------------------------------------------------------------------------*/
// level 0 fixes x_0 = 1 and x_1 = 0; level 1 asks x_0 <= 0 (it cannot:
// slack 1, value 1/2) and x_1 <= 5 (already satisfied: slack 0); level 2
// asks x_0 + x_2 <= -2, i.e., x_2 <= -3 (satisfiable: value 0), and level 3
// x_2 = 0, which the inequality of level 2 keeps at -3 (value 9/2)

static Instance slack_instance( void )
{
 Instance in;
 in.name = "infeasible and satisfied tasks";
 in.n = 4;
 in.eps = 1e-3;
 Mat A0 = Mat::Zero( 2 , 4 );
 A0( 0 , 0 ) = 1;
 A0( 1 , 1 ) = 1;
 add_level( in , A0 , Vec::Unit( 2 , 0 ) , Mat( 0 , 4 ) , Vec( 0 ) );
 Mat C1 = Mat::Zero( 2 , 4 );
 C1( 0 , 0 ) = 1;
 C1( 1 , 1 ) = 1;
 Vec d1( 2 );
 d1 << 0 , 5;
 add_level( in , Mat( 0 , 4 ) , Vec( 0 ) , C1 , d1 );
 Mat C2 = Mat::Zero( 1 , 4 );
 C2( 0 , 0 ) = 1;
 C2( 0 , 2 ) = 1;
 add_level( in , Mat( 0 , 4 ) , Vec( 0 ) , C2 , Vec::Constant( 1 , -2 ) );
 Mat A3 = Mat::Zero( 1 , 4 );
 A3( 0 , 2 ) = 1;
 add_level( in , A3 , Vec::Zero( 1 ) , Mat( 0 , 4 ) , Vec( 0 ) );
 return( in );
 }

/*--------------------------------------------------------------------------*/
// dependent rows: in level 0 a row repeated and the sum of two rows; in
// level 1 two equal rows asking for different values (least squares: their
// mean) and a repeated inequality; in level 2 the inequalities of level 1
// again, tighter

static Instance dependent_instance( void )
{
 std::mt19937 g( 5 );
 Instance in;
 in.name = "dependent rows";
 in.n = 6;
 in.eps = 1e-3;
 Mat A0 = random_matrix( g , 2 , 6 , 0.8 );
 Mat A0d( 4 , 6 );
 A0d << A0 , A0.row( 0 ) , A0.row( 0 ) + A0.row( 1 );
 Vec b0 = random_vector( g , 2 );
 Vec b0d( 4 );
 b0d << b0 , b0( 0 ) , b0( 0 ) + b0( 1 );
 add_level( in , A0d , b0d , Mat( 0 , 6 ) , Vec( 0 ) );
 Mat r = random_matrix( g , 1 , 6 , 0.8 );
 Mat A1( 2 , 6 );
 A1 << r , r;
 Vec b1( 2 );
 b1 << 1 , 3;
 Mat c = random_matrix( g , 1 , 6 , 0.8 );
 Mat C1( 2 , 6 );
 C1 << c , c;
 add_level( in , A1 , b1 , C1 , Vec::Constant( 2 , -1 ) );
 add_level( in , Mat( 0 , 6 ) , Vec( 0 ) , C1 , Vec::Constant( 2 , -2 ) );
 add_level( in , Mat::Identity( 6 , 6 ) , Vec::Zero( 6 ) , Mat( 0 , 6 ) ,
            Vec( 0 ) );
 return( in );
 }

/*--------------------------------------------------------------------------*/

static Instance weighted_instance( void )
{
 Instance in = random_instance( 11 );
 in.name = "weights";
 std::mt19937 g( 12 );
 std::uniform_real_distribution< double > u( 0.5 , 3 );
 for( int p = 0 ; p < in.levels() ; ++p ) {
  in.we.push_back( Vec( in.b[ p ].size() ) );
  in.wi.push_back( Vec( in.d[ p ].size() ) );
  for( auto & x : in.we.back() )
   x = u( g );
  for( auto & x : in.wi.back() )
   x = u( g );
  }
 return( in );
 }

/*--------------------------------------------------------------------------*/
// solves in with the HQPSolver of the config, compares them with the
// reference, and returns the number of errors

static int solve_and_check( const Instance & in , int & usable )
{
 const RefSolution ref = reference( in ) , ref0 = reference( in , false );
 if( ! ( ref.ok && ref0.ok ) ) {
  std::cout << in.name << ": the reference failed -> KO" << std::endl;
  return( 1 );
  }

 HQPBlock block;
 load( block , in );
 auto slvrs = attach( block , config );

 int errors = 0;
 usable = 0;
 for( auto s : slvrs ) {
  if( ! s->get_inner_Solver() ) {
   std::cout << in.name << " [" << label( s ) << "]: no inner Solver in "
             << "this build, skipped" << std::endl;
   continue;
   }
  ++usable;
  const int st = s->compute();
  if( st != Solver::kOK ) {
   std::cout << in.name << " [" << label( s ) << "]: status " << st
             << " -> KO" << std::endl;
   ++errors;
   continue;
   }
  errors += check( in.name , s , ref , ref0 );
  }

 block.unregister_Solvers( true );
 return( errors );
 }

/*--------------------------------------------------------------------------*/
// the expected values of the levels of slack_instance()

static int check_slack_values( void )
{
 Instance in = slack_instance();
 in.eps = 1e-9;
 const RefSolution ref = reference( in );
 const double expected[] = { 0 , 0.5 , 0 , 4.5 };
 int errors = 0;
 for( int p = 0 ; p < 4 ; ++p )
  if( std::abs( ref.value[ p ] - expected[ p ] ) > 1e-6 ) {
   std::cout << "reference of " << in.name << ": level " << p << " value "
             << ref.value[ p ] << " instead of " << expected[ p ]
             << " -> KO" << std::endl;
   ++errors;
   }
 if( ! errors )
  std::cout << "reference of " << in.name << ": values 0, 1/2, 0, 9/2 "
            << "-> OK" << std::endl;
 return( errors );
 }

/*--------------------------------------------------------------------------*/
// a step of Model Predictive Control: new b, d, A and C on the same
// HQPBlock, compared with the reference and with a new HQPBlock

static int mpc_step( void )
{
 Instance in = random_instance( 21 );
 in.name = "MPC step";
 HQPBlock block;
 load( block , in );
 auto slvrs = attach( block , config );
 int errors = 0;
 for( auto s : slvrs )
  if( s->get_inner_Solver() )
   s->compute();

 // the new data: b and d moved, A and C perturbed, an entry out of the
 // pattern of a row added and one of the pattern zeroed
 std::mt19937 g( 22 );
 Instance nw = in;
 nw.name = "MPC step, after the change";
 for( int p = 0 ; p < in.levels() ; ++p ) {
  nw.b[ p ] += 0.2 * random_vector( g , nw.b[ p ].size() );
  nw.d[ p ] += 0.2 * random_vector( g , nw.d[ p ].size() );
  for( auto * M : { & nw.A[ p ] , & nw.C[ p ] } ) {
   for( int i = 0 ; i < M->rows() ; ++i )
    for( int j = 0 ; j < M->cols() ; ++j )
     if( ( *M )( i , j ) != 0 )
      ( *M )( i , j ) *= 1 + 0.1 * random_vector( g , 1 )( 0 );
   if( M->rows() ) {
    for( int j = 0 ; j < M->cols() ; ++j )
     if( ( *M )( 0 , j ) == 0 ) {
      ( *M )( 0 , j ) = 0.5;
      break;
      }
    for( int j = M->cols() - 1 ; j >= 0 ; --j )
     if( ( *M )( M->rows() - 1 , j ) != 0 ) {
      ( *M )( M->rows() - 1 , j ) = 0;
      break;
      }
    }
   }
  std::vector< double > b( nw.b[ p ].data() ,
                           nw.b[ p ].data() + nw.b[ p ].size() );
  std::vector< double > d( nw.d[ p ].data() ,
                           nw.d[ p ].data() + nw.d[ p ].size() );
  block.set_b( p , b );
  block.set_d( p , d );
  block.set_A( p , nw.A[ p ].sparseView() );
  block.set_C( p , nw.C[ p ].sparseView() );
  }

 const RefSolution ref = reference( nw ) , ref0 = reference( nw , false );
 HQPBlock fresh;
 load( fresh , nw );
 auto fslvrs = attach( fresh , config );

 for( std::size_t i = 0 ; i < slvrs.size() ; ++i ) {
  auto s = slvrs[ i ];
  if( ! s->get_inner_Solver() )
   continue;
  const int st = s->compute();
  const int fst = fslvrs[ i ]->compute();
  if( ( st != Solver::kOK ) || ( fst != Solver::kOK ) ) {
   std::cout << nw.name << " [" << label( s ) << "]: status " << st
             << " (from scratch " << fst << ") -> KO" << std::endl;
   ++errors;
   continue;
   }
  errors += check( nw.name , s , ref , ref0 );
  errors += check( nw.name + ", from scratch" , fslvrs[ i ] , ref , ref0 );
  }

 block.unregister_Solvers( true );
 fresh.unregister_Solvers( true );
 return( errors );
 }

/*--------------------------------------------------------------------------*/
// serialize() and deserialize() give back the same data

static int round_trip( const Instance & in )
{
 HQPBlock block;
 load( block , in );
 const auto file = ( std::filesystem::temp_directory_path() /
                     "HQPBlock_round_trip.nc4" ).string();
 block.serialize( file , eBlockFile );
 auto back = dynamic_cast< HQPBlock * >( Block::deserialize( file ) );
 std::filesystem::remove( file );

 bool ok = back && ( back->get_num_var() == block.get_num_var() ) &&
  ( back->get_num_levels() == block.get_num_levels() ) &&
  ( back->get_b() == block.get_b() ) && ( back->get_d() == block.get_d() ) &&
  ( back->get_we() == block.get_we() ) &&
  ( back->get_wi() == block.get_wi() ) &&
  ( back->get_regularization() == block.get_regularization() );
 if( ok )
  for( Block::Index p = 0 ; p < block.get_num_levels() ; ++p ) {
   const Mat a = Mat( back->get_A( p ) ) - Mat( block.get_A( p ) );
   const Mat c = Mat( back->get_C( p ) ) - Mat( block.get_C( p ) );
   if( ( a.size() && ( a.cwiseAbs().maxCoeff() != 0 ) ) ||
       ( c.size() && ( c.cwiseAbs().maxCoeff() != 0 ) ) )
    ok = false;
   }
 delete back;
 std::cout << "netCDF round trip of " << in.name
           << ( ok ? " -> OK" : " -> KO" ) << std::endl;
 return( ok ? 0 : 1 );
 }

/*--------------------------------------------------------------------------*/

int main( int argc , char ** argv )
{
 int errors = 0 , usable = 0 , u;

 errors += check_slack_values();

 std::vector< Instance > cases;
 for( unsigned seed = 1 ; seed <= 6 ; ++seed )
  cases.push_back( random_instance( seed ) );
 cases.push_back( kinds_instance() );
 cases.push_back( slack_instance() );
 cases.push_back( dependent_instance() );
 cases.push_back( weighted_instance() );

 for( const auto & in : cases ) {
  errors += solve_and_check( in , u );
  usable = std::max( usable , u );
  }

 if( usable )
  errors += mpc_step();

 errors += round_trip( kinds_instance() );
 errors += round_trip( weighted_instance() );

 if( ! usable )
  std::cout << "no HQPSolver with an inner Solver: only the reference and "
            << "the netCDF format have been checked" << std::endl;

 std::cout << ( errors ? "HQPBlock: FAILED" : "HQPBlock: all tests passed" )
           << std::endl;
 return( errors ? 1 : 0 );
 }

/*--------------------------------------------------------------------------*/
/*---------------------------- End File test.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
