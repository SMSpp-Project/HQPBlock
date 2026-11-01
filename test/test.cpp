/*--------------------------------------------------------------------------*/
/*------------------------------ File test.cpp -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Unit test of HQPBlock, which needs the core library only: what is
 * checked is what the Block itself does, on small hierarchies written here
 * whose degenerate parts are a level of equalities only, one of
 * inequalities only, an empty one, rows with no entries and weights
 * different from 1, and HQPStageSolver, the Solver of the module that
 * needs nothing else. The cases are:
 *
 * - load() and its errors (no variables, no levels, sizes that do not
 *   agree, a negative regularization), and the sizes and starts of the rows
 *   of each level;
 *
 * - the abstract representation: the ColVariable x, v and w, the row of
 *   each task with its slack as first active Variable, the coefficients and
 *   the right-hand sides weighted, and the Objective of level 0;
 *
 * - reset_levels() and next_level(): the state of each level, the slacks
 *   fixed at their values at the current x (plus the tolerance on the
 *   inequalities), the rows relaxed and the coefficients of the Objective,
 *   also for an abstract representation generated after the levels have
 *   been reset, and the errors of next_level();
 *
 * - the Modification issued by set_b(), set_d(), set_A() (in and out of
 *   the pattern of a row, an entry set to 0), set_C(), set_we(), set_wi(),
 *   set_regularization(), set_center(), reset_levels() and next_level(),
 *   counted by a FakeSolver: only what changes is issued, nothing with
 *   eNoMod, and a change made before the abstract representation is there
 *   is found in it once generated;
 *
 * - the netCDF format: the round trip of hierarchies with empty levels and
 *   empty rows, with no equality or no inequality rows at all and with no
 *   entries in any row, which of the optional variables are written, and
 *   the errors of deserialize() on a file whose rows do not add up, whose
 *   starts decrease or whose column index is out of range (which
 *   Block::deserialize() reports, giving no Block);
 *
 * - print(), whose lines are compared whole;
 *
 * - the weights of the variables in the regularization: their coefficients
 *   in the Objective, also after set_center() and set_regularization(),
 *   their netCDF variable and the errors of load();
 *
 * - refix_level(), which raises the inequality slacks of a fixed level and
 *   fixes again the equality ones, and get_Solution();
 *
 * - the HQPDistBlock of a small hierarchy split between 2 owners, with
 *   local rows, coupling rows (a tie of entries, an owner given), a row with
 *   no entries and a column copied by both owners: the owners of the rows,
 *   the columns and copies of each sub-Block, the consensus rows, the weights
 *   of the regularization, the rows of the sub-Blocks, the x and the slacks
 *   of the hierarchy against those of the HQPBlock of the same data, the
 *   copies brought to their owners and the slacks fixed again by
 *   next_level(), a change of the data in the pattern (Modification of the
 *   sub-Block) and out of it (an NBModification first, then the sub-Blocks
 *   anew), the netCDF format and the errors of load();
 *
 * - the same HQPDistBlock with the structure kBenders, read from
 *   structure-benders.txt: the coupling rows and their slacks in the root,
 *   the t, the y and the link rows of each owner, their relaxation and the
 *   slacks of the coupling rows along the levels, a change of the data in
 *   and out of the pattern of a coupling row (an inequality and an equality
 *   one), and the structure refused once the abstract representation is
 *   there.
 *
 * - the stage structure: load() and its errors, the getters, the netCDF
 *   format with and without the stages of the rows, a file without it,
 *   and print();
 *
 * - the recursion of HQPStageSolver::StageEQP against the dense KKT system
 *   of the whole horizon on random problems: rows from 0 to more than the
 *   inputs at each stage (constraints-to-go), as many rows as inputs, rows
 *   dependent in a stage and through the dynamics, rows on x_0 and x_N, an
 *   x_0 of size 0, sizes that change along the stages, one stage, stages
 *   with no input, the size of the 8 robots of the radial scenario, an
 *   inconsistent dependent row (reported) and no regularization; the
 *   trajectory, the objective, the rows and the stationarity with the
 *   multipliers;
 *
 * - HQPStageSolver on the levels of hierarchies whose values are known (an
 *   integrator with input limits, smoothness and goals, with the stages of
 *   the rows derived and given; the initial state as a column; dependent
 *   rows of a task and of the levels above), on two steps of Model
 *   Predictive Control (set_b(), then set_A() on the dynamics) against a
 *   new HQPBlock, the warm start taking no more iterations than the cold
 *   one, what it refuses (no stage structure, a level above the dynamics,
 *   a row of the dynamics with two states of its stage, a row out of its
 *   given stage only at its level, a Block that is not an HQPBlock, wrong
 *   parameters) and the iteration limit;
 *
 * - with instance files as arguments, only HQPStageSolver on each of them,
 *   against the solution of each level that it carries.
 *
 * The HQPSolver, which needs an inner Solver from another module, is tested
 * by the suite HQPBlock of the system tests of SMS++.
 *
 * \author Donato Meoli \n
 *         Dipartimento di Informatica \n
 *         Universita' di Pisa \n
 */
/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <netcdf>

#include "DQuadFunction.h"

#include "FakeSolver.h"

#include "HQPBlock.h"

#include "HQPDistBlock.h"

#include "HQPStageSolver.h"

#include "ColVariableSolution.h"

#include "LinearFunction.h"

/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

using Mat = Eigen::MatrixXd;
using Vec = std::vector< double >;
using Index = Block::Index;

static int errors = 0;

#define CHECK( c ) \
 do { if( ! ( c ) ) { \
  std::cout << "  " << __FILE__ << ":" << __LINE__ << ": " #c " -> KO" \
            << std::endl; \
  ++errors; } } while( 0 )

/*--------------------------------------------------------------------------*/
// true if f() throws an E

template< class E >
static bool throws( const std::function< void( void ) > & f )
{
 try {
  f();
  }
 catch( const E & ) {
  return( true );
  }
 catch( ... ) {
  return( false );
  }
 return( false );
 }

/*--------------------------------------------------------------------------*/
// the data of a hierarchy, dense, to be loaded into an HQPBlock

struct Data {
 Index n = 0;
 std::vector< Mat > A , C;
 std::vector< Vec > b , d , we , wi;
 double eps = 0;
 HQPBlock::StageData st;
 };

static HQPBlock::SparseMatrix sparse( const Mat & M )
{
 return( M.sparseView() );
 }

static void load( HQPBlock & block , const Data & D )
{
 std::vector< HQPBlock::SparseMatrix > A , C;
 for( const auto & M : D.A )
  A.push_back( sparse( M ) );
 for( const auto & M : D.C )
  C.push_back( sparse( M ) );
 auto b = D.b , d = D.d , we = D.we , wi = D.wi;
 auto st = D.st;
 block.load( D.n , std::move( A ) , std::move( b ) , std::move( C ) ,
	     std::move( d ) , std::move( we ) , std::move( wi ) , D.eps , {} ,
	     std::move( st ) );
 }

/*--------------------------------------------------------------------------*/
// 3 variables, 4 levels: level 0 has two equalities, the second with no
// entries, and no inequality; level 1 has no equality and two inequalities,
// the first with no entries; level 2 is empty; level 3 has an equality of
// weight 2 and an inequality whose weight is not given (hence 1)

static Data edge_data( void )
{
 Data D;
 D.n = 3;
 D.eps = 0.5;
 Mat A0( 2 , 3 );
 A0 << 1 , -2 , 0 ,
       0 , 0 , 0;
 Mat C1( 2 , 3 );
 C1 << 0 , 0 , 0 ,
       0 , 3 , 1;
 Mat A3( 1 , 3 ) , C3( 1 , 3 );
 A3 << 1 , 1 , 1;
 C3 << -1 , 0 , 4;
 D.A = { A0 , Mat( 0 , 3 ) , Mat( 0 , 3 ) , A3 };
 D.b = { { 1 , 2 } , {} , {} , { 3 } };
 D.C = { Mat( 0 , 3 ) , C1 , Mat( 0 , 3 ) , C3 };
 D.d = { {} , { -1 , 2 } , {} , { 0.5 } };
 D.we = { {} , {} , {} , { 2 } };
 D.wi = { {} , {} , {} , {} };
 return( D );
 }

/*--------------------------------------------------------------------------*/
// one level of equalities, one of inequalities, no weights

static Data small_data( void )
{
 Data D;
 D.n = 2;
 Mat A0( 1 , 2 ) , C1( 2 , 2 );
 A0 << 1 , 1;
 C1 << 1 , 0 ,
       0 , -1;
 D.A = { A0 , Mat( 0 , 2 ) };
 D.b = { { 2 } , {} };
 D.C = { Mat( 0 , 2 ) , C1 };
 D.d = { {} , { 0.5 , 0 } };
 return( D );
 }

/*--------------------------------------------------------------------------*/
// the LinearFunction of the row i (among all) of an equality (eq) or
// inequality task

static const LinearFunction * row_function( const HQPBlock & block , bool eq ,
                                            Index i )
{
 const auto & row = eq ? block.get_eq()[ i ] : block.get_ineq()[ i ];
 return( static_cast< const LinearFunction * >(
          const_cast< FRowConstraint & >( row ).get_function() ) );
 }

/*--------------------------------------------------------------------------*/
// the row i (among all) of the kind eq agrees with the data of block: its
// slack first with coefficient -1, then the weighted entries of the row,
// and the weighted right-hand side

static bool row_agrees( const HQPBlock & block , bool eq , Index i )
{
 Index p = 0;
 while( ( eq ? block.get_eq_start( p + 1 ) : block.get_ineq_start( p + 1 ) )
        <= i )
  ++p;
 const Index h = i - ( eq ? block.get_eq_start( p )
                          : block.get_ineq_start( p ) );
 const double wg = eq ? block.get_we()[ i ] : block.get_wi()[ i ];
 const Mat M( eq ? block.get_A( p ) : block.get_C( p ) );
 const auto lf = row_function( block , eq , i );
 const auto & slack = eq ? block.get_v()[ i ] : block.get_w()[ i ];
 if( ( lf->get_active_var( 0 ) != & slack ) ||
     ( lf->get_coefficient( 0 ) != -1 ) )
  return( false );

 // the coefficient of x_j among the active Variable, 0 if not there
 Vec cf( block.get_num_var() , 0 );
 for( Index k = 1 ; k < lf->get_num_active_var() ; ++k ) {
  const auto x = static_cast< const ColVariable * >(
                                                 lf->get_active_var( k ) );
  const auto j = x - block.get_x().data();
  if( ( j < 0 ) || ( Index( j ) >= block.get_num_var() ) )
   return( false );
  cf[ j ] += lf->get_coefficient( k );
  }
 for( Index j = 0 ; j < block.get_num_var() ; ++j )
  if( cf[ j ] != wg * M( h , j ) )
   return( false );

 const auto & row = eq ? block.get_eq()[ i ] : block.get_ineq()[ i ];
 if( eq )
  return( ( row.get_lhs() == wg * block.get_b()[ i ] ) &&
          ( row.get_rhs() == wg * block.get_b()[ i ] ) );
 return( ( row.get_lhs() == -Inf< double >() ) &&
         ( row.get_rhs() == wg * block.get_d()[ i ] ) );
 }

/*--------------------------------------------------------------------------*/
// all the rows agree with the data

static bool rows_agree( const HQPBlock & block )
{
 for( Index i = 0 ; i < block.get_eq().size() ; ++i )
  if( ! row_agrees( block , true , i ) )
   return( false );
 for( Index i = 0 ; i < block.get_ineq().size() ; ++i )
  if( ! row_agrees( block , false , i ) )
   return( false );
 return( true );
 }

/*--------------------------------------------------------------------------*/
// the DQuadFunction of the Objective

static const DQuadFunction * objective( HQPBlock & block )
{
 return( static_cast< const DQuadFunction * >(
          static_cast< FRealObjective * >( block.get_objective() )->
          get_function() ) );
 }

/*--------------------------------------------------------------------------*/
// the Objective is that of the current state of the levels, times sc: (
// - eps c , eps / 2 ) on x, ( 0 , 1 / 2 ) on the slacks of the levels in
// it, given by in[ p ], ( 0 , 0 ) on the others, and the constant eps || c
// ||^2 / 2

static bool objective_agrees( HQPBlock & block ,
                              const std::vector< bool > & in ,
                              double sc = 1 )
{
 const auto f = objective( block );
 const Index n = block.get_num_var() , ne = block.get_eq().size();
 const double eps = block.get_regularization();
 const auto & c = block.get_center();
 if( f->get_num_active_var() != n + ne + block.get_ineq().size() )
  return( false );
 double c2 = 0;
 for( auto cj : c )
  c2 += cj * cj;
 if( std::abs( f->get_constant_term() - sc * eps * c2 / 2 ) > 1e-15 )
  return( false );
 for( Index j = 0 ; j < n ; ++j )
  if( ( f->get_active_var( j ) != & block.get_x()[ j ] ) ||
      ( f->get_linear_coefficient( j ) !=
        ( c.empty() ? 0.0 : - sc * eps * c[ j ] ) ) ||
      ( f->get_quadratic_coefficient( j ) != sc * eps / 2 ) )
   return( false );
 for( Index p = 0 ; p < block.get_num_levels() ; ++p ) {
  const double q = in[ p ] ? sc / 2 : 0.0;
  for( Index i = block.get_eq_start( p ) ; i < block.get_eq_start( p + 1 ) ;
       ++i )
   if( ( f->get_active_var( n + i ) != & block.get_v()[ i ] ) ||
       ( f->get_linear_coefficient( n + i ) != 0 ) ||
       ( f->get_quadratic_coefficient( n + i ) != q ) )
    return( false );
  for( Index i = block.get_ineq_start( p ) ;
       i < block.get_ineq_start( p + 1 ) ; ++i )
   if( ( f->get_active_var( n + ne + i ) != & block.get_w()[ i ] ) ||
       ( f->get_linear_coefficient( n + ne + i ) != 0 ) ||
       ( f->get_quadratic_coefficient( n + ne + i ) != q ) )
    return( false );
  }
 return( true );
 }

/*--------------------------------------------------------------------------*/
// the slacks of level p are fixed (at the values in v and w, if given) or
// free, and its rows relaxed or not

static bool level_is( const HQPBlock & block , Index p , bool fixed ,
                      bool relaxed , const Vec * v = nullptr ,
                      const Vec * w = nullptr )
{
 for( Index i = block.get_eq_start( p ) ; i < block.get_eq_start( p + 1 ) ;
      ++i ) {
  const auto & x = block.get_v()[ i ];
  if( ( x.is_fixed() != fixed ) ||
      ( block.get_eq()[ i ].is_relaxed() != relaxed ) )
   return( false );
  if( v && ( x.get_value() != ( *v )[ i - block.get_eq_start( p ) ] ) )
   return( false );
  }
 for( Index i = block.get_ineq_start( p ) ;
      i < block.get_ineq_start( p + 1 ) ; ++i ) {
  const auto & x = block.get_w()[ i ];
  if( ( x.is_fixed() != fixed ) ||
      ( block.get_ineq()[ i ].is_relaxed() != relaxed ) )
   return( false );
  if( w && ( x.get_value() != ( *w )[ i - block.get_ineq_start( p ) ] ) )
   return( false );
  }
 return( true );
 }

/*--------------------------------------------------------------------------*/

static void generate( HQPBlock & block )
{
 block.generate_abstract_variables();
 block.generate_abstract_constraints();
 block.generate_objective();
 }

/*--------------------------------------------------------------------------*/
// the x of the HQPBlock, writable

static std::vector< ColVariable > & x_of( HQPBlock & block )
{
 return( * block.get_static_variable_v< ColVariable >( "x" ) );
 }

/*--------------------------------------------------------------------------*/
/*--------------------------------- TESTS ----------------------------------*/
/*--------------------------------------------------------------------------*/

static void test_load( void )
{
 std::cout << "load() and the sizes of the levels" << std::endl;
 const int e0 = errors;

 // the errors
 auto bad = []( const std::function< void( Data & ) > & change ) {
  Data D = edge_data();
  change( D );
  HQPBlock block;
  return( throws< std::invalid_argument >( [ & ]() { load( block , D ); } ) );
  };
 CHECK( bad( []( Data & D ) { D.n = 0; } ) );
 CHECK( bad( []( Data & D ) { D.A.clear(); D.b.clear(); D.C.clear();
                              D.d.clear(); D.we.clear(); D.wi.clear(); } ) );
 CHECK( bad( []( Data & D ) { D.b.pop_back(); } ) );
 CHECK( bad( []( Data & D ) { D.we.pop_back(); } ) );
 CHECK( bad( []( Data & D ) { D.A[ 3 ] = Mat::Zero( 1 , 4 ); } ) );
 CHECK( bad( []( Data & D ) { D.C[ 1 ] = Mat::Zero( 2 , 2 ); } ) );
 CHECK( bad( []( Data & D ) { D.b[ 0 ].push_back( 0 ); } ) );
 CHECK( bad( []( Data & D ) { D.d[ 3 ].clear(); } ) );
 CHECK( bad( []( Data & D ) { D.we[ 3 ] = { 1 , 1 }; } ) );
 CHECK( bad( []( Data & D ) { D.wi[ 1 ] = { 1 }; } ) );
 CHECK( bad( []( Data & D ) { D.eps = -1; } ) );
 CHECK( bad( []( Data & D ) { D.eps = std::nan( "" ); } ) );

 // the sizes
 HQPBlock block;
 load( block , edge_data() );
 CHECK( block.get_num_var() == 3 );
 CHECK( block.get_num_levels() == 4 );
 const Index m[] = { 2 , 0 , 0 , 1 } , l[] = { 0 , 2 , 0 , 1 };
 const Index es[] = { 0 , 2 , 2 , 2 , 3 } , is[] = { 0 , 0 , 2 , 2 , 3 };
 for( Index p = 0 ; p < 4 ; ++p ) {
  CHECK( block.get_num_eq_rows( p ) == m[ p ] );
  CHECK( block.get_num_ineq_rows( p ) == l[ p ] );
  }
 for( Index p = 0 ; p <= 4 ; ++p ) {
  CHECK( block.get_eq_start( p ) == es[ p ] );
  CHECK( block.get_ineq_start( p ) == is[ p ] );
  }
 CHECK( block.get_b() == Vec( { 1 , 2 , 3 } ) );
 CHECK( block.get_d() == Vec( { -1 , 2 , 0.5 } ) );
 CHECK( block.get_we() == Vec( { 1 , 1 , 2 } ) );
 CHECK( block.get_wi() == Vec( { 1 , 1 , 1 } ) );
 CHECK( block.get_regularization() == 0.5 );
 CHECK( block.get_center().empty() );
 CHECK( block.get_level() == 0 );
 CHECK( ! block.anyone_there() );

 // the weights may also be not given at all
 Data D = edge_data();
 D.we.clear();
 D.wi.clear();
 load( block , D );
 CHECK( block.get_we() == Vec( { 1 , 1 , 1 } ) );

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/

static void test_abstract( void )
{
 std::cout << "the abstract representation" << std::endl;
 const int e0 = errors;

 HQPBlock block;
 load( block , edge_data() );
 generate( block );
 CHECK( block.anyone_there() );
 CHECK( block.get_x().size() == 3 );
 CHECK( block.get_v().size() == 3 );
 CHECK( block.get_w().size() == 3 );
 CHECK( block.get_eq().size() == 3 );
 CHECK( block.get_ineq().size() == 3 );
 for( const auto & x : block.get_x() )
  CHECK( ( ! x.is_positive() ) && ( ! x.is_negative() ) &&
         ( ! x.is_fixed() ) );
 for( const auto & v : block.get_v() )
  CHECK( ( ! v.is_positive() ) && ( ! v.is_negative() ) &&
         ( ! v.is_fixed() ) );
 for( const auto & w : block.get_w() )
  CHECK( w.is_positive() && ( ! w.is_fixed() ) );
 CHECK( rows_agree( block ) );

 // the empty rows have the slack alone, the others one entry per non-zero
 const Index ne[] = { 3 , 1 , 4 } , ni[] = { 1 , 3 , 3 };
 for( Index i = 0 ; i < 3 ; ++i ) {
  CHECK( row_function( block , true , i )->get_num_active_var() == ne[ i ] );
  CHECK( row_function( block , false , i )->get_num_active_var() ==
         ni[ i ] );
  }
 // e.g., the row of weight 2: 2 x_0 + 2 x_1 + 2 x_2 - v_2 = 6
 CHECK( block.get_eq()[ 2 ].get_rhs() == 6 );
 CHECK( row_function( block , true , 2 )->get_coefficient( 1 ) == 2 );
 for( const auto & r : block.get_eq() )
  CHECK( ! r.is_relaxed() );
 for( const auto & r : block.get_ineq() )
  CHECK( ! r.is_relaxed() );

 CHECK( block.get_objective()->get_sense() == Objective::eMin );
 CHECK( objective_agrees( block , { true , false , false , false } ) );

 // generating it again changes nothing
 const auto f = objective( block );
 generate( block );
 CHECK( objective( block ) == f );

 // a center given before the generation is in the Objective
 HQPBlock other;
 load( other , edge_data() );
 other.set_center( { 1 , -2 , 4 } );
 generate( other );
 CHECK( objective_agrees( other , { true , false , false , false } ) );
 CHECK( objective( other )->get_linear_coefficient( 2 ) == -2 );

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/

static void test_levels( void )
{
 std::cout << "reset_levels() and next_level()" << std::endl;
 const int e0 = errors;

 HQPBlock block;
 load( block , edge_data() );

 // next_level() needs the Variable
 CHECK( throws< std::logic_error >( [ & ]() { block.next_level(); } ) );

 // the levels reset before the generation are found in it
 block.reset_levels( true );
 CHECK( block.relaxes_lower() );
 generate( block );
 CHECK( level_is( block , 0 , false , false ) );
 for( Index p = 1 ; p < 4 ; ++p ) {
  CHECK( block.get_level_state( p ) == HQPBlock::kOff );
  CHECK( level_is( block , p , true , true ) );
  }
 CHECK( objective_agrees( block , { true , true , true , true } ) );

 // a point x and the slacks of each level there
 auto & x = x_of( block );
 x[ 0 ].set_value( 2 );
 x[ 1 ].set_value( -1 );
 x[ 2 ].set_value( 1 );
 Vec v , w;
 // level 0: v = ( 2 + 2 - 1 , 0 - 2 ) = ( 3 , -2 ), no w
 CHECK( block.level_slacks( 0 , v , w ) == 6.5 );
 CHECK( ( v == Vec( { 3 , -2 } ) ) && w.empty() );
 // level 1: w = ( max( 0 , 0 + 1 ) , max( 0 , -3 + 1 - 2 ) ) = ( 1 , 0 )
 CHECK( block.level_slacks( 1 , v , w ) == 0.5 );
 CHECK( v.empty() && ( w == Vec( { 1 , 0 } ) ) );
 // level 2: nothing
 CHECK( block.level_slacks( 2 , v , w ) == 0 );
 CHECK( v.empty() && w.empty() );
 // level 3: v = 2 ( 2 - 3 ) = -2, w = max( 0 , -2 + 4 - 0.5 ) = 1.5
 CHECK( block.level_slacks( 3 , v , w ) == 3.125 );
 CHECK( ( v == Vec( { -2 } ) ) && ( w == Vec( { 1.5 } ) ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.level_slacks( 4 , v , w ); } ) );

 // from level 0 to 1: the slacks of 0 fixed at their values, level 1 free,
 // its rows enforced, and in the Objective
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.next_level( -1 ); } ) );
 block.next_level( 0.25 );
 CHECK( block.get_level() == 1 );
 const Vec v0 = { 3 , -2 };
 CHECK( block.get_level_state( 0 ) == HQPBlock::kFixed );
 CHECK( level_is( block , 0 , true , false , & v0 ) );
 CHECK( block.get_level_state( 1 ) == HQPBlock::kFree );
 CHECK( level_is( block , 1 , false , false ) );
 CHECK( level_is( block , 3 , true , true ) );
 CHECK( objective_agrees( block , { true , true , true , true } ) );

 // from 1 to 2: the tolerance is added to the w of level 1
 block.next_level( 0.25 );
 const Vec w1 = { 1.25 , 0.25 };
 CHECK( level_is( block , 1 , true , false , nullptr , & w1 ) );
 CHECK( level_is( block , 2 , false , false ) );

 // from 2 (empty) to 3, then no further
 block.next_level();
 CHECK( block.get_level() == 3 );
 CHECK( level_is( block , 3 , false , false ) );
 CHECK( throws< std::logic_error >( [ & ]() { block.next_level(); } ) );

 // back to level 0 with the rows of the lower levels enforced: all the
 // slacks free, only those of level 0 in the Objective
 block.reset_levels( false );
 CHECK( block.get_level() == 0 );
 CHECK( ! block.relaxes_lower() );
 for( Index p = 0 ; p < 4 ; ++p ) {
  CHECK( block.get_level_state( p ) == HQPBlock::kFree );
  CHECK( level_is( block , p , false , false ) );
  }
 CHECK( objective_agrees( block , { true , false , false , false } ) );

 // the slacks of the levels above the current one stay in the Objective
 block.next_level();
 block.next_level();
 CHECK( objective_agrees( block , { true , true , true , false } ) );

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/
// counts the Modification of each kind in the list of the FakeSolver, then
// empties it

struct ModCount {
 int rhs = 0;      // RowConstraintMod
 int relax = 0;    // ConstraintMod relaxing or enforcing a row
 int var = 0;      // VariableMod
 int coeff = 0;    // C05FunctionModLin, a change of coefficients of a row
 int added = 0;    // C05FunctionModVarsAddd, Variable added to a row
 int obj = 0;      // a change of the DQuadFunction of the Objective
 int other = 0;    // anything else

 [[nodiscard]] int total( void ) const {
  return( rhs + relax + var + coeff + added + obj + other );
  }
 };

static ModCount count( FakeSolver * s , const DQuadFunction * f )
{
 ModCount c;
 auto & l = s->get_Modification_list();
 for( const auto & m : l ) {
  if( std::dynamic_pointer_cast< RowConstraintMod >( m ) )
   ++c.rhs;
  else if( std::dynamic_pointer_cast< ConstraintMod >( m ) )
   ++c.relax;
  else if( std::dynamic_pointer_cast< VariableMod >( m ) )
   ++c.var;
  else if( auto fm = std::dynamic_pointer_cast< FunctionMod >( m ) ) {
   if( fm->function() == f )
    ++c.obj;
   else if( std::dynamic_pointer_cast< C05FunctionModLin >( m ) )
    ++c.coeff;
   else
    ++c.other;
   }
  else if( std::dynamic_pointer_cast< C05FunctionModVarsAddd >( m ) )
   ++c.added;
  else
   ++c.other;
  }
 l.clear();
 return( c );
 }

/*--------------------------------------------------------------------------*/

static void test_modifications( void )
{
 std::cout << "the Modification of the changes of the data" << std::endl;
 const int e0 = errors;

 HQPBlock block;
 load( block , edge_data() );
 generate( block );
 auto s = new FakeSolver();
 block.register_Solver( s );
 const auto f = objective( block );
 ModCount c;

 // b: only the row that changes
 block.set_b( 0 , { 1 , 5 } );
 c = count( s , f );
 CHECK( ( c.rhs == 1 ) && ( c.total() == 1 ) );
 CHECK( block.get_b() == Vec( { 1 , 5 , 3 } ) );
 CHECK( rows_agree( block ) );
 block.set_b( 0 , { 1 , 5 } );
 CHECK( count( s , f ).total() == 0 );
 block.set_b( 3 , { 7 } , eNoMod );
 CHECK( count( s , f ).total() == 0 );
 CHECK( block.get_eq()[ 2 ].get_rhs() == 14 );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_b( 1 , { 1 } ); } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_b( 4 , {} ); } ) );

 // d, on the level with an empty row too
 block.set_d( 1 , { 0 , -1 } );
 c = count( s , f );
 CHECK( ( c.rhs == 2 ) && ( c.total() == 2 ) );
 CHECK( rows_agree( block ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_d( 2 , { 1 } ); } ) );

 // A: a coefficient in the pattern changed, an entry out of it added to the
 // empty row, which gets a new active Variable, and nothing else
 Mat A0( 2 , 3 );
 A0 << 1 , -3 , 0 ,
       0 , 0 , 4;
 block.set_A( 0 , sparse( A0 ) );
 c = count( s , f );
 CHECK( ( c.coeff == 1 ) && ( c.added == 1 ) && ( c.total() == 2 ) );
 CHECK( row_function( block , true , 1 )->get_num_active_var() == 2 );
 CHECK( rows_agree( block ) );
 // an entry set to 0 stays in the pattern, with coefficient 0
 A0( 0 , 0 ) = 0;
 block.set_A( 0 , sparse( A0 ) );
 c = count( s , f );
 CHECK( ( c.coeff == 1 ) && ( c.total() == 1 ) );
 CHECK( row_function( block , true , 0 )->get_num_active_var() == 3 );
 CHECK( Mat( block.get_A( 0 ) )( 0 , 0 ) == 0 );
 CHECK( rows_agree( block ) );
 // the same matrix again changes nothing
 block.set_A( 0 , sparse( A0 ) );
 CHECK( count( s , f ).total() == 0 );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_A( 0 , sparse( Mat::Zero( 1 , 3 ) ) ); } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_A( 0 , sparse( Mat::Zero( 2 , 2 ) ) ); } ) );

 // C, on the level with no equality
 Mat C1( 2 , 3 );
 C1 << 1 , 0 , 0 ,
       0 , 3 , 2;
 block.set_C( 1 , sparse( C1 ) );
 c = count( s , f );
 CHECK( ( c.coeff == 1 ) && ( c.added == 1 ) && ( c.total() == 2 ) );
 CHECK( rows_agree( block ) );

 // the weights: coefficients and right-hand sides of the rows
 block.set_we( 3 , { 3 } );
 c = count( s , f );
 CHECK( ( c.coeff == 1 ) && ( c.rhs == 1 ) && ( c.total() == 2 ) );
 CHECK( block.get_we() == Vec( { 1 , 1 , 3 } ) );
 CHECK( rows_agree( block ) );
 block.set_wi( 3 , { 0.5 } );
 c = count( s , f );
 CHECK( ( c.coeff == 1 ) && ( c.rhs == 1 ) && ( c.total() == 2 ) );
 CHECK( rows_agree( block ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_wi( 3 , {} ); } ) );

 // the regularization and its center: the coefficients of x
 block.set_regularization( 2 );
 c = count( s , f );
 CHECK( ( c.obj == 1 ) && ( c.total() == 1 ) );
 block.set_regularization( 2 );
 CHECK( count( s , f ).total() == 0 );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_regularization( -1 ); } ) );
 // the linear coefficients and the constant
 block.set_center( { 1 , 2 , 3 } );
 c = count( s , f );
 CHECK( ( c.obj == 2 ) && ( c.total() == 2 ) );
 CHECK( objective_agrees( block , { true , false , false , false } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_center( { 1 } ); } ) );
 block.set_center( {} );
 CHECK( count( s , f ).total() == 2 );
 CHECK( objective_agrees( block , { true , false , false , false } ) );
 // with no regularization the center changes nothing in the Objective
 block.set_regularization( 0 );
 count( s , f );
 block.set_center( { 1 , 1 , 1 } );
 CHECK( count( s , f ).total() == 0 );
 CHECK( objective_agrees( block , { true , false , false , false } ) );
 block.set_center( {} );

 // the levels: resetting to the state they are in issues nothing
 block.reset_levels( false );
 CHECK( count( s , f ).total() == 0 );
 // relaxing the lower levels: a slack fixed and a row relaxed for each row
 // of levels 1 to 3, and their slacks in the Objective (one Modification
 // for each kind of rows of a level that has some)
 block.reset_levels( true );
 c = count( s , f );
 CHECK( ( c.var == 4 ) && ( c.relax == 4 ) && ( c.obj == 3 ) &&
        ( c.total() == 11 ) );
 // to level 1: the two slacks of level 0 fixed, those of level 1 unfixed
 // and its rows enforced; its slacks are in the Objective already
 block.next_level();
 c = count( s , f );
 CHECK( ( c.var == 4 ) && ( c.relax == 2 ) && ( c.total() == 6 ) );
 // with eNoMod nothing is issued, and the state changes all the same
 block.next_level( 0 , eNoMod );
 CHECK( count( s , f ).total() == 0 );
 CHECK( block.get_level() == 2 );

 block.unregister_Solvers( true );

 // the changes made before the abstract representation is there are found
 // in it once generated
 HQPBlock late;
 load( late , edge_data() );
 late.set_b( 0 , { 4 , 4 } );
 late.set_d( 3 , { -2 } );
 late.set_A( 0 , sparse( A0 ) );
 late.set_C( 1 , sparse( C1 ) );
 late.set_we( 0 , { 2 , 3 } );
 late.set_wi( 1 , { 5 , 6 } );
 late.set_regularization( 4 );
 generate( late );
 CHECK( rows_agree( late ) );
 CHECK( objective_agrees( late , { true , false , false , false } ) );

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/
// the group of the HQPBlock in a file written by serialize()

static netCDF::NcGroup block_group( netCDF::NcFile & f )
{
 return( f.getGroups().begin()->second );
 }

/*--------------------------------------------------------------------------*/
// serialize() and deserialize() give back the same data, and the optional
// dimensions and variables written are the expected ones

static void round_trip( const std::string & name , const Data & D ,
                        bool eq , bool ineq , bool eqel , bool weight ,
                        bool reg )
{
 const int e0 = errors;
 HQPBlock block;
 load( block , D );
 const auto file = ( std::filesystem::temp_directory_path() /
                     "HQPBlock_unit_test.nc4" ).string();
 block.serialize( file , eBlockFile );

 {
  netCDF::NcFile f( file , netCDF::NcFile::read );
  auto g = block_group( f );
  CHECK( g.getDim( "NumEqRows" ).isNull() != eq );
  CHECK( g.getDim( "NumIneqRows" ).isNull() != ineq );
  CHECK( g.getDim( "NumEqElements" ).isNull() != eqel );
  CHECK( g.getVar( "EqWeight" ).isNull() != weight );
  CHECK( g.getVar( "Regularization" ).isNull() != reg );
  }

 auto back = dynamic_cast< HQPBlock * >( Block::deserialize( file ) );
 std::filesystem::remove( file );
 CHECK( back );
 if( back ) {
  CHECK( back->get_num_var() == block.get_num_var() );
  CHECK( back->get_num_levels() == block.get_num_levels() );
  CHECK( back->get_b() == block.get_b() );
  CHECK( back->get_d() == block.get_d() );
  CHECK( back->get_we() == block.get_we() );
  CHECK( back->get_wi() == block.get_wi() );
  CHECK( back->get_regularization() == block.get_regularization() );
  for( Index p = 0 ; p <= block.get_num_levels() ; ++p ) {
   CHECK( back->get_eq_start( p ) == block.get_eq_start( p ) );
   CHECK( back->get_ineq_start( p ) == block.get_ineq_start( p ) );
   }
  for( Index p = 0 ; p < block.get_num_levels() ; ++p ) {
   CHECK( ( back->get_A( p ).rows() == block.get_A( p ).rows() ) &&
          ( back->get_A( p ).cols() == block.get_A( p ).cols() ) &&
          ( Mat( back->get_A( p ) ) == Mat( block.get_A( p ) ) ) );
   CHECK( ( back->get_C( p ).rows() == block.get_C( p ).rows() ) &&
          ( back->get_C( p ).cols() == block.get_C( p ).cols() ) &&
          ( Mat( back->get_C( p ) ) == Mat( block.get_C( p ) ) ) );
   }
  // and its abstract representation is that of the data
  generate( *back );
  CHECK( rows_agree( *back ) );
  delete back;
  }
 std::cout << " netCDF round trip of " << name
           << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/
// a file of D changed by chg() is refused by deserialize()

static void refused( const std::string & name , const Data & D ,
                     const std::function< void( netCDF::NcGroup & ) > & chg )
{
 const int e0 = errors;
 HQPBlock block;
 load( block , D );
 const auto file = ( std::filesystem::temp_directory_path() /
                     "HQPBlock_unit_test_bad.nc4" ).string();
 block.serialize( file , eBlockFile );
 {
  netCDF::NcFile f( file , netCDF::NcFile::write );
  auto g = block_group( f );
  chg( g );
  }
 // the reader throws, and Block::deserialize() reports it and gives nothing
 {
  netCDF::NcFile f( file , netCDF::NcFile::read );
  auto g = block_group( f );
  HQPBlock back;
  CHECK( throws< std::invalid_argument >( [ & ]() {
   back.deserialize( g ); } ) );
  }
 std::cout << "  ";
 auto back = Block::deserialize( file );
 CHECK( ! back );
 delete back;
 std::filesystem::remove( file );
 std::cout << " deserialize() refuses " << name
           << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/

static void test_netcdf( void )
{
 std::cout << "the netCDF format" << std::endl;

 // empty levels, empty rows, a weight not 1, a regularization
 round_trip( "the edge cases" , edge_data() , true , true , true , true ,
             true );

 // no inequality rows at all, no weight, no regularization
 Data noi = edge_data();
 noi.C = { Mat( 0 , 3 ) , Mat( 0 , 3 ) , Mat( 0 , 3 ) , Mat( 0 , 3 ) };
 noi.d = { {} , {} , {} , {} };
 noi.we.clear();
 noi.wi.clear();
 noi.eps = 0;
 round_trip( "no inequality rows" , noi , true , false , true , false ,
             false );

 // no equality rows at all
 Data noe = edge_data();
 noe.A = { Mat( 0 , 3 ) , Mat( 0 , 3 ) , Mat( 0 , 3 ) , Mat( 0 , 3 ) };
 noe.b = { {} , {} , {} , {} };
 noe.we.clear();
 round_trip( "no equality rows" , noe , false , true , false , false ,
             true );

 // equality rows with no entries at all: NumEqElements is not there
 Data noel = edge_data();
 noel.A = { Mat::Zero( 2 , 3 ) , Mat( 0 , 3 ) , Mat( 0 , 3 ) ,
            Mat::Zero( 1 , 3 ) };
 round_trip( "rows with no entries" , noel , true , true , false , true ,
             true );

 // a single level with a single variable and nothing else
 Data one;
 one.n = 1;
 one.A = { Mat( 0 , 1 ) };
 one.b = { {} };
 one.C = { Mat( 0 , 1 ) };
 one.d = { {} };
 round_trip( "a level with no rows" , one , false , false , false , false ,
             false );

 // the errors
 refused( "rows that do not add up" , edge_data() ,
          []( netCDF::NcGroup & g ) {
           const unsigned int r[] = { 1 , 0 , 0 , 1 };
           g.getVar( "EqRows" ).putVar( r ); } );
 refused( "decreasing starts" , edge_data() ,
          []( netCDF::NcGroup & g ) {
           const unsigned int s[] = { 0 , 2 , 1 };
           g.getVar( "IneqStart" ).putVar( s ); } );
 refused( "a column out of range" , edge_data() ,
          []( netCDF::NcGroup & g ) {
           const std::vector< size_t > at = { 0 };
           g.getVar( "EqIndex" ).putVar( at , 3u ); } );
 }

/*--------------------------------------------------------------------------*/

static void test_print( void )
{
 std::cout << "print()" << std::endl;
 const int e0 = errors;

 HQPBlock block;
 Data D = small_data();
 D.we = { { 2 } , {} };
 D.eps = 0.25;
 load( block , D );

 std::ostringstream s0 , s1;
 block.print( s0 );
 block.print( s1 , 1 );
 const std::string head =
  "HQPBlock with 2 variables and 2 levels, regularization 0.25\n"
  " level 0: 1 equality rows, 0 inequality rows\n";
 const std::string head1 = " level 1: 0 equality rows, 2 inequality rows\n";
 CHECK( s0.str() == head + head1 );
 CHECK( s1.str() == head + "  2 * ( 1 x_0 1 x_1 - v = 2 )\n" + head1 +
        "  1 * ( 1 x_0 - w <= 0.5 )\n"
        "  1 * ( -1 x_1 - w <= 0 )\n" );
 if( errors != e0 )
  std::cout << s1.str();

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }


/*--------------------------------------------------------------------------*/

static void test_reg_weights( void )
{
 std::cout << "the weights of the variables in the regularization"
           << std::endl;
 const int e0 = errors;

 Data D = small_data();
 D.eps = 0.5;
 HQPBlock block;
 {
  std::vector< HQPBlock::SparseMatrix > A , C;
  for( const auto & M : D.A ) A.push_back( sparse( M ) );
  for( const auto & M : D.C ) C.push_back( sparse( M ) );
  auto b = D.b , d = D.d;
  block.load( D.n , std::move( A ) , std::move( b ) , std::move( C ) ,
              std::move( d ) , {} , {} , D.eps , { 0.25 , 2 } );
  }
 generate( block );
 const auto f = objective( block );
 CHECK( f->get_quadratic_coefficient( 0 ) == 0.5 * 0.25 / 2 );
 CHECK( f->get_quadratic_coefficient( 1 ) == 0.5 * 2 / 2 );
 block.set_center( { 1 , -1 } );
 CHECK( f->get_linear_coefficient( 0 ) == - 0.5 * 0.25 );
 CHECK( f->get_linear_coefficient( 1 ) == 0.5 * 2 );
 block.set_regularization( 2 );
 CHECK( f->get_quadratic_coefficient( 0 ) == 2 * 0.25 / 2 );
 CHECK( f->get_linear_coefficient( 1 ) == 2 * 2 );

 // the netCDF variable, written only if a weight is not 1
 const auto file = ( std::filesystem::temp_directory_path() /
                     "HQPBlock_unit_test_rw.nc4" ).string();
 block.serialize( file , eBlockFile );
 {
  netCDF::NcFile nf( file , netCDF::NcFile::read );
  CHECK( ! block_group( nf ).getVar( "RegularizationWeight" ).isNull() );
  }
 auto back = dynamic_cast< HQPBlock * >( Block::deserialize( file ) );
 std::filesystem::remove( file );
 CHECK( back &&
        ( back->get_regularization_weights() == Vec( { 0.25 , 2 } ) ) );
 delete back;

 // the errors: wrong size, a weight that is not positive
 for( const Vec & rw : { Vec( { 1 } ) , Vec( { 1 , 0 } ) } ) {
  HQPBlock bad;
  std::vector< HQPBlock::SparseMatrix > A , C;
  for( const auto & M : D.A ) A.push_back( sparse( M ) );
  for( const auto & M : D.C ) C.push_back( sparse( M ) );
  auto b = D.b , d = D.d;
  auto w = rw;
  CHECK( throws< std::invalid_argument >( [ & ]() {
   bad.load( D.n , std::move( A ) , std::move( b ) , std::move( C ) ,
             std::move( d ) , {} , {} , 1 , std::move( w ) ); } ) );
  }

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/

static void test_refix( void )
{
 std::cout << "refix_level() and get_Solution()" << std::endl;
 const int e0 = errors;

 HQPBlock block;
 load( block , small_data() );
 generate( block );
 auto & x = x_of( block );
 block.reset_levels( true );
 CHECK( throws< std::logic_error >( [ & ]() { block.refix_level( 0 ); } ) );
 x[ 0 ].set_value( 1 );
 x[ 1 ].set_value( 0.5 );
 block.next_level();  // v_0 fixed at 1 + 0.5 - 2 = -0.5
 CHECK( block.get_v()[ 0 ].get_value() == -0.5 );

 // x moves: the fixed v_0 follows it
 x[ 0 ].set_value( 2 );
 block.refix_level( 0 );
 CHECK( block.get_v()[ 0 ].get_value() == 0.5 );
 CHECK( block.get_v()[ 0 ].is_fixed() );

 // get_Solution() reads x and the slacks, and writes them back
 auto sol = block.get_Solution( nullptr , false );
 CHECK( dynamic_cast< ColVariableSolution * >( sol ) );
 x[ 0 ].set_value( -7 );
 sol->write( & block );
 CHECK( x[ 0 ].get_value() == 2 );
 delete sol;

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/
// 5 columns, owners { 0 , 0 , 1 , 1 , 0 }, 4 levels:
// - level 0: x_0 + x_1 = 1 (local to 0), x_2 - x_3 = 0 (local to 1), and a
//   row with no entries (owned by 0);
// - level 1: x_1 + x_2 <= 0, a tie between 0 and 1 (owned by 0), and
//   x_0 + 2 x_2 + 3 x_3 <= 1 (2 entries of 1, owned by 1);
// - level 2: x_4 + x_3 = 2, a tie owned by 0 unless given otherwise;
// - level 3: empty, so that the coupling row of level 1 is fixed again.
// Owner 0 copies x_2 and x_3, owner 1 copies x_0; x_2 has mu = 1, x_0 too.

// the scale of the whole Objective, which leaves the values of the levels
// as they are

static void test_scale( void )
{
 std::cout << "scale of the Objective" << std::endl;
 const int e0 = errors;
 HQPBlock block;
 load( block , small_data() );
 // before the Objective is there, then on it
 block.set_objective_scale( 2 );
 generate( block );
 CHECK( objective_agrees( block , { true , false , false , false } , 2 ) );
 block.set_objective_scale( 1 );
 CHECK( objective_agrees( block , { true , false , false , false } ) );
 block.set_objective_scale( 2 );
 Vec c( block.get_num_var() , 0.5 );
 block.set_center( c );
 CHECK( objective_agrees( block , { true , false , false , false } , 2 ) );
 Vec v , w , v1 , w1;
 const double val = block.level_slacks( 0 , v , w );
 block.set_objective_scale( 1 );
 CHECK( block.level_slacks( 0 , v1 , w1 ) == val );
 CHECK( objective_agrees( block , { true , false , false , false } ) );
 for( double bad : { 0.0 , -1.0 , Inf< double >() } )
  CHECK( throws< std::invalid_argument >( [ & ]() {
          block.set_objective_scale( bad ); } ) );
 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/

static Data dist_data( void )
{
 Data D;
 D.n = 5;
 D.eps = 0.1;
 Mat A0( 3 , 5 ) , C1( 2 , 5 ) , A2( 1 , 5 );
 A0 << 1 , 1 , 0 , 0 , 0 ,
       0 , 0 , 1 , -1 , 0 ,
       0 , 0 , 0 , 0 , 0;
 C1 << 0 , 1 , 1 , 0 , 0 ,
       1 , 0 , 2 , 3 , 0;
 A2 << 0 , 0 , 0 , 1 , 1;
 D.A = { A0 , Mat( 0 , 5 ) , A2 , Mat( 0 , 5 ) };
 D.b = { { 1 , 0 , 0.5 } , {} , { 2 } , {} };
 D.C = { Mat( 0 , 5 ) , C1 , Mat( 0 , 5 ) , Mat( 0 , 5 ) };
 D.d = { {} , { 0 , 1 } , {} , {} };
 D.wi = { {} , { 1 , 2 } , {} , {} };
 return( D );
 }

/*--------------------------------------------------------------------------*/
// the LinearFunction of the k-th consensus row has the copy with 1 and the
// owner's variable with -1, both sides 0

static bool consensus_agrees( const HQPDistBlock & db )
{
 const auto & cons = db.get_consensus();
 Index k = 0;
 for( Index r = 0 ; r < db.get_num_robots() ; ++r ) {
  const auto & cols = db.get_robot_columns( r );
  for( Index h = db.get_num_own( r ) ; h < cols.size() ; ++h , ++k ) {
   if( k >= cons.size() )
    return( false );
   const auto j = cols[ h ];
   const auto o = db.get_column_owner()[ j ];
   const auto & oc = db.get_robot_columns( o );
   const Index ho = std::find( oc.begin() , oc.end() , j ) - oc.begin();
   auto lf = dynamic_cast< const LinearFunction * >(
                                             cons[ k ].get_function() );
   if( ( ! lf ) || ( lf->get_num_active_var() != 2 ) ||
       ( lf->get_active_var( 0 ) != & db.get_robot( r )->get_x()[ h ] ) ||
       ( lf->get_coefficient( 0 ) != 1 ) ||
       ( lf->get_active_var( 1 ) != & db.get_robot( o )->get_x()[ ho ] ) ||
       ( lf->get_coefficient( 1 ) != -1 ) ||
       ( cons[ k ].get_lhs() != 0 ) || ( cons[ k ].get_rhs() != 0 ) )
    return( false );
   }
  }
 return( k == cons.size() );
 }

/*--------------------------------------------------------------------------*/

static void test_dist( void )
{
 std::cout << "HQPDistBlock" << std::endl;
 const int e0 = errors;
 const Data D = dist_data();
 HQPBlock central;
 load( central , D );

 HQPDistBlock db;
 db.load( central , 2 , { 0 , 0 , 1 , 1 , 0 } );
 CHECK( db.get_num_robots() == 2 );
 CHECK( db.get_num_var() == 5 );
 CHECK( db.get_num_levels() == 4 );
 CHECK( db.get_eq_owner() == HQPDistBlock::Vec_Index( { 0 , 1 , 0 , 0 } ) );
 CHECK( db.get_ineq_owner() == HQPDistBlock::Vec_Index( { 0 , 1 } ) );
 CHECK( db.get_robot_columns( 0 ) ==
        HQPDistBlock::Vec_Index( { 0 , 1 , 4 , 2 , 3 } ) );
 CHECK( db.get_num_own( 0 ) == 3 );
 CHECK( db.get_robot_columns( 1 ) ==
        HQPDistBlock::Vec_Index( { 2 , 3 , 0 } ) );
 CHECK( db.get_num_own( 1 ) == 2 );
 // x_0 copied by 1, x_2 and x_3 by 0: weights 1/2 on them, 1 on the others
 CHECK( db.get_robot( 0 )->get_regularization_weights() ==
        Vec( { 0.5 , 1 , 1 , 0.5 , 0.5 } ) );
 CHECK( db.get_robot( 1 )->get_regularization_weights() ==
        Vec( { 0.5 , 0.5 , 0.5 } ) );
 // the rows of the sub-Blocks, level by level
 CHECK( ( db.get_robot( 0 )->get_num_eq_rows( 0 ) == 2 ) &&
        ( db.get_robot( 1 )->get_num_eq_rows( 0 ) == 1 ) &&
        ( db.get_robot( 0 )->get_num_ineq_rows( 1 ) == 1 ) &&
        ( db.get_robot( 1 )->get_num_ineq_rows( 1 ) == 1 ) &&
        ( db.get_robot( 0 )->get_num_eq_rows( 2 ) == 1 ) &&
        ( db.get_robot( 1 )->get_num_eq_rows( 2 ) == 0 ) );
 CHECK( db.get_robot( 1 )->get_wi() == Vec( { 2 } ) );
 CHECK( ( Mat( db.get_robot( 1 )->get_C( 1 ) ) ==
          ( Mat( 1 , 3 ) << 2 , 3 , 1 ).finished() ) );

 // the abstract representation: no Variable, the consensus rows, an empty
 // Objective
 db.generate_abstract_variables();
 db.generate_abstract_constraints();
 db.generate_objective();
 CHECK( db.get_static_variable_groups().empty() );
 CHECK( db.get_consensus().size() == 3 );
 CHECK( consensus_agrees( db ) );
 CHECK( db.get_objective() &&
        ( db.get_objective()->get_num_active_var() == 0 ) );

 // x and the slacks of the hierarchy are those of the HQPBlock at the same
 // x, whatever the copies are
 generate( central );
 const Vec xv = { 0.3 , -1 , 2 , 0.5 , 4 };
 db.set_x_values( xv );
 central.set_x_values( xv );
 Vec x;
 db.get_x_values( x );
 CHECK( x == xv );
 // the copy of x_2 in owner 0, moved away from it
 auto & cx = const_cast< ColVariable & >( db.get_robot( 0 )->get_x()[ 3 ] );
 CHECK( cx.get_value() == 2 );
 cx.set_value( 9 );
 for( Index p = 0 ; p < 4 ; ++p ) {
  Vec v , w , vc , wc;
  CHECK( db.level_slacks( p , v , w ) == central.level_slacks( p , vc , wc ) );
  CHECK( ( v == vc ) && ( w == wc ) );
  }

 // next_level(): the copies take the values of the owners, the distance is
 // recorded, and the slacks of the coupling rows are those at the owners
 auto s = new FakeSolver();
 db.register_Solver( s );
 db.reset_levels( true );
 db.next_level();
 CHECK( db.get_robot( 0 )->get_x()[ 3 ].get_value() == 2 );
 CHECK( db.get_consensus_residual( 0 ) == 7 );
 db.next_level();
 CHECK( db.get_level() == 2 );
 // owner 1 owns x_0 + 2 x_2 + 3 x_3 <= 1, weight 2: 2 ( 0.3 + 4 + 1.5 - 1 )
 CHECK( std::abs( db.get_robot( 1 )->get_w()[ 0 ].get_value() - 9.6 ) <
        1e-12 );
 // the owner moves x_0: the next level brings the copy of owner 1 to it,
 // and the slack of level 1 that 1 has fixed rises to 2 ( 2.3 + 5.5 - 1 )
 const_cast< ColVariable & >( db.get_robot( 0 )->get_x()[ 0 ] ).set_value(
                                                                         2.3 );
 db.next_level();
 CHECK( db.get_robot( 1 )->get_x()[ 2 ].get_value() == 2.3 );
 CHECK( std::abs( db.get_robot( 1 )->get_w()[ 0 ].get_value() - 13.6 ) <
        1e-12 );
 CHECK( db.get_robot( 1 )->get_w()[ 0 ].is_fixed() );
 CHECK( throws< std::logic_error >( [ & ]() { db.next_level(); } ) );

 // set_slack_values() writes each slack in the sub-Block that owns it
 db.reset_levels( false );
 db.set_slack_values( 0 , { 7 , 8 , 9 } , {} );
 CHECK( ( db.get_robot( 0 )->get_v()[ 0 ].get_value() == 7 ) &&
        ( db.get_robot( 1 )->get_v()[ 0 ].get_value() == 8 ) &&
        ( db.get_robot( 0 )->get_v()[ 1 ].get_value() == 9 ) );

 // a change in the pattern is a Modification of the sub-Block that owns
 // the row; one out of it (x_4 in x_2 - x_3 = 0, which 1 owns and still
 // owns after, without having x_4) builds the sub-Blocks anew, the
 // NBModification first
 s->get_Modification_list().clear();
 db.set_d( 1 , { 0 , 3 } );
 CHECK( db.get_robot( 1 )->get_d() == Vec( { 3 } ) );
 CHECK( ! s->get_Modification_list().empty() );
 CHECK( std::none_of( s->get_Modification_list().begin() ,
                      s->get_Modification_list().end() ,
                      []( const auto & m ) {
                       return( bool( std::dynamic_pointer_cast<
                                       const NBModification >( m ) ) ); } ) );
 s->get_Modification_list().clear();
 Mat A0 = D.A[ 0 ];
 A0( 1 , 4 ) = 5;
 db.set_A( 0 , sparse( A0 ) );
 CHECK( ( ! s->get_Modification_list().empty() ) &&
        std::dynamic_pointer_cast< const NBModification >(
                                 s->get_Modification_list().front() ) );
 CHECK( db.get_eq_owner()[ 1 ] == 1 );
 CHECK( db.get_robot_columns( 1 ) ==
        HQPDistBlock::Vec_Index( { 2 , 3 , 0 , 4 } ) );
 CHECK( consensus_agrees( db ) );
 db.unregister_Solvers( true );

 // the netCDF format: the owners of the rows only if given
 const auto file = ( std::filesystem::temp_directory_path() /
                     "HQPDistBlock_unit_test.nc4" ).string();
 for( bool given : { false , true } ) {
  HQPDistBlock dg;
  if( given )
   dg.load( central , 2 , { 0 , 0 , 1 , 1 , 0 } , { 0 , 1 , 0 , 1 } );
  else
   dg.load( central , 2 , { 0 , 0 , 1 , 1 , 0 } );
  dg.serialize( file , eBlockFile );
  {
   netCDF::NcFile nf( file , netCDF::NcFile::read );
   auto g = block_group( nf );
   CHECK( ! g.getVar( "ColumnOwner" ).isNull() );
   CHECK( g.getVar( "EqRowOwner" ).isNull() != given );
   CHECK( g.getVar( "IneqRowOwner" ).isNull() );
   CHECK( ! g.getGroup( "HQPBlock" ).isNull() );
   }
  auto back = dynamic_cast< HQPDistBlock * >( Block::deserialize( file ) );
  CHECK( back );
  if( back ) {
   CHECK( back->get_eq_owner() == dg.get_eq_owner() );
   CHECK( back->get_ineq_owner() == dg.get_ineq_owner() );
   for( Index r = 0 ; r < 2 ; ++r )
    CHECK( back->get_robot_columns( r ) == dg.get_robot_columns( r ) );
   CHECK( back->get_central().get_b() == central.get_b() );
   delete back;
   }
  // the given owners move the row of level 2 to 1
  if( given )
   CHECK( dg.get_robot( 1 )->get_num_eq_rows( 2 ) == 1 );
  }
 std::filesystem::remove( file );

 // the errors of load()
 HQPDistBlock bad;
 CHECK( throws< std::invalid_argument >( [ & ]() {
  bad.load( central , 2 , { 0 , 0 , 2 , 1 , 0 } ); } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  bad.load( central , 3 , { 0 , 0 , 1 , 1 , 0 } ); } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  bad.load( central , 2 , { 0 , 0 , 1 , 1 } ); } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  bad.load( central , 2 , { 0 , 0 , 1 , 1 , 0 } , { 1 , 1 , 0 , 1 } ); } ) );

 std::ostringstream os;
 db.print( os );
 CHECK( os.str().rfind( "HQPDistBlock with 2 sub-Blocks and 4 copies" , 0 )
        == 0 );

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }


/*--------------------------------------------------------------------------*/
// the coefficients of the active Variable of the LinearFunction of a row, in
// the order given, all and only

static bool row_is( const FRowConstraint & row ,
                    const std::vector< std::pair< const ColVariable * ,
                                                  double > > & cf ,
                    double lhs , double rhs )
{
 auto lf = dynamic_cast< const LinearFunction * >( row.get_function() );
 if( ( ! lf ) || ( lf->get_num_active_var() != cf.size() ) ||
     ( row.get_lhs() != lhs ) || ( row.get_rhs() != rhs ) )
  return( false );
 for( Index a = 0 ; a < cf.size() ; ++a )
  if( ( lf->get_active_var( a ) != cf[ a ].first ) ||
      ( lf->get_coefficient( a ) != cf[ a ].second ) )
   return( false );
 return( true );
 }

/*--------------------------------------------------------------------------*/

static void test_dist_benders( void )
{
 std::cout << "HQPDistBlock with the structure kBenders" << std::endl;
 const int e0 = errors;
 const Data D = dist_data();
 HQPBlock central;
 load( central , D );
 auto strc = Configuration::deserialize( "structure-benders.txt" );

 HQPDistBlock db;
 db.load( central , 2 , { 0 , 0 , 1 , 1 , 0 } );
 db.set_structure( strc );
 CHECK( db.get_structure() == HQPDistBlock::kBenders );
 // x_3 + x_4 = 2 (level 2), x_1 + x_2 <= 0 and x_0 + 2 x_2 + 3 x_3 <= 1
 // (level 1, weight 2) are the coupling rows, the equality one first
 CHECK( ( db.get_coupling_rows() ==
          std::vector< std::pair< bool , Index > >(
           { { true , 3 } , { false , 0 } , { false , 1 } } ) ) );
 CHECK( db.get_robot_columns( 0 ) ==
        HQPDistBlock::Vec_Index( { 0 , 1 , 4 } ) );
 CHECK( db.get_robot_columns( 1 ) == HQPDistBlock::Vec_Index( { 2 , 3 } ) );
 CHECK( ( db.get_robot( 0 )->get_num_eq_rows( 0 ) == 2 ) &&
        ( db.get_robot( 1 )->get_num_eq_rows( 0 ) == 1 ) &&
        ( db.get_robot( 0 )->get_num_ineq_rows( 1 ) == 0 ) &&
        ( db.get_robot( 1 )->get_num_ineq_rows( 1 ) == 0 ) &&
        ( db.get_robot( 0 )->get_num_eq_rows( 2 ) == 0 ) );
 CHECK( db.get_robot( 0 )->get_regularization_weights() ==
        Vec( { 1 , 1 , 1 } ) );

 db.generate_abstract_variables();
 db.generate_abstract_constraints();
 db.generate_objective();
 // the y of x_3 and x_4, the t of the inequality coupling rows in the
 // other columns: x_1 and x_2 of x_1 + x_2 <= 0, x_0 and x_2 of the other
 const auto & t = * db.get_static_variable_v< ColVariable >( "t" );
 const auto & sl = * db.get_static_variable_v< ColVariable >( "s" );
 const auto & y = * db.get_static_variable_v< ColVariable >( "y" );
 CHECK( ( t.size() == 4 ) && ( sl.size() == 3 ) && ( y.size() == 2 ) );
 CHECK( ( ! sl[ 0 ].is_positive() ) && sl[ 1 ].is_positive() );
 const auto & cr = * db.get_static_constraint_v< FRowConstraint >(
                                                               "coupling" );
 const double Inf_d = Inf< double >();
 CHECK( row_is( cr[ 0 ] , { { & sl[ 0 ] , -1 } , { & y[ 0 ] , 1 } ,
                            { & y[ 1 ] , 1 } } , 2 , 2 ) );
 CHECK( row_is( cr[ 2 ] , { { & sl[ 2 ] , -1 } , { & y[ 0 ] , 6 } ,
                            { & t[ 2 ] , 2 } , { & t[ 3 ] , 2 } } ,
                - Inf_d , 2 ) );
 // the parts of the inequality rows first, then the links of the y
 const auto & x0 = db.get_robot( 0 )->get_x();
 const auto & x1 = db.get_robot( 1 )->get_x();
 auto & l0 = db.get_robot( 0 )->get_link_rows();
 auto & l1 = db.get_robot( 1 )->get_link_rows();
 CHECK( ( l0.size() == 3 ) && ( l1.size() == 3 ) );
 CHECK( row_is( l0.front() , { { & x0[ 1 ] , 1 } , { & t[ 0 ] , -1 } } ,
                - Inf_d , 0 ) );
 CHECK( row_is( l0.back() , { { & x0[ 2 ] , 1 } , { & y[ 1 ] , -1 } } ,
                0 , 0 ) );
 CHECK( row_is( * std::next( l1.begin() ) ,
                { { & x1[ 0 ] , 2 } , { & t[ 3 ] , -1 } } , - Inf_d , 0 ) );
 CHECK( row_is( l1.back() , { { & x1[ 1 ] , 1 } , { & y[ 0 ] , -1 } } ,
                0 , 0 ) );
 // at level 0 the link rows of levels 1 and 2 are relaxed
 for( const auto & l : { & l0 , & l1 } )
  for( const auto & row : *l )
   CHECK( row.is_relaxed() );

 // the levels: relaxed and fixed to 0 below, then the rows of level 1
 auto s = new FakeSolver();
 db.register_Solver( s );
 db.reset_levels( true );
 CHECK( cr[ 1 ].is_relaxed() && sl[ 1 ].is_fixed() );
 db.set_x_values( { 0.3 , -1 , 2 , 0.5 , 4 } );
 db.next_level();
 CHECK( ( ! cr[ 1 ].is_relaxed() ) && ( ! sl[ 1 ].is_fixed() ) );
 CHECK( ! l0.front().is_relaxed() );
 // the y of x_4 is only in the row of level 2, that of x_3 also in level 1
 CHECK( l0.back().is_relaxed() && ( ! l1.back().is_relaxed() ) );
 db.next_level();
 // s_2 fixed at the residual at the x of the owners, 2 ( 0.3 + 4 + 1.5 - 1 )
 CHECK( sl[ 2 ].is_fixed() &&
        ( std::abs( sl[ 2 ].get_value() - 9.6 ) < 1e-12 ) );
 CHECK( ! l0.back().is_relaxed() );
 // the y have the values of the owners, the t those of their parts, i.e.,
 // their link rows hold at 0, and the free s_0 is the residual of its row,
 // 0.5 + 4 - 2
 CHECK( ( y[ 0 ].get_value() == 0.5 ) && ( y[ 1 ].get_value() == 4 ) );
 for( const auto & l : { & l0 , & l1 } )
  for( auto & row : *l ) {
   auto lf = static_cast< LinearFunction * >( row.get_function() );
   lf->compute( true );
   CHECK( std::abs( lf->get_value() ) < 1e-12 );
   }
 CHECK( ( ! sl[ 0 ].is_fixed() ) &&
        ( std::abs( sl[ 0 ].get_value() - 2.5 ) < 1e-12 ) );
 // the Objective of the root has all its Variable, t, s and y in this
 // order, the t and the y at 0 and the s of level 2 at 1 / 2; scaled by 2,
 // so are those of the root and of the owners
 const auto ro = static_cast< const DQuadFunction * >(
            static_cast< FRealObjective * >( db.get_objective() )->
            get_function() );
 CHECK( ( ro->get_num_active_var() == 9 ) &&
        ( ro->get_active_var( 0 ) == & t[ 0 ] ) &&
        ( ro->get_active_var( 4 ) == & sl[ 0 ] ) &&
        ( ro->get_active_var( 7 ) == & y[ 0 ] ) &&
        ( ro->get_quadratic_coefficient( 4 ) == 0.5 ) &&
        ( ro->get_quadratic_coefficient( 0 ) == 0 ) &&
        ( ro->get_quadratic_coefficient( 7 ) == 0 ) );
 const double q0 = objective( * db.get_robot( 0 ) )->
                                               get_quadratic_coefficient( 0 );
 db.set_objective_scale( 2 );
 CHECK( ( ro->get_quadratic_coefficient( 4 ) == 1 ) &&
        ( ro->get_quadratic_coefficient( 0 ) == 0 ) &&
        ( objective( * db.get_robot( 0 ) )->get_quadratic_coefficient( 0 ) ==
          2 * q0 ) );
 db.set_objective_scale( 1 );
 CHECK( ro->get_quadratic_coefficient( 4 ) == 0.5 );
 for( Index p = 0 ; p < 4 ; ++p ) {
  Vec v , w , vc , wc;
  generate( central );
  central.set_x_values( { 0.3 , -1 , 2 , 0.5 , 4 } );
  CHECK( db.level_slacks( p , v , w ) == central.level_slacks( p , vc , wc ) );
  }

 // a change in the pattern: the side of a coupling row and the coefficients
 // of a part of it, by Modification
 s->get_Modification_list().clear();
 db.set_d( 1 , { 0 , 3 } );
 CHECK( cr[ 2 ].get_rhs() == 6 );
 Mat C1 = D.C[ 1 ];
 C1( 1 , 2 ) = 5;
 C1( 1 , 3 ) = 4;
 db.set_C( 1 , sparse( C1 ) );
 CHECK( row_is( * std::next( l1.begin() ) ,
                { { & x1[ 0 ] , 5 } , { & t[ 3 ] , -1 } } , - Inf_d , 0 ) );
 CHECK( row_is( cr[ 2 ] , { { & sl[ 2 ] , -1 } , { & y[ 0 ] , 8 } ,
                            { & t[ 2 ] , 2 } , { & t[ 3 ] , 2 } } ,
                - Inf_d , 6 ) );
 // and the coefficients of an equality one, on the y
 Mat A2 = D.A[ 2 ];
 A2( 0 , 4 ) = 3;
 db.set_A( 2 , sparse( A2 ) );
 CHECK( row_is( cr[ 0 ] , { { & sl[ 0 ] , -1 } , { & y[ 0 ] , 1 } ,
                            { & y[ 1 ] , 3 } } , 2 , 2 ) );
 CHECK( std::none_of( s->get_Modification_list().begin() ,
                      s->get_Modification_list().end() ,
                      []( const auto & m ) {
                       return( bool( std::dynamic_pointer_cast<
                                       const NBModification >( m ) ) ); } ) );
 // out of it: x_4, which has a y, in x_1 + x_2 <= 0
 s->get_Modification_list().clear();
 C1( 0 , 4 ) = 1;
 db.set_C( 1 , sparse( C1 ) );
 CHECK( ( ! s->get_Modification_list().empty() ) &&
        std::dynamic_pointer_cast< const NBModification >(
                                 s->get_Modification_list().front() ) );
 CHECK( db.get_robot( 0 )->get_link_rows().size() == 3 );
 // and so does a column with no y in an equality one
 s->get_Modification_list().clear();
 A2( 0 , 2 ) = 1;
 db.set_A( 2 , sparse( A2 ) );
 CHECK( ( ! s->get_Modification_list().empty() ) &&
        std::dynamic_pointer_cast< const NBModification >(
                                 s->get_Modification_list().front() ) );
 CHECK( db.get_static_variable_v< ColVariable >( "y" )->size() == 3 );
 db.unregister_Solvers( true );

 // the structure cannot change once the abstract representation is there
 HQPDistBlock dc;
 dc.load( central , 2 , { 0 , 0 , 1 , 1 , 0 } );
 dc.generate_abstract_variables();
 CHECK( throws< std::logic_error >( [ & ]() { dc.set_structure( strc ); } ) );

 std::ostringstream os;
 db.print( os );
 CHECK( os.str().rfind( "HQPDistBlock with 2 sub-Blocks and 3 coupling rows" ,
                        0 ) == 0 );
 delete strc;

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/
/*------------------------- THE STAGE STRUCTURE ----------------------------*/
/*--------------------------------------------------------------------------*/
// a single integrator over 2 stages, columns u_0 , u_1 , s_1 , s_2 (s_0 = 0
// is not a column): level 0 the dynamics s_{k+1} = s_k + u_k, level 1 the
// limits |u_k| <= 1, level 2 the smoothness |u_1 - u_0| <= 0.5 (a row of
// stage 1 on u_0), level 3 the goal s_2 = 5, which the limits keep at 2
// (value 9/2), level 4 s_1 = 0, which the goal keeps at 1 (value 1/2)

static Data integrator_data( void )
{
 Data D;
 D.n = 4;
 D.eps = 1e-10;
 Mat A0( 2 , 4 ) , C1( 4 , 4 ) , C2( 2 , 4 ) , A3( 1 , 4 ) , A4( 1 , 4 );
 A0 << -1 , 0 , 1 , 0 , 0 , -1 , -1 , 1;
 C1 << 1 , 0 , 0 , 0 , -1 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , -1 , 0 , 0;
 C2 << -1 , 1 , 0 , 0 , 1 , -1 , 0 , 0;
 A3 << 0 , 0 , 0 , 1;
 A4 << 0 , 0 , 1 , 0;
 D.A = { A0 , Mat( 0 , 4 ) , Mat( 0 , 4 ) , A3 , A4 };
 D.b = { { 0 , 0 } , {} , {} , { 5 } , { 0 } };
 D.C = { Mat( 0 , 4 ) , C1 , C2 , Mat( 0 , 4 ) , Mat( 0 , 4 ) };
 D.d = { {} , { 1 , 1 , 1 , 1 } , { 0.5 , 0.5 } , {} , {} };
 D.st.num_stages = 2;
 D.st.var_stage = { 0 , 1 , 1 , 2 };
 D.st.var_state = { 0 , 0 , 1 , 1 };
 return( D );
 }

/*--------------------------------------------------------------------------*/
// one stage with the initial state as a column, s_0 , u_0 , s_1: level 0
// the dynamics s_1 = s_0 + u_0, level 1 s_0 = 3 (a row of stage 0 on x_0
// only), level 2 |u_0| <= 1, level 3 s_1 = 0, which the others keep at 2
// (value 2)

static Data initial_state_data( void )
{
 Data D;
 D.n = 3;
 D.eps = 1e-10;
 Mat A0( 1 , 3 ) , A1( 1 , 3 ) , C2( 2 , 3 ) , A3( 1 , 3 );
 A0 << -1 , -1 , 1;
 A1 << 1 , 0 , 0;
 C2 << 0 , 1 , 0 , 0 , -1 , 0;
 A3 << 0 , 0 , 1;
 D.A = { A0 , A1 , Mat( 0 , 3 ) , A3 };
 D.b = { { 0 } , { 3 } , {} , { 0 } };
 D.C = { Mat( 0 , 3 ) , Mat( 0 , 3 ) , C2 , Mat( 0 , 3 ) };
 D.d = { {} , {} , { 1 , 1 } , {} };
 D.st.num_stages = 1;
 D.st.var_stage = { 0 , 0 , 1 };
 D.st.var_state = { 1 , 0 , 1 };
 return( D );
 }

/*--------------------------------------------------------------------------*/
// the levels of a hierarchy solved one after the other by an HQPStageSolver
// each, as HQPSolver does with intProxCenter 1 and intRelaxLower 1; the
// Solvers are registered to the HQPBlock, so that they see its
// Modification, and kept from a run to the next one

struct Cascade
{
 HQPBlock & block;
 std::vector< HQPStageSolver * > s;
 long iter = 0;   // the iterations of the last run()
 int status = 0;  // the first status not kOK, kOK if none

 explicit Cascade( HQPBlock & b ) : block( b ) { }

 ~Cascade() { block.unregister_Solvers( true ); }

 HQPStageSolver * solver( Index p ) {
  while( s.size() <= p ) {
   auto x = dynamic_cast< HQPStageSolver * >(
     Solver::new_Solver( "HQPStageSolver" ) );
   x->set_Block( &block );
   block.register_Solver( x );
   s.push_back( x );
   }
  return( s[ p ] );
  }

 // the value of each level
 std::vector< double > run( void ) {
  generate( block );
  block.reset_levels( true );
  block.set_center( {} );
  iter = 0;
  status = Solver::kOK;
  std::vector< double > val;
  for( Index p = 0 ; p < block.get_num_levels() ; ++p ) {
   auto x = solver( p );
   const int st = x->compute();
   iter += x->get_elapsed_iterations();
   if( ( st != Solver::kOK ) && ( status == Solver::kOK ) )
    status = st;
   x->get_var_solution();
   Vec xv , v , w;
   block.get_x_values( xv );
   val.push_back( block.level_slacks( p , v , w ) );
   if( p + 1 < block.get_num_levels() ) {
    block.set_center( xv );
    block.next_level();
    }
   }
  return( val );
  }
 };

/*--------------------------------------------------------------------------*/
// the values of the levels are the expected ones, to tol

static bool values_are( const std::vector< double > & val ,
			const std::vector< double > & exp , double tol )
{
 if( val.size() != exp.size() )
  return( false );
 for( Index p = 0 ; p < val.size() ; ++p )
  if( std::abs( val[ p ] - exp[ p ] ) > tol * std::max( 1.0 , exp[ p ] ) ) {
   std::cout << "  level " << p << " value " << val[ p ] << " instead of "
	     << exp[ p ] << std::endl;
   return( false );
   }
 return( true );
 }

/*--------------------------------------------------------------------------*/

static void test_stage_data( void )
{
 std::cout << "the stage structure of HQPBlock" << std::endl;
 const int e0 = errors;

 // the getters
 HQPBlock block;
 load( block , integrator_data() );
 CHECK( block.get_num_stages() == 2 );
 CHECK( block.get_var_stage() == std::vector< Index >( { 0 , 1 , 1 , 2 } ) );
 CHECK( block.get_var_state() == std::vector< Index >( { 0 , 0 , 1 , 1 } ) );
 CHECK( block.get_eq_stage().empty() && block.get_ineq_stage().empty() );
 CHECK( block.get_dynamics_level() == 0 );

 // no stage structure: an HQPBlock as before
 load( block , edge_data() );
 CHECK( block.get_num_stages() == 0 );
 CHECK( block.get_var_stage().empty() && block.get_var_state().empty() );

 // the errors of load()
 auto bad = []( const std::function< void( Data & ) > & change ) {
  Data D = integrator_data();
  change( D );
  HQPBlock b;
  return( throws< std::invalid_argument >( [ & ]() { load( b , D ); } ) );
  };
 CHECK( bad( []( Data & D ) { D.st.var_stage.pop_back(); } ) );
 CHECK( bad( []( Data & D ) { D.st.var_state.push_back( 0 ); } ) );
 CHECK( bad( []( Data & D ) { D.st.var_state[ 0 ] = 2; } ) );
 // an input of stage N, a state beyond N
 CHECK( bad( []( Data & D ) { D.st.var_stage[ 1 ] = 2; } ) );
 CHECK( bad( []( Data & D ) { D.st.var_stage[ 3 ] = 3; } ) );
 CHECK( bad( []( Data & D ) { D.st.eq_stage = { 0 , 0 }; } ) );
 CHECK( bad( []( Data & D ) { D.st.ineq_stage.assign( 6 , 2 ); } ) );
 CHECK( bad( []( Data & D ) { D.st.dynamics_level = 5; } ) );
 CHECK( bad( []( Data & D ) { D.st.num_stages = 0; } ) );
 CHECK( bad( []( Data & D ) {
  D.st = HQPBlock::StageData();
  D.st.dynamics_level = 1;
  } ) );
 CHECK( ! bad( []( Data & D ) { D.st.eq_stage = { 0 , 1 , 1 , 0 }; } ) );

 // the netCDF format: what is written, and the round trip
 const auto file =
   ( std::filesystem::temp_directory_path() / "HQPBlock_unit_test_stages.nc4" )
     .string();
 for( int given = 0 ; given < 2 ; ++given ) {
  Data D = integrator_data();
  if( given ) {
   D.st.eq_stage = { 0 , 1 , 1 , 0 };
   D.st.ineq_stage = { 0 , 0 , 1 , 1 , 1 , 1 };
   D.st.dynamics_level = 0;
   }
  load( block , D );
  block.serialize( file , eBlockFile );
  {
   netCDF::NcFile f( file , netCDF::NcFile::read );
   auto g = block_group( f );
   CHECK( g.getDim( "NumStages" ).getSize() == 2 );
   CHECK( ! g.getVar( "VarStage" ).isNull() );
   CHECK( ! g.getVar( "VarState" ).isNull() );
   CHECK( ! g.getVar( "DynamicsLevel" ).isNull() );
   CHECK( g.getVar( "EqStage" ).isNull() != bool( given ) );
   CHECK( g.getVar( "IneqStage" ).isNull() != bool( given ) );
   }
  auto back = dynamic_cast< HQPBlock * >( Block::deserialize( file ) );
  CHECK( back );
  if( back ) {
   CHECK( back->get_num_stages() == 2 );
   CHECK( back->get_var_stage() == block.get_var_stage() );
   CHECK( back->get_var_state() == block.get_var_state() );
   CHECK( back->get_eq_stage() == block.get_eq_stage() );
   CHECK( back->get_ineq_stage() == block.get_ineq_stage() );
   CHECK( back->get_dynamics_level() == 0 );
   delete back;
   }
  }

 // a file without stage data has none of these variables
 load( block , edge_data() );
 block.serialize( file , eBlockFile );
 {
  netCDF::NcFile f( file , netCDF::NcFile::read );
  auto g = block_group( f );
  CHECK( g.getDim( "NumStages" ).isNull() );
  CHECK( g.getVar( "VarStage" ).isNull() );
  CHECK( g.getVar( "DynamicsLevel" ).isNull() );
  }

 // VarState missing, or a wrong DynamicsLevel, is refused
 load( block , integrator_data() );
 block.serialize( file , eBlockFile );
 {
  netCDF::NcFile f( file , netCDF::NcFile::write );
  auto g = block_group( f );
  g.getVar( "DynamicsLevel" ).putVar( std::vector< size_t >() , 7u );
  }
 {
  netCDF::NcFile f( file , netCDF::NcFile::read );
  auto g = block_group( f );
  HQPBlock b;
  CHECK( throws< std::invalid_argument >( [ & ]() { b.deserialize( g ); } ) );
  }
 std::filesystem::remove( file );

 // print()
 std::ostringstream s0 , s1;
 block.print( s0 );
 block.print( s1 , 1 );
 CHECK( s0.str().find( " 2 stages, the dynamics at level 0\n" ) !=
	std::string::npos );
 CHECK( s1.str().find( "  stage of each column (s state, u input): u0 u1 "
		       "s1 s2\n" ) != std::string::npos );

 std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/
/*---------------------- THE RECURSION OF StageEQP -------------------------*/
/*--------------------------------------------------------------------------*/

using EVec = Eigen::VectorXd;
using EQP = HQPStageSolver::StageEQP;

// the kinds of dependent rows of a random StageEQP
enum Dependent {
 kNone = 0 ,
 kInStage = 1 ,
 kThroughDynamics = 2 ,
 kInconsistent = 4
 };

/*--------------------------------------------------------------------------*/
// a random StageEQP with nx[ k ] states and nu[ k ] inputs at stage k,
// mrow[ k ] rows (k = 0 .. N, mrow[ N ] those on x_N), ntask task rows per
// stage in the cost and the regularization eps; the rows hold at a random
// trajectory, unless kInconsistent, which moves one dependent row; the
// coefficients of the rows on x are multiplied by gs, so that the inputs
// they fix do not make the closed loop unstable

static EQP random_eqp( unsigned seed , const std::vector< Index > & nx ,
		       const std::vector< Index > & nu ,
		       const std::vector< Index > & mrow , Index ntask ,
		       double eps , int dep , double gs = 1 )
{
 std::mt19937 g( seed );
 std::normal_distribution< double > nd( 0 , 1 );
 auto rnd = [ & ]( Index r , Index c ) {
  Mat M( r , c );
  for( Index i = 0 ; i < r ; ++i )
   for( Index j = 0 ; j < c ; ++j )
    M( i , j ) = nd( g );
  return( M );
  };

 const Index N = nu.size();
 EQP e;
 e.resize( N );
 std::vector< EVec > xb( N + 1 ) , ub( N );
 xb[ 0 ] = rnd( nx[ 0 ] , 1 );
 for( Index k = 0 ; k < N ; ++k ) {
  e.A[ k ] =
    Mat::Identity( nx[ k + 1 ] , nx[ k ] ) + 0.05 * rnd( nx[ k + 1 ] ,
							 nx[ k ] );
  e.B[ k ] = rnd( nx[ k + 1 ] , nu[ k ] );
  e.c[ k ] = 0.1 * rnd( nx[ k + 1 ] , 1 );
  ub[ k ] = rnd( nu[ k ] , 1 );
  xb[ k + 1 ] = e.A[ k ] * xb[ k ] + e.B[ k ] * ub[ k ] + e.c[ k ];
  }
 for( Index k = 0 ; k <= N ; ++k ) {
  const Index m = ( k < N ) ? nu[ k ] : 0;
  const Mat T = rnd( ntask , nx[ k ] + m );
  const EVec t = rnd( ntask , 1 );
  const Mat Hs =
    T.transpose() * T + eps * Mat::Identity( nx[ k ] + m , nx[ k ] + m );
  const EVec hs = -T.transpose() * t;
  e.Q[ k ] = Hs.topLeftCorner( nx[ k ] , nx[ k ] );
  e.q[ k ] = hs.head( nx[ k ] );
  if( k < N ) {
   e.R[ k ] = Hs.bottomRightCorner( m , m );
   e.S[ k ] = Hs.bottomLeftCorner( m , nx[ k ] );
   e.r[ k ] = hs.tail( m );
   }
  e.G[ k ] = gs * rnd( mrow[ k ] , nx[ k ] );
  e.H[ k ] = rnd( mrow[ k ] , m );
  if( ( dep & kInStage ) && ( mrow[ k ] >= 2 ) ) {
   // a combination of the first two rows
   const Index r = e.G[ k ].rows();
   e.G[ k ].conservativeResize( r + 1 , Eigen::NoChange );
   e.H[ k ].conservativeResize( r + 1 , Eigen::NoChange );
   e.G[ k ].row( r ) = e.G[ k ].row( 0 ) + 2 * e.G[ k ].row( 1 );
   e.H[ k ].row( r ) = e.H[ k ].row( 0 ) + 2 * e.H[ k ].row( 1 );
   }
  e.g[ k ] = e.G[ k ] * xb[ k ];
  if( k < N )
   e.g[ k ] += e.H[ k ] * ub[ k ];
  }
 if( dep & kThroughDynamics )
  // a row of stage k + 1 on x_{k+1} only, written also at stage k
  for( Index k = 0 ; k + 1 < N ; k += 2 ) {
   if( ! nx[ k + 1 ] )
    continue;
   const Mat a = rnd( 1 , nx[ k + 1 ] );
   auto add = [ & ]( Index s , const Mat & Gr , const Mat & Hr , double v ) {
    const Index r = e.G[ s ].rows();
    e.G[ s ].conservativeResize( r + 1 , Eigen::NoChange );
    e.H[ s ].conservativeResize( r + 1 , Eigen::NoChange );
    e.g[ s ].conservativeResize( r + 1 );
    e.G[ s ].row( r ) = Gr;
    e.H[ s ].row( r ) = Hr;
    e.g[ s ]( r ) = v;
    };
   add( k + 1 , a , Mat::Zero( 1 , nu[ k + 1 ] ) , ( a * xb[ k + 1 ] )( 0 ) );
   add( k , a * e.A[ k ] , a * e.B[ k ] ,
	( a * ( xb[ k + 1 ] - e.c[ k ] ) )( 0 ) );
   }
 if( dep & kInconsistent )
  for( Index k = 0 ; k <= N ; ++k )
   if( e.g[ k ].size() > mrow[ k ] ) {
    e.g[ k ]( e.g[ k ].size() - 1 ) += 1e-3;
    break;
    }
 // the rows of x_N have no input
 e.H[ N ].resize( e.G[ N ].rows() , 0 );
 return( e );
 }

/*--------------------------------------------------------------------------*/
// the KKT system of the whole horizon, z = ( x_0 , u_0 , x_1 , ... , x_N ),
// solved by a complete orthogonal decomposition (dependent rows allowed);
// returns the trajectory

static void dense_kkt( const EQP & e , std::vector< EVec > & x ,
		       std::vector< EVec > & u )
{
 const Index N = e.num_stages();
 std::vector< Index > xi( N + 1 ) , ui( N );
 Index nz = 0 , m = 0;
 for( Index k = 0 ; k <= N ; ++k ) {
  xi[ k ] = nz;
  nz += e.Q[ k ].rows();
  if( k < N ) {
   ui[ k ] = nz;
   nz += e.R[ k ].rows();
   m += e.A[ k ].rows();
   }
  m += e.G[ k ].rows();
  }
 Mat K = Mat::Zero( nz + m , nz + m );
 EVec rhs = EVec::Zero( nz + m );
 Index row = nz;
 for( Index k = 0 ; k <= N ; ++k ) {
  const Index nx = e.Q[ k ].rows();
  K.block( xi[ k ] , xi[ k ] , nx , nx ) = e.Q[ k ];
  rhs.segment( xi[ k ] , nx ) = -e.q[ k ];
  auto put = [ & ]( Index r , Index c , const Mat & M ) {
   K.block( r , c , M.rows() , M.cols() ) = M;
   K.block( c , r , M.cols() , M.rows() ) = M.transpose();
   };
  if( k < N ) {
   const Index n_u = e.R[ k ].rows();
   K.block( ui[ k ] , ui[ k ] , n_u , n_u ) = e.R[ k ];
   K.block( ui[ k ] , xi[ k ] , n_u , nx ) = e.S[ k ];
   K.block( xi[ k ] , ui[ k ] , nx , n_u ) = e.S[ k ].transpose();
   rhs.segment( ui[ k ] , n_u ) = -e.r[ k ];
   const Index nd = e.A[ k ].rows();
   put( row , xi[ k + 1 ] , -Mat::Identity( nd , nd ) );
   put( row , xi[ k ] , e.A[ k ] );
   put( row , ui[ k ] , e.B[ k ] );
   rhs.segment( row , nd ) = -e.c[ k ];
   row += nd;
   }
  const Index mg = e.G[ k ].rows();
  if( mg ) {
   put( row , xi[ k ] , e.G[ k ] );
   if( k < N )
    put( row , ui[ k ] , e.H[ k ] );
   rhs.segment( row , mg ) = e.g[ k ];
   row += mg;
   }
  }
 const EVec sol =
   Eigen::CompleteOrthogonalDecomposition< Mat >( K ).solve( rhs );
 x.assign( N + 1 , EVec() );
 u.assign( N , EVec() );
 for( Index k = 0 ; k <= N ; ++k ) {
  x[ k ] = sol.segment( xi[ k ] , e.Q[ k ].rows() );
  if( k < N )
   u[ k ] = sol.segment( ui[ k ] , e.R[ k ].rows() );
  }
 }

/*--------------------------------------------------------------------------*/
// the objective of e at a trajectory

static double eqp_value( const EQP & e , const std::vector< EVec > & x ,
			 const std::vector< EVec > & u )
{
 const Index N = e.num_stages();
 double v = 0;
 for( Index k = 0 ; k <= N ; ++k ) {
  v += 0.5 * x[ k ].dot( e.Q[ k ] * x[ k ] ) + e.q[ k ].dot( x[ k ] );
  if( k < N )
   v += u[ k ].dot( e.S[ k ] * x[ k ] ) +
	0.5 * u[ k ].dot( e.R[ k ] * u[ k ] ) + e.r[ k ].dot( u[ k ] );
  }
 return( v );
 }

/*--------------------------------------------------------------------------*/
// the largest residual of the dynamics and of the rows at the solution of e

static double eqp_residual( const EQP & e )
{
 const Index N = e.num_stages();
 double r = 0;
 for( Index k = 0 ; k <= N ; ++k ) {
  EVec g = e.G[ k ] * e.x[ k ] - e.g[ k ];
  if( k < N ) {
   g += e.H[ k ] * e.u[ k ];
   r = std::max(
     r ,
       ( e.A[ k ] * e.x[ k ] + e.B[ k ] * e.u[ k ] + e.c[ k ] - e.x[ k + 1 ] )
	  .lpNorm< Eigen::Infinity >() );
   }
  if( g.size() )
   r = std::max( r , g.lpNorm< Eigen::Infinity >() );
  }
 return( r );
 }

/*--------------------------------------------------------------------------*/
// the largest residual of the stationarity conditions with the multipliers
// of e: the costates backward from lambda_N, the conditions on each u_k and
// on the free x_0

static double eqp_stationarity( const EQP & e )
{
 const Index N = e.num_stages();
 EVec lam = e.Q[ N ] * e.x[ N ] + e.q[ N ] + e.G[ N ].transpose() * e.mu[ N ];
 double r = 0;
 for( Index k = N ; k-- > 0 ; ) {
  const EVec gu = e.R[ k ] * e.u[ k ] + e.S[ k ] * e.x[ k ] + e.r[ k ] +
		  e.B[ k ].transpose() * lam +
		  e.H[ k ].transpose() * e.mu[ k ];
  if( gu.size() )
   r = std::max( r , gu.lpNorm< Eigen::Infinity >() );
  lam = e.Q[ k ] * e.x[ k ] + e.S[ k ].transpose() * e.u[ k ] + e.q[ k ] +
	e.A[ k ].transpose() * lam + e.G[ k ].transpose() * e.mu[ k ];
  }
 if( lam.size() )
  r = std::max( r , lam.lpNorm< Eigen::Infinity >() );
 return( r );
 }

/*--------------------------------------------------------------------------*/
// the recursion against the dense KKT system on a random StageEQP: the same
// trajectory (to xtol, relative), objective, rows satisfied and multipliers
// satisfying the stationarity conditions (to the same); kInconsistent must
// be reported

static void eqp_case( const std::string & name , unsigned seed ,
		      const std::vector< Index > & nx ,
		      const std::vector< Index > & nu ,
		      const std::vector< Index > & mrow , Index ntask ,
		      double eps , int dep , double xtol = 1e-7 ,
		      double gs = 1 )
{
 const int e0 = errors;
 EQP e = random_eqp( seed , nx , nu , mrow , ntask , eps , dep , gs );
 const double incons = e.solve( 0 );
 if( dep & kInconsistent ) {
  CHECK( incons > 1e-5 );
  std::cout << " recursion, " << name << ": inconsistency " << incons
	    << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  return;
  }

 std::vector< EVec > xk , uk;
 dense_kkt( e , xk , uk );
 double dx = 0 , sc = 1;
 for( Index k = 0 ; k < xk.size() ; ++k ) {
  if( xk[ k ].size() ) {
   dx = std::max( dx , ( e.x[ k ] - xk[ k ] ).lpNorm< Eigen::Infinity >() );
   sc = std::max( sc , xk[ k ].lpNorm< Eigen::Infinity >() );
   }
  if( ( k < uk.size() ) && uk[ k ].size() ) {
   dx = std::max( dx , ( e.u[ k ] - uk[ k ] ).lpNorm< Eigen::Infinity >() );
   sc = std::max( sc , uk[ k ].lpNorm< Eigen::Infinity >() );
   }
  }
 const double fr = eqp_value( e , e.x , e.u ) , fk = eqp_value( e , xk , uk );
 const double res = eqp_residual( e ) , sta = eqp_stationarity( e );
 CHECK( incons < 1e-9 );
 CHECK( dx <= xtol * sc );
 CHECK( std::abs( fr - fk ) <= 1e-9 * std::max( 1.0 , std::abs( fk ) ) );
 CHECK( res <= 1e-9 * sc );
 CHECK( sta <= xtol * sc );
 if( dep & ( kInStage | kThroughDynamics ) )
  CHECK( e.dropped > 0 );
 std::cout << " recursion, " << name << ": |x - KKT| " << dx << ", objective "
	   << fr << " vs " << fk << ", rows " << res << ", stationarity "
	   << sta << ", dropped " << e.dropped << ", to go " << e.to_go
	   << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
 }

/*--------------------------------------------------------------------------*/

static void test_recursion( void )
{
 std::cout << "the Riccati recursion of HQPStageSolver::StageEQP" << std::endl;
 auto rep = []( Index n , Index v ) {
  return( std::vector< Index >( n , v ) );
  };

 // the size of one robot of the radial scenario, 0 to 2 rows per stage
 std::vector< Index > m10( 11 );
 for( Index k = 0 ; k <= 10 ; ++k )
  m10[ k ] = k % 3;
 eqp_case( "one robot" , 1 , rep( 11 , 4 ) , rep( 10 , 2 ) , m10 , 2 , 1e-6 ,
	   kNone );

 // more rows than inputs: constraints-to-go at every stage
 eqp_case( "more rows than inputs" , 2 , rep( 9 , 4 ) , rep( 8 , 2 ) ,
	   { 3 , 0 , 3 , 1 , 3 , 0 , 3 , 0 , 1 } , 2 , 1e-6 , kNone );

 // as many rows as inputs: no freedom left in u
 eqp_case( "inputs fixed by the rows" , 3 , rep( 7 , 3 ) , rep( 6 , 2 ) ,
	   { 2 , 2 , 2 , 2 , 2 , 2 , 0 } , 1 , 1e-6 , kNone , 1e-7 , 0.1 );

 // dependent rows in a stage, and through the dynamics
 eqp_case( "dependent rows in a stage" , 4 , rep( 9 , 4 ) , rep( 8 , 2 ) ,
	   { 2 , 3 , 0 , 2 , 2 , 3 , 0 , 2 , 0 } , 2 , 1e-6 , kInStage );
 eqp_case( "rows dependent through the dynamics" , 5 , rep( 9 , 4 ) , rep( 8 ,
									   2 ) ,
	   { 1 , 0 , 1 , 0 , 1 , 0 , 1 , 0 , 0 } , 2 , 1e-6 ,
      kThroughDynamics );
 eqp_case( "both" , 6 , rep( 9 , 4 ) , rep( 8 , 2 ) , { 2 , 2 , 2 , 2 , 2 , 2 ,
	   2 , 2 , 0 } ,
	   2 , 1e-6 , kInStage | kThroughDynamics );

 // rows on x_N, a free x_0 constrained by the rows of stage 0
 eqp_case( "rows on x_0 and x_N" , 7 , rep( 6 , 3 ) , rep( 5 , 1 ) ,
	   { 2 , 1 , 0 , 1 , 0 , 2 } , 1 , 1e-6 , kNone );

 // x_0 of size 0, as with no initial state among the columns, and sizes
 // that change along the stages
 eqp_case( "x_0 of size 0" , 8 , { 0 , 2 , 3 , 3 , 4 , 2 } , { 2 , 1 , 2 , 3 ,
	   1 } ,
	   { 1 , 0 , 2 , 1 , 3 , 1 } , 1 , 1e-6 , kNone );

 // a single stage, and stages with no input
 eqp_case( "one stage" , 9 , { 3 , 3 } , { 2 } , { 1 , 1 } , 2 , 1e-6 ,
	   kNone );
 eqp_case( "stages with no input" , 10 , rep( 5 , 2 ) , { 1 , 0 , 2 , 0 } ,
	   { 1 , 1 , 0 , 1 , 0 } , 2 , 1e-6 , kNone );

 // the size of the 8 robots of the radial scenario
 std::vector< Index > m8( 11 );
 for( Index k = 0 ; k <= 10 ; ++k )
  m8[ k ] = ( 5 * k ) % 17;
 m8[ 10 ] = 3;
 eqp_case( "eight robots" , 11 , rep( 11 , 32 ) , rep( 10 , 16 ) , m8 , 16 ,
	   1e-6 ,
	   kNone , 1e-3 );

 // a dependent row that is not consistent
 eqp_case( "an inconsistent row" , 12 , rep( 5 , 3 ) , rep( 4 , 2 ) ,
	   { 2 , 2 , 2 , 2 , 0 } , 1 , 1e-6 , kInStage | kInconsistent );

 // no regularization and one task row: Z singular, the minimum-norm step
 {
  const int e0 = errors;
  EQP e =
    random_eqp( 13 , rep( 5 , 3 ) , rep( 4 , 3 ) , { 1 , 0 , 1 , 0 , 0 } , 1 ,
		0 , kNone );
  e.solve( 0 );
  std::vector< EVec > xk , uk;
  dense_kkt( e , xk , uk );
  const double fr = eqp_value( e , e.x , e.u ) , fk = eqp_value( e , xk , uk );
  CHECK( eqp_residual( e ) <= 1e-8 );
  CHECK( fr <= fk + 1e-8 * std::max( 1.0 , std::abs( fk ) ) );
  std::cout << " recursion, no regularization: objective " << fr << " vs "
	    << fk << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  }
 }

/*--------------------------------------------------------------------------*/
/*------------------------- THE HQPStageSolver -----------------------------*/
/*--------------------------------------------------------------------------*/

static void test_stage_solver( void )
{
 std::cout << "HQPStageSolver" << std::endl;

 // the levels of the integrator, whose values are known; with the stages
 // of the rows given, the same
 for( int given = 0 ; given < 2 ; ++given ) {
  const int e0 = errors;
  Data D = integrator_data();
  if( given ) {
   D.st.eq_stage = { 0 , 1 , 1 , 0 };
   D.st.ineq_stage = { 0 , 0 , 1 , 1 , 1 , 1 };
   }
  HQPBlock block;
  load( block , D );
  Cascade cs( block );
  const auto val = cs.run();
  CHECK( cs.status == Solver::kOK );
  CHECK( values_are( val , { 0 , 0 , 0 , 4.5 , 0.5 } , 1e-7 ) );
  Vec x;
  block.get_x_values( x );
  CHECK( std::abs( x[ 0 ] - 1 ) + std::abs( x[ 1 ] - 1 ) +
	   std::abs( x[ 2 ] - 1 ) + std::abs( x[ 3 ] - 2 ) <
	 1e-6 );
  std::cout << " the integrator"
	    << ( given ? ", stages of the rows given" : "" )
	    << ": values 0, 0, 0, "
	    << "9/2, 1/2 in " << cs.iter << " iterations"
	    << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  }

 // the initial state among the columns: x_0 is chosen too
 {
  const int e0 = errors;
  HQPBlock block;
  load( block , initial_state_data() );
  Cascade cs( block );
  const auto val = cs.run();
  CHECK( cs.status == Solver::kOK );
  CHECK( values_are( val , { 0 , 0 , 0 , 2 } , 1e-7 ) );
  std::cout << " the initial state as a column: values 0, 0, 0, 2"
	    << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  }

 // dependent rows: two equal rows of the goal asking for 5 and 7 (their
 // mean, value (9 + 25) / 2), which the next levels keep as two dependent
 // hard rows, and the goal repeated in the last level
 {
  const int e0 = errors;
  Data D = integrator_data();
  Mat A3( 2 , 4 ) , A4( 2 , 4 );
  A3 << 0 , 0 , 0 , 1 , 0 , 0 , 0 , 1;
  A4 << 0 , 0 , 1 , 0 , 0 , 0 , 0 , 1;
  D.A[ 3 ] = A3;
  D.b[ 3 ] = { 5 , 7 };
  D.A[ 4 ] = A4;
  D.b[ 4 ] = { 0 , 5 };
  HQPBlock block;
  load( block , D );
  Cascade cs( block );
  const auto val = cs.run();
  CHECK( cs.status == Solver::kOK );
  CHECK( values_are( val , { 0 , 0 , 0 , 17 , 5 } , 1e-7 ) );
  std::cout << " dependent rows: values 0, 0, 0, 17, 5"
	    << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  }

 // a step of Model Predictive Control: the goal moves (set_b(), the
 // dynamics kept) and then the dynamics change (set_A(), computed anew);
 // the warm start takes no more iterations than the cold one, and both agree
 // with a new Solver of the new data
 {
  const int e0 = errors;
  Data D = integrator_data();
  HQPBlock warm , cold;
  load( warm , D );
  load( cold , D );
  Cascade cw( warm ) , cc( cold );
  for( Index p = 0 ; p < warm.get_num_levels() ; ++p )
   cc.solver( p )->set_par( HQPStageSolver::intWarmStart , 0 );
  cw.run();
  cc.run();
  long iw = 0 , ic = 0;
  for( int k = 0 ; k < 2 ; ++k ) {
   if( k == 0 ) {
    warm.set_b( 3 , { 1.5 } );
    cold.set_b( 3 , { 1.5 } );
    }
   else {
    Mat A0( 2 , 4 );
    A0 << -0.5 , 0 , 1 , 0 , 0 , -0.5 , -1 , 1;
    warm.set_A( 0 , sparse( A0 ) );
    cold.set_A( 0 , sparse( A0 ) );
    }
   const auto vw = cw.run();
   iw += cw.iter;
   const auto vc = cc.run();
   ic += cc.iter;
   // a new HQPBlock with the new data
   HQPBlock fresh;
   Data F = D;
   F.b[ 3 ] = { 1.5 };
   if( k ) {
    F.A[ 0 ] << -0.5 , 0 , 1 , 0 , 0 , -0.5 , -1 , 1;
    }
   load( fresh , F );
   Cascade cf( fresh );
   const auto vf = cf.run();
   CHECK( ( cw.status == Solver::kOK ) && ( cc.status == Solver::kOK ) );
   CHECK( values_are( vw , vf , 1e-9 ) );
   CHECK( values_are( vc , vf , 1e-9 ) );
   }
  CHECK( iw <= ic );
  std::cout << " two steps of Model Predictive Control: iterations warm " << iw
	    << ", cold " << ic << ( errors == e0 ? " -> OK" : " -> KO" )
	    << std::endl;
  }

 // what does not fit, and the parameters
 {
  const int e0 = errors;
  auto refused = []( HQPBlock & block , Index level ) {
   Cascade cs( block );
   generate( block );
   block.reset_levels( true );
   for( Index p = 0 ; p < level ; ++p ) {
    cs.solver( p )->compute();
    cs.solver( p )->get_var_solution();
    block.next_level();
    }
   return( throws< std::invalid_argument >(
     [ & ]() { cs.solver( level )->compute(); } ) );
   };

  // no stage structure
  HQPBlock b0;
  load( b0 , small_data() );
  CHECK( refused( b0 , 0 ) );

  // a level above that of the dynamics
  Data D1 = integrator_data();
  std::swap( D1.A[ 0 ] , D1.A[ 3 ] );
  std::swap( D1.b[ 0 ] , D1.b[ 3 ] );
  D1.st.dynamics_level = 3;
  HQPBlock b1;
  load( b1 , D1 );
  CHECK( refused( b1 , 0 ) );

  // a row of the dynamics with two states of its stage
  Data D2 = integrator_data();
  D2.A[ 0 ]( 1 , 2 ) = 1;
  D2.A[ 0 ]( 0 , 3 ) = 1;
  HQPBlock b2;
  load( b2 , D2 );
  CHECK( refused( b2 , 0 ) );

  // a row out of its given stage (s_2 at stage 0): refused at its level,
  // not before
  Data D3 = integrator_data();
  D3.st.eq_stage = { 0 , 1 , 0 , 0 };
  HQPBlock b3;
  load( b3 , D3 );
  CHECK( ! refused( b3 , 2 ) );
  HQPBlock b4;
  load( b4 , D3 );
  CHECK( refused( b4 , 3 ) );

  // a Block that is not an HQPBlock
  HQPDistBlock db;
  db.load( b0 , 2 , { 0 , 1 } );
  HQPStageSolver s;
  s.set_Block( &db );
  CHECK( throws< std::invalid_argument >( [ & ]() { s.compute(); } ) );
  s.set_Block( nullptr );

  // the parameters
  CHECK( s.get_int_par( HQPStageSolver::intWarmStart ) == 1 );
  CHECK( s.get_int_par( Solver::intMaxIter ) == 10000 );
  CHECK( s.get_dbl_par( HQPStageSolver::dblRankTol ) == 0 );
  CHECK( s.int_par_str2idx( "intWarmStart" ) == HQPStageSolver::intWarmStart );
  CHECK( s.dbl_par_str2idx( "dblRankTol" ) == HQPStageSolver::dblRankTol );
  CHECK( s.dbl_par_idx2str( HQPStageSolver::dblDualTol ) == "dblDualTol" );
  CHECK( throws< std::invalid_argument >(
    [ & ]() { s.set_par( HQPStageSolver::intWarmStart , 2 ); } ) );
  CHECK( throws< std::invalid_argument >(
    [ & ]() { s.set_par( HQPStageSolver::dblRankTol , -1.0 ); } ) );

  // an iteration limit stops a level before its optimum
  HQPBlock b5;
  load( b5 , integrator_data() );
  Cascade c5( b5 );
  for( Index p = 0 ; p < b5.get_num_levels() ; ++p )
   c5.solver( p )->set_par( Solver::intMaxIter , 1 );
  c5.run();
  CHECK( c5.status == Solver::kStopIter );

  std::cout << " what does not fit, the parameters and the iteration limit"
	    << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  }
 }

/*--------------------------------------------------------------------------*/
/*-------------------------- THE INSTANCE FILES ----------------------------*/
/*--------------------------------------------------------------------------*/
// the levels of the HQPBlock of an instance file solved by HQPStageSolver,
// against those at the solution of each level that the file carries (the
// group HQPReference, written by the code that generated it)

static void test_file( const std::string & file )
{
 const int e0 = errors;
 const std::string nm = std::filesystem::path( file ).stem().string();
 auto block = dynamic_cast< HQPBlock * >( Block::deserialize( file ) );
 if( ! block ) {
  std::cout << " " << nm << ": cannot read the HQPBlock -> KO" << std::endl;
  ++errors;
  return;
  }
 CHECK( block->get_num_stages() > 0 );

 const Index P = block->get_num_levels() , n = block->get_num_var();
 Mat X( P , n );
 {
  netCDF::NcFile f( file , netCDF::NcFile::read );
  Eigen::Matrix< double , Eigen::Dynamic , Eigen::Dynamic ,
    Eigen::RowMajor > Xr(
    P , n );
  f.getGroup( "HQPReference" ).getVar( "X" ).getVar( Xr.data() );
  X = Xr;
  }

 std::vector< double > ref;
 for( Index p = 0 ; p < P ; ++p ) {
  Vec x( n );
  for( Index j = 0 ; j < n ; ++j )
   x[ j ] = X( p , j );
  generate( *block );
  block->set_x_values( x );
  Vec v , w;
  ref.push_back( block->level_slacks( p , v , w ) );
  }

 {
  Cascade cs( *block );
  const auto val = cs.run();
  CHECK( cs.status == Solver::kOK );
  CHECK( values_are( val , ref , 1e-5 ) );
  Vec x;
  block->get_x_values( x );
  double dx = 0;
  for( Index j = 0 ; j < n ; ++j )
   dx = std::max( dx , std::abs( x[ j ] - X( P - 1 , j ) ) );
  std::cout << " " << nm << ": " << P << " levels, " << n << " columns, "
	    << block->get_num_stages() << " stages, " << cs.iter
	    << " iterations, |x - file| " << dx << ", values";
  for( Index p = 0 ; p < P ; ++p )
   std::cout << " " << val[ p ] << " (" << ref[ p ] << ")";
  std::cout << ( errors == e0 ? " -> OK" : " -> KO" ) << std::endl;
  }
 delete block;
 }

/*--------------------------------------------------------------------------*/

int main( int argc , char ** argv )
{
 // the instance files given, if any, and nothing else
 if( argc > 1 ) {
  std::cout << "HQPStageSolver on the instance files" << std::endl;
  for( int i = 1 ; i < argc ; ++i )
   test_file( argv[ i ] );
  std::cout << ( errors ? "HQPBlock: FAILED" : "HQPBlock: all tests passed" )
	    << std::endl;
  return( errors ? 1 : 0 );
  }

 test_load();
 test_abstract();
 test_levels();
 test_modifications();
 test_netcdf();
 test_print();
 test_reg_weights();
 test_refix();
 test_scale();
 test_dist();
 test_dist_benders();
 test_stage_data();
 test_recursion();
 test_stage_solver();

 std::cout << ( errors ? "HQPBlock: FAILED" : "HQPBlock: all tests passed" )
           << std::endl;
 return( errors ? 1 : 0 );
 }

/*--------------------------------------------------------------------------*/
/*---------------------------- End File test.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
