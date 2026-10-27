/*--------------------------------------------------------------------------*/
/*------------------------------ File test.cpp -----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Unit test of HQPBlock, which needs the core library only: nothing is
 * solved, and what is checked is what the Block itself does, on small
 * hierarchies written here whose degenerate parts are a level of equalities
 * only, one of inequalities only, an empty one, rows with no entries and
 * weights different from 1. The cases are:
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
 *   anew), the netCDF format and the errors of load().
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
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <netcdf>

#include "DQuadFunction.h"

#include "FakeSolver.h"

#include "HQPBlock.h"

#include "HQPDistBlock.h"

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
 block.load( D.n , std::move( A ) , std::move( b ) , std::move( C ) ,
             std::move( d ) , std::move( we ) , std::move( wi ) , D.eps );
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
// the Objective is that of the current state of the levels: ( - eps c ,
// eps / 2 ) on x, ( 0 , 1 / 2 ) on the slacks of the levels in it, given by
// in[ p ], ( 0 , 0 ) on the others

static bool objective_agrees( HQPBlock & block ,
                              const std::vector< bool > & in )
{
 const auto f = objective( block );
 const Index n = block.get_num_var() , ne = block.get_eq().size();
 const double eps = block.get_regularization();
 const auto & c = block.get_center();
 if( f->get_num_active_var() != n + ne + block.get_ineq().size() )
  return( false );
 for( Index j = 0 ; j < n ; ++j )
  if( ( f->get_active_var( j ) != & block.get_x()[ j ] ) ||
      ( f->get_linear_coefficient( j ) !=
        ( c.empty() ? 0.0 : - eps * c[ j ] ) ) ||
      ( f->get_quadratic_coefficient( j ) != eps / 2 ) )
   return( false );
 for( Index p = 0 ; p < block.get_num_levels() ; ++p ) {
  const double q = in[ p ] ? 0.5 : 0.0;
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
 block.set_center( { 1 , 2 , 3 } );
 c = count( s , f );
 CHECK( ( c.obj == 1 ) && ( c.total() == 1 ) );
 CHECK( objective_agrees( block , { true , false , false , false } ) );
 CHECK( throws< std::invalid_argument >( [ & ]() {
  block.set_center( { 1 } ); } ) );
 block.set_center( {} );
 CHECK( count( s , f ).total() == 1 );
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

int main( int argc , char ** argv )
{
 test_load();
 test_abstract();
 test_levels();
 test_modifications();
 test_netcdf();
 test_print();
 test_reg_weights();
 test_refix();
 test_dist();

 std::cout << ( errors ? "HQPBlock: FAILED" : "HQPBlock: all tests passed" )
           << std::endl;
 return( errors ? 1 : 0 );
 }

/*--------------------------------------------------------------------------*/
/*---------------------------- End File test.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
