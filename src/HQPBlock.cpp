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

#include <numeric>

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
// of the netCDF group, as described in HQPBlock::deserialize()

void read_rows( const netCDF::NcGroup & group , const std::string & pfx ,
                std::size_t n , std::size_t P ,
                std::vector< HQPBlock::SparseMatrix > & M ,
                std::vector< std::vector< double > > & rhs ,
                std::vector< std::vector< double > > & wgh )
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

 std::vector< unsigned int > rows;
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
// into the netCDF group, as described in HQPBlock::deserialize()

void write_rows( netCDF::NcGroup & group , const std::string & pfx ,
                 const netCDF::NcDim & levels ,
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
 group.addVar( pfx + "Rows" , netCDF::NcUint() , levels ).putVar(
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
                     Vec_double && rw )
{
 guts_of_load( n , std::move( A ) , std::move( b ) , std::move( C ) ,
               std::move( d ) , std::move( we ) , std::move( wi ) , eps ,
               std::move( rw ) );

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
                             Vec_double && rw )
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

 guts_of_destructor();

 f_n = n;
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

 // the NBModification is issued by Block::deserialize()
 guts_of_load( n , std::move( A ) , std::move( b ) , std::move( C ) ,
               std::move( d ) , std::move( we ) , std::move( wi ) , eps ,
               std::move( rw ) );

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

 write_rows( group , "Eq" , dl , v_A , v_b , v_we );
 write_rows( group , "Ineq" , dl , v_C , v_d , v_wi );

 if( f_eps != 0 )
  group.addVar( "Regularization" , netCDF::NcDouble() ).putVar( &f_eps );

 if( std::any_of( v_rw.begin() , v_rw.end() ,
                  []( double r ) { return( r != 1 ); } ) )
  group.addVar( "RegularizationWeight" , netCDF::NcDouble() , dn ).putVar(
                                                               v_rw.data() );

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

 add_static_constraint( v_eq , "eq" );
 add_static_constraint( v_in , "ineq" );
 add_dynamic_constraint( v_link , "link" );

 AR |= HasCns;

 }  // end( HQPBlock::generate_abstract_constraints )

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
 for( Index h = 0 ; h < v.size() ; ++h )
  v_v[ v_eq_start[ p ] + h ].set_value( v[ h ] );
 for( Index h = 0 ; h < w.size() ; ++h )
  v_w[ v_in_start[ p ] + h ].set_value( w[ h ] );
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
 for( auto & row : v_link )
  row.clear();
 f_obj.clear();

 reset_static_constraints();
 reset_dynamic_constraints();
 reset_static_variables();
 reset_objective();

 v_eq.clear();
 v_in.clear();
 v_link.clear();
 v_cols.clear();
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
 const Index ne = v_b.size();
 std::vector< int > pos( f_n , -1 );

 for( Index h = 0 ; h < Index( Mp.rows() ) ; ++h ) {
  const Index i = first + h;
  const double wg = eq ? v_we[ i ] : v_wi[ i ];
  auto & row = eq ? v_eq[ i ] : v_in[ i ];
  auto lf = static_cast< LinearFunction * >( row.get_function() );
  auto & cols = v_cols[ eq ? i : ne + i ];

  // the new coefficient of each active Variable, the slack excluded
  Vec_double nc( cols.size() , 0 );
  for( Index k = 0 ; k < cols.size() ; ++k )
   pos[ cols[ k ] ] = k;
  LinearFunction::v_coeff_pair added;
  for( SparseMatrix::InnerIterator it( Mp , h ) ; it ; ++it )
   if( pos[ it.col() ] >= 0 )
    nc[ pos[ it.col() ] ] = wg * it.value();
   else
    added.emplace_back( & v_x[ it.col() ] , wg * it.value() );
  for( auto c : cols )
   pos[ c ] = -1;

  // only the coefficients that change are sent
  Vec_double chg;
  LinearFunction::Subset nms;
  for( Index k = 0 ; k < cols.size() ; ++k )
   if( lf->get_coefficient( k + 1 ) != nc[ k ] ) {
    chg.push_back( nc[ k ] );
    nms.push_back( k + 1 );
    }
  if( ! nms.empty() )
   lf->modify_coefficients( std::move( chg ) , std::move( nms ) , true ,
                            issueMod );

  if( ! added.empty() ) {
   for( const auto & a : added )
    cols.push_back( std::distance( v_x.data() , a.first ) );
   lf->add_variables( std::move( added ) , issueMod );
   }

  if( eq ) {
   if( row.get_rhs() != wg * v_b[ i ] )
    row.set_both( wg * v_b[ i ] , issueMod );
   }
  else
   if( row.get_rhs() != wg * v_d[ i ] )
    row.set_rhs( wg * v_d[ i ] , issueMod );
  }
 }  // end( HQPBlock::set_rows )

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
/*------------------------ End File HQPBlock.cpp ---------------------------*/
/*--------------------------------------------------------------------------*/
