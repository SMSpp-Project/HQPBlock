/*--------------------------------------------------------------------------*/
/*------------------------- File HQPDistBlock.cpp --------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the HQPDistBlock class, which implements the Block
 * concept [see Block.h] for a hierarchical (lexicographic) Quadratic Program
 * whose variables are split among sub-Blocks.
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

#include "HQPDistBlock.h"

#include <map>

#include "ColVariableSolution.h"

#include "SMSnetCDF.h"

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register HQPDistBlock in the Block factory

SMSpp_insert_in_factory_cpp_1( HQPDistBlock );

/*--------------------------------------------------------------------------*/
/*--------------------------- LOCAL FUNCTIONS ------------------------------*/
/*--------------------------------------------------------------------------*/

namespace {

using Index = Block::Index;
using Vec_Index = HQPDistBlock::Vec_Index;
using SparseMatrix = HQPBlock::SparseMatrix;

// the owner of a row out of the owners of its columns: the one with most
// entries in it; between two with as many, r among r < s if ( s - r ) mod R
// <= R / 2, so that each owner owns about half of the pairs it is in, and
// the smallest one among more than two; 0 for a row with no entries

Index row_owner( const SparseMatrix & M , Eigen::Index h ,
                 const Vec_Index & col_owner , Index R )
{
 std::map< Index , Index > cnt;
 for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
  ++cnt[ col_owner[ it.col() ] ];
 if( cnt.empty() )
  return( 0 );

 Index best = 0;
 for( const auto & [ r , c ] : cnt )
  best = std::max( best , c );
 Vec_Index cand;
 for( const auto & [ r , c ] : cnt )
  if( c == best )
   cand.push_back( r );

 if( cand.size() == 2 )
  return( ( ( cand[ 1 ] - cand[ 0 ] ) % R <= R / 2 ) ? cand[ 0 ]
                                                     : cand[ 1 ] );
 return( cand.front() );
 }

/*--------------------------------------------------------------------------*/
// a copy of the data of an HQPBlock into another one

void copy_data( const HQPBlock & from , HQPBlock & to )
{
 const Index P = from.get_num_levels();
 std::vector< SparseMatrix > A , C;
 std::vector< std::vector< double > > b , d , we , wi;
 for( Index p = 0 ; p < P ; ++p ) {
  A.push_back( from.get_A( p ) );
  C.push_back( from.get_C( p ) );
  const auto e = from.get_eq_start( p ) , i = from.get_ineq_start( p );
  const auto me = from.get_num_eq_rows( p ) , mi = from.get_num_ineq_rows( p );
  b.emplace_back( from.get_b().begin() + e , from.get_b().begin() + e + me );
  d.emplace_back( from.get_d().begin() + i , from.get_d().begin() + i + mi );
  we.emplace_back( from.get_we().begin() + e ,
                   from.get_we().begin() + e + me );
  wi.emplace_back( from.get_wi().begin() + i ,
                   from.get_wi().begin() + i + mi );
  }
 auto rw = from.get_regularization_weights();
 to.load( from.get_num_var() , std::move( A ) , std::move( b ) ,
          std::move( C ) , std::move( d ) , std::move( we ) , std::move( wi ) ,
          from.get_regularization() , std::move( rw ) );
 }

/*--------------------------------------------------------------------------*/
// the owners in a netCDF variable of the group, if it is there

bool read_owner( const netCDF::NcGroup & group , const std::string & dim ,
                 const std::string & var , Vec_Index & owner )
{
 std::size_t m = 0;
 ::deserialize_dim( group , dim , m );
 if( ( ! m ) || group.getVar( var ).isNull() )
  return( false );
 std::vector< unsigned int > o;
 ::deserialize( group , var , m , o , false );
 owner.assign( o.begin() , o.end() );
 return( true );
 }

/*--------------------------------------------------------------------------*/
// the owners in a netCDF variable over a new dimension of the group

void write_owner( netCDF::NcGroup & group , const std::string & dim ,
                  const std::string & var , const Vec_Index & owner )
{
 std::vector< unsigned int > o( owner.begin() , owner.end() );
 auto dm = group.addDim( dim , o.size() );
 group.addVar( var , netCDF::NcUint() , dm ).putVar( o.data() );
 }

}  // end( unnamed namespace )

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void HQPDistBlock::load( const HQPBlock & central , Index R ,
                         Vec_Index && col_owner , Vec_Index && eq_owner ,
                         Vec_Index && in_owner )
{
 guts_of_destructor();

 copy_data( central , f_central );
 f_R = R;
 v_col_owner = std::move( col_owner );
 v_eq_owner = std::move( eq_owner );
 v_in_owner = std::move( in_owner );
 f_given_eq = ! v_eq_owner.empty();
 f_given_in = ! v_in_owner.empty();
 build();

 regenerate();

 }  // end( HQPDistBlock::load( memory ) )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::deserialize( const netCDF::NcGroup & group )
{
 std::size_t R = 0 , n = 0;
 ::deserialize_dim( group , "NumRobots" , R , false );
 ::deserialize_dim( group , "NumVar" , n , false );

 Vec_Index col_owner , eq_owner , in_owner;
 if( ! read_owner( group , "NumVar" , "ColumnOwner" , col_owner ) )
  throw( std::invalid_argument( "HQPDistBlock::deserialize: no "
                                "ColumnOwner" ) );
 read_owner( group , "NumEqRows" , "EqRowOwner" , eq_owner );
 read_owner( group , "NumIneqRows" , "IneqRowOwner" , in_owner );

 const auto sub = group.getGroup( "HQPBlock" );
 if( sub.isNull() )
  throw( std::invalid_argument( "HQPDistBlock::deserialize: no group "
                                "HQPBlock" ) );
 HQPBlock central;
 central.deserialize( sub );
 if( central.get_num_var() != n )
  throw( std::invalid_argument( "HQPDistBlock::deserialize: NumVar differs "
                                "from that of the group HQPBlock" ) );

 guts_of_destructor();

 copy_data( central , f_central );
 f_R = R;
 v_col_owner = std::move( col_owner );
 v_eq_owner = std::move( eq_owner );
 v_in_owner = std::move( in_owner );
 f_given_eq = ! v_eq_owner.empty();
 f_given_in = ! v_in_owner.empty();
 build();

 // whoever is listening has to find the abstract representation there; the
 // NBModification is issued by Block::deserialize()
 if( anyone_there() ) {
  generate_abstract_variables();
  generate_abstract_constraints();
  generate_objective();
  }

 Block::deserialize( group );

 }  // end( HQPDistBlock::deserialize )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::serialize( netCDF::NcGroup & group ) const
{
 Block::serialize( group );

 group.addDim( "NumRobots" , f_R );
 write_owner( group , "NumVar" , "ColumnOwner" , v_col_owner );
 if( f_given_eq && ( ! v_eq_owner.empty() ) )
  write_owner( group , "NumEqRows" , "EqRowOwner" , v_eq_owner );
 if( f_given_in && ( ! v_in_owner.empty() ) )
  write_owner( group , "NumIneqRows" , "IneqRowOwner" , v_in_owner );

 auto sub = group.addGroup( "HQPBlock" );
 f_central.serialize( sub );

 }  // end( HQPDistBlock::serialize )

/*--------------------------------------------------------------------------*/
/*-------------------- Methods for the abstract representation -------------*/
/*--------------------------------------------------------------------------*/

void HQPDistBlock::generate_abstract_variables( Configuration * stvv )
{
 if( AR & HasVar )  // the Variable are there already
  return;           // nothing to do

 for( auto b : v_Block )
  b->generate_abstract_variables();

 AR |= HasVar;

 }  // end( HQPDistBlock::generate_abstract_variables )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::generate_abstract_constraints( Configuration * stcc )
{
 if( AR & HasCns )  // the Constraint are there already
  return;           // nothing to do

 generate_abstract_variables();
 for( auto b : v_Block )
  b->generate_abstract_constraints();

 auto var = [ & ]( Index r , Index j ) {
  return( const_cast< ColVariable * >(
                   & get_robot( r )->get_x()[ v_pos[ r ][ j ] ] ) );
  };

 v_cons = std::vector< FRowConstraint >( v_copy.size() );
 for( Index k = 0 ; k < v_copy.size() ; ++k ) {
  const auto [ r , j ] = v_copy[ k ];
  LinearFunction::v_coeff_pair cf;
  cf.emplace_back( var( r , j ) , 1.0 );
  cf.emplace_back( var( v_col_owner[ j ] , j ) , -1.0 );
  v_cons[ k ].set_function( new LinearFunction( std::move( cf ) ) , eNoMod );
  v_cons[ k ].set_both( 0 , eNoMod );
  }
 add_static_constraint( v_cons , "consensus" );

 AR |= HasCns;

 }  // end( HQPDistBlock::generate_abstract_constraints )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::generate_objective( Configuration * objc )
{
 if( AR & HasObj )  // the Objective is there already
  return;           // nothing to do

 for( auto b : v_Block )
  b->generate_objective();

 f_obj.set_function( new LinearFunction() , eNoMod );
 f_obj.set_sense( Objective::eMin , eNoMod );
 set_objective( & f_obj , eNoMod );

 AR |= HasObj;

 }  // end( HQPDistBlock::generate_objective )

/*--------------------------------------------------------------------------*/
/*----------------------- Methods for handling the levels ------------------*/
/*--------------------------------------------------------------------------*/

void HQPDistBlock::reset_levels( bool relax_lower , ModParam issueMod )
{
 for( Index r = 0 ; r < f_R ; ++r )
  get_robot( r )->reset_levels( relax_lower , issueMod );
 f_level = 0;
 v_delta.clear();
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::next_level( double tol , ModParam issueMod )
{
 if( f_level + 1 >= get_num_levels() )
  throw( std::logic_error( "HQPDistBlock::next_level: the current level is "
                           "the last one" ) );
 if( ! ( AR & HasVar ) )
  throw( std::logic_error( "HQPDistBlock::next_level: the Variable are not "
                           "there" ) );

 // the copies take the values of the owners
 Vec_double x;
 get_x_values( x );
 double delta = 0;
 for( const auto & [ r , j ] : v_copy ) {
  auto & var = const_cast< ColVariable & >(
                            get_robot( r )->get_x()[ v_pos[ r ][ j ] ] );
  delta = std::max( delta , std::abs( var.get_value() - x[ j ] ) );
  var.set_value( x[ j ] );
  }
 v_delta.resize( f_level + 1 , 0 );
 v_delta[ f_level ] = delta;

 // the rows of the fixed levels are satisfied at the values of the owners
 for( Index r = 0 ; r < f_R ; ++r )
  if( v_cols[ r ].size() > v_nown[ r ] ) {
   auto rb = get_robot( r );
   for( Index q = 0 ; q < f_level ; ++q )
    if( rb->get_level_state( q ) == HQPBlock::kFixed )
     rb->refix_level( q , issueMod );
   }

 for( Index r = 0 ; r < f_R ; ++r )
  get_robot( r )->next_level( tol , issueMod );
 ++f_level;

 }  // end( HQPDistBlock::next_level )

/*--------------------------------------------------------------------------*/

double HQPDistBlock::level_slacks( Index p , Vec_double & v ,
                                   Vec_double & w ) const
{
 if( p >= get_num_levels() )
  throw( std::invalid_argument( "HQPDistBlock::level_slacks: level " +
                                std::to_string( p ) + " out of range" ) );
 Vec_double x;
 get_x_values( x );

 double val = 0;
 for( int k = 0 ; k < 2 ; ++k ) {
  const bool eq = ( k == 0 );
  const auto & M = eq ? f_central.get_A( p ) : f_central.get_C( p );
  const auto & rhs = eq ? f_central.get_b() : f_central.get_d();
  const auto & wgh = eq ? f_central.get_we() : f_central.get_wi();
  const Index first = eq ? f_central.get_eq_start( p )
                         : f_central.get_ineq_start( p );
  auto & s = eq ? v : w;
  s.resize( M.rows() );
  for( Eigen::Index h = 0 ; h < M.rows() ; ++h ) {
   double mx = 0;
   for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
    mx += it.value() * x[ it.col() ];
   s[ h ] = wgh[ first + h ] * ( mx - rhs[ first + h ] );
   if( ! eq )
    s[ h ] = std::max( 0.0 , s[ h ] );
   val += s[ h ] * s[ h ];
   }
  }

 return( val / 2 );

 }  // end( HQPDistBlock::level_slacks )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_center( const Vec_double & c , ModParam issueMod )
{
 if( ( ! c.empty() ) && ( c.size() != get_num_var() ) )
  throw( std::invalid_argument( "HQPDistBlock::set_center: wrong size" ) );

 for( Index r = 0 ; r < f_R ; ++r ) {
  Vec_double cr;
  if( ! c.empty() )
   for( auto j : v_cols[ r ] )
    cr.push_back( c[ j ] );
  get_robot( r )->set_center( cr , issueMod );
  }
 }  // end( HQPDistBlock::set_center )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::get_x_values( Vec_double & x ) const
{
 x.assign( get_num_var() , 0 );
 if( ! ( AR & HasVar ) )
  return;
 for( Index j = 0 ; j < x.size() ; ++j ) {
  const auto r = v_col_owner[ j ];
  x[ j ] = get_robot( r )->get_x()[ v_pos[ r ][ j ] ].get_value();
  }
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_x_values( const Vec_double & x )
{
 if( x.size() != get_num_var() )
  throw( std::invalid_argument( "HQPDistBlock::set_x_values: wrong size" ) );
 if( ! ( AR & HasVar ) )
  return;
 for( Index r = 0 ; r < f_R ; ++r ) {
  Vec_double xr;
  for( auto j : v_cols[ r ] )
   xr.push_back( x[ j ] );
  get_robot( r )->set_x_values( xr );
  }
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_slack_values( Index p , const Vec_double & v ,
                                     const Vec_double & w )
{
 if( ( v.size() != f_central.get_num_eq_rows( p ) ) ||
     ( w.size() != f_central.get_num_ineq_rows( p ) ) )
  throw( std::invalid_argument( "HQPDistBlock::set_slack_values: wrong "
                                "size" ) );
 if( ! ( AR & HasVar ) )
  return;

 std::vector< Vec_double > vr( f_R ) , wr( f_R );
 for( Index r = 0 ; r < f_R ; ++r ) {
  vr[ r ].resize( get_robot( r )->get_num_eq_rows( p ) );
  wr[ r ].resize( get_robot( r )->get_num_ineq_rows( p ) );
  }
 const auto e = f_central.get_eq_start( p );
 const auto i = f_central.get_ineq_start( p );
 for( Index h = 0 ; h < v.size() ; ++h )
  vr[ v_eq_owner[ e + h ] ][ v_eq_pos[ e + h ] ] = v[ h ];
 for( Index h = 0 ; h < w.size() ; ++h )
  wr[ v_in_owner[ i + h ] ][ v_in_pos[ i + h ] ] = w[ h ];
 for( Index r = 0 ; r < f_R ; ++r )
  get_robot( r )->set_slack_values( p , vr[ r ] , wr[ r ] );

 }  // end( HQPDistBlock::set_slack_values )

/*--------------------------------------------------------------------------*/
/*-------------------------- Methods for changing data ---------------------*/
/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_b( Index p , const Vec_double & b , ModParam issueMod )
{
 f_central.set_b( p , b );
 update_rows( true , p , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_d( Index p , const Vec_double & d , ModParam issueMod )
{
 f_central.set_d( p , d );
 update_rows( false , p , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_A( Index p , const SparseMatrix & A ,
                          ModParam issueMod )
{
 f_central.set_A( p , A );
 update_rows( true , p , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_C( Index p , const SparseMatrix & C ,
                          ModParam issueMod )
{
 f_central.set_C( p , C );
 update_rows( false , p , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_we( Index p , const Vec_double & we ,
                           ModParam issueMod )
{
 f_central.set_we( p , we );
 update_rows( true , p , issueMod );
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::set_wi( Index p , const Vec_double & wi ,
                           ModParam issueMod )
{
 f_central.set_wi( p , wi );
 update_rows( false , p , issueMod );
 }

/*--------------------------------------------------------------------------*/
/*----------------------- Methods for handling Solution --------------------*/
/*--------------------------------------------------------------------------*/

Solution * HQPDistBlock::get_Solution( Configuration * solc , bool emptys )
{
 auto sol = new ColVariableSolution;
 if( ! emptys )
  sol->read( this );
 return( sol );
 }

/*--------------------------------------------------------------------------*/
/*---------------------------- Methods for printing ------------------------*/
/*--------------------------------------------------------------------------*/

void HQPDistBlock::print( std::ostream & output , char vlvl ) const
{
 output << "HQPDistBlock with " << f_R << " sub-Blocks and "
        << v_copy.size() << " copies of variables" << std::endl;
 for( Index r = 0 ; r < f_R ; ++r ) {
  const auto rb = get_robot( r );
  Index me = 0 , mi = 0;
  for( Index p = 0 ; p < rb->get_num_levels() ; ++p ) {
   me += rb->get_num_eq_rows( p );
   mi += rb->get_num_ineq_rows( p );
   }
  output << " sub-Block " << r << ": " << v_nown[ r ] << " variables, "
         << v_cols[ r ].size() - v_nown[ r ] << " copies, " << me
         << " equality rows, " << mi << " inequality rows" << std::endl;
  }
 if( vlvl > 0 )
  f_central.print( output , vlvl );
 }  // end( HQPDistBlock::print )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/

void HQPDistBlock::guts_of_destructor( void )
{
 // the consensus rows point to the Variable of the sub-Blocks
 for( auto & row : v_cons )
  row.clear();
 f_obj.clear();

 reset_static_constraints();
 reset_objective();
 v_cons.clear();

 for( auto b : v_Block )
  delete b;
 v_Block.clear();

 AR = 0;
 }

/*--------------------------------------------------------------------------*/

void HQPDistBlock::build( void )
{
 const Index n = f_central.get_num_var();
 const Index P = f_central.get_num_levels();
 const Index ne = f_central.get_eq_start( P );
 const Index ni = f_central.get_ineq_start( P );

 // check the owners of the columns
 if( ! f_R )
  throw( std::invalid_argument( "HQPDistBlock::build: no owners" ) );
 if( v_col_owner.size() != n )
  throw( std::invalid_argument( "HQPDistBlock::build: wrong size of the "
                                "owners of the columns" ) );
 v_nown.assign( f_R , 0 );
 for( auto r : v_col_owner ) {
  if( r >= f_R )
   throw( std::invalid_argument( "HQPDistBlock::build: an owner of a "
                                 "column is out of range" ) );
  ++v_nown[ r ];
  }
 for( Index r = 0 ; r < f_R ; ++r )
  if( ! v_nown[ r ] )
   throw( std::invalid_argument( "HQPDistBlock::build: owner " +
                                 std::to_string( r ) + " has no column" ) );

 // the owners of the rows, chosen unless given, and the copies they ask
 v_pos.assign( f_R , std::vector< int >( n , -1 ) );
 for( Index j = 0 ; j < n ; ++j )
  v_pos[ v_col_owner[ j ] ][ j ] = 0;

 for( int k = 0 ; k < 2 ; ++k ) {
  const bool eq = ( k == 0 );
  auto & owner = eq ? v_eq_owner : v_in_owner;
  const bool given = eq ? f_given_eq : f_given_in;
  const Index m = eq ? ne : ni;
  if( given && ( owner.size() != m ) )
   throw( std::invalid_argument( std::string( "HQPDistBlock::build: wrong "
                                 "size of the owners of the " ) +
                                 ( eq ? "equality" : "inequality" ) +
                                 " rows" ) );
  if( ! given )
   owner.assign( m , 0 );
  for( Index p = 0 ; p < P ; ++p ) {
   const auto & M = eq ? f_central.get_A( p ) : f_central.get_C( p );
   const Index first = eq ? f_central.get_eq_start( p )
                          : f_central.get_ineq_start( p );
   for( Eigen::Index h = 0 ; h < M.rows() ; ++h ) {
    auto & r = owner[ first + h ];
    if( ! given )
     r = row_owner( M , h , v_col_owner , f_R );
    else {
     if( r >= f_R )
      throw( std::invalid_argument( "HQPDistBlock::build: an owner of a "
                                    "row is out of range" ) );
     bool has = ( M.row( h ).nonZeros() == 0 );
     for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
      if( v_col_owner[ it.col() ] == r )
       has = true;
     if( ! has )
      throw( std::invalid_argument( "HQPDistBlock::build: the owner of a "
                                    "row owns none of its columns" ) );
     }
    for( SparseMatrix::InnerIterator it( M , h ) ; it ; ++it )
     if( v_pos[ r ][ it.col() ] < 0 )
      v_pos[ r ][ it.col() ] = 0;
    }
   }
  }

 // the columns of each owner: its own ones, then its copies
 v_cols.assign( f_R , {} );
 v_copy.clear();
 Vec_Index mu( n , 0 );
 for( Index r = 0 ; r < f_R ; ++r ) {
  auto & cols = v_cols[ r ];
  for( Index j = 0 ; j < n ; ++j )
   if( v_col_owner[ j ] == r )
    cols.push_back( j );
  for( Index j = 0 ; j < n ; ++j )
   if( ( v_col_owner[ j ] != r ) && ( v_pos[ r ][ j ] >= 0 ) ) {
    cols.push_back( j );
    v_copy.emplace_back( r , j );
    ++mu[ j ];
    }
  for( Index k = 0 ; k < cols.size() ; ++k )
   v_pos[ r ][ cols[ k ] ] = k;
  }

 // the position of each row among the rows of its level of its owner
 for( int k = 0 ; k < 2 ; ++k ) {
  const bool eq = ( k == 0 );
  const auto & owner = eq ? v_eq_owner : v_in_owner;
  auto & pos = eq ? v_eq_pos : v_in_pos;
  pos.assign( owner.size() , 0 );
  for( Index p = 0 ; p < P ; ++p ) {
   Vec_Index cnt( f_R , 0 );
   const Index first = eq ? f_central.get_eq_start( p )
                          : f_central.get_ineq_start( p );
   const Index m = eq ? f_central.get_num_eq_rows( p )
                      : f_central.get_num_ineq_rows( p );
   for( Index i = first ; i < first + m ; ++i )
    pos[ i ] = cnt[ owner[ i ] ]++;
   }
  }

 // the sub-Blocks
 const auto & rwc = f_central.get_regularization_weights();
 for( Index r = 0 ; r < f_R ; ++r ) {
  std::vector< SparseMatrix > A( P ) , C( P );
  std::vector< Vec_double > b( P ) , d( P ) , we( P ) , wi( P );
  for( Index p = 0 ; p < P ; ++p ) {
   robot_rows( r , true , p , A[ p ] , b[ p ] , we[ p ] );
   robot_rows( r , false , p , C[ p ] , d[ p ] , wi[ p ] );
   }
  Vec_double rw;
  for( auto j : v_cols[ r ] )
   rw.push_back( ( rwc.empty() ? 1.0 : rwc[ j ] ) / ( 1 + mu[ j ] ) );

  auto rb = new HQPBlock( this );
  rb->load( v_cols[ r ].size() , std::move( A ) , std::move( b ) ,
            std::move( C ) , std::move( d ) , std::move( we ) ,
            std::move( wi ) , f_central.get_regularization() ,
            std::move( rw ) );
  v_Block.push_back( rb );
  }

 f_level = 0;
 v_delta.clear();

 }  // end( HQPDistBlock::build )

/*--------------------------------------------------------------------------*/

bool HQPDistBlock::robot_rows( Index r , bool eq , Index p ,
                               SparseMatrix & M , Vec_double & rhs ,
                               Vec_double & wgh ) const
{
 const auto & Mc = eq ? f_central.get_A( p ) : f_central.get_C( p );
 const auto & owner = eq ? v_eq_owner : v_in_owner;
 const auto & rc = eq ? f_central.get_b() : f_central.get_d();
 const auto & wc = eq ? f_central.get_we() : f_central.get_wi();
 const Index first = eq ? f_central.get_eq_start( p )
                        : f_central.get_ineq_start( p );

 std::vector< Eigen::Triplet< double > > trp;
 rhs.clear();
 wgh.clear();
 bool ok = true;
 for( Eigen::Index h = 0 ; h < Mc.rows() ; ++h ) {
  const Index i = first + h;
  if( owner[ i ] != r )
   continue;
  for( SparseMatrix::InnerIterator it( Mc , h ) ; it ; ++it ) {
   const int k = v_pos[ r ][ it.col() ];
   if( k < 0 )
    ok = false;
   else
    trp.emplace_back( rhs.size() , k , it.value() );
   }
  rhs.push_back( rc[ i ] );
  wgh.push_back( wc[ i ] );
  }
 M.resize( rhs.size() , v_cols[ r ].size() );
 M.setFromTriplets( trp.begin() , trp.end() );
 M.makeCompressed();
 return( ok );

 }  // end( HQPDistBlock::robot_rows )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::update_rows( bool eq , Index p , ModParam issueMod )
{
 std::vector< SparseMatrix > M( f_R );
 std::vector< Vec_double > rhs( f_R ) , wgh( f_R );
 bool ok = true;
 for( Index r = 0 ; r < f_R ; ++r )
  ok = robot_rows( r , eq , p , M[ r ] , rhs[ r ] , wgh[ r ] ) && ok;

 if( ! ok ) {  // a column that a sub-Block has not got: all anew
  // the NBModification comes first, so that a Solver that drops what it
  // has built on the old sub-Blocks (e.g., an HQPSolver its inner Solver)
  // finds them still there
  const bool there = anyone_there();
  if( there )
   add_Modification( std::make_shared< NBModification >( this ) );
  guts_of_destructor();
  build();
  if( there ) {
   generate_abstract_variables();
   generate_abstract_constraints();
   generate_objective();
   }
  return;
  }

 for( Index r = 0 ; r < f_R ; ++r ) {
  auto rb = get_robot( r );
  if( eq ) {
   rb->set_we( p , wgh[ r ] , issueMod );
   rb->set_A( p , M[ r ] , issueMod );
   rb->set_b( p , rhs[ r ] , issueMod );
   }
  else {
   rb->set_wi( p , wgh[ r ] , issueMod );
   rb->set_C( p , M[ r ] , issueMod );
   rb->set_d( p , rhs[ r ] , issueMod );
   }
  }
 }  // end( HQPDistBlock::update_rows )

/*--------------------------------------------------------------------------*/

void HQPDistBlock::regenerate( void )
{
 // whoever is listening has to find the abstract representation there
 if( anyone_there() ) {
  generate_abstract_variables();
  generate_abstract_constraints();
  generate_objective();
  add_Modification( std::make_shared< NBModification >( this ) );
  }
 }

/*--------------------------------------------------------------------------*/
/*---------------------- End File HQPDistBlock.cpp -------------------------*/
/*--------------------------------------------------------------------------*/
