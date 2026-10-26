/*--------------------------------------------------------------------------*/
/*-------------------------- File HQPSolver.cpp ----------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Implementation of the HQPSolver class, a Solver for the hierarchical
 * (lexicographic) Quadratic Program of an HQPBlock.
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

#include <iomanip>

#include "HQPSolver.h"

/*--------------------------------------------------------------------------*/
/*------------------------- NAMESPACE AND USING ----------------------------*/
/*--------------------------------------------------------------------------*/

using namespace SMSpp_di_unipi_it;

/*--------------------------------------------------------------------------*/
/*----------------------------- STATIC MEMBERS -----------------------------*/
/*--------------------------------------------------------------------------*/

// register HQPSolver in the Solver factory

SMSpp_insert_in_factory_cpp_0( HQPSolver );

/*--------------------------------------------------------------------------*/
/*---------------------- CONSTRUCTOR AND DESTRUCTOR ------------------------*/
/*--------------------------------------------------------------------------*/

HQPSolver::~HQPSolver()
{
 drop_inner();
 delete f_bsc;
 }

/*--------------------------------------------------------------------------*/
/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
/*--------------------------------------------------------------------------*/

void HQPSolver::set_Block( Block * block )
{
 if( block == f_Block )
  return;

 drop_inner();
 Solver::set_Block( block );
 v_stat.clear();
 v_xl.clear();
 v_vl.clear();
 v_wl.clear();
 v_val.clear();
 f_status = kUnEval;

 if( f_Block && f_bsc )
  create_inner();
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::set_par( idx_type par , int value )
{
 switch( par ) {
  case( int_HQP_Prox ):
   f_prox = value ? 1 : 0;
   break;
  case( int_HQP_RelaxLower ):
   f_relax = value ? 1 : 0;
   break;
  case( intLogVerb ):
   f_verb = value;
   break;
  default:
   Solver::set_par( par , value );
  }
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::set_par( idx_type par , double value )
{
 if( par != dbl_HQP_SlackTol ) {
  Solver::set_par( par , value );
  return;
  }
 if( ! ( value >= 0 ) )
  throw( std::invalid_argument( "HQPSolver::set_par: negative "
                                "dbl_HQP_SlackTol" ) );
 f_tol = value;
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::set_par( idx_type par , std::string && value )
{
 if( par != str_HQP_BSCfg ) {
  Solver::set_par( par , std::move( value ) );
  return;
  }

 BlockSolverConfig * cfg = nullptr;
 if( ! value.empty() ) {
  auto c = Configuration::deserialize( value );
  cfg = dynamic_cast< BlockSolverConfig * >( c );
  if( ! cfg ) {
   delete c;
   throw( std::invalid_argument( "HQPSolver::set_par: unable to create a "
                                 "BlockSolverConfig out of " + value ) );
   }
  }

 drop_inner();
 delete f_bsc;
 f_bsc = cfg;
 f_bsc_name = std::move( value );

 if( f_Block && f_bsc )
  create_inner();
 }

/*--------------------------------------------------------------------------*/
/*--------------------- METHODS FOR SOLVING THE MODEL ----------------------*/
/*--------------------------------------------------------------------------*/

int HQPSolver::compute( bool changedvars )
{
 lock();

 v_stat.clear();
 v_xl.clear();
 v_vl.clear();
 v_wl.clear();
 v_val.clear();

 auto hb = dynamic_cast< HQPBlock * >( f_Block );
 if( ( ! hb ) || ( ! f_inner ) || ( ! hb->get_num_levels() ) ) {
  unlock();
  return( f_status = kError );
  }

 // the Block is lock()-ed only while it is changed, since the inner Solver
 // has to read_lock() it in its compute()
 const bool owned = hb->is_owned_by( f_id );
 auto lock_block = [ & ]() {
  if( ( ! owned ) && ( ! hb->lock( f_id ) ) )
   throw( std::runtime_error( "HQPSolver::compute: unable to lock the "
                              "Block" ) );
  };
 auto unlock_block = [ & ]() {
  if( ! owned )
   hb->unlock( f_id );
  };

 const int verb = f_verb;
 const Index P = hb->get_num_levels();
 const Index n = hb->get_num_var();

 lock_block();
 hb->reset_levels( f_relax );
 hb->set_center( {} );
 unlock_block();

 f_status = kOK;
 for( Index p = 0 ; p < P ; ++p ) {
  const int st = f_inner->compute();
  v_stat.push_back( st );
  if( ( st < kOK ) || ( st >= kError ) || ( ! f_inner->has_var_solution() ) )
  {
   f_status = ( ( st >= kOK ) && ( st < kError ) ) ? kError : st;
   if( f_log && verb )
    *f_log << "HQPSolver: level " << p << " stopped with status " << st
           << std::endl;
   break;
   }
  if( ( st != kOK ) && ( f_status == kOK ) )
   f_status = st;

  lock_block();
  f_inner->get_var_solution();

  Vec_double x( n );
  for( Index j = 0 ; j < n ; ++j )
   x[ j ] = hb->get_x()[ j ].get_value();
  Vec_double v , w;
  v_val.push_back( hb->level_slacks( p , v , w ) );
  v_vl.push_back( std::move( v ) );
  v_wl.push_back( std::move( w ) );

  if( f_log && verb )
   *f_log << "HQPSolver: level " << std::setw( 3 ) << p << " status "
          << std::setw( 3 ) << st << " value " << std::setprecision( 12 )
          << v_val.back() << std::endl;

  if( p + 1 < P ) {
   if( f_prox )
    hb->set_center( x );
   hb->next_level( f_tol );
   }
  v_xl.push_back( std::move( x ) );
  unlock_block();
  }

 unlock();
 return( f_status );

 }  // end( HQPSolver::compute )

/*--------------------------------------------------------------------------*/
/*---------------------- METHODS FOR READING RESULTS -----------------------*/
/*--------------------------------------------------------------------------*/

void HQPSolver::get_var_solution( Configuration * solc )
{
 auto hb = dynamic_cast< HQPBlock * >( f_Block );
 if( ( ! hb ) || v_xl.empty() )
  return;

 // the ColVariable are written as they are, but a fixed slack only takes
 // the value it is fixed at [see ColVariable::set_value()]
 auto & x = const_cast< std::vector< ColVariable > & >( hb->get_x() );
 auto & v = const_cast< std::vector< ColVariable > & >( hb->get_v() );
 auto & w = const_cast< std::vector< ColVariable > & >( hb->get_w() );
 for( Index j = 0 ; j < x.size() ; ++j )
  x[ j ].set_value( v_xl.back()[ j ] );
 for( Index p = 0 ; p < v_xl.size() ; ++p ) {
  for( Index h = 0 ; h < v_vl[ p ].size() ; ++h )
   v[ hb->get_eq_start( p ) + h ].set_value( v_vl[ p ][ h ] );
  for( Index h = 0 ; h < v_wl[ p ].size() ; ++h )
   w[ hb->get_ineq_start( p ) + h ].set_value( v_wl[ p ][ h ] );
  }
 }  // end( HQPSolver::get_var_solution )

/*--------------------------------------------------------------------------*/
/*--------------------------- PROTECTED METHODS ----------------------------*/
/*--------------------------------------------------------------------------*/
// the first Solver of the BlockSolverConfig that is in the factory becomes
// the inner Solver, with its ComputeConfig (if any)

void HQPSolver::create_inner( void )
{
 const auto & names = f_bsc->get_SolverNames();
 const auto & cfgs = f_bsc->get_SolverConfigs();
 for( Index i = 0 ; i < names.size() ; ++i ) {
  if( ! Solver::has_Solver( names[ i ] ) )
   continue;
  f_inner = Solver::new_Solver( names[ i ] );
  if( ( i < cfgs.size() ) && cfgs[ i ] )
   f_inner->set_ComputeConfig( cfgs[ i ] );
  f_inner->set_Block( f_Block );
  return;
  }
 }

/*--------------------------------------------------------------------------*/

void HQPSolver::drop_inner( void )
{
 if( ! f_inner )
  return;
 f_inner->set_Block( nullptr );
 delete f_inner;
 f_inner = nullptr;
 }

/*--------------------------------------------------------------------------*/
/*------------------------ End File HQPSolver.cpp --------------------------*/
/*--------------------------------------------------------------------------*/
