/*--------------------------------------------------------------------------*/
/*-------------------------- File HQPDistBlock.h ---------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Header file for the *concrete* class HQPDistBlock, which implements the
 * Block concept [see Block.h] for a hierarchical (lexicographic) Quadratic
 * Program whose variables are split among sub-Blocks, e.g., the robots of a
 * multi-robot Model Predictive Control, so that it can be solved by
 * decomposition.
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
/*----------------------------- DEFINITIONS --------------------------------*/
/*--------------------------------------------------------------------------*/

#ifndef __HQPDistBlock
 #define __HQPDistBlock
                      /* self-identification: #endif at the end of the file */

/*--------------------------------------------------------------------------*/
/*------------------------------ INCLUDES ----------------------------------*/
/*--------------------------------------------------------------------------*/

#include "HQPBlock.h"

/*--------------------------------------------------------------------------*/
/*------------------------------ NAMESPACE ---------------------------------*/
/*--------------------------------------------------------------------------*/

/// namespace for the Structured Modeling System++ (SMS++)
namespace SMSpp_di_unipi_it
{
/*--------------------------------------------------------------------------*/
/*-------------------------------- CLASSES ---------------------------------*/
/*--------------------------------------------------------------------------*/
/** @defgroup HQPDistBlock_CLASSES Classes in HQPDistBlock.h
 *  @{ */

/*--------------------------------------------------------------------------*/
/*--------------------------- CLASS HQPDistBlock ---------------------------*/
/*--------------------------------------------------------------------------*/
/// a hierarchical QP split among sub-Blocks, one for each owner of columns
/** The HQPDistBlock class implements the Block concept [see Block.h] for
 * the hierarchical (lexicographic) Quadratic Program of an HQPBlock [see
 * HQPBlock.h], the centralized one, whose \f$ n \f$ variables are split
 * among \f$ R \f$ owners (e.g., the robots of a multi-robot Model Predictive
 * Control), so that the hierarchy can be solved by decomposition, level by
 * level, by an HQPSolver [see HQPSolver.h] whose inner Solver is, e.g., a
 * LagrangianDualSolver.
 *
 * The data is the centralized hierarchy and the owner \f$ o( j ) \in \{ 0 ,
 * \ldots , R - 1 \} \f$ of each column j. Each row (of any level, equality
 * or inequality) has an owner too: the owner of all its columns if they all
 * have the same (a local row), otherwise (a coupling row) one among those
 * of its columns, the one with most entries in the row and, between two
 * with as many, the one that makes each owner own about half of the pairs
 * it is in (r between r and s if \f$ ( s - r ) \bmod R \leq \lfloor R / 2
 * \rfloor \f$, the smallest one among more than two); the owners of the
 * rows can also be given. Owner r copies column j, with \f$ o( j ) \neq r
 * \f$, if j has an entry in a row that r owns; \f$ \mathcal{C}_r \f$ is the
 * set of these columns, and \f$ \mu_j \f$ the number of owners that copy j.
 *
 * Owner r is a sub-Block, an HQPBlock whose variables are those it owns
 * followed by those in \f$ \mathcal{C}_r \f$, both in increasing order, and
 * whose rows are those it owns, with the same levels as the centralized
 * one; each row and its slack is in one sub-Block only, so that its term in
 * the Objective is counted once, as in the centralized one. The
 * regularization of a column is split among its \f$ 1 + \mu_j \f$ copies,
 * each having weight \f$ r_j / ( 1 + \mu_j ) \f$ [see
 * HQPBlock::get_regularization_weights()], so that on the consensus, where
 * all the copies have the value of the owner, the sum of the Objectives of
 * the sub-Blocks is the Objective of the centralized hierarchy.
 *
 * The HQPDistBlock has no Variable and an empty Objective, and its static
 * Constraint are the consensus rows, an FRowConstraint with a
 * LinearFunction
 *
 * \f[
 *   x^{(r)}_j - x^{(o(j))}_j = 0 \qquad j \in \mathcal{C}_r \;,\;
 *   r = 0 , \ldots , R - 1 \;\;,
 * \f]
 *
 * for each copy, which a LagrangianDualSolver relaxes: its Lagrangian Dual
 * has one LagBFunction for each owner, whose Objective at each level is
 * strictly convex if \f$ \varepsilon > 0 \f$. With the consensus rows all in
 * the model (e.g., a :MILPSolver on the whole tree) the problem of each
 * level is exactly that of the centralized hierarchy.
 *
 * The levels are those of the sub-Blocks, moved together [see
 * HQPLevelBlock]: the \f$ x \f$ of the hierarchy is that of the owners, the
 * slacks of a row are those of the sub-Block that owns it. Moving from a
 * level to the next one first gives the copies the values of the owners,
 * since a Solver that relaxes the consensus rows leaves them only close to
 * each other, then fixes again the slacks of the levels already fixed at
 * these values [see HQPBlock::refix_level()], so that the \f$ x \f$ of the
 * owners is feasible for all of them, and finally moves each sub-Block to
 * the next level [see next_level()].
 *
 * The data can be changed with the methods of the class [see set_b() and
 * the others], which change those of the sub-Blocks by Modification; a
 * change of the pattern of a row that brings in a column its owner does not
 * have builds the sub-Blocks anew, with an NBModification. */

class HQPDistBlock : public HQPLevelBlock
{
/*----------------------- PUBLIC PART OF THE CLASS -------------------------*/

 public:

/*---------------------------- PUBLIC TYPES --------------------------------*/

 /// the row-major sparse matrix of the coefficients of a level
 using SparseMatrix = HQPBlock::SparseMatrix;

 using Vec_Index = std::vector< Index >;  ///< a vector of Index

/*------------------------------ CONSTRUCTOR -------------------------------*/
 /// constructor of HQPDistBlock, taking a pointer to the father Block
 /** Constructor of HQPDistBlock. It accepts a pointer to the father Block
  * (defaulting to nullptr, so that this can also be used as the void
  * constructor required by the Block factory). The HQPDistBlock is empty
  * until load() or deserialize() is called. */

 explicit HQPDistBlock( Block * father = nullptr )
  : HQPLevelBlock( father ) , f_R( 0 ) , f_given_eq( false ) ,
    f_given_in( false ) , f_level( 0 ) , AR( 0 ) {}

/*------------------------------ DESTRUCTOR --------------------------------*/
 /// destructor of HQPDistBlock: deletes the sub-Blocks

 virtual ~HQPDistBlock() { guts_of_destructor(); }

/*-------------------------- OTHER INITIALIZATIONS -------------------------*/
 /// loads the HQPDistBlock out of a centralized hierarchy and the owners
 /** Loads the HQPDistBlock out of the data of the centralized hierarchy in
  * \p central (which is copied, and whose abstract representation, if any,
  * is not used), the number \p R of owners and the owner of each column in
  * \p col_owner (of size central.get_num_var(), each entry smaller than \p
  * R, each owner having at least one column). The owners of the equality
  * and of the inequality rows, all the levels one after the other as in
  * HQPBlock::get_b(), can be given in \p eq_owner and \p in_owner (empty
  * meaning that they are chosen as described in the general notes), and
  * each of them must own a column of the row if the row has any. If the
  * abstract representation is there already it is constructed anew, and an
  * NBModification is issued. Exception is thrown if the sizes do not
  * agree. */

 void load( const HQPBlock & central , Index R , Vec_Index && col_owner ,
            Vec_Index && eq_owner = {} , Vec_Index && in_owner = {} );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// extends Block::deserialize( netCDF::NcGroup )
 /** Extends Block::deserialize( netCDF::NcGroup ) to the specific format of
  * an HQPDistBlock. Besides the mandatory "type" attribute of any :Block,
  * the group contains:
  *
  * - the dimension "NumRobots", the number \f$ R \f$ of owners;
  *
  * - the dimension "NumVar" and the variable "ColumnOwner" (netCDF::NcUint,
  *   indexed over "NumVar"), the owner of each column;
  *
  * - the dimension "NumEqRows" and the variable "EqRowOwner"
  *   (netCDF::NcUint, indexed over "NumEqRows"), the owner of each equality
  *   row, optional (chosen as described in the general notes if not there),
  *   and so "NumIneqRows" and "IneqRowOwner" for the inequality rows;
  *
  * - the group "HQPBlock", the centralized hierarchy in the format of
  *   HQPBlock [see HQPBlock::deserialize()], with the same "NumVar". */

 void deserialize( const netCDF::NcGroup & group ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// loads the HQPDistBlock out of an istream: not implemented

 void load( std::istream & input , char frmt = 0 ) override
 {
  throw( std::logic_error( "HQPDistBlock::load: no text format, use the "
                           "netCDF one" ) );
  }

/*---------------------- Methods for handling Variable ---------------------*/
 /// generates the ColVariable of the sub-Blocks; the HQPDistBlock has none

 void generate_abstract_variables( Configuration * stvv = nullptr )
  override;

/*--------------------- Methods for handling Constraint --------------------*/
 /// generates the consensus rows, after the Constraint of the sub-Blocks
 /** Generates the Constraint of the sub-Blocks and the static
  * FRowConstraint of the consensus rows (named "consensus"), one for each
  * owner r and each column j in \f$ \mathcal{C}_r \f$, in this order, each
  * with a LinearFunction whose active Variable are the copy \f$ x^{(r)}_j
  * \f$ (coefficient 1) and the \f$ x_j \f$ of the owner of j (coefficient
  * -1), and both sides 0. */

 void generate_abstract_constraints( Configuration * stcc = nullptr )
  override;

/*---------------------- Methods for handling Objective --------------------*/
 /// generates the Objective of the sub-Blocks and an empty one
 /** Generates the Objective of the sub-Blocks and that of the HQPDistBlock,
  * a minimization FRealObjective with a LinearFunction with no active
  * Variable. */

 void generate_objective( Configuration * objc = nullptr ) override;

/*-------------- Methods for checking the state of the HQPDistBlock --------*/
 /// returns true if any part of the abstract representation is there

 bool anyone_there( void ) const override
 {
  return( AR ? true : Block::anyone_there() );
  }

/*----------------------- Methods for reading the data ---------------------*/
 /// returns the centralized hierarchy, without abstract representation

 [[nodiscard]] const HQPBlock & get_central( void ) const {
  return( f_central );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number \f$ R \f$ of owners, i.e., of sub-Blocks

 [[nodiscard]] Index get_num_robots( void ) const { return( f_R ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the sub-Block (an HQPBlock) of owner r

 [[nodiscard]] HQPBlock * get_robot( Index r ) const {
  return( static_cast< HQPBlock * >( v_Block[ r ] ) );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the owner of each column

 [[nodiscard]] const Vec_Index & get_column_owner( void ) const {
  return( v_col_owner );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the owner of each equality row, as HQPBlock::get_b()

 [[nodiscard]] const Vec_Index & get_eq_owner( void ) const {
  return( v_eq_owner );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the owner of each inequality row, as get_eq_owner()

 [[nodiscard]] const Vec_Index & get_ineq_owner( void ) const {
  return( v_in_owner );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the columns of owner r: its own ones, then its copies
 /** Returns the column of the centralized hierarchy of each variable of
  * the sub-Block of owner r: those that r owns first, then those in \f$
  * \mathcal{C}_r \f$, both in increasing order. */

 [[nodiscard]] const Vec_Index & get_robot_columns( Index r ) const {
  return( v_cols[ r ] );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number of variables that owner r owns

 [[nodiscard]] Index get_num_own( Index r ) const { return( v_nown[ r ] ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the consensus rows

 [[nodiscard]] const std::vector< FRowConstraint > & get_consensus( void )
  const { return( v_cons ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the largest distance of a copy from its owner at level p
 /** Returns \f$ \max_{r,j} | x^{(r)}_j - x_{o(j)} | \f$ at the solution of
  * level p, i.e., when next_level() has moved the HQPDistBlock from level p
  * to level p + 1 (0 if it has not, or if there is no copy). */

 [[nodiscard]] double get_consensus_residual( Index p ) const {
  return( p < v_delta.size() ? v_delta[ p ] : 0 );
  }

/*--------------------- Methods for handling the levels --------------------*/
 /// returns the number \f$ n \f$ of variables of the centralized hierarchy

 [[nodiscard]] Index get_num_var( void ) const override {
  return( f_central.get_num_var() );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the number \f$ P \f$ of levels

 [[nodiscard]] Index get_num_levels( void ) const override {
  return( f_central.get_num_levels() );
  }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// returns the current level

 [[nodiscard]] Index get_level( void ) const { return( f_level ); }

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// brings all the sub-Blocks to the problem of level 0
 /** Calls HQPBlock::reset_levels() of each sub-Block. */

 void reset_levels( bool relax_lower = false ,
                    ModParam issueMod = eModBlck ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// moves all the sub-Blocks from level p to level p + 1
 /** Moves all the sub-Blocks from the current level p to level p + 1, where
  * p + 1 < get_num_levels() must hold:
  *
  * - the copies are given the values of their owners (with no
  *   Modification), the largest distance being recorded [see
  *   get_consensus_residual()];
  *
  * - the slacks of the levels before p that are fixed are fixed again at
  *   these values [see HQPBlock::refix_level()], which changes only those of
  *   the coupling rows, so that the \f$ x \f$ of the owners satisfies all
  *   the rows whose slacks are fixed whatever the distance of the copies
  *   from the owners was;
  *
  * - each sub-Block is moved to level p + 1 [see HQPBlock::next_level()],
  *   the slacks of level p being fixed at these values, with \p tol.
  *
  * The changes are issued as Modification according to \p issueMod. */

 void next_level( double tol = 0 , ModParam issueMod = eModBlck ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// computes the slacks of level p at the current \f$ x \f$ of the owners
 /** Writes in \p v and \p w the slacks of level p of the centralized
  * hierarchy at the current \f$ x \f$ of the owners [see get_x_values()],
  * and returns the value \f$ \|v\|^2 / 2 + \|w\|^2 / 2 \f$ of level p
  * there [see HQPBlock::level_slacks()]. */

 double level_slacks( Index p , Vec_double & v ,
                      Vec_double & w ) const override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the center of the regularization of all the sub-Blocks
 /** Gives each variable of each sub-Block the entry of \p c (of size \f$ n
  * \f$, or 0 meaning \f$ c = 0 \f$) of its column, copies included. */

 void set_center( const Vec_double & c ,
                  ModParam issueMod = eModBlck ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// writes in \p x the current values of the variables of the owners

 void get_x_values( Vec_double & x ) const override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// writes \p x in the variables of the owners and in all the copies

 void set_x_values( const Vec_double & x ) override;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// writes the slacks of level p in those of the sub-Blocks that own them

 void set_slack_values( Index p , const Vec_double & v ,
                        const Vec_double & w ) override;

/*-------------------------- Methods for changing data ---------------------*/
 /// changes \f$ b_p \f$ [see HQPBlock::set_b()]

 void set_b( Index p , const Vec_double & b , ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes \f$ d_p \f$ [see HQPBlock::set_d()]

 void set_d( Index p , const Vec_double & d , ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes \f$ A_p \f$ [see HQPBlock::set_A()]
 /** Changes the matrix \f$ A_p \f$ of level p into \p A, which must have
  * the same size. If each row of level p has its entries in columns of its
  * sub-Block, the sub-Blocks change by Modification [see
  * HQPBlock::set_A()]; otherwise the HQPDistBlock is loaded anew [see
  * load()], the owners of the rows being chosen anew unless they were
  * given, and an NBModification is issued before the old sub-Blocks are
  * deleted, so that a Solver can drop what it has built on them while they
  * are there (as HQPSolver does with its inner Solvers). */

 void set_A( Index p , const SparseMatrix & A ,
             ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes \f$ C_p \f$, as set_A() does for \f$ A_p \f$

 void set_C( Index p , const SparseMatrix & C ,
             ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the diagonal of \f$ W^e_p \f$ [see HQPBlock::set_we()]

 void set_we( Index p , const Vec_double & we ,
              ModParam issueMod = eModBlck );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// changes the diagonal of \f$ W^i_p \f$ [see HQPBlock::set_wi()]

 void set_wi( Index p , const Vec_double & wi ,
              ModParam issueMod = eModBlck );

/*----------------------- Methods for handling Solution --------------------*/
 /// returns a ColVariableSolution of the ColVariable of the sub-Blocks
 /** Returns a ColVariableSolution of all the ColVariable of the sub-Blocks,
  * read from them if ! \p emptys; \p solc is ignored. */

 Solution * get_Solution( Configuration * solc = nullptr ,
                          bool emptys = true ) override;

/*---------------------------- Methods for printing ------------------------*/
 /// prints the HQPDistBlock on an ostream with the given verbosity
 /** Prints the sizes of the HQPDistBlock and of each sub-Block and, if \p
  * vlvl > 0, also the centralized hierarchy [see HQPBlock::print()]. */

 void print( std::ostream & output , char vlvl = 0 ) const override;

/*--------------------------- Methods for saving ---------------------------*/
 /// extends Block::serialize( netCDF::NcGroup )
 /** Extends Block::serialize( netCDF::NcGroup ) to the specific format of
  * an HQPDistBlock [see deserialize()]; the owners of the rows are written
  * only if they were given. */

 void serialize( netCDF::NcGroup & group ) const override;

 using Block::serialize;  // the other versions, e.g., to a file

/*-------------------- PROTECTED PART OF THE CLASS -------------------------*/

 protected:

/*-------------------------- PROTECTED METHODS -----------------------------*/

 /// deletes the sub-Blocks and the consensus rows, keeping the data

 void guts_of_destructor( void );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// builds the sub-Blocks out of the data and of the owners
 /** Checks the owners, chooses those of the rows not given, finds the
  * copies, and builds the sub-Blocks (without their abstract
  * representation), after having deleted the old ones, if any. */

 void build( void );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// the rows of level p of one kind of owner r, in its columns
 /** Writes in \p M, \p rhs and \p wgh the matrix, the right-hand sides and
  * the weights of the rows of level p, the equality ones if \p eq and the
  * inequality ones otherwise, that owner r owns, in the columns of its
  * sub-Block; returns false if one of them has an entry in a column that
  * is not of the sub-Block. */

 bool robot_rows( Index r , bool eq , Index p , SparseMatrix & M ,
                  Vec_double & rhs , Vec_double & wgh ) const;

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// brings the rows of level p of one kind of all the sub-Blocks to the data
 /** Brings the rows of level p, the equality ones if \p eq, of each
  * sub-Block to those of the centralized hierarchy, by the methods of
  * HQPBlock; if a row has an entry in a column that its sub-Block does not
  * have, the HQPDistBlock is loaded anew instead. */

 void update_rows( bool eq , Index p , ModParam issueMod );

/*- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 /// regenerates the abstract representation if there was one
 /** If someone is listening, generates the abstract representation of the
  * new sub-Blocks and the consensus rows, and issues an NBModification. */

 void regenerate( void );

/*--------------------------- PROTECTED FIELDS -----------------------------*/

 HQPBlock f_central;         ///< the centralized hierarchy (data only)

 Index f_R;                  ///< number of owners (sub-Blocks)

 Vec_Index v_col_owner;      ///< owner of each column
 Vec_Index v_eq_owner;       ///< owner of each equality row
 Vec_Index v_in_owner;       ///< owner of each inequality row
 bool f_given_eq;            ///< the owners of the equality rows were given
 bool f_given_in;            ///< those of the inequality rows were given

 std::vector< Vec_Index > v_cols;   ///< the columns of each owner
 Vec_Index v_nown;                  ///< the number of own columns of each

 /// for each owner, the position of each column in its sub-Block
 /** v_pos[ r ][ j ] is the index of column j among the variables of the
  * sub-Block of owner r, or -1 if it has not got it. */
 std::vector< std::vector< int > > v_pos;

 /// the position of each row among the rows of its level of its owner
 Vec_Index v_eq_pos;
 Vec_Index v_in_pos;                ///< as v_eq_pos, the inequality rows

 /// for each consensus row, the owner r and the column j of the copy
 std::vector< std::pair< Index , Index > > v_copy;

 std::vector< FRowConstraint > v_cons;  ///< the consensus rows

 FRealObjective f_obj;       ///< the empty Objective

 Index f_level;              ///< the current level
 Vec_double v_delta;         ///< consensus residual of each level left

 unsigned char AR;           ///< bit-wise coded: what abstract is there

 static constexpr unsigned char HasVar = 1;
 ///< first bit of AR == 1 if the Variable have been constructed
 static constexpr unsigned char HasObj = 2;
 ///< second bit of AR == 1 if the Objective has been constructed
 static constexpr unsigned char HasCns = 4;
 ///< third bit of AR == 1 if the Constraint have been constructed

/*--------------------- PRIVATE PART OF THE CLASS --------------------------*/

 private:

/*-------------------------- PRIVATE METHODS -------------------------------*/

 SMSpp_insert_in_factory_h;  // insert HQPDistBlock in the Block factory

/*--------------------------------------------------------------------------*/

 };  // end( class( HQPDistBlock ) )

/** @} end( group( HQPDistBlock_CLASSES ) ) */

/*--------------------------------------------------------------------------*/

 }  // end( namespace SMSpp_di_unipi_it )

/*--------------------------------------------------------------------------*/

#endif  /* HQPDistBlock.h included */

/*--------------------------------------------------------------------------*/
/*------------------------ End File HQPDistBlock.h -------------------------*/
/*--------------------------------------------------------------------------*/
