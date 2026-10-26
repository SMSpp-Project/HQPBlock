/*--------------------------------------------------------------------------*/
/*--------------------------- File HQPBlock.cppm ---------------------------*/
/*--------------------------------------------------------------------------*/
/** @file
 * Interface of the C++20 module smspp.hqpblock, which exports every name that
 * the headers of HQPBlock declare at namespace scope, together with the
 * modules it depends on; a translation unit can then write
 *
 *     import smspp.hqpblock;
 *
 * in place of the #include of the headers. The macros (e.g., those of the
 * factory in SMSFactory.h) are not exported by a module, so a file that uses
 * them still includes the header that defines them.
 *
 * This file is written by SMS++/cmake/smspp_module_exports.py, which reads
 * the names from the AST of the headers: it is not to be edited by hand.
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
/*------------------------- GLOBAL MODULE FRAGMENT -------------------------*/
/*--------------------------------------------------------------------------*/

module;

#include "HQPSolver.h"

/*--------------------------------------------------------------------------*/
/*----------------------------- MODULE PURVIEW -----------------------------*/
/*--------------------------------------------------------------------------*/

export module smspp.hqpblock;

export import smspp;

export namespace SMSpp_di_unipi_it {

 // HQPBlock.h
 using SMSpp_di_unipi_it::HQPBlock;

 // HQPSolver.h
 using SMSpp_di_unipi_it::HQPSolver;
}

/*--------------------------------------------------------------------------*/
/*------------------------- End File HQPBlock.cppm -------------------------*/
/*--------------------------------------------------------------------------*/
