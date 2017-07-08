/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */

#ifndef FIRSTORDERBOUNDCONSTRAINTSCOSTFUNCTION_H
#define FIRSTORDERBOUNDCONSTRAINTSCOSTFUNCTION_H
#include <shogun/lib/config.h>
#include <shogun/optimization/FirstOrderCostFunction.h>
namespace shogun
{

/** @brief The first order cost function base class with bound constrains.
 *
 * This class gives the interface used in
 * a first-order gradient-based bound constrained minimizer
 *
 */
class FirstOrderBoundConstraintsCostFunction: public FirstOrderCostFunction
{
public:
	virtual ~FirstOrderBoundConstraintsCostFunction() {};

	/** Get the lower bound of variables 
	 * 
	 * Usually the length of the bound should equal to the length of target variables.
	 *
	 * If the length of the bound is 1,
	 * the bound constrain is applied to all target variables.
	 *
	 * @return the lower bound
	 */
	virtual SGVector<float64_t> get_lower_bound()=0;

	/** Get the upper bound of variables 
	 *
	 * Usually the length of the bound should equal to the length of target variables.
	 *
	 * If the length of the bound is 1,
	 * the bound constrain is applied to all target variables.
	 *
	 * @return the upper bound
	 */
	virtual SGVector<float64_t> get_upper_bound()=0;
};

}

#endif