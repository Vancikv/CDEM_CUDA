#include "aux_functions.h"

double load_function(FLOAT_TYPE tau)
{
	if (tau < 1.0)
	{
		return coef1*(tau*tau*tau) + coef2 * (tau*tau);
	}
	else
	{
		return 1.0;
	}
}

