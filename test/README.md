# Testing

MC/DC has a few testing suits to ensure operability during development and deployment. 
The goal of testing is to ensure that if they pass you have a full operable MC/DC install
that will produce correct results no matter the arc, OS, or mode.

Each tests directory has it's own specific documentation that goes into further detail about
the how what and why but currently our testing is in a few different catagories

## Unit

Unit testing is the process where you test the smallest functional unit of code.
It is intended to make sure that a hypothetical `add_one()` function will do exactly as it 
is documented and expected to do. Ideally development happens in a "testing driven way"
where in developers first write unit tests for functions then the functions them selves but
this is really how work is actually done. In MC/DC we expect all of our input_.py functions
to be unit tested to ensure that we cantingly support already generated input decks no matter the
version.

Ideally this will extend to actual transport functions but for now it is what it is.

## Verification

Verification tests are a forum of integration tests in which we compare results from MC/DC to known solutions 
produced either via an analytic solution (AZURV1) or via an over resolved benchmark solution.
We then look at the convergence rate of MC/DC as we increase the number of particles to see if it converges as expected (for analog Monte Carlo $\mathcal(O) = 1/\sqrt{N}$). If so we have "

## Regression

Regression tests are a forum of integration test that will call into MC/DC and run some kind of problem
and test MC/DC as a whole pice of software.
For regression tests specifically we compare old solutions (i.e. regressed versions of MC/DC) to what MC/DC
is currently producing with new functions and features.
As we can fix the random number seed we know that the particle tracks should be *exactly* reproduced.
We compare what MC/DC produces across operable modes (numba, python, gpu) to stored results in `.h5` files.
This is the most common test we use in PRs.

When we implement a new feature that alters the random number sequence thus changing the results in the `.h5` files we run our verification test suite to ensure everything is still working as expected. 

## Performance

Performance testing is handled in another associated directory

## Other Testing not currently implemented

Some other notable types of testing that exist out there are
* Validation
* Integration

# CI and DevOps Implementations
We try to automate our testing as much as possible.
Currently in public using github actions we allow for automated unit (Python only), and regression testing (in numba, Python, and MPI mode).
These two sets of tests will cover many bugs but not all. 
**Every test is written in blood.**

We have a github runner which will manually execute the verification tests (only if required and requested). 
This runner takes forever to run and it is recommended this is done on a cluster.

The GPU and Performance tests are ran externally and manually using resources in other directories.