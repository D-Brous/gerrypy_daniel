<><><><><> Set-up Instructions <><><><><>
1. Make sure you have virtualenv installed. If you don't, install with:
<pip install virtualenv>
2. <virtualenv -p python3.9.6 venv>
3. <source venv/bin/activate>
4. <pip install -r requirements.txt>
5. Activate gurobi license

<><><><><> Terminology <><><><><><><>

- *ix* := index
- *id* := identification number
- *cgu* := Census Geographical Unit. In this codebase, cgu refers to a particular unit of geography on a map, where that unit could be of several different granularity levels. The granularity levels implemented here, from coarsest to finest, are counties, tracts, block groups, and blocks.
- node *capacity* := Total number of districts that a node can split into.

<><><><><> How To Use <><><><><>

To run tests:
- python -m unittest
