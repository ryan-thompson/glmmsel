## Test environments
* ubuntu 24.04, R 4.5.0
* win-builder (release and devel)

## R CMD check results
0 errors | 0 warnings | 1 note

> checking compilation flags used ... NOTE
  Compilation used the following non-portable flag(s):
    ‘-mno-omit-leaf-frame-pointer’

* This note only appears on the Ubuntu test environment and is due to use of Rcpp.
