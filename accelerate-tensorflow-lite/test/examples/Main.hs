module Main where

import Control.Monad (replicateM)
import System.Random (randomRIO)

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Interpreter as I
import qualified Data.Array.Accelerate.TensorFlow as TFCPU
import qualified Data.Array.Accelerate.TensorFlow.Lite as TPU


-- This is matrix-matrix multiplication. We represent a matrix as a vector of
-- rows, i.e. the inner dimension of a Matrix is a row.
matmat :: A.Acc (A.Matrix Float) -> A.Acc (A.Matrix Float) -> A.Acc (A.Matrix Float)
matmat a b =
  let A.I2 k m = A.shape a
      A.I2 _ n = A.shape b
  in A.sum $
       A.generate (A.I3 k n m) $ \(A.I3 i j p) ->
         a A.! A.I2 i p * b A.! A.I2 p j

genmatrix :: A.DIM2 -> IO (A.Matrix Float)
genmatrix dim@(A.Z A.:. n A.:. m) = A.fromList dim <$> replicateM (n * m) (randomRIO (0, 10))

main :: IO ()
main = do
  -- Inputs
  let dimA = A.Z A.:. 3 A.:. 2
      dimB = A.Z A.:. 2 A.:. 4
      dimC = A.Z A.:. 3 A.:. 4  -- result dimension
  let a1 = A.fromList dimA
             [1, 2
             ,3, 4
             ,5, 6]
      b1 = A.fromList dimB
             [1, 0, 1, 0
             ,0, 1, 0, 1]

  -- representative sample input
  samples <- replicateM 10 $ do
    a <- genmatrix dimA
    b <- genmatrix dimB
    return (a TPU.:-> b TPU.:-> TPU.Result dimC)

  -- First let's try it in the accelerate interpreter
  putStrLn "## Running in the interpreter"
  print $ I.runN matmat a1 b1

  -- Then run it on the CPU using TensorFlow
  putStrLn "## Running on TensorFlow native CPU"
  print $ TFCPU.runN matmat a1 b1

  -- Then let's run it on a TPU, the easy way
  putStrLn "## Running on TPU, easy"
  do let model = TPU.compile matmat samples
     print $ TPU.execute model a1 b1

  -- And then the hard way, which scales better to multiple model executions
  putStrLn "## Running on TPU, better"
  do TPU.withConverterPy $ \converter -> do
       TPU.withDeviceContext $ do
         let model = TPU.compileWith converter matmat samples
         print $ TPU.execute model a1 b1
