module Main where

import Prelude as P
import Data.Array.Accelerate as A
import Data.Array.Accelerate.TensorFlow as TPU

import Data.Array.Accelerate.Sugar.Shape

zipWithTest :: IO ()
zipWithTest = do
    let res = TPU.runN (A.zipWith (+)) intShapes outShape xs ys
    putStrLn "Is zipWith (+) correct?"
    print $ res
  where
    base = [0..]
    intShapes = shapeToList <$> [shape, shape]
    outShape  = shapeToList <$> [shape]
    shape = Z :. size :. size
    xs = fromList shape (                        base :: [Float])
    ys = fromList shape (P.map (\x -> x * 2 + 1) base :: [Float])
    size = 5

main :: IO ()
main = do
  _ <- zipWithTest
  putStrLn "Nothing"
