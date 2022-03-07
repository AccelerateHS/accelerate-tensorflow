{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeOperators #-}
module Main where

import Prelude as P

import Control.Monad (unless)

import Test.Tasty
import Test.Tasty.HUnit

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Sugar.Shape
import Data.Array.Accelerate.TensorFlow.Lite as TPU

main :: IO ()
main = do
  defaultMain tests

tests :: TestTree
tests = testGroup "Tests" [unitTests]

unitTests :: TestTree
unitTests = testGroup "Unit tests"
  [ mapTests
  , zipWithTests
  , foldTests
  ]

asAccelerateArray xs = fromList (Z :. P.length xs) xs

mapTests :: TestTree
mapTests = testGroup "Map Tests"
  [ testCase "Map +1 over single Float" $
      map' (+1) ([0..] :: [Float]) 5 @?= fromList (Z :. 5 :. 5) [2]
  ]
  where
    map' f xs len =
      let
        shape = Z :. len :. len
        xs' = fromList shape xs
        reprData = [xs' :-> Result shape]
        model = TPU.compile (A.map (+1)) reprData
      in TPU.execute model xs'

zipWithTests :: TestTree
zipWithTests = testGroup "ZipWith Tests"
  [ testCase "ZipWith (+) [0] [0]" $
      zipWith' (+) zeroList [0..] (Z :. 1) @?=~ [0]
  , testCase "ZipWith (+) [0] [1] " $
      zipWith' (+) zeroList [1..] (Z :. 1) @?=~ [1]
  , testCase "ZipWith (+) [1,1] [1,2] " $
      zipWith' (+) oneList  [1..] (Z :. 2) @?=~ [2, 3]
  , testCase "ZipWith (+) [1,2] [1,1] " $
      zipWith' (+) [1..] oneList (Z :. 2) @?=~ [2, 3]
  , testCase "ZipWith (+) [1,1..] [1,2..100]" $
      zipWith' (+) oneList [1..] (Z :. 100) @?=~ [2, 3..101]
  , testCase "ZipWith (+) [[1]] [[1]]" $
      zipWith' (+) oneList [1..] (Z :. 1 :. 1) @?=~ [2]
  , testCase "ZipWith (*) [0] [1]" $
      zipWith' (*) zeroList [1..] (Z :. 1) @?=~ [0]
  , testCase "ZipWith (*) [1, 1, 1] [1, 2, 3]" $
      zipWith' (*) oneList [1..] (Z :. 3) @?=~ [1, 2, 3]
  ]
  where
    zeroList, oneList :: [Float]
    zeroList = [0, 0..]
    oneList  = [1, 1..]

    zipWith' f xs' ys' shape =
      let
        xs = fromList shape xs'
        ys = fromList shape ys'
        reprData = [xs :-> ys :-> Result shape]
        model = TPU.compile (A.zipWith f) reprData
      in toList $ TPU.execute model xs ys

foldTests :: TestTree
foldTests = testGroup "Fold Tests"
  [ testCase "Fold (+) [0]" $
      fold' (+) ascList (Z :. 1) @?=~ [0]
  , testCase "Fold (+) [0, 1, 2, 3, 4, 5]" $
      fold' (+) ascList (Z :. 6) @?=~ [15]
  , testCase "Fold (*) [0]" $
      fold' (*) ascList (Z :. 1) @?=~ [0]
  , testCase "Fold (*) [0, 1, 2, 3, 4, 5]" $
      fold' (*) ascList (Z :. 6) @?=~ [0]
  , testCase "Fold (+) [[0]]" $
      fold' (+) ascList (Z :. 1 :. 1) @?=~ [0]
  , testCase "Fold (+) [[0, 1, 2], [3, 4, 5]]" $
      fold' (+) ascList (Z :. 2 :. 3) @?=~ [3, 12]
  ]
  where
    ascList :: [Float]
    ascList = [0..]

    fold' :: Shape sh => (Exp Float -> Exp Float -> Exp Float) -> [Float] -> (sh :. Int) -> [Float]
    fold' f xs' shape =
      let
        stripShape :: (sh :. Int) -> sh
        stripShape (x :. _) = x

        xs = fromList shape xs'
        reprData = [xs :-> Result $ stripShape shape]
        model = TPU.compile (A.fold f 0) reprData
      in toList $ TPU.execute model xs

-- The TPU has _much_ lower precision than an actual float! As such, we need
-- something better to be able to assert the correctness of the TPU results
-- than (==). Here is my best attempt as of now
assertCloseEnough :: (P.Ord a, P.Floating a, Show a) => String -> [a] -> [a] -> [a] -> Assertion
assertCloseEnough preface expected actual precision =
    unless (closeEnough precision actual expected) (assertFailure msg)
  where
    msg =
          (if P.null preface then "" else preface P.++ "\n")
      P.++  "expected: " P.++ show expected P.++ "\n but got: " P.++ show actual
    closeEnough precision actual expected =
      (P.&&) (P.length precision P.== P.length actual P.&& P.length actual P.== P.length expected) $
        P.and $ P.zipWith3 (\e a p -> e - p P.<= a P.&& a P.<= e + p) expected precision actual

(@?=~) :: (P.Ord a, P.Floating a, Show a) => [a] -> [a] -> Assertion
actual @?=~ expected = assertCloseEnough "" expected actual ((*0.05) <$> expected)

