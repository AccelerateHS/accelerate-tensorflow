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
-- import Data.Array.Accelerate.TensorFlow.Lite.Sugar.Shapes as TPUS

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
  , mathTests
  ]

asAccelerateArray xs = fromList (Z :. P.length xs) xs

mapTests :: TestTree
mapTests = testGroup "Map Tests"
  [ testCase "Map +1 over single Float" $
      map' (1+) ([0..] :: [Float]) (Z :. 1) @?= [1]
  , testCase "Sin [0, PI/6, PI/4, PI/3]" $
      map' sin sinList (Z :. 4) @?=~ [sqrt 0.0 / 2.0, sqrt 1.0 / 2.0, sqrt 2.0 / 2.0, sqrt 3.0 / 2.0]
  , testCase "Sin [[0, PI/6], [PI/4, PI/3]]" $
      map' sin sinList (Z :. 2 :. 2) @?=~ [sqrt 0.0 / 2.0, sqrt 1.0 / 2.0, sqrt 2.0 / 2.0, sqrt 3.0 / 2.0]
  ]
  where
    sinList :: [Float]
    sinList = 0 : ((pi /) <$> [6, 4, 3, 2])

    map' f xs' shape =
      let
        xs = fromList shape xs'
        reprData = [xs :-> Result shape]
        model = TPU.compile (A.map f) reprData
      in toList $ TPU.execute model xs

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

mathTests :: TestTree
mathTests = testGroup "Math Function Tests (using map)"
  [ testGroup "Sin" $
      [ testCase "Sin [0]" $
          map' sin sinList (Z :. 1) @?=~ [0]
      , testCase "Sin [0, PI/6, PI/4, PI/3, PI/2]" $
          map' sin sinList (Z :. 5) @?=~ [sqrt 0.0 / 2.0, sqrt 1.0 / 2.0, sqrt 2.0 / 2.0, sqrt 3.0 / 2.0, sqrt 4.0 / 2.0]
      ]
  , testGroup "Cos" $
      [ testCase "Cos [0]" $
          map' cos sinList (Z :. 1) @?=~ [1]
      , testCase "Cos [0, PI/6, PI/4, PI/3, PI/2]" $
          map' cos sinList (Z :. 5) @?=~ [sqrt 4.0 / 2.0, sqrt 3.0 / 2.0, sqrt 2.0 / 2.0, sqrt 1.0 / 2.0, sqrt 0.0 / 2.0]
      ]
  , testGroup "Max" $
      [ testCase "Max [0] [-3]" $
          zipWith' A.max maxListLeft maxListRight (Z :. 1) @?=~ [0]
      , testCase "Max [0, -3, 1, -6, 2, -9] [-3, 0, -6, 1, -9, 2]" $
          zipWith' A.max maxListLeft maxListRight (Z :. 6) @?=~ [0, 0, 1, 1, 2, 2]
      ]
  ]
  where
    sinList, maxListLeft, maxListRight :: [Float]
    sinList = 0 : ((pi /) <$> [6, 4, 3, 2])
    maxListLeft  =      merge [0..] $ (*(-1)) <$> [3, 6..]
    maxListRight = flip merge [0..] $ (*(-1)) <$> [3, 6..]

    merge :: [a] -> [a] -> [a]
    merge [] xs = xs
    merge xs [] = xs
    merge (x:xs) (y:ys) = x:y:merge xs ys

    map' :: Shape sh => (Exp Float -> Exp Float) -> [Float] -> sh -> [Float]
    map' f xs' shape =
      let
        xs = fromList shape xs'
        reprData = [xs :-> Result shape]
        model = TPU.compile (A.map f) reprData
      in toList $ TPU.execute model xs

    zipWith' :: Shape sh => (Exp Float -> Exp Float -> Exp Float) -> [Float] -> [Float] -> sh -> [Float]
    zipWith' f xs' ys' shape =
      let
        xs = fromList shape xs'
        ys = fromList shape ys'
        reprData = [xs :-> ys :-> Result shape]
        model = TPU.compile (A.zipWith f) reprData
      in toList $ TPU.execute model xs ys

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

