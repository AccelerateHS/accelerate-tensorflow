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
  , generateTests
  , transposeTests
  , uncurryTests
  -- , dotpTests
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

generateTests :: TestTree
generateTests = testGroup "Generate Tests"
  [ testCase "generate (I1 1) (const 1.0)" $
      generate' (Z :. 1)  (I1 1)  (const 1.0) @?=~ [1.0]
  , testCase "generate (I1 10) (const 1.0)" $
      generate' (Z :. 10) (I1 10) (const 1.0) @?=~ [1.0 .. 10.0]
  , testCase "generate (I1 10) ( \\(I1 n) -> fromIntegral n)" $
      generate' (Z :. 10) (I1 10) ( \(I1 n) -> A.fromIntegral n) @?=~ [0.0 .. 9.0]
  ]
  where
    generate' :: Shape sh => sh -> Exp sh -> (Exp sh -> Exp Float) -> [Float]
    generate' sh esh f =
      let
        xs = A.generate esh f
        reprData = [Result sh]
        model = TPU.compile xs reprData
      in toList $ TPU.execute model

transposeTests :: TestTree
transposeTests = testGroup "Transpose Tests"
  [ testCase "transpose [[1]]" $
      --transpose' (Z :. 1 :. 1) (A.fromList [1.0, 1.0..]) `areEqual` [[1.0]]
      transpose' (Z :. 1 :. 1) [1.0, 1.0..] @?=~ [1.0]
  , testCase "transpose [[1, 1], [1, 1]]" $
      transpose' (Z :. 2 :. 2) [1.0, 1.0..] @?=~ [1.0, 1.0, 1.0, 1.0]
  , testCase "transpose [[1, 2]]" $
      transpose' (Z :. 1 :. 2) [1.0, 2.0]   @?=~ [1.0, 2.0]
  , testCase "transpose [[1], [2]]" $
      transpose' (Z :. 2 :. 1) [1.0, 2.0]   @?=~ [1.0, 2.0]
  , testCase "transpose [[1, 2, 3], [4, 5, 6]]" $
      transpose' (Z :. 2 :. 3) [1.0, 2.0..] @?=~ [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
  , testCase "transpose [[1, 2], [3, 4], [5, 6]]" $
      transpose' (Z :. 3 :. 2) [1.0, 2.0..] @?=~ [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
  , testCase "transpose [[1, 2, 3], [4, 5, 6], [7, 8, 9]]" $
      transpose' (Z :. 3 :. 3) [1.0, 2.0..] @?=~ [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
  , testCase "transpose [[1], [2], [3], [4], [5], [6]]" $
      transpose' (Z :. 6 :. 1) [1.0, 2.0..] @?=~ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  , testCase "transpose [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]" $
      transpose' (Z :. 6 :. 2) [1.0, 2.0..] @?=~ [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
  , testCase "transpose [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]" $
      transpose' (Z :. 2 :. 6) [1.0, 2.0..] @?=~ [1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0]
  ]
  where
    transpose' :: DIM2 -> [Float] -> [Float]
    transpose' sh xs' =
      let
        xs = fromList sh xs'
        reprData = [xs :-> Result sh]
        model = TPU.compile A.transpose reprData
      in toList $ TPU.execute model xs

    areEqual :: [[Float]] -> [[Float]] -> Assertion
    areEqual xss yss =
      let
        -- Check if the outer lists have the same shape and if all inner lists
        -- have the same shape as well.
        xsl = P.map P.length xss
        ysl = P.map P.length yss

        -- Check if each element of the inner lists are equal to the element in
        -- the same position in the other list
        eqs = P.zipWith (@?=~) xss yss

        -- Make sure that all elements are equal to each other
        eq  = sequence_ eqs

        -- Make a nice message so the user knows what the issue is...
        listLenMsg =
            "The list we found does not have the correct lengths:\nFound lengths: "
         P.++ show xsl
         P.++ "\nExpected lengths: "
         P.++ show ysl
      in do
        unless (xsl P.== ysl) (assertFailure listLenMsg)
        eq

uncurryTests :: TestTree
uncurryTests = testGroup "Uncurry Tests"
  [ testCase "uncurry curriedDot [0] [1]" $
      uncurry' (\as bs -> A.fold (+) 0 $ A.zipWith (*) as bs) [0] [1]
  , testCase "uncurry curriedDot [0, 1] [1, 3]" $
      uncurry' (\as bs -> A.fold (+) 0 $ A.zipWith (*) as bs) [0, 1] [1, 3]
  , testCase "uncurry curriedDot [0, 1, 2, 3, 4, 5] [1, 3, 5, 7, 9, 11]" $
      uncurry' (\as bs -> A.fold (+) 0 $ A.zipWith (*) as bs) [0, 1, 2, 3, 4, 5] [1, 3, 5, 7, 9, 11]
  ]
  where
    uncurry'
      :: (Acc (Array (Z :. Int) Float) -> Acc (Array (Z :. Int) Float) -> Acc (Array Z Float))
      -> [Float]
      -> [Float]
      -> Assertion
    uncurry' f as bs =
      let
        sh@(s :. _) = Z :. P.length as
	xs = fromList sh as
	ys = fromList sh bs
	tu = use (xs, ys)
	reprData = [(xs, ys) :-> Result s]
	model = TPU.compile (A.uncurry f) reprData
      in toList (TPU.execute model (xs, ys)) @?=~ [P.foldr (+) 0 (P.zipWith (*) as bs)]

dotpTests :: TestTree
dotpTests = testGroup "Dot Product Tests"
  [ testCase "dot [0] [1]" $
      dotTest 1
  , testCase "dot [0, 1] [1, 3]" $
      dotTest 2
  , testCase "dot [0, 1, 2] [1, 3, 5]" $
      dotTest 3
  , testCase "dot [0, 1, 2, 3] [1, 3, 5, 7]" $
      dotTest 4
  , testCase "dot [0, 1, 2, 3, 4] [1, 3, 5, 7, 9]" $
      dotTest 5
  , testCase "dot [0, 1, 2, 3, 4, 5, 6] [1, 3, 5, 7, 9, 11, 13]" $
      dotTest 7
  ]
  where
    zeroList, oneList :: [Float]
    zeroList = [0..]
    oneList  = [1, 3..]

    dot xs' ys' shape@(s :. _) =
      let
        xs = fromList shape ys'
        ys = fromList shape ys'
        reprData = [xs :-> ys :-> Result s]
        model = TPU.compile (\as bs -> A.fold (+) 0 $ A.zipWith (*) as bs) reprData
      in toList $ TPU.execute model xs ys
    dotTest n = dot zeroList oneList (Z :. n) @?=~ [P.foldr (+) 0 (P.zipWith (*) (P.take n zeroList) oneList)]

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
foldTests = testGroup "Fold-like Tests"
  [ testGroup "Fold Tests"
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
  , testGroup "Sum Tests"
    [ testCase "Sum [0]" $
        sum' ascList (Z :. 1) @?=~ [0]
    , testCase "Sum [0..5]" $
        sum' ascList (Z :. 6) @?=~ [15]
    , testCase "Sum [[0..2], [3..5]]" $
        sum' ascList (Z :. 2 :. 3) @?=~ [3, 12]
    ]
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

    sum' :: Shape sh => [Float] -> (sh :. Int) -> [Float]
    sum' xs' shape =
      let
        stripShape :: (sh :. Int) -> sh
        stripShape (x :. _) = x

        xs = fromList shape xs'
        reprData = [xs :-> Result $ stripShape shape]
        model = TPU.compile A.sum reprData
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

