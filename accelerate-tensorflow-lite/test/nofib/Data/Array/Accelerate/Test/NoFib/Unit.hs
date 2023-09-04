{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.Map
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Unit (

  test_unit

) where

import Data.Array.Accelerate.Test.NoFib.Base

import Data.Array.Accelerate                                        as A
import Data.Array.Accelerate.TensorFlow.Lite                        ( Args(..) )

-- import Data.Array.Accelerate.Sugar.Shape

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Prelude                                                      as P


test_unit :: TestContext -> TestTree
test_unit tc =
  testGroup "unit"
    [ testProperty "sum_generate_2" $ prop_sum_generate_2 tc
    , testProperty "generate_2" $ prop_generate_2 tc
    , tree_sum3mod4 tc
    ]

prop_sum_generate_2 :: TestContext -> Property
prop_sum_generate_2 tc = property $ do
  sh   <- forAll (Gen.filter (\(Z :. _ :. m) -> m P.> 0) dim2)
  -- let sh = Z :. 2 :. 5
  collect sh
  let sh' = let Z :. n :. _ = sh in Z :. n
  ndat <- forAll (Gen.int (Range.linear 10 16))
  dat1 <- forAll (Gen.list (Range.singleton ndat) (array ForSample sh f32))
  dat2 <- forAll (Gen.list (Range.singleton ndat) (array ForSample sh f32))
  xs1  <- forAll (array ForInput sh f32)
  xs2  <- forAll (array ForInput sh f32)
  let f a b = A.sum (A.generate (A.constant sh) (\i -> a ! i + b ! i))
  -- let f a b = A.sum (A.zipWith (+) a b)
  tpuTestCase tc f (P.zipWith (\a b -> a :-> b :-> Result sh') dat1 dat2) xs1 xs2

prop_generate_2 :: TestContext -> Property
prop_generate_2 tc = property $ do
  sh   <- forAll (Gen.filter (\(Z :. _ :. m) -> m P.> 0) dim2)
  collect sh
  ndat <- forAll (Gen.int (Range.linear 10 16))
  dat1 <- forAll (Gen.list (Range.singleton ndat) (array ForSample sh f32))
  dat2 <- forAll (Gen.list (Range.singleton ndat) (array ForSample sh f32))
  xs1  <- forAll (array ForInput sh f32)
  xs2  <- forAll (array ForInput sh f32)
  let f a b = A.generate (A.constant sh) (\i -> a ! i + b ! i)
  tpuTestCase tc f (P.zipWith (\a b -> a :-> b :-> Result sh) dat1 dat2) xs1 xs2

tree_sum3mod4 :: TestContext -> TestTree
tree_sum3mod4 tc =
  localOption (HedgehogTestLimit (Just 1)) $
  testGroup "sum3mod4" $
    concat [[testDim3 2 2 2]
           ,[testDim3 1 1 n | n <- [1..20]]
           ,[testDim3 2 1 n | n <- [1..20]]
           ,[testDim3 3 1 n | n <- [1..20]]
           ,[testDim3 4 1 n | n <- [1..20]]
           ,[testDim3 5 1 n | n <- [1..20]]
           ,[testDim3 3 2 n | n <- [1..20]]
           ,[testDim3 1 1 n | n <- [1..20]]
           ,[testDim3 1 2 n | n <- [1..20]]
           ,[testDim3 1 3 n | n <- [1..20]]
           ,[testDim2 1 n | n <- [1..20]]
           ,[testDim2 2 n | n <- [1..20]]
           ,[testDim2 3 n | n <- [1..20]]
           ]
  where
    testDim2 :: Int -> Int -> TestTree
    testDim2 a b =
      testProperty (show a P.++ " " P.++ show b) $
        prop_sum (return (Z :. a :. b)) f32

    testDim3 :: Int -> Int -> Int -> TestTree
    testDim3 a b c =
      testProperty (show a P.++ " " P.++ show b P.++ " " P.++ show c) $
        prop_sum (return (Z :. a :. b :. c)) f32

    prop_sum
        :: (P.Eq sh, Show sh, Shape sh, A.Num e, Show e, Similar e)
        => Gen (sh :. Int)
        -> (WhichData -> Gen e)
        -> Property
    prop_sum dim e =
      property $ do
        sh@(shtail :. _) <- forAll dim
        ndat <- forAll (Gen.int (Range.linear 10 16))
        dat <- forAll (do arrs <- Gen.list (Range.singleton ndat) (array ForSample sh e)
                          return [arr :-> Result shtail | arr <- arrs])
        xs  <- forAll (array ForInput sh e)
        tpuTestCase tc A.sum dat xs
