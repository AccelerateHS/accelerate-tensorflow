{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
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
