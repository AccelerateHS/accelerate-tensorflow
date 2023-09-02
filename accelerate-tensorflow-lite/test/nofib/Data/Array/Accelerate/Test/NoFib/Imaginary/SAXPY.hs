{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ParallelListComp    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Imaginary.SAXPY
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Imaginary.SAXPY (

  test_saxpy

) where

import Data.Array.Accelerate.Test.NoFib.Base

import Data.Array.Accelerate                                        as A
import Data.Array.Accelerate.TensorFlow.Lite                        as TPU

import Data.Array.Accelerate.Sugar.Elt

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Control.Monad
import Prelude                                                      as P


test_saxpy :: TestContext -> TestTree
test_saxpy tc =
  testGroup "saxpy"
    [ testElt f32
    ]
    where
      testElt :: forall e. (P.Eq e, P.Num e, A.Num e, Show e, Similar e)
              => (WhichData -> Gen e)
              -> TestTree
      testElt e =
        testProperty (show (eltR @e)) $ prop_saxpy tc e

prop_saxpy
    :: (P.Eq e, P.Num e, A.Num e, Show e, Similar e)
    => TestContext
    -> (WhichData -> Gen e)
    -> Property
prop_saxpy tc e =
  property $ do
    sh  <- forAll dim1
    dat <- forAllWith (const "sample-data") (generate_sample_data_saxpy sh e)
    α   <- forAll (e ForInput)
    xs  <- forAll (array ForInput sh e)
    ys  <- forAll (array ForInput sh e)
    let saxpy = A.zipWith (\x y -> constant α * x + y)
    tpuTestCase tc saxpy dat xs ys

generate_sample_data_saxpy
  :: Elt e
  => DIM1
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Vector e -> Vector e -> Vector e))
generate_sample_data_saxpy sh e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  ys <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> y :-> Result sh | x <- xs | y <- ys ]

