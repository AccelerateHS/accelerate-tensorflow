{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Imaginary.SASUM
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Imaginary.SASUM (

  test_sasum

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

import Prelude                                                      as P


test_sasum :: TestContext -> TestTree
test_sasum tc =
  testGroup "sasum"
    [ testElt f32
    ]
    where
      testElt :: forall e. (A.Num e, Show e, Similar e)
              => (WhichData -> Gen e)
              -> TestTree
      testElt e =
        testProperty (show (eltR @e)) $ prop_sasum tc e

prop_sasum
    :: (A.Num e, Show e, Similar e)
    => TestContext
    -> (WhichData -> Gen e)
    -> Property
prop_sasum tc e =
  property $ do
    sh  <- forAll dim1
    dat <- forAll (generate_sample_data_sasum sh e)
    xs  <- forAll (array ForInput sh e)
    let sasum = A.fold (+) 0 . A.map abs
    tpuTestCase tc sasum dat xs

generate_sample_data_sasum
  :: Elt e
  => DIM1
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Vector e -> Scalar e))
generate_sample_data_sasum sh e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> Result Z | x <- xs ]

