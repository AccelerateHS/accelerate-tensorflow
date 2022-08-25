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
import Data.Array.Accelerate.Interpreter                            as I
import Data.Array.Accelerate.TensorFlow.Lite                        as TPU

import Data.Array.Accelerate.Sugar.Elt

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Prelude                                                      as P


test_sasum :: TestTree
test_sasum =
  testGroup "sasum"
    [ testElt f32
    ]
    where
      testElt :: forall e. (A.Num e, Show e, Similar e)
              => Gen e
              -> TestTree
      testElt e =
        testProperty (show (eltR @e)) $ prop_sasum e

prop_sasum
    :: (A.Num e, Show e, Similar e)
    => Gen e
    -> Property
prop_sasum e =
  property $ do
    sh  <- forAll dim1
    dat <- forAllWith (const "sample-data") (generate_sample_data_sasum sh e)
    xs  <- forAll (array sh e)
    let
        sasum = A.fold (+) 0 . A.map abs
        !ref  = I.runN sasum
        !tpu  = TPU.compile sasum dat
    --
    TPU.execute tpu xs ~~~ ref xs

generate_sample_data_sasum
  :: Elt e
  => DIM1
  -> Gen e
  -> Gen (RepresentativeData (Vector e -> Scalar e))
generate_sample_data_sasum sh e = do
  i  <- Gen.int (Range.linear 1 16)
  xs <- Gen.list (Range.singleton i) (array sh e)
  return [ x :-> Result Z | x <- xs ]

