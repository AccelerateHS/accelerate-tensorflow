{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ParallelListComp    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Imaginary.DotP
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Imaginary.DotP (

  test_dotp

) where

import Data.Array.Accelerate.Smart                                  ( ($$) )
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


test_dotp :: TestTree
test_dotp =
  testGroup "dotp"
    [ testElt f32
    ]
    where
      testElt :: forall e. (A.Num e, Show e, Similar e)
              => (WhichData -> Gen e)
              -> TestTree
      testElt e =
        testProperty (show (eltR @e)) $ prop_dotp e

prop_dotp
    :: (A.Num e, Show e, Similar e)
    => (WhichData -> Gen e)
    -> Property
prop_dotp e =
  property $ do
    sh  <- forAll dim1
    dat <- forAllWith (const "sample-data") (generate_sample_data_dotp sh e)
    xs  <- forAll (array ForInput sh e)
    ys  <- forAll (array ForInput sh e)
    let
        dotp = A.sum $$ A.zipWith (*)
        !ref = I.runN dotp
        !tpu = TPU.compile dotp dat
    --
    TPU.execute tpu xs ys ~~~ ref xs ys

generate_sample_data_dotp
  :: Elt e
  => DIM1
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Vector e -> Vector e -> Scalar e))
generate_sample_data_dotp sh e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  ys <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> y :-> Result Z | x <- xs | y <- ys ]

