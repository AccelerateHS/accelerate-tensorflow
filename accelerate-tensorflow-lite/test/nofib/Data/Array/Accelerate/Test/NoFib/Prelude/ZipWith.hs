{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ParallelListComp    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.ZipWith
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude.ZipWith (

  test_zipWith

) where

import Data.Array.Accelerate.Test.NoFib.Base

import Data.Array.Accelerate                                        as A
import Data.Array.Accelerate.Interpreter                            as I
import Data.Array.Accelerate.TensorFlow.Lite                        as TPU

import Data.Array.Accelerate.Sugar.Shape

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Prelude                                                      as P


test_zipWith :: TestTree
test_zipWith =
  testGroup "zipWith"
    [ testDim dim0
    , testDim dim1
    , testDim dim2
    ]
    where
      testDim :: forall sh. (Shape sh, Show sh, P.Eq sh)
              => Gen sh
              -> TestTree
      testDim dim =
        testGroup ("DIM" P.++ show (rank @sh))
          [ testProperty "add" $ prop_zipWith (+) dim f32
          , testProperty "sub" $ prop_zipWith (-) dim f32
          , testProperty "mul" $ prop_zipWith (*) dim f32
          , testProperty "min" $ prop_zipWith A.min dim f32
          , testProperty "max" $ prop_zipWith A.max dim f32
          ]


prop_zipWith
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
    => (Exp e -> Exp e -> Exp e)
    -> Gen sh
    -> Gen e
    -> Property
prop_zipWith f dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
    xs  <- forAll (array sh e)
    ys  <- forAll (array sh e)
    let !ref = I.runN (A.zipWith f)
        !tpu = TPU.compile (A.zipWith f) dat
    --
    TPU.execute tpu xs ys ~~~ ref xs ys

generate_sample_data
  :: (Shape sh, Elt e)
  => sh
  -> Gen e
  -> Gen (RepresentativeData (Array sh e -> Array sh e -> Array sh e))
generate_sample_data sh e = do
  i  <- Gen.int (Range.linear 1 16)
  xs <- Gen.list (Range.singleton i) (array sh e)
  ys <- Gen.list (Range.singleton i) (array sh e)
  return [ x :-> y :-> Result sh | x <- xs | y <- ys ]

