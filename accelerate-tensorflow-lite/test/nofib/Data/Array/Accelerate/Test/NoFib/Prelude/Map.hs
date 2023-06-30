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

module Data.Array.Accelerate.Test.NoFib.Prelude.Map (

  test_map

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


test_map :: TestTree
test_map =
  testGroup "map"
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
          [ testProperty "rem" $ prop_map (+1) dim i32
          , testProperty "sin"   $ prop_map sin dim f32
          , testProperty "cos"   $ prop_map cos dim f32
          , testProperty "sqrt"  $ prop_map sqrt dim (fmap abs . f32)
          ]


prop_map
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
    => (Exp e -> Exp e)
    -> Gen sh
    -> (WhichData -> Gen e)
    -> Property
prop_map f dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (array ForInput sh e)
    let !ref = I.runN (A.map f)
        !tpu = TPU.compile (A.map f) dat
    --
    TPU.execute tpu xs ~~~ ref xs

generate_sample_data
  :: (Shape sh, Elt e)
  => sh
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array sh e -> Array sh e))
generate_sample_data sh e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> Result sh | x <- xs ]

