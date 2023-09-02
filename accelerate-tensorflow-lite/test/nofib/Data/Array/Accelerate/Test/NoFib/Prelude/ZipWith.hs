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
import Data.Array.Accelerate.TensorFlow.Lite                        as TPU

import Data.Array.Accelerate.Sugar.Shape

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Prelude                                                      as P


test_zipWith :: TestContext -> TestTree
test_zipWith tc =
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
          [ testProperty "add" $ prop_zipWith tc (+) dim f32
          , testProperty "sub" $ prop_zipWith tc (-) dim f32
          , testProperty "mul" $ prop_zipWith tc (*) dim f32
          , testProperty "min" $ prop_zipWith tc A.min dim f32
          , testProperty "max" $ prop_zipWith tc A.max dim f32
          ]


prop_zipWith
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
    => TestContext
    -> (Exp e -> Exp e -> Exp e)
    -> Gen sh
    -> (WhichData -> Gen e)
    -> Property
prop_zipWith tc f dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
    xs  <- forAll (array ForInput sh e)
    ys  <- forAll (array ForInput sh e)
    tpuTestCase tc (A.zipWith f) dat xs ys

generate_sample_data
  :: (Shape sh, Elt e)
  => sh
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array sh e -> Array sh e -> Array sh e))
generate_sample_data sh e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  ys <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> y :-> Result sh | x <- xs | y <- ys ]

