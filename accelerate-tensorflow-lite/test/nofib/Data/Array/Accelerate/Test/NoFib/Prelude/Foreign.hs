{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.Generate
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude.Foreign (

  test_foreign

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


test_foreign :: TestContext -> TestTree
test_foreign tc =
  testGroup "foreign"
    [ testDim dim1
    , testDim dim2
    , testDim dim3
    ]
    where
      testDim :: forall sh. (Shape sh, Show sh, P.Eq sh)
              => Gen (sh:.Int)
              -> TestTree
      testDim dim =
        testGroup ("DIM" P.++ show (rank @sh))
          [ testProperty "argmin" $ prop_min tc dim f32
          , testProperty "argmax" $ prop_max tc dim i16
          ]

prop_min
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e, A.Ord e)
    => TestContext
    -> Gen (sh:.Int)
    -> (WhichData -> Gen e)
    -> Property
prop_min tc dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (array ForInput sh e)
    let !f   = argMin
    tpuTestCase tc f dat xs

prop_max
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e, A.Ord e)
    => TestContext
    -> Gen (sh:.Int)
    -> (WhichData -> Gen e)
    -> Property
prop_max tc dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (array ForInput sh e)
    let !f   = argMax
    tpuTestCase tc f dat xs


generate_sample_data
  :: (Shape sh, Elt e)
  => (sh:.Int)
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array (sh:.Int) e -> Array sh (Int32, e)))
generate_sample_data (sh:.sz) e = do
  xs <- Gen.list (Range.linear 10 16) (array ForSample (sh:.sz) e)
  return [ x :-> Result sh | x <- xs ]

