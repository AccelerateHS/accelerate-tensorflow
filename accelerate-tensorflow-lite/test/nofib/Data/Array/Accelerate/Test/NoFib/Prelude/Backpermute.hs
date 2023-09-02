{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ParallelListComp    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.Backpermute
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude.Backpermute (

  test_backpermute

) where

import Data.Array.Accelerate.Test.NoFib.Base

import Data.Array.Accelerate                                        as A
import Data.Array.Accelerate.TensorFlow.Lite                        as TPU

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Prelude                                                      as P


test_backpermute :: TestContext -> TestTree
test_backpermute tc =
  testGroup "backpermute"
    [ testDIM2
    ]
    where
      testDIM2 :: TestTree
      testDIM2 =
        testGroup "DIM2"
          [ testProperty "transpose" $ prop_transpose tc f32
          ]

prop_transpose
    :: (Elt e, Show e, Similar e)
    => TestContext
    -> (WhichData -> Gen e)
    -> Property
prop_transpose tc e =
  property $ do
    sh  <- forAll dim2
    dat <- forAllWith (const "sample-data") (generate_sample_data_transpose sh e)
    xs  <- forAll (array ForInput sh e)
    tpuTestCase tc A.transpose dat xs

generate_sample_data_transpose
  :: Elt e
  => DIM2
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array DIM2 e -> Array DIM2 e))
generate_sample_data_transpose sh@(Z :. h :. w) e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> Result (Z :. w :. h) | x <- xs ]

