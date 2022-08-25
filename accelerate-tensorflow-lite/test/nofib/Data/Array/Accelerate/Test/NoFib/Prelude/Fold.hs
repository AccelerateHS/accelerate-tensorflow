{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.Fold
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude.Fold (

  test_fold

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


test_fold :: TestTree
test_fold =
  testGroup "fold"
    [ testDim dim1
    , testDim dim2
    , testDim dim3
    ]
    where
      testDim :: forall sh. (Shape sh, Show sh, P.Eq sh)
              => Gen (sh :. Int)
              -> TestTree
      testDim dim =
        testGroup ("DIM" P.++ show (rank @(sh :. Int)))
          [ testProperty "sum" $ test_sum dim f32
          ]

test_sum
    :: (P.Eq sh, Show sh, Shape sh, A.Num e, Show e, Similar e)
    => Gen (sh :. Int)
    -> Gen e
    -> Property
test_sum dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
    xs  <- forAll (array sh e)
    let f    = A.sum
        !ref = I.runN f
        !tpu = TPU.compile f dat
    --
    TPU.execute tpu xs ~~~ ref xs

generate_sample_data
  :: (Shape sh, Elt e)
  => (sh :. Int)
  -> Gen e
  -> Gen (RepresentativeData (Array (sh :. Int) e -> Array sh e))
generate_sample_data (sh :. sz) e = do
  i  <- Gen.int (Range.linear 1 16)
  xs <- Gen.list (Range.singleton i) (array (sh :. sz) e)
  return [ x :-> Result sh | x <- xs ]

