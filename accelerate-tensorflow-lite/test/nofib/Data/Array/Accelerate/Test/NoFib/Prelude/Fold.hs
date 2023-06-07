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

import Data.Array.Accelerate.Sugar.Shape                            as S

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
          [ testProperty "sum"     $ prop_fold (+) 0 dim f32
          , testProperty "product" $ prop_fold1 (*) dim f32
          , testProperty "minimum" $ prop_fold1 A.min dim f32
          , testProperty "maximum" $ prop_fold1 A.max dim f32
          ]

prop_fold
    :: (P.Eq sh, Show sh, Shape sh, A.Num e, Show e, Similar e)
    => (Exp e -> Exp e -> Exp e)
    -> Exp e
    -> Gen (sh :. Int)
    -> (WhichData -> Gen e)
    -> Property
prop_fold f z dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (array ForInput sh e)
    let acc  = A.fold f z
        !ref = I.runN acc
        !tpu = TPU.compile acc dat
    --
    TPU.execute tpu xs ~~~ ref xs

prop_fold1
    :: (P.Eq sh, Show sh, Shape sh, A.Num e, Show e, Similar e)
    => (Exp e -> Exp e -> Exp e)
    -> Gen (sh :. Int)
    -> (WhichData -> Gen e)
    -> Property
prop_fold1 f dim e =
  property $ do
    sh  <- forAll (dim `except` \sh -> S.size sh P.== 0)
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (array ForInput sh e)
    let acc  = A.fold1 f
        !ref = I.runN acc
        !tpu = TPU.compile acc dat
    --
    TPU.execute tpu xs ~~~ ref xs

generate_sample_data
  :: (Shape sh, Elt e)
  => (sh :. Int)
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array (sh :. Int) e -> Array sh e))
generate_sample_data (sh :. sz) e = do
  i  <- Gen.int (Range.linear 1 16)
  xs <- Gen.list (Range.singleton i) (array ForSample (sh :. sz) e)
  return [ x :-> Result sh | x <- xs ]

