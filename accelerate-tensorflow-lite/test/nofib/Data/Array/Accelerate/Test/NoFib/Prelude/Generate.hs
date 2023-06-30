{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.Generate
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude.Generate (

  test_generate

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


test_generate :: TestTree
test_generate =
  testGroup "generate"
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
          [ testProperty "fill16" $ prop_fill dim i16
          , testProperty "fill8" $ prop_fill dim i8
          , testProperty "mod19" $ prop_mod19 dim int
          ]

prop_fill
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
    => Gen sh
    -> (WhichData -> Gen e)
    -> Property
prop_fill dim e =
  property $ do
    sh  <- forAll dim
    x   <- forAll (e ForInput)
    dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
    let f    = A.fill (A.constant sh) (A.constant x)
        !ref = I.runN f
        !tpu = TPU.compile f dat
    --
    TPU.execute tpu ~~~ ref

prop_mod19
    :: (P.Eq sh, Show sh, Shape sh)
    => Gen sh
    -> (WhichData -> Gen Int)
    -> Property
prop_mod19 dim e =
  property $ do
    sh  <- forAll dim
    x   <- forAll (e ForInput)
    dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
    let f    = A.generate (A.constant sh) (\ix -> A.toIndex (A.constant sh) ix `A.rem` A.constant 19)
        !ref = I.runN f
        !tpu = TPU.compile f dat
    --
    TPU.execute tpu ~~~ ref



-- prop_generate
--     :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
--     => (Exp sh -> Exp e)
--     -> Gen sh
--     -> Gen e
--     -> Property
-- prop_generate f dim e =
--   property $ do
--     sh  <- forAll dim
--     dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
--     let !ref = I.runN (A.generate (A.constant sh) f)
--         !tpu = TPU.compile (A.generate (A.constant sh) f) dat
--     --
--     TPU.execute tpu ~~~ ref

generate_sample_data
  :: (Shape sh, Elt e)
  => sh
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array sh e))
generate_sample_data sh _e = do
  Gen.list (Range.linear 10 16) (Gen.constant (Result sh))

