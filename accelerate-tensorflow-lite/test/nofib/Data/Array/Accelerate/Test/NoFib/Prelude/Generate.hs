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
          [ testProperty "fill" $ prop_fill dim f32
          ]

prop_fill
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
    => Gen sh
    -> Gen e
    -> Property
prop_fill dim e =
  property $ do
    sh  <- forAll dim
    x   <- forAll e
    dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
    let f    = A.fill (A.constant sh) (A.constant x)
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
  -> Gen e
  -> Gen (RepresentativeData (Array sh e))
generate_sample_data sh _e = do
  Gen.list (Range.linear 1 16) (Gen.constant (Result sh))

