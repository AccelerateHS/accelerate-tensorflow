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
import Data.Array.Accelerate.TensorFlow.Lite                        as TPU

import Data.Array.Accelerate.Sugar.Shape

import Hedgehog
import qualified Hedgehog.Gen                                       as Gen
import qualified Hedgehog.Range                                     as Range

import Test.Tasty
import Test.Tasty.Hedgehog

import Prelude                                                      as P


test_generate :: TestContext -> TestTree
test_generate tc =
  testGroup "generate"
    [ testDim dim0
    , testDim dim1
    , testDim dim2
    , testProperty "fromintegral" $ prop_generate tc (\(I1 i) -> A.fromIntegral @Int @Int32 i) dim1
    , testProperty "genid" $ prop_generate tc (\(I1 i) -> I1 i) dim1
    ]
    where
      testDim :: forall sh. (Shape sh, Show sh, P.Eq sh, Similar sh)
              => Gen sh
              -> TestTree
      testDim dim =
        testGroup ("DIM" P.++ show (rank @sh))
          [ testProperty "fill32" $ prop_fill tc dim f32
          , testProperty "fill16" $ prop_fill tc dim i16
          , testProperty "fill8" $ prop_fill tc dim i8
          , testProperty "mod19" $ prop_mod19 tc dim
          , testProperty "noop_i32" $ prop_noop tc dim i32
          , testProperty "noop_i64" $ prop_noop tc dim i64
          , testProperty "noop_f32" $ prop_noop tc dim f32
          ]

prop_fill
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e, P.Num e)
    => TestContext
    -> Gen sh
    -> (WhichData -> Gen e)
    -> Property
prop_fill tc dim e =
  property $ do
    sh  <- forAll dim
    x   <- forAll (e ForInput)
    dat <- forAllWith (const "sample-data") (generate_sample_data sh)
    let f    = A.fill (A.constant sh) (A.constant x)
    tpuTestCase tc f dat



-- prop_id 
--     :: forall sh . (P.Eq sh, Show sh, Shape sh, Similar sh, Elt sh)
--     => Gen sh
--     -> (WhichData -> Gen e)
--     -> Property
-- prop_id = prop_generate id
prop_generate
    :: forall sh e. (P.Eq sh, Show sh, Shape sh, Elt e, Similar e, Show e, P.Eq e)
    => TestContext
    -> (Exp sh -> Exp e)
    -> Gen sh
    -> Property
prop_generate tc g dim =
  property $ do
    sh  <- forAll dim
    dat <- forAllWith (const "sample-data") (generate_sample_data sh)
    let f :: Acc (Array sh e)
        f = A.generate (A.constant sh) g
    tpuTestCase tc f dat


prop_noop
    :: forall sh e . (P.Eq sh, Show sh, Shape sh, Elt e, Similar e, Show e, P.Eq e)
    => TestContext
    -> Gen sh
    -> (WhichData -> Gen e)
    -> Property
prop_noop tc dim e =
  property $ do
    sh  <- forAll dim
    dat <- forAll (do i  <- Gen.int (Range.linear 10 16)
                      xs <- Gen.list (Range.singleton i) (array ForSample sh e)
                      return [ x :-> Result sh | x <- xs ])
    inp <- forAll (array ForInput sh e)

    let f :: Acc (Array sh e) -> Acc (Array sh e)
        f xs = xs
    tpuTestCase tc f dat inp


prop_mod19
    :: forall sh. (P.Eq sh, Show sh, Shape sh)
    => TestContext
    -> Gen sh
    -> Property
prop_mod19 tc dim =
  property $ do
    sh  <- forAll dim
    dat <- forAllWith (const "sample-data") (generate_sample_data sh)
    let f    = A.generate (A.constant sh) (\ix -> A.toIndex (A.constant sh) ix `A.rem` A.constant 19)
    tpuTestCase tc f dat


-- prop_generate
--     :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
--     => TestContext
--     -> (Exp sh -> Exp e)
--     -> Gen sh
--     -> Gen e
--     -> Property
-- prop_generate tc f dim e =
--   property $ do
--     sh  <- forAll dim
--     dat <- forAllWith (const "sample-data") (generate_sample_data sh e)
--     tpuTestCase tc (A.generate (A.constant sh) f) dat

generate_sample_data
  :: (Shape sh, Elt e)
  => sh
  -- -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array sh e))
generate_sample_data sh = do
  Gen.list (Range.linear 10 16) (Gen.constant (Result sh))

