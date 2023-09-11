{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
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

import Control.Monad.IO.Class
import Prelude                                                      as P


test_foreign :: TestContext -> TestTree
test_foreign tc =
  testGroup "foreign"
    [ testDim dim1
    , testDim dim2
    , testDim dim3
    ]
    where
      testDim :: forall sh. (Shape2 sh, Show sh, P.Eq sh)
              => Gen (sh:.Int)
              -> TestTree
      testDim dim =
        testGroup ("DIM" P.++ show (rank @sh))
          [ testProperty "argmin" $ prop_min tc dim f32
          , testProperty "argmax" $ prop_max tc dim i16
          , testProperty "append_i32" $ prop_app tc dim i32
          , testProperty "append_f32" $ prop_app tc dim f32
          ]


prop_app
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e)
    => TestContext
    -> Gen (sh:.Int)
    -> (WhichData -> Gen e)
    -> Property
prop_app tc dim e =
  property $ do
    sh1 <- forAll (Gen.filter (\(_ :. n) -> n P.> 0) dim)
    sh2 <- forAll (Gen.filter (\(_ :. n) -> n P.> 0) dim)
    ndat <- forAll (Gen.int (Range.linear 10 16))
    dat1 <- forAll (Gen.list (Range.singleton ndat) (array ForSample sh1 e))
    dat2 <- forAll (Gen.list (Range.singleton ndat) (array ForSample sh2 e))
    xs <- forAll (array ForInput sh1 e)
    ys <- forAll (array ForInput sh2 e)
    let sh = appendresultshape sh1 sh2
    tpuTestCase tc append (P.zipWith (\a b -> a :-> b :-> Result sh) dat1 dat2) xs ys
  where
    appendresultshape (sh1:.sz1) (sh2:.sz2) = Data.Array.Accelerate.Sugar.Shape.intersect sh1 sh2 :. sz1+sz2

prop_min
    :: (P.Eq sh, Show sh, Shape2 sh, Elt e, Show e, Similar e, A.Ord e, P.Ord e, P.Num e)
    => TestContext
    -> Gen (sh:.Int)
    -> (WhichData -> Gen e)
    -> Property
prop_min tc dim e =
  property $ do
    sh  <- forAll (Gen.filter (\(_ :. n) -> n P.> 0) dim)
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (Gen.filter (wellDefinedMax negate) $ array ForInput sh e)
    let !f   = argMin
    tpuTestCase tc f dat xs

prop_max
    :: (P.Eq sh, Show sh, Shape2 sh, Elt e, Show e, Similar e, A.Ord e, P.Ord e)
    => TestContext
    -> Gen (sh:.Int)
    -> (WhichData -> Gen e)
    -> Property
prop_max tc dim e =
  property $ do
    sh  <- forAll (Gen.filter (\(_ :. n) -> n P.> 0) dim)
    dat <- forAll (generate_sample_data sh e)
    xs  <- forAll (Gen.filter (wellDefinedMax id) $ array ForInput sh e)
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

-- The comparison is done after mapping the element function.
wellDefinedMax :: (Shape2 sh, Elt e, P.Ord e') => (e -> e') -> Array (sh:.Int) e -> Bool
wellDefinedMax conv arr =
  let sh :. n = arrayShape arr
      vectors = [[conv (arr `indexArray` (ix :. i)) | i <- [0 .. n - 1]] | ix <- enumerateShape sh]
      wellDefinedMaxVector l = P.length (P.filter (P.>= P.maximum l) l) P.<= 1
  in P.all wellDefinedMaxVector vectors

-- Such metaprogramming is either very annoying or impossible with the
-- Accelerate userland API, so we reimplement some stuff here.
class Shape sh => Shape2 sh where
  enumerateShape :: sh -> [sh]
instance Shape2 Z where
  enumerateShape Z = [Z]
instance Shape2 sh => Shape2 (sh:.Int) where
  enumerateShape (sh :. n) = [ix :. i | ix <- enumerateShape sh, i <- [0 .. n - 1]]
