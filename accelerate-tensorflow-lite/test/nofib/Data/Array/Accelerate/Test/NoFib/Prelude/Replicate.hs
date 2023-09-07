{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ParallelListComp    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Prelude.Replicate
-- Copyright   : [2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Prelude.Replicate (

  test_replicate

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


test_replicate :: TestContext -> TestTree
test_replicate tc =
  testGroup "replicate"
    [ testDIM2
    , testDIM5
    ]
    where
      testDIM2 :: TestTree
      testDIM2 =
        testGroup "DIM2"
          [ testOne (SliceFixed' (SliceAll'   SliceNil'))
          , testOne (SliceAll'   (SliceFixed' SliceNil'))
          , testOne (SliceFixed' (SliceFixed' SliceNil'))
          ]

      testDIM5 :: TestTree
      testDIM5 =
        testGroup "DIM5"
          [ testOne (SliceFixed' (SliceAll'   (SliceFixed' (SliceFixed' (SliceAll'   SliceNil')))))
          , testOne (SliceAll'   (SliceAll'   (SliceAll'   (SliceAll'   (SliceAll'   SliceNil')))))
          , testOne (SliceFixed' (SliceFixed' (SliceFixed' (SliceFixed' (SliceFixed' SliceNil')))))
          , testOne (SliceAll'   (SliceFixed' (SliceAll'   (SliceAll'   (SliceFixed' SliceNil')))))
          , testOne (SliceAll'   (SliceAll'   (SliceFixed' (SliceAll'   (SliceAll'   SliceNil')))))
          , testOne (SliceAll'   (SliceAll'   (SliceAll'   (SliceAll'   (SliceFixed' SliceNil')))))
          , testOne (SliceFixed' (SliceAll'   (SliceAll'   (SliceAll'   (SliceAll'   SliceNil')))))
          ]

      testOne :: (Show dim, Shape slice, FullShape ix ~ dim, SliceShape ix ~ slice, Elt ix, Slice ix, P.Eq dim)
              => SliceIndex' ix slice dim -> TestTree
      testOne = \ix -> testProperty ("replicate_" P.++ sliceString ix "") $ prop_replicate tc f32 ix
        where sliceString :: SliceIndex' ix slice dim -> ShowS
              sliceString SliceNil' = showString "Z"
              sliceString (SliceAll' ix) = sliceString ix . showString "A"
              sliceString (SliceFixed' ix) = sliceString ix . showString "n"

prop_replicate
    :: (Elt e, Show e, Similar e, Show dim, Shape slice, FullShape ix ~ dim, SliceShape ix ~ slice, Elt ix, Slice ix, P.Eq dim)
    => TestContext
    -> (WhichData -> Gen e)
    -> SliceIndex' ix slice dim
    -> Property
prop_replicate tc e slix = property $ do
  fullsh <- forAll (Gen.filter (checkAll (P.>= 2) slix) (genSliceDim slix))
  let slsh = sliceOf slix fullsh
  dat <- forAll (generate_sample_data slsh fullsh e)
  xs  <- forAll (array ForInput slsh e)
  tpuTestCase tc (A.replicate (A.constant (slixOf slix fullsh))) dat xs

data SliceIndex' ix slice dim where
  SliceNil'   :: SliceIndex' A.Z A.Z A.Z
  SliceAll'   :: SliceIndex' ix slice dim -> SliceIndex' (ix A.:. A.All) (slice A.:. Int) (dim A.:. Int)
  SliceFixed' :: SliceIndex' ix slice dim -> SliceIndex' (ix A.:. Int)    slice           (dim A.:. Int)

checkAll :: (Int -> Bool) -> SliceIndex' ix slice dim -> dim -> Bool
checkAll _ SliceNil' _ = True
checkAll f (SliceAll' ix) (sh A.:. n) = f n P.&& checkAll f ix sh
checkAll f (SliceFixed' ix) (sh A.:. n) = f n P.&& checkAll f ix sh

genSliceDim :: SliceIndex' ix slice dim -> Gen dim
genSliceDim SliceNil' = pure A.Z
genSliceDim (SliceAll' ix) = (A.:.) <$> genSliceDim ix <*> Gen.int (Range.linear 1 10)
genSliceDim (SliceFixed' ix) = (A.:.) <$> genSliceDim ix <*> Gen.int (Range.linear 1 10)

slixOf :: SliceIndex' ix slice dim -> dim -> ix
slixOf SliceNil' A.Z = A.Z
slixOf (SliceAll' ix) (sh A.:. _) = slixOf ix sh A.:. A.All
slixOf (SliceFixed' ix) (sh A.:. n) = slixOf ix sh A.:. n

sliceOf :: SliceIndex' ix slice dim -> dim -> slice
sliceOf SliceNil' A.Z = A.Z
sliceOf (SliceAll' ix) (sh A.:. n) = sliceOf ix sh A.:. n
sliceOf (SliceFixed' ix) (sh A.:. _) = sliceOf ix sh

generate_sample_data
  :: (Elt e, Shape sh, Shape fullsh)
  => sh
  -> fullsh
  -> (WhichData -> Gen e)
  -> Gen (RepresentativeData (Array sh e -> Array fullsh e))
generate_sample_data sh fullsh e = do
  i  <- Gen.int (Range.linear 10 16)
  xs <- Gen.list (Range.singleton i) (array ForSample sh e)
  return [ x :-> Result fullsh | x <- xs ]

