{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.Test.NoFib.Misc
-- Copyright   : [2023] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.Test.NoFib.Misc
  where

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


test_misc :: TestTree
test_misc =
  testGroup "misc"
    [ test_arguments
    ]

test_arguments :: TestTree
test_arguments =
  testGroup "arguments"
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
          [ testProperty "missing" $ prop_missing_args dim f32
          ]

prop_missing_args
    :: (P.Eq sh, Show sh, Shape sh, Elt e, Show e, Similar e, A.Num e)
    => Gen sh
    -> (WhichData -> Gen e)
    -> Property
prop_missing_args dim e =
  property $ do
    sh   <- forAll dim

    let genInput wd = do
          xss' <- Gen.list (Range.singleton 7) $ array wd sh e
          let [arr1, arr2, arr3, arr4, arr5, arr6, arr7] = P.map indexArray xss'

          let arg1 = fromFunction sh (\i -> (arr1 i, (arr2 i, arr3 i)))
              arg2 = fromFunction sh (\i -> arr4 i)
              arg3 = fromFunction sh (\i -> (arr5 i, arr6 i))
              arg4 = fromFunction sh (\i -> arr7 i)
          return (arg1, arg2, arg3, arg4)

    numsamples <- forAll (Gen.int (Range.linear 10 16))
    samples <- forAll (Gen.list (Range.singleton numsamples) (genInput ForSample))
    let makeReprArgs (dat1, dat2, dat3, dat4) =
          dat1 :-> dat2 :-> dat3 :-> dat4 :-> Result sh
    let dat = P.map makeReprArgs samples

    (inp1, inp2, inp3, inp4) <- forAll (genInput ForInput)

    -- which inputs will be used?
    mask <- forAll (Gen.list (Range.singleton 7) Gen.bool)

    let f = \(T2 x1 (T2 x2 x3)) x4 (T2 x5 x6) x7 ->
              let values = [x1, x2, x3, x4, x5, x6, x7]
              in P.sum $ P.zipWith (\b x -> if b then x else 0) mask values

    let !ref = I.runN (A.zipWith4 f)
        !tpu = TPU.compile (A.zipWith4 f) dat

    TPU.execute tpu inp1 inp2 inp3 inp4 ~~~ ref inp1 inp2 inp3 inp4
