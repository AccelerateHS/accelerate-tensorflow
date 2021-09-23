{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE NoImplicitPrelude   #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite
  where

import Data.Array.Accelerate
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Representation.Type
import qualified Data.Array.Accelerate.Representation.Array         as R

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor

import qualified TensorFlow.Core                                    as TF

import System.IO.Unsafe


run :: forall arrs. Arrays arrs => Acc arrs -> arrs
run | Fetchable <- fetchable @arrs
    = toArr
    . unsafePerformIO . TF.runSession . TF.run
    . buildAcc
    . convertAcc


data Fetchable t where
  Fetchable :: TF.Fetchable (Tensors t) t => Fetchable t

fetchable :: forall arrs. Arrays arrs => Fetchable (ArraysR arrs)
fetchable =
  let go :: R.ArraysR a -> Fetchable a
      go TupRunit                = Fetchable
      go (TupRsingle R.ArrayR{}) = Fetchable
      go (TupRpair aR bR)
        | Fetchable <- go aR
        , Fetchable <- go bR
        = Fetchable
  in
  go (arraysR @arrs)

