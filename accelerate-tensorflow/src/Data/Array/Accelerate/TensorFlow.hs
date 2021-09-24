{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow (

  run,

) where

import Data.Array.Accelerate                                        ( Acc )
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import qualified Data.Array.Accelerate.Representation.Array         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import qualified TensorFlow.Core                                    as TF

import System.IO.Unsafe


-- | Run a complete embedded program using the default TensorFlow backend
--
run :: forall arrs. Arrays arrs => Acc arrs -> arrs
run | FetchableDict <- fetchableDict @arrs
    = toArr
    . unsafePerformIO . TF.runSession . TF.run
    . buildAcc
    . convertAcc

data FetchableDict t where
  FetchableDict :: TF.Fetchable (Tensors t) t => FetchableDict t

fetchableDict :: forall arrs. Arrays arrs => FetchableDict (ArraysR arrs)
fetchableDict =
  let go :: R.ArraysR a -> FetchableDict a
      go TupRunit                = FetchableDict
      go (TupRsingle R.ArrayR{}) = FetchableDict
      go (TupRpair aR bR)
        | FetchableDict <- go aR
        , FetchableDict <- go bR
        = FetchableDict
  in
  go (arraysR @arrs)

