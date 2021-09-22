{-# LANGUAGE NoImplicitPrelude #-}
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
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Sugar.Array

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen

import qualified TensorFlow.Core                                    as TF

import System.IO.Unsafe


run :: (Shape sh, Elt e)
    => Acc (Array sh e)
    -> Array sh e
run = Array
    . unsafePerformIO . TF.runSession . TF.run
    . buildAcc
    . convertAcc

