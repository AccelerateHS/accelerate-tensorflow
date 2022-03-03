{-# LANGUAGE GADTs #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Model
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Model
  where

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Trafo.Sharing

import Data.Array.Accelerate.TensorFlow.CodeGen.AST

import Data.Array.Accelerate.TensorFlow.Lite.Representation.Args
import Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes

import Data.ByteString                                                        ( ByteString )


data Model f where
  Model :: AfunctionRepr a f r
        -> ModelAfun r
        -> ByteString
        -> Model f

data ModelAfun f where
  Mbody :: ArraysR a -> Shapes a                     -> ModelAfun a
  Mlam  :: ALeftHandSide a aenv aenv' -> ModelAfun b -> ModelAfun (a -> b)

modelAfun
    :: AfunctionRepr a f r
    -> OpenTfun aenv r
    -> Args r
    -> ModelAfun r
modelAfun AfunctionReprBody       (Tbody aR _) (Aresult _ sh)  = Mbody aR sh
modelAfun (AfunctionReprLam lamR) (Tlam lhs f) (Aparam _ _ xs) = Mlam lhs (modelAfun lamR f xs)
modelAfun _ _ _ = error "impossible"

