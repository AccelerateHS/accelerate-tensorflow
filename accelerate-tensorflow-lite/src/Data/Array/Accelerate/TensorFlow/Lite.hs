{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
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

import Data.Array.Accelerate                                        ( Acc )
import Data.Array.Accelerate.Sugar.Array
import Data.Array.Accelerate.Trafo.Sharing
import qualified Data.Array.Accelerate.Representation.Array         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.TensorFlow.Lite.Compile

import Text.Printf


compileAfun :: Afunction f => f -> IO FilePath
compileAfun
  = compileTfun
  . buildAfun
  . convertAfun

-- saveAcc :: forall arrs. Arrays arrs => Acc arrs -> IO ()
-- saveAcc acc =
--   let model = buildAcc (convertAcc acc)
--    in save_model (arraysR @arrs) model

-- saveAfun :: forall f. Afunction f => f -> IO ()
-- saveAfun acc =
--   let model = buildAfun (convertAfun acc)

--       go :: AfunctionRepr g (AfunctionR g) (ArraysFunctionR g)
--          -> OpenTfun aenv (ArraysFunctionR g)
--          -> IO ()
--       go AfunctionReprBody       (Tbody bR b) = save_model bR b
--       go (AfunctionReprLam lamR) (Tlam _ f)   = go lamR f
--       go _                       _            = error "impossible"
--   in
--   go (afunctionRepr @f) model

-- save_model :: R.ArraysR arrs -> Tensors arrs -> IO ()
-- save_model bR b = do
--   let graph = graph_of_model bR b
--   tflite <- convert_to_tflite graph
--   printf "path: %s\n" tflite

