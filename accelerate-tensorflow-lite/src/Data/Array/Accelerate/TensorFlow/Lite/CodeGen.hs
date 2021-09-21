{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen
  where

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor
import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Exp

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Array


buildPreOpenAcc
    :: forall aenv sh e.
       Val aenv
    -> OpenAcc aenv (Array sh e)
    -> Tensor sh e
buildPreOpenAcc aenv (OpenAcc pacc) =
  let
      buildA :: OpenAcc aenv (Array sh' e') -> Tensor sh' e'
      buildA = buildPreOpenAcc aenv
  in
  case pacc of
    Map tR f xs
      | Lam lhs (Body b)  <- f
      , xs'@(Tensor sh _) <- buildA xs
      -> buildOpenExp sh (Empty `push` (lhs, xs')) aenv b

