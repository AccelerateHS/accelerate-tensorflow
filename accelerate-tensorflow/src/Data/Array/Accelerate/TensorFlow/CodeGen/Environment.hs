{-# LANGUAGE GADTs #-}
{-# OPTIONS_HADDOCK hide #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Environment
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Environment
  where

import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Representation.Array


-- Valuation for an environment
--
data Val env where
  Empty :: Val ()
  Push  :: Val env -> TensorArrayData t -> Val (env, t)

data Aval aenv where
  Aempty :: Aval ()
  Apush  :: Aval aenv -> Tensors t -> Aval (aenv, t)

-- Push a set of variables into an environment
--
push :: Val env -> (ELeftHandSide t env env', TensorArrayData t) -> Val env'
push env (LeftHandSideWildcard _, _     ) = env
push env (LeftHandSideSingle _  , a     ) = env `Push` a
push env (LeftHandSidePair l1 l2, (a, b)) = push env (l1, a) `push` (l2, b)

apush :: Aval env -> (ALeftHandSide t env env', Tensors t) -> Aval env'
apush env (LeftHandSideWildcard _, _     ) = env
apush env (LeftHandSideSingle _  , a     ) = env `Apush` a
apush env (LeftHandSidePair l1 l2, (a, b)) = apush env (l1, a) `apush` (l2, b)

-- Projection of a value from a valuation using a de Bruijn index
--
prj :: Idx env t -> Val env -> TensorArrayData t
prj ZeroIdx       (Push _   v) = v
prj (SuccIdx idx) (Push val _) = prj idx val

aprj :: Idx env (Array sh e) -> Aval env -> Tensor sh e
aprj ZeroIdx       (Apush _   v) = v
aprj (SuccIdx idx) (Apush val _) = aprj idx val

