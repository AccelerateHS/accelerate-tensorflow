{-# LANGUAGE GADTs #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Environment
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Environment
  where

import Data.Array.Accelerate.TensorFlow.Lite.CodeGen.Tensor

import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.AST.LeftHandSide


-- Valuation for an environment
--
data Val env where
  Empty :: Val ()
  Push  :: Val env -> TensorArrayData t -> Val (env, t)

-- Push a set of variables into an environment
--
push :: Val env -> (LeftHandSide s t env env', TensorArrayData t) -> Val env'
push env (LeftHandSideWildcard _, _     ) = env
push env (LeftHandSideSingle _  , a     ) = env `Push` a
push env (LeftHandSidePair l1 l2, (a, b)) = push env (l1, a) `push` (l2, b)

-- Projection of a value from a valuation using a de Bruijn index
--
prj :: Idx env t -> Val env -> TensorArrayData t
prj ZeroIdx       (Push _   v) = v
prj (SuccIdx idx) (Push val _) = prj idx val

