{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim
-- Copyright   : [2023] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim (
  -- * Tensor wrapper
  Tensor,
  wrap, wrap1, wrap2,
  unwrap,

  -- * Wrapped operations
  wrapConcat,
  wrapPack,

  -- * Inspecting the collected graph
  tensorGraph,
  -- tensorHead, tensorHead',
  dotGraph,
  SomeTensor,
) where

import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim.IdGen

import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Ops                                     as TF

import Data.Complex                                                 ( Complex )
import Data.Foldable
import Data.Int
import Data.Map.Strict                                              ( Map )
import qualified Data.Map.Strict                                    as Map
import Data.Typeable


data Tsil a = Nil | Snoc (Tsil a) a
  deriving (Show, Eq, Ord, Functor, Foldable, Traversable)

data Argument where
  TensArg  :: (Typeable e, Show e) => Tensor e -> Argument
  OtherArg :: (Typeable a, Show a) => a        -> Argument
deriving instance Show Argument

data Tensor e = Tensor
  { tTens :: TF.Tensor TF.Build e
  , tId :: Int
  , _tFun :: String
  , tArgs :: Tsil Argument }

instance Show (Tensor e) where
  showsPrec p (Tensor _ i fun args) = showParen (p > 10) $
    showString ("Tensor <TF.Tensor> " ++ show i ++ " " ++ show fun ++ " ") .
    showsPrec 11 (toList args)

instance (TF.TensorType e, Num e, Typeable e, Show e
         ,TF.OneOf '[Double, Float, Int32, Int64, Complex Float, Complex Double] e)
    => Num (Tensor e) where
  -- right-hand sides taken from the Num instance for TF.Tensor in tensorflow-ops:TensorFlow.Ops
  (+) = wrap "add" TF.add
  (*) = wrap "mul" TF.mul
  (-) = wrap "sub" TF.sub
  abs = wrap "abs" TF.abs
  fromInteger = wrap1 "scalar" TF.scalar . fromInteger
  signum = wrap "sign" TF.sign
  negate = wrap "neg" TF.neg

unwrap :: Tensor e -> TF.Tensor TF.Build e
unwrap = tTens

wrap :: ShimTensFun f
     => String  -- ^ function name
     -> f -> ShimmedTensFun f
wrap name = wrap' name Nil

wrap1 :: (Typeable a, Show a, ShimTensFun f) => String -> (a -> f) -> a -> ShimmedTensFun f
wrap1 name f arg = wrap' name (Nil `Snoc` OtherArg arg) (f arg)

wrap2 :: (Typeable a1, Show a1, Typeable a2, Show a2, ShimTensFun f)
      => String -> (a1 -> a2 -> f) -> a1 -> a2 -> ShimmedTensFun f
wrap2 name f arg1 arg2 = wrap' name (Nil `Snoc` OtherArg arg1 `Snoc` OtherArg arg2) (f arg1 arg2)

class ShimTensFun f where
  type ShimmedTensFun f

  wrap' :: String -> Tsil Argument -> f -> ShimmedTensFun f

instance ShimTensFun (TF.Tensor TF.Build e) where
  type ShimmedTensFun (TF.Tensor TF.Build e) = Tensor e
  wrap' name args tens = Tensor tens (genId tens) name args

instance (Typeable e, Show e, t ~ TF.Tensor TF.Build, ShimTensFun f) => ShimTensFun (t e -> f) where
  type ShimmedTensFun (t e -> f) = Tensor e -> ShimmedTensFun f
  wrap' name args f = \wtens -> wrap' name (args `Snoc` TensArg wtens) (f (tTens wtens))

wrapConcat :: (Typeable t, Show t, TF.TensorType t) => Tensor Int32 -> [Tensor t] -> Tensor t
wrapConcat axis l =
  let tens = TF.concat (unwrap axis) (map unwrap l)
  in Tensor tens
            (genId tens)
            "concat"
            (foldl' Snoc (Nil `Snoc` TensArg axis) (map TensArg l))

wrapPack :: (Typeable t, Show t, TF.TensorType t) => [Tensor t] -> Tensor t
wrapPack l =
  let tens = TF.pack (map unwrap l)
  in Tensor tens
            (genId tens)
            "pack"
            (foldl' Snoc Nil (map TensArg l))

data SomeTensor = forall a. SomeTensor (Tensor a)
deriving instance Show SomeTensor

-- -- | Format a tensor, inserting only info about the operation itself and a
-- -- brief summary of its children. Suitable to display about every tensor in a
-- -- graph.
-- tensorHead :: Tensor a -> String
-- tensorHead (Tensor _ i fun args) =
--   "<" ++ show i ++ "> " ++ fun
--   ++ (case args of Nil -> "" ; _ -> ":")
--   ++ concat [(' ' :) $ case arg of
--                          OtherArg x -> showsPrec 11 x ""
--                          TensArg (Tensor _ j fun' _) -> "(<" ++ show j ++ "> " ++ fun' ++ ")"
--             | arg <- toList args]

-- -- | @tensorHead' (SomeTensor tens) = 'tensorHead' tens@
-- tensorHead' :: SomeTensor -> String
-- tensorHead' (SomeTensor t) = tensorHead t

dotGraph :: Map Int SomeTensor -> String
dotGraph mp =
  concat ["digraph G {"
         ,concatMap genNode nodes
         ,concatMap genEdges nodes
         ,"}"]
  where
    nodes = Map.elems mp
    genNode (SomeTensor (Tensor _ i fun args)) = show i ++ " [label=\"" ++ lab ++ "\"];"
      where lab = "(" ++ show i ++ ") " ++ fun ++ concatMap fmtArg args
            fmtArg (OtherArg x) = " " ++ concatMap escape (showsPrec 11 x "")
              where escape '"' = "\\\""
                    escape '\\' = "\\\\"
                    escape c = [c]
            fmtArg (TensArg (Tensor _ j _ _)) = " <" ++ show j ++ ">"
    genEdges (SomeTensor (Tensor _ i _ args)) = concat $ zipWith fmtArg (toList args) indices
      where fmtArg OtherArg{} _ = ""
            fmtArg (TensArg (Tensor _ j _ _)) 0 = show i ++ " -> " ++ show j ++ " [color=\"red\"];"
            fmtArg (TensArg (Tensor _ j _ _)) _ = show i ++ " -> " ++ show j ++ ";"
            isTens OtherArg{} = False ; isTens TensArg{} = True
            indices = scanl (+) 0 (map (fromEnum . isTens) (toList args))

tensorGraph :: Tensor a -> Map Int SomeTensor
tensorGraph = go mempty
  where
    go :: Map Int SomeTensor -> Tensor a -> Map Int SomeTensor
    go mp tens = case Map.lookup (tId tens) mp of
      Just _ -> mp
      Nothing ->
        let process (TensArg t) m = go m t
            process (OtherArg _) m = m
        in foldr (.) id (map process (toList (tArgs tens))) (Map.insert (tId tens) (SomeTensor tens) mp)
