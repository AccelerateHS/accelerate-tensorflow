{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE PolyKinds           #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Lite.Model
-- Copyright   : [2021..2022] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Lite.Model (

  Model(..), encodeModel, decodeModel,
  ModelAfun(..), modelAfun,

) where

import Data.Array.Accelerate.AST
import Data.Array.Accelerate.AST.LeftHandSide                                 ( LeftHandSide(..), lhsToTupR )
import Data.Array.Accelerate.Analysis.Match
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Smart                                  as Smart
import qualified Data.Array.Accelerate.Sugar.Array                            as Sugar

import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Base

import Data.Array.Accelerate.TensorFlow.Lite.Representation.Args
import Data.Array.Accelerate.TensorFlow.Lite.Representation.Shapes

import Unsafe.Coerce
import Data.ByteString                                                        ( ByteString )
import Data.Serialize
import qualified Data.ByteString                                              as B


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


encodeModel :: Model f -> ByteString
encodeModel (Model _ f g) = runPut $ do
  putModelAfun f
  putInt64le (fromIntegral (B.length g))
  putByteString g

decodeModel :: forall f. Afunction f => ByteString -> Either String (Model (AfunctionR f))
decodeModel buffer =
  let fR    = afunctionRepr @f
      model = do
        f <- getModelAfun
        n <- getInt64le
        g <- getByteString (fromIntegral n)
        return (f, g)
  in
  case runGet model buffer of
    Left err            -> Left err
    Right (Exists f, g)
      | Just Refl <- checkModelAfun fR f -> Right (Model fR f g)
      | otherwise                        -> Left "Couldn't match expected type `TODO' with actual type `TODO'"

checkModelAfun :: AfunctionRepr a f r -> ModelAfun s -> Maybe (r :~: s)
checkModelAfun fR@AfunctionReprBody (Mbody aR _)
  | Just Refl <- checkAfunctionReprBody fR aR
  = Just Refl
checkModelAfun fR@(AfunctionReprLam lamR) (Mlam lhs f)
  | Just Refl <- checkAfunctionReprLam fR (lhsToTupR lhs)
  , Just Refl <- checkModelAfun lamR f
  = Just Refl
checkModelAfun _ _
  = Nothing

checkAfunctionReprBody
    :: forall a b. Sugar.Arrays a
    => AfunctionRepr (Smart.Acc a) a (Sugar.ArraysR a)
    -> ArraysR b
    -> Maybe (Sugar.ArraysR a :~: b)
checkAfunctionReprBody AfunctionReprBody bR
  | Just Refl <- matchArraysR (Sugar.arraysR @a) bR
  = Just Refl
checkAfunctionReprBody _ _
  = Nothing

checkAfunctionReprLam
    :: forall a b br breprr c. Sugar.Arrays a
    => AfunctionRepr (Smart.Acc a -> b) (a -> br) (Sugar.ArraysR a -> breprr)
    -> ArraysR c
    -> Maybe (Sugar.ArraysR a :~: c)
checkAfunctionReprLam AfunctionReprLam{} cR
  | Just Refl <- matchArraysR (Sugar.arraysR @a) cR
  = Just Refl
checkAfunctionReprLam _ _
  = Nothing


data Exists f where
  Exists :: f a -> Exists f

-- data Exists2 f where
--   Exists2 :: f a b -> Exists2 f

data Exists3 f where
  Exists3 :: f a b c -> Exists3 f


putModelAfun :: ModelAfun f -> Put
putModelAfun (Mbody aR sh) = putWord8 0 >> putArraysR aR >> putShapes aR sh
putModelAfun (Mlam lhs f)  = putWord8 1 >> putALeftHandSide lhs >> putModelAfun f

getModelAfun :: Get (Exists ModelAfun)
getModelAfun = do
  tag <- getWord8
  case tag of
    0 -> do
      Exists aR <- getArraysR
      sh        <- getShapes aR
      return $ Exists (Mbody aR sh)
    1 -> do
      Exists3 lhs <- getALeftHandSide
      Exists f    <- getModelAfun
      return $ Exists (Mlam lhs f)
    _ -> fail "invalid ModelAfun"

putShapes :: ArraysR a -> Shapes a -> Put
putShapes TupRunit                    ()     = return ()
putShapes (TupRpair aR bR)            (a, b) = putShapes aR a >> putShapes bR b
putShapes (TupRsingle (ArrayR shR _)) sh     = putShape shR sh

getShapes :: ArraysR a -> Get (Shapes a)
getShapes TupRunit                    = return ()
getShapes (TupRpair aR bR)            = (,) <$> getShapes aR <*> getShapes bR
getShapes (TupRsingle (ArrayR shR _)) = getShape shR

putShape :: ShapeR sh -> sh -> Put
putShape ShapeRz          ()      = return ()
putShape (ShapeRsnoc szR) (sz, s) = putInt64le (fromIntegral s) >> putShape szR sz

getShape :: ShapeR sh -> Get sh
getShape ShapeRz          = return ()
getShape (ShapeRsnoc szR) = do
  sh <- getInt64le
  sz <- getShape szR
  return (sz, fromIntegral sh)

putALeftHandSide :: ALeftHandSide a aenv aenv' -> Put
putALeftHandSide = putLeftHandSide putArrayR

getALeftHandSide :: Get (Exists3 ALeftHandSide)
getALeftHandSide = getLeftHandSide getArrayR

putLeftHandSide :: (forall u. s u -> Put) -> LeftHandSide s v env env' -> Put
putLeftHandSide f = \case
  LeftHandSideSingle s   -> putWord8 0 >> f s
  LeftHandSideWildcard t -> putWord8 1 >> putTupR f t
  LeftHandSidePair a b   -> putWord8 2 >> putLeftHandSide f a >> putLeftHandSide f b

getLeftHandSide :: Get (Exists s) -> Get (Exists3 (LeftHandSide s))
getLeftHandSide f = do
  tag <- getWord8
  case tag of
    0 -> do
      Exists s <- f
      return $ Exists3 (LeftHandSideSingle s)
    1 -> do
      Exists t <- getTupR f
      return $ Exists3 (LeftHandSideWildcard t)
    2 -> do
      Exists3 a <- getLeftHandSide f
      Exists3 b <- getLeftHandSide f
      return $ Exists3 (LeftHandSidePair a (unsafeCoerce b)) -- TLM: what the heck? Providing either 'a' or 'b' is fine, but not both?
    _ -> fail "invalid LeftHandSide"

putArraysR :: ArraysR a -> Put
putArraysR = putTupR putArrayR

getArraysR :: Get (Exists ArraysR)
getArraysR = getTupR getArrayR

putArrayR :: ArrayR a -> Put
putArrayR (ArrayR shR eR) = putShapeR shR >> putTypeR eR

getArrayR :: Get (Exists ArrayR)
getArrayR = do
  Exists shR <- getShapeR
  Exists eR  <- getTypeR
  return $ Exists (ArrayR shR eR)

putShapeR :: ShapeR sh -> Put
putShapeR ShapeRz         = putWord8 0
putShapeR (ShapeRsnoc sh) = putWord8 1 >> putShapeR sh

getShapeR :: Get (Exists ShapeR)
getShapeR = do
  tag <- getWord8
  case tag of
    0 -> return $ Exists ShapeRz
    1 -> do
      Exists sh <- getShapeR
      return $ Exists (ShapeRsnoc sh)
    _ -> fail "invalid ShapeR"

putTypeR :: Putter (TypeR a)
putTypeR = putTupR putScalarType

getTypeR :: Get (Exists TypeR)
getTypeR = getTupR getScalarType

putTupR :: (forall t. s t -> Put) -> TupR s a -> Put
putTupR f = \case
  TupRunit     -> putWord8 0
  TupRsingle s -> putWord8 1 >> f s
  TupRpair a b -> putWord8 2 >> putTupR f a >> putTupR f b

getTupR :: Get (Exists s) -> Get (Exists (TupR s))
getTupR f = do
  tag <- getWord8
  case tag of
    0 -> return (Exists TupRunit)
    1 -> do
      Exists s <- f
      return $ Exists (TupRsingle s)
    2 -> do
      Exists a <- getTupR f
      Exists b <- getTupR f
      return $ Exists (TupRpair a b)
    _ -> fail "invalid TupR"

putScalarType :: ScalarType t -> Put
putScalarType (SingleScalarType t) = putWord8 0 >> putSingleType t
putScalarType VectorScalarType{}   = unsupported "SIMD-vector types"

getScalarType :: Get (Exists ScalarType)
getScalarType = do
  tag <- getWord8
  case tag of
    0 -> do
      Exists t <- getSingleType
      return $ Exists (SingleScalarType t)
    _ -> fail "invalid ScalarType"

putSingleType :: SingleType t -> Put
putSingleType (NumSingleType t) = putNumType t

getSingleType :: Get (Exists SingleType)
getSingleType = do
  Exists t <- getNumType
  return $ Exists (NumSingleType t)

putNumType :: NumType t -> Put
putNumType (IntegralNumType t) = putWord8 0 >> putIntegralType t
putNumType (FloatingNumType t) = putWord8 1 >> putFloatingType t

getNumType :: Get (Exists NumType)
getNumType = do
  tag <- getWord8
  case tag of
    0 -> do
      Exists t <- getIntegralType
      return $ Exists (IntegralNumType t)
    1 -> do
      Exists t <- getFloatingType
      return $ Exists (FloatingNumType t)
    _ -> fail "invalid NumType"

putIntegralType :: IntegralType t -> Put
putIntegralType TypeInt    = putWord8 0
putIntegralType TypeInt8   = putWord8 1
putIntegralType TypeInt16  = putWord8 2
putIntegralType TypeInt32  = putWord8 3
putIntegralType TypeInt64  = putWord8 4
putIntegralType TypeWord   = putWord8 5
putIntegralType TypeWord8  = putWord8 6
putIntegralType TypeWord16 = putWord8 7
putIntegralType TypeWord32 = putWord8 8
putIntegralType TypeWord64 = putWord8 9

getIntegralType :: Get (Exists IntegralType)
getIntegralType = do
  tag <- getWord8
  case tag of
    0 -> return $ Exists TypeInt
    1 -> return $ Exists TypeInt8
    2 -> return $ Exists TypeInt16
    3 -> return $ Exists TypeInt32
    4 -> return $ Exists TypeInt64
    5 -> return $ Exists TypeWord
    6 -> return $ Exists TypeWord8
    7 -> return $ Exists TypeWord16
    8 -> return $ Exists TypeWord32
    9 -> return $ Exists TypeWord64
    _ -> fail "invalid IntegralType"

putFloatingType :: FloatingType t -> Put
putFloatingType TypeHalf   = putWord8 0
putFloatingType TypeFloat  = putWord8 1
putFloatingType TypeDouble = putWord8 2

getFloatingType :: Get (Exists FloatingType)
getFloatingType = do
  tag <- getWord8
  case tag of
    0 -> return $ Exists TypeHalf
    1 -> return $ Exists TypeFloat
    2 -> return $ Exists TypeDouble
    _ -> fail "invalid FloatingType"

