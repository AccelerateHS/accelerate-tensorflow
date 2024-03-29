{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns        #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE LambdaCase          #-}
{-# LANGUAGE OverloadedStrings   #-}
{-# LANGUAGE PatternSynonyms     #-}
{-# LANGUAGE RankNTypes          #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
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
  runN,

  argMin, argMax, append

) where

import Data.Array.Accelerate                                        ( Acc, Exp, Shape, Array, (:.)(..)
                                                                    , pattern T2, pattern (::.) )
import qualified Data.Array.Accelerate                              as A
import Data.Array.Accelerate.AST                                    ( ALeftHandSide )
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Array.Data
import Data.Array.Accelerate.Array.Unique
import Data.Array.Accelerate.Lifetime
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Sugar.Array                            ( Arrays(..), ArraysR(..) )
import Data.Array.Accelerate.Trafo.Sharing
import Data.Array.Accelerate.Trafo.Simplify
import Data.Array.Accelerate.Type
import qualified Data.Array.Accelerate.Representation.Array         as R
import qualified Data.Array.Accelerate.Representation.Shape         as R

import Data.Array.Accelerate.TensorFlow.CodeGen
import Data.Array.Accelerate.TensorFlow.CodeGen.AST
import Data.Array.Accelerate.TensorFlow.CodeGen.Tensor
import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Tensor.Shim as Sh
import Data.Array.Accelerate.TensorFlow.TypeDicts
import Data.Array.Accelerate.TensorFlow.CodeGen.Foreign

import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields     as TF
import qualified TensorFlow.Build                                   as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Internal.FFI                            as Internal

import Control.Monad.State
import Data.Functor.Identity
import qualified Data.Set                                           as Set
import Foreign.ForeignPtr
import Foreign.Storable
import Lens.Family2 (view)
import System.Environment                                           ( lookupEnv )
import System.IO.Unsafe
import Text.Printf
import qualified Data.Text                                          as T
import qualified Data.Vector.Storable                               as V

import qualified Data.Array.Accelerate.TensorFlow.CodeGen.Arithmetic as Ar

-- | Run a complete embedded program using the default TensorFlow backend
--
run :: forall arrs. Arrays arrs => Acc arrs -> arrs
run | FetchableDict <- fetchableDict @arrs
    = toArr
    . unsafePerformIO . TF.runSession . TF.run
    . debugPrintTFGraph (arraysR @arrs)
    . buildAcc
    . simplifyAcc
    . convertAcc

{-# NOINLINE debugPrintTFGraph #-}
debugPrintTFGraph :: R.ArraysR a -> Tensors a -> Tensors a
debugPrintTFGraph = \rep tenss -> unsafePerformIO $ do
  lookupEnv "ACCELERATE_TF_PRINT_TFGRAPH" >>= \case
    Just val | not (null val) -> do
      let graphs = go rep tenss
      putStrLn $ "Rendered TF graphs (" ++ show (length graphs) ++ "):"
      forM_ graphs $ \s -> putStrLn $ "- " ++ s
      return tenss
    _ -> return tenss
  where
    go :: R.ArraysR a -> Tensors a -> [String]
    go (TupRsingle (R.ArrayR shR eR)) (Tensor _ sht at) =
      go2 (shapeType shR) sht ++ go2 eR at
    go TupRunit () = []
    go (TupRpair r1 r2) (x, y) = go r1 x ++ go r2 y

    go2 :: TypeR a -> TArrayDataR Sh.Tensor a -> [String]
    go2 (TupRsingle sR) tens = buildTypeDictsScalar sR $ [Sh.dotGraph (Sh.tensorGraph tens)]
    go2 TupRunit () = []
    go2 (TupRpair r1 r2) (x, y) = go2 r1 x ++ go2 r2 y

-- | Prepare an embedded array program for execution on the default
-- TensorFlow backend
--
runN :: forall f. Afunction f => f -> AfunctionR f
runN acc =
  let
      !model = buildAfun afun
      !afun  = simplifyAfun
             . convertAfun
             $ acc

      nodeNames = modelNodeNames model
      actualInputs = Set.fromList $ filter ("input" `T.isPrefixOf`) nodeNames

      eval :: forall aenvtop g. AfunctionRepr g (AfunctionR g) (ArraysFunctionR g)
           -> OpenTfun aenvtop (ArraysFunctionR g)
           -> Int
           -> [TF.Feed]
           -> AfunctionR g
      eval AfunctionReprBody (Tbody bR b) _ aenv
        | FetchableDict <- fetchableDictR bR
        = toArr
        . unsafePerformIO
        . TF.runSession
        . TF.runWithFeeds aenv
        . debugPrintTFGraph (arraysR @(AfunctionR g))
        $ b
      eval (AfunctionReprLam lamR) (Tlam lhs f) skip aenv = \arr ->
        let
            go :: ALeftHandSide t aenv aenv' -> t -> [TF.Feed] -> State Int [TF.Feed]
            go LeftHandSideWildcard{}                   _                  env = return env
            go (LeftHandSidePair aR bR)                 (a, b)             env = go bR b =<< go aR a env
            go (LeftHandSideSingle (R.ArrayR _shR _eR)) (R.Array _sh _adata) env = state $ \i ->
              let
                  sh'    = evalState (shape _shR _sh) 0
                  adata' = evalState (array _eR _adata) 0

                  shape :: ShapeR sh -> sh -> State Int [TF.Feed]
                  shape ShapeRz          ()     = return []
                  shape (ShapeRsnoc shR) (t, h) = do
                    j <- get
                    modify (+1)

                    t' <- shape shR t

                    let name = T.pack (printf "input%d_shape%d" i j)
                    return $
                      if name `Set.member` actualInputs
                        then let tensorDataBytes = V.fromListN 1 [fromIntegral h :: ScalarTensorDataR Int]
                                 tensorDataShape = TF.Shape []
                             in TF.feed (TF.tensorValueFromName name)
                                        (TF.encodeTensorData tensorDataShape tensorDataBytes)
                                : t'
                        else t'

                  array :: TypeR t -> ArrayData t -> State Int [TF.Feed]
                  array TupRunit         ()     = return []
                  array (TupRsingle aR)  a      = buildTypeDictsScalar aR feed a
                  array (TupRpair aR bR) (a, b) = do
                    a' <- array aR a
                    b' <- array bR b
                    return (a' ++ b')

                  feed :: forall t s. (Storable t, TF.TensorType s, s ~ ScalarTensorDataR t) => UniqueArray t -> State Int [TF.Feed]
                  feed ua = do
                    j <- get
                    modify (+1)

                    let name = T.pack (printf "input%d_adata%d" i j)
                    return $
                      if name `Set.member` actualInputs
                        then let fp                   = unsafeGetValue (uniqueArrayData ua)
                                 tensorDataBytes      = V.unsafeFromForeignPtr0 (castForeignPtr fp :: ForeignPtr Word8) (R.size _shR _sh * sizeOf (undefined :: t))
                                 tensorDataType       = TF.tensorType (undefined :: s)
                                 tensorDataDimensions = [ fromIntegral x :: ScalarTensorDataR Int | x <- reverse (R.shapeToList _shR _sh) ]
                             in [TF.feed (TF.tensorValueFromName name)
                                         (TF.TensorData (Internal.TensorData {..}))]
                        else []
              in
              (sh' ++ adata' ++ env, i+1)

            (aenv', next) = runState (go lhs (fromArr arr) aenv) skip
        in
        eval lamR f next aenv'
      eval _ _ _ _ = error "impossible"
  in
  eval (afunctionRepr @f) model 0 []

modelNodeNames :: OpenTfun aenv t -> [T.Text]
modelNodeNames (Tlam _ f)         = modelNodeNames f
modelNodeNames (Tbody arrR model) =
  let
      go :: TF.MonadBuild m => R.ArraysR a -> Tensors a -> m ()
      go TupRunit                ()                               = return ()
      go (TupRpair aR bR)        (a, b)                           = go aR a >> go bR b
      go (TupRsingle R.ArrayR{}) (Tensor (R.ArrayR shR aR) sht t) = array (shapeType shR) sht >> array aR t

      array :: TF.MonadBuild m => TypeR t -> TensorArrayData t -> m ()
      array TupRunit         ()     = return ()
      array (TupRpair aR bR) (a, b) = array aR a >> array bR b
      array (TupRsingle aR)  a      = buildTypeDictsScalar aR $
                                        TF.render (Sh.unwrap a) >> return ()
  in runIdentity $ TF.evalBuildT $ do
       go arrR model
       map (view TF.name) <$> TF.flushNodeBuffer


data FetchableDict t where
  FetchableDict :: TF.Fetchable (Tensors t) t => FetchableDict t

fetchableDict :: forall arrs. Arrays arrs => FetchableDict (ArraysR arrs)
fetchableDict = fetchableDictR (arraysR @arrs)

fetchableDictR :: R.ArraysR a -> FetchableDict a
fetchableDictR TupRunit                = FetchableDict
fetchableDictR (TupRsingle R.ArrayR{}) = FetchableDict
fetchableDictR (TupRpair aR bR)
  | FetchableDict <- fetchableDictR aR
  , FetchableDict <- fetchableDictR bR
  = FetchableDict

-- TODO: put the fallback implementation in Accelerate's Prelude as "argMin" and "argMax", and export "argMinTPU","argMaxTPU" with ForeignAcc from here
argMinMax :: (A.Ord a, Shape sh) => MinMax -> Acc (Array (sh :. Int) a) -> Acc (Array sh (Int32, a))
argMinMax minMax xs = let ys = argMinMax' xs
                        in A.imap (\ix y -> T2 y (xs A.! (ix ::. (A.fromIntegral y)))) ys
  where
    argMinMax' :: (Shape sh, A.Ord a) => Acc (Array (sh :. Int) a) -> Acc (Array sh Int32)
    argMinMax' = A.foreignAcc
      (ForeignAcc "argminmax" $ tpuArgMinMax minMax) --TPU
      (   A.map (\(T2 (_ ::. i) _) -> A.fromIntegral i) 
        . A.fold1 (\(T2 a x) (T2 b y) -> A.ifThenElse (x `minOrMax` y) (T2 a x) (T2 b y))
        . A.imap T2) -- fallback (interpreter/CPU/GPU)
    minOrMax :: A.Ord a => Exp a -> Exp a -> Exp Bool
    minOrMax = case minMax of
      Min -> (A.<)
      Max -> (A.>)

argMin, argMax :: (A.Ord a, Shape sh) => Acc (Array (sh :. Int) a) -> Acc (Array sh (Int32, a))
argMin = argMinMax Min
argMax = argMinMax Max

-- Specialised instance, because naive translation through `generate` results in a 'select' over two generates that both include out-of-bounds indexing.
append :: (Shape sh, A.Elt e) => Acc (Array (sh :. Int) e) -> Acc (Array (sh :. Int) e) -> Acc (Array (sh :. Int) e)
append xs ys = A.foreignAcc
  (ForeignAcc "append" tensorflowappend)
  (\(T2 xs' ys') -> xs' A.++ ys')
  (backpermuteToSameSize $ T2 xs ys)
  where
    tensorflowappend :: (((), Tensor (sh,Int) e), Tensor (sh,Int) e) -> Tensor (sh,Int) e
    tensorflowappend (((), Tensor (R.ArrayR (ShapeRsnoc shR) eR) (sh,szl) l)
                         , Tensor (R.ArrayR _                _ ) (_ ,szr) r) = 
                           Tensor (R.ArrayR (ShapeRsnoc shR) eR) (sh, szl+szr) (go eR l r)
      where
        go :: TypeR e -> TensorArrayData e -> TensorArrayData e -> TensorArrayData e
        go TupRunit () () = ()
        go (TupRpair l r) (l1,r1) (l2,r2) = (go l l1 l2, go r r1 r2)
        go (TupRsingle t) x y = buildTypeDictsScalar t $ Sh.wrapConcat (fromIntegral $ rank shR) [x, y]
        -- zipmin :: forall e. TypeR e -> TensorArrayData e -> TensorArrayData e -> TensorArrayData e
        -- zipmin TupRunit () () = ()
        -- zipmin (TupRpair l r) (l1,r1) (l2,r2) = (zipmin l l1 l2, zipmin r r1 r2)
        -- zipmin (TupRsingle t) x y = buildTypeDictsScalar t $ Ar.min (singleType @e) (x, y)

    backpermuteToSameSize :: (Shape sh, A.Elt e) => Acc (Array (sh:.Int) e, Array (sh:.Int) e) -> Acc (Array (sh:.Int) e, Array (sh:.Int) e)
    backpermuteToSameSize (T2 xs ys) = T2 xs' ys'
      where
        xs' = A.backpermute (sh::.szx) id xs
        ys' = A.backpermute (sh::.szy) id ys
        shx ::. szx = A.shape xs
        shy ::. szy = A.shape ys
        sh = A.intersect shx shy
