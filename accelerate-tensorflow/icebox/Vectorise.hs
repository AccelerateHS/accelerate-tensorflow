{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE GADTs                #-}
{-# LANGUAGE LambdaCase           #-}
{-# LANGUAGE RankNTypes           #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE StandaloneDeriving   #-}
{-# LANGUAGE TypeApplications     #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE UndecidableInstances #-}
-- |
-- Module      : Data.Array.Accelerate.TensorFlow.Vectorise
-- Copyright   : [2021] The Accelerate Team
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <trevor.mcdonell@gmail.com>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.TensorFlow.Vectorise
  where

import Data.Array.Accelerate.AST                                    hiding ( liftPreOpenAcc )
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.AST.Var
import Data.Array.Accelerate.Pretty                                 ()
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Shape                   hiding ( size )
import Data.Array.Accelerate.Representation.Slice
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Sugar.Elt
import Data.Array.Accelerate.Trafo.Substitution
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.Type

import Prelude                                                      hiding ( replicate )

-- import qualified Data.Array.Accelerate as A
-- import qualified Data.Array.Accelerate.Trafo.Sharing as A

-- xs :: A.Acc (A.Vector Float)
-- xs = A.use $ A.fromList (A.Z A.:. 10) [0..]

-- foo :: A.Acc (A.Vector Float) -> A.Acc (A.Vector Float)
-- foo = A.map (+1)
--     . A.map (*2)
--     . A.map (\x -> sqrt (1-x) / pi)

-- foo' :: Acc (Vector Float)
-- foo' = A.convertAfun foo
-- foo' = A.convertAcc (foo xs)

-- -- fooL = let sz = A.size xs
-- --         in r  = foo xs



-- The size parameter in the lifting transform.
--
type Size aenv = Exp aenv Int

-- type family Extend env t where
--   Extend env ()     = env
--   Extend env (a, b) = (Extend (Extend env a) b)
--   Extend env a      = (env, a)

-- Encodes the relationship between the old environments and the new
-- environments during the lifting transform
--
data Context env env' aenv aenv' where
  -- Base context
  EmptyC    :: Context env env aenv aenv

  -- -- A scalar expression
  -- PushExpC  :: TypeR e
  --           -> Context env env' aenv aenv'
  --           -> Context (Extend env e) (Extend env' e) aenv aenv'

  -- -- An array computation
  -- PushAccC  :: ArraysR a
  --           -> Context env env' aenv aenv'
  --           -> Context env env' (Extend aenv a) (Extend aenv' a)

  -- A lifted scalar expression
  PushLExpC :: ELeftHandSide e env env_e
            -> Context env   env' aenv aenv'
            -> Context env_e env' aenv (aenv', Vector e)

  -- -- A lifted array computation
  -- PushLAccC :: ArraysR a
  --           -> Context env env' aenv aenv'
  --           -> Context env env' (Extend aenv a) (Extend aenv' (Segmented a))


weakenWithContext :: Context env env' aenv aenv' -> aenv :> aenv'
weakenWithContext EmptyC          = weakenId
-- weakenWithContext (PushExpC _ c)  = weakenWithContext c
-- weakenWethContext (PushAccC aR c) =
weakenWithContext (PushLExpC _ c) = weakenSucc' (weakenWithContext c) -- weakenWithLHS alhs .> weakenWithContext c

data Strength
  = Aggressive
  | Conservative
  | HoistOnly
  | Nested
  deriving Show

data Lifted acc aenv t
    = Avoided (acc aenv t)
    | Lifted  (acc aenv (Segmented t))

deriving instance Show (LiftedOpenAcc () t)

type LiftedOpenAcc aenv t = Lifted OpenAcc aenv t

type family Segmented t where
  Segmented ()           = Scalar Int
  Segmented (Array sh e) = (Vector sh, Vector e)
  Segmented (a, b)       = (Segmented a, Segmented b)


type LiftAcc acc
    = forall aenv aenvL t.
      Strength
   -> Context () () aenv aenvL
   -> Size aenvL
   -> acc aenv t
   -> Lifted acc aenvL t


-- segmentedSize
--     :: ArraysR t
--     -> ArrayVars aenv (Segmented t)
--     -> Size aenv
-- segmentedSize TupRunit              (TupRsingle a)              = Index a Nil
-- segmentedSize (TupRsingle ArrayR{}) (TupRpair (TupRsingle s) _) = ShapeSize dim1 (Shape s)
-- segmentedSize (TupRpair aR1 aR2) (TupRpair a1 a2)            = _


-- liftPreOpenAfun1
--     :: Strength
--     -> Context () () aenv aenvL
--     -> OpenAfun aenv  (a -> b)
--     -> OpenAfun aenvL (Segmented a -> Segmented b)
-- liftPreOpenAfun1 strength context (Alam lhs (Abody b)) =
--   let size =
--       aR   = lhsToTupR lhs

liftPreOpenAcc
    :: forall aenv aenvL t.
       Strength
    -> Context () () aenv aenvL
    -> OpenAcc aenv t
    -> LiftedOpenAcc aenvL t
liftPreOpenAcc strength context (OpenAcc pacc) =
  let
      infixr 0 $^
      f $^ a = f (OpenAcc a)

      segmented :: Lifted OpenAcc aenv' a -> OpenAcc aenv' (Segmented a)
      segmented (Avoided a) = unavoidable (arraysR a) (Const (scalarType @Int) 10) a -- XXX TODO size context?!
      segmented (Lifted l)  = l

      unavoidable
          :: ArraysR a
          -> Size aenv'
          -> OpenAcc aenv' a
          -> OpenAcc aenv' (Segmented a)
      unavoidable aR size a
        | DeclareVars lhs k vars <- declareVars aR
        = let
              flatten :: ArrayVar aenv' (Array sh e) -> OpenAcc aenv' (Vector e)
              flatten v@(Var (ArrayR shR _) _)
                | ShapeRsnoc ShapeRz <- shR = OpenAcc $ Avar v
                | otherwise                 = OpenAcc $ Reshape dim1 (Nil `Pair` ShapeSize shR (Shape v)) $^ Avar v

              replicate :: ArrayVar aenv' (Vector e) -> Size aenv' -> OpenAcc aenv' (Matrix e)
              replicate v s =
                let slice = SliceFixed (SliceAll SliceNil)
                    slix  = (Nil `Pair` Nil) `Pair` s
                 in OpenAcc $ Replicate slice slix $^ Avar v

              go :: Size aenv' -> ArrayVars aenv' a -> OpenAcc aenv' (Segmented a)
              go s TupRunit         = OpenAcc $ Unit (eltR @Int) s
              go s (TupRpair v1 v2) = OpenAcc $ Apair (go s v1) (go s v2)
              go s (TupRsingle v)
                | Var (ArrayR shR eR) _ <- v
                = let values =
                        let eR1 = ArrayR dim1 eR
                            eR2 = ArrayR dim2 eR
                            s'  = weaken (weakenSucc weakenId) s
                        in
                        OpenAcc $ Alet (LeftHandSideSingle eR1) (flatten v) . OpenAcc
                                $ Alet (LeftHandSideSingle eR2) (replicate (Var eR1 ZeroIdx) s')
                                $ flatten (Var eR2 ZeroIdx)
                      segs =
                        let aR = ArrayR dim1 (shapeType shR)
                            sh = Pair Nil s
                            f  = Lam (LeftHandSideWildcard (shapeType dim1)) (Body (Shape v))
                        in
                        OpenAcc $ Generate aR sh f
                  in
                  OpenAcc $ Apair segs values
          in
          OpenAcc $ Alet lhs a
                  $ go (weaken k size) (vars weakenId)

      liftA :: OpenAcc aenv a -> LiftedOpenAcc aenvL a
      liftA = liftPreOpenAcc strength context

      liftE :: Context env env' aenv aenv'
            -> Size aenv'
            -> OpenExp env aenv e
            -> OpenAcc aenv' (Vector e)
      liftE = liftExp strength

      liftF1 :: Fun aenv (a -> b) -> OpenAcc (aenvL, Vector a) (Vector b)
      liftF1 (Lam lhs (Body b)) =
        let aR   = ArrayR dim1 (lhsToTupR lhs)
            -- alhs = LeftHandSideSingle aR
         in
         liftE (PushLExpC lhs context) (ShapeSize dim1 (Shape (Var aR ZeroIdx))) b
      liftF1 _ = error "writing a type-checker is hard"

      -- aletL :: ALeftHandSide a aenv aenv'
      --       -> OpenAcc aenv  a
      --       -> OpenAcc aenv' b
      --       -> LiftedOpenAcc aenvL b
      -- aletL lhs bnd body =
      --   let bndL  = liftA bnd
      --       bodyL = liftA

      useL :: ArrayR (Array sh e)
           -> Array sh e
           -> LiftedOpenAcc aenvL (Array sh e)
      useL aR a = Avoided (OpenAcc (Use aR a))

      mapL :: TypeR b
           -> Fun aenv (a -> b)
           -> OpenAcc aenv (Array sh a)
           -> LiftedOpenAcc aenvL (Array sh b)
      mapL bR f a =
        let ArrayR shR aR = arrayR a
            shR'          = shapeType shR
            fL            = liftF1 f
            aL            = liftA a
            lhs1          = LeftHandSideSingle (ArrayR dim1 shR') `LeftHandSidePair` LeftHandSideSingle (ArrayR dim1 aR)
            lhs2          = LeftHandSideSingle (ArrayR dim1 bR)
            seg           = Avar (Var (ArrayR dim1 shR') (SuccIdx (SuccIdx ZeroIdx)))
            arr           = Avar (Var (ArrayR dim1 bR) ZeroIdx)
        in
        Lifted $^ Alet lhs1 (segmented aL)
               $^ Alet lhs2 (weaken (sink (weakenSucc weakenId)) fL)
               $^ Apair (OpenAcc seg) (OpenAcc arr)

      -- generateL :: ArrayR (Array sh e)
      --           -> Exp aenv sh
      --           -> Fun aenv (sh -> e)
      --           -> LiftedOpenAcc aenvL (Array sh e)
      -- generateL aR@(ArrayR shR eR) sh f =
      --   let shR'          = shapeType shR
      --       sz            = Const (scalarType @Int) 10 -- XXX ???
      --       shL           = liftE context sz sh
      --       fL            = liftF1 f
      --       lhs1          = LeftHandSideSingle (ArrayR dim1 shR')
      --   in
      --   Lifted $^ Alet lhs1 shL (error "TODO: generateL")

      replicateL :: SliceIndex slix sl co sh
                 -> Exp aenv slix
                 -> OpenAcc aenv (Array sl e)
                 -> LiftedOpenAcc aenvL (Array sh e)
      replicateL sl slix a
        | DeclareVars lhs k v <- declareVars (shapeType (sliceDomainR sl))
        = let
              aR    = arrayR a
              slix' = weaken (weakenSucc weakenId) slix
              sh    = IndexFull sl slix' (Shape a0)
              p     = Lam lhs (Body (IndexSlice sl (weakenE k slix') (expVars (v weakenId))))
              a0    = Var aR ZeroIdx
          in
          liftA $^ Alet (LeftHandSideSingle aR) a
                $^ Backpermute (sliceDomainR sl) sh p
                $^ Avar a0

  in
  case pacc of
    -- Alet lhs bnd body                 -> undefined
    -- Avar v                            -> undefined
    -- Apair xs ys                       -> undefined
    -- Anil                              -> undefined
    -- Apply aR f xs                     -> undefined
    -- Aforeign aR asm f xs              -> undefined
    -- Acond p xs ys                     -> undefined
    -- Awhile p f xs                     -> undefined
    -- Atrace m xs ys                    -> undefined
    Use aR a                          -> useL aR a
    -- Unit eR e                         -> undefined
    -- Reshape shR sh a                  -> undefined
    -- Generate aR sh f                  -> undefined
    -- Transform aR sh p f xs            -> undefined
    Replicate slice slix sl           -> replicateL slice slix sl
    -- Slice sliceIndex sh slix          -> undefined
    Map tR f xs                       -> mapL tR f xs
    -- ZipWith tR f xs ys                -> undefined
    -- Fold f z xs                       -> undefined
    -- FoldSeg iR f z xs ss              -> undefined
    -- Scan dir f z xs                   -> undefined
    -- Scan' dir f z xs                  -> undefined
    -- Permute f d p xs                  -> undefined
    -- Backpermute shR sh p xs           -> undefined
    -- Stencil sR tR f b xs              -> undefined
    -- Stencil2 sR1 sR2 tR f b1 xs b2 ys -> undefined


liftExp
    :: forall env envL aenv aenvL t.
       Strength
    -> Context env envL aenv aenvL
    -> Size aenvL
    -> OpenExp env aenv t
    -> OpenAcc aenvL (Vector t)
liftExp strength context size =
  let
      liftE :: OpenExp env aenv e -> OpenAcc aenvL (Vector e)
      liftE = liftExp strength context size

      liftF1 :: TypeR a
             -> (forall env''. OpenExp env'' aenv' a -> OpenExp env'' aenv' b)
             -> OpenFun env' aenv' (a -> b)
      liftF1 aR f
        | DeclareVars lhs _ v <- declareVars aR
        = Lam lhs (Body (f (expVars (v weakenId))))

      liftF2 :: TypeR a
             -> TypeR b
             -> (forall env''. OpenExp env'' aenv' a -> OpenExp env'' aenv' b -> OpenExp env'' aenv' c)
             -> OpenFun env' aenv' (a -> b -> c)
      liftF2 aR bR f
        | DeclareVars lhsA _ vA <- declareVars aR
        , DeclareVars lhsB _ vB <- declareVars bR
        = Lam lhsA
        $ Lam lhsB
        $ Body (f (expVars (vA (weakenWithLHS lhsB))) (expVars (vB weakenId)))

      fillE :: Exp aenv e -> PreOpenAcc OpenAcc aenvL (Vector e)
      fillE x =
        let eR  = expType x
            aR  = ArrayR dim1 eR
            lhs = LeftHandSideWildcard (shapeType dim1)
         in Generate aR (Pair Nil size) (Lam lhs (Body (weaken (weakenWithContext context) x)))

      letL :: ELeftHandSide a env env' -> OpenExp env aenv a -> OpenExp env' aenv b -> PreOpenAcc OpenAcc aenvL (Vector b)
      letL lhs bnd body =
        let bnd'  = liftE bnd
            body' = liftExp strength (PushLExpC lhs context) (weaken (weakenSucc weakenId) size) body
            -- alhs  = LeftHandSideSingle aR
            eR    = lhsToTupR lhs
            aR    = ArrayR dim1 eR
         in
         Alet (LeftHandSideSingle aR) bnd' body'

      -- XXX: This isn't right! The Var represents a single scalar, but the
      -- LHS includes the entire tuple of scalar vars, equivalent to
      -- a single array var! So we need to do a bit more deconstruction to
      -- project out the field we care about from the lifted array. UGH!
      --
      -- varL :: TypeR e
      --      -> Context env env' aenv aenv'
      --      -> Idx env e
      --      -> (env'  :> envL)
      --      -> (aenv' :> aenvL)
      --      -> PreOpenAcc OpenAcc aenvL (Vector e)
      -- varL = error "TODO: varL"
      -- varL eR (PushLExpC _ (LeftHandSideSingle aR@(ArrayR _ eR')) _) ZeroIdx kE kA
      --   | Just Refl <- matchTypeR eR eR'    -- why?
      --   = weaken kA (Avar (Var aR ZeroIdx))

      -- varL eR (PushLExpC _ _ c) (SuccIdx ix) kE kA
      --   = varL eR c ix _ _

      varL :: Context env env' aenv aenv'
           -> ExpVar env e
           -> (env'  :> envL)
           -> (aenv' :> aenvL)
           -> PreOpenAcc OpenAcc aenvL (Vector e)
      varL (PushLExpC lhs c) v@(Var bR _) kE kA =
        let aR = lhsToTupR lhs
            f  = Lam lhs (Body (Evar v))
            a  = OpenAcc (Avar (Var (ArrayR dim1 aR) (weaken kA ZeroIdx)))
         in Map (TupRsingle bR) f a

      -- varL (PushLExpC _ (LeftHandSideSingle aR@(ArrayR _ eR)) c) (Var eR' _) kE kA
      --   | Just Refl <- matchTypeR eR (TupRsingle eR')
      --   = weaken kA (Avar (Var aR ZeroIdx))

      -- varL (PushLExpC _ _ c) (Var _ (SuccIdx _))

      -- varL (PushLExpC lhs alhs c) (Var eR ZeroIdx) kE kA =
      --   let aR = ArrayR dim1 (TupRsingle eR)
            -- v  = Var aR ZeroIdx               :: aenv' ~ (aenv'', Vector e) => ArrayVar aenv' (Vector e)
            -- v' = weaken kA (Avar v)
         -- in
         -- _

      --   = weaken (kA .> weakenWithLHS alhs .> weakenWithContext c)
      --   $ Avar (Var (ArrayR dim1 (TupRsingle eR)) ZeroIdx)

      -- varL (PushLExpC (LeftHandSideSingle _) c) (Var eR ZeroIdx) kE kA = weaken (weakenSucc kA .> weakenWithContext c) (Avar (Var (ArrayR dim1 (TupRsingle eR)) ZeroIdx))

  in
  OpenAcc . \case
    Let lhs bnd body              -> letL lhs bnd body
    Evar v                        -> varL context v weakenId weakenId
    Foreign tR asm f x            -> Map tR (liftF1 (expType x) (Foreign tR asm f)) (liftE x)
    Pair x y                      -> let aR = expType x
                                         bR = expType y
                                         cR = TupRpair aR bR
                                      in ZipWith cR (liftF2 aR bR Pair) (liftE x) (liftE y)
    Nil                           -> fillE Nil
    -- VecPack vR x                  -> undefined
    -- VecUnpack vR x                -> undefined
    -- IndexSlice sliceIndex slix sh -> undefined
    -- IndexFull sliceIndex slix sl  -> undefined
    ToIndex shR sh ix             -> let tR = shapeType shR
                                      in ZipWith (eltR @Int) (liftF2 tR tR (ToIndex shR)) (liftE sh) (liftE ix)
    FromIndex shR sh i            -> let tR = shapeType shR
                                      in ZipWith tR (liftF2 tR (eltR @Int) (FromIndex shR)) (liftE sh) (liftE i)
    -- Case tag xs x                 -> undefined
    -- Cond p t e                    -> undefined
    -- While p f x                   -> undefined
    Const t x                     -> fillE (Const t x)
    PrimConst x                   -> fillE (PrimConst x)
    PrimApp f x                   -> let (aR, bR) = primFunType f
                                      in Map bR (liftF1 aR (PrimApp f)) (liftE x)
    Index v ix                    -> let Var (ArrayR _ eR) _ = v
                                      in Map eR (liftF1 (expType ix) (Index (weaken (weakenWithContext context) v))) (liftE ix)
    LinearIndex v ix              -> let Var (ArrayR _ eR) _ = v
                                      in Map eR (liftF1 (eltR @Int) (LinearIndex (weaken (weakenWithContext context) v))) (liftE ix)
    Shape v                       -> fillE (Shape v)
    ShapeSize shR sh              -> let tR = shapeType shR
                                      in Map (eltR @Int) (liftF1 tR (ShapeSize shR)) (liftE sh)
    Undef t                       -> fillE (Undef t)
    Coerce tA tB a                -> Map (TupRsingle tB) (liftF1 (TupRsingle tA) (Coerce tA tB)) (liftE a)

