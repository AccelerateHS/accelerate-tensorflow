{-# LANGUAGE LambdaCase #-}
module Data.Array.Accelerate.TensorFlow.Lite.ConverterPy (
  ConverterPy,
  withConverterPy,
  ConverterSettings(..),
  defaultConverterSettings,
  withConverterPy',
  runConverterJob,
) where

import qualified Proto.Tensorflow.Core.Framework.Graph              as TF
import qualified Proto.Tensorflow.Core.Framework.Graph_Fields       as TF
import qualified Proto.Tensorflow.Core.Framework.NodeDef_Fields     as TF

import Control.Concurrent                                           ( forkIO, threadDelay )
import Control.Exception
import Control.Monad                                                ( when, forM_ )
import Data.ByteString                                              ( ByteString )
import qualified Data.ByteString.Lazy                               as LB
import qualified Data.ByteString.Lazy.Char8                         as LB8
import Data.ByteString.Builder                                      ( Builder )
import Data.IORef
import Data.ProtoLens
import Data.Word
import Lens.Family2
import System.Exit
import System.FilePath
import System.IO
import System.IO.Temp
import System.Posix.IO                                              ( handleToFd, closeFd )
import System.Process
import System.Timeout
import Text.Printf
import qualified Data.ByteString                                    as B
import qualified Data.ByteString.Builder                            as B
import qualified Data.Text                                          as T
import qualified Data.Text.Encoding                                 as TE

import Paths_accelerate_tensorflow_lite


-- TODO: The job interface between this module and converter.py is simply
-- lifted from what it was before (command-line arguments). This is nonsense
-- and the input files, at least the data file, should just go over stdin
-- instead of via a temporary directory. TFLiteConverter.from_frozen_graph
-- seems to only accept a _file name_ instead of just a buffer, so perhaps the
-- graph does need to be written to a file.


-- | The representation of a running converter.py process.
--
-- Every distinct 'ConverterPy' represents a distinct process. Access to these
-- processes is NOT thread-safe! See 'runConverterJob' for details.
data ConverterPy = ConverterPy ConverterSettings (IORef (Maybe CPImpl))
-- This is an IORef so that we can mutate the contents in-place. Nothing
-- indicates that no process has been started yet; presumably the last job
-- failed, and we want to wait until a new job happens to start a new
-- converter. Just indiates a running process.

data CPImpl = CPImpl
    ProcessHandle
    Handle  -- ^ stdin
    Handle  -- ^ tflite output stream
    (IORef LB.ByteString)  -- ^ stderr

data ConverterSettings = ConverterSettings
    { csVerbose :: Bool }
  deriving (Show)

defaultConverterSettings :: ConverterSettings
defaultConverterSettings = ConverterSettings False

-- | Every call to this function starts a new converter.py process.
-- The Python process exits after the subcomputation returns.
--
-- This version initialises the 'ConverterSettings' to
-- 'defaultConverterSettings'.
withConverterPy :: (ConverterPy -> IO a) -> IO a
withConverterPy = withConverterPy' defaultConverterSettings

-- | Every call to this function starts a new converter.py process.
-- The Python process exits after the subcomputation returns.
withConverterPy' :: ConverterSettings -> (ConverterPy -> IO a) -> IO a
withConverterPy' settings = bracket (openConverterPy settings) closeConverterPy

openConverterPy :: ConverterSettings -> IO ConverterPy
openConverterPy settings = do
  script     <- getDataFileName "converter.py"
  python_exe <- getDataFileName "tf-python-venv/bin/python3"

  -- (read, write)
  (outpipe0, outpipe1) <- createPipe
  outpipe1fd <- handleToFd outpipe1  -- this invalidates outpipe1

  let cp = (proc python_exe [script, show (toInteger outpipe1fd)])
             { std_in = CreatePipe
             , std_out = if csVerbose settings then Inherit else CreatePipe
             , std_err = if csVerbose settings then Inherit else CreatePipe }
  (Just inh, mouth, merrh, ph) <- createProcess cp

  -- now close our copy of the write end of the pipe so that the Python process has the only copy
  closeFd outpipe1fd

  -- ignore stdout output
  forM_ mouth $ \outh ->
    forkIO $
      let loop = do
            bs <- B.hGetSome outh 4000
            if B.length bs == 0 then return () else loop
      in loop

  -- collect stderr output (but it will be cleared repeatedly)
  errstream <- newIORef mempty
  forM_ merrh $ \errh ->
    forkIO $
      let loop = do
            bs <- B.hGetSome errh 4000
            if B.length bs == 0
              then return ()
              else do
                atomicModifyIORef' errstream (\acc -> (acc <> LB.fromStrict bs, ()))
                loop
      in loop

  cpref <- newIORef (Just (CPImpl ph inh outpipe0 errstream))
  return (ConverterPy settings cpref)

closeConverterPy :: ConverterPy -> IO ()
closeConverterPy (ConverterPy _ cpref) =
  readIORef cpref >>= \case
    Nothing -> return ()
    Just (CPImpl ph inh _ errstream) -> do
      B.hPut inh (B.singleton 2)
      hFlush inh
      ex <- waitForProcess ph
      case ex of
        ExitFailure r -> do
          errs <- readIORef errstream
          error $ printf "converter.py exited with code %d\n%s" r (TE.decodeUtf8 (LB.toStrict errs))
        ExitSuccess   -> return ()

-- | Returns the .tflite file contents as a binary blob.
--
-- Note that this function is NOT thread-safe! You can use separate
-- 'ConverterPy' objects simultaneously just fine, but do not use
-- 'runConverterJob' concurrently on a single 'ConverterPy'.
runConverterJob :: ConverterPy -> TF.GraphDef -> Builder -> IO ByteString
runConverterJob converter graph reprdata = do
  CPImpl _ inh outh stderrstream <- ensureConverter converter
  withTemporaryDirectory "acctflite-conv" $ \tmpdir -> do
    let graphFname = tmpdir </> "model.pb"
        dataFname = tmpdir </> "data.bin"
    B.writeFile graphFname (encodeMessage graph)
    withBinaryFile dataFname WriteMode $ \h -> B.hPutBuilder h reprdata

    let names   = map (view TF.name) (graph ^. TF.node)
        inputs  = filter (T.isPrefixOf (T.pack "input")) names
        outputs = filter (T.isPrefixOf (T.pack "output")) names

    B.hPut inh (B.singleton 1)
    B.hPutBuilder inh (serialiseJobSpec (JobSpec graphFname inputs outputs dataFname))
    hFlush inh

    reslenBS <- B.hGet outh 8
    when (B.length reslenBS < 8) $ do
      -- wait a bit for errors to accumulate
      threadDelay 50000  -- 50ms
      -- print errors
      errs <- readIORef stderrstream
      hPutStrLn stderr (LB8.unpack errs)
      -- make sure the next job will start a new converter.py process
      killConverterPy converter
      ioError (userError "Unexpected EOF from converter.py")

    -- hPutStrLn stderr $ "HS: reslenBS = " ++ show reslenBS
    let reslen = fromIntegral (fromWord64LE reslenBS)
    -- hPutStrLn stderr $ "HS: reslen = " ++ show reslen
    tflitebuf <- B.hGet outh reslen

    -- successful completion, so clear the stderr stream
    atomicWriteIORef stderrstream mempty

    return tflitebuf

ensureConverter :: ConverterPy -> IO CPImpl
ensureConverter converter@(ConverterPy settings cpref) =
  readIORef cpref >>= \case
    Nothing -> do
      ConverterPy _ newcpref <- openConverterPy settings
      readIORef newcpref >>= atomicWriteIORef cpref
      ensureConverter converter
    Just cpimpl -> return cpimpl

-- | Stop the currently-running process, if any, and set the 'CPImpl' to
-- Nothing.
killConverterPy :: ConverterPy -> IO ()
killConverterPy (ConverterPy _ cpref) =
  readIORef cpref >>= \case
    Nothing -> return ()
    Just (CPImpl ph _ _ _) -> do
      -- don't wait on the process being terminated
      _ <- forkIO $ do
        _ <- terminateAfterAWhile ph
        return ()
      -- process stopped, so it's not available anymore
      atomicWriteIORef cpref Nothing

data JobSpec = JobSpec
    FilePath  -- ^ graph def file
    [T.Text]  -- ^ input node names
    [T.Text]  -- ^ output node names
    FilePath  -- ^ representative data file
  deriving (Show)

serialiseJobSpec :: JobSpec -> Builder
serialiseJobSpec (JobSpec graphdeffile inputs outputs reprdatafile) =
  mconcat
    [buildList (map B.word8 (B.unpack (TE.encodeUtf8 (T.pack graphdeffile))))
    ,buildList (map (buildList . map B.char7 . T.unpack) inputs)
    ,buildList (map (buildList . map B.char7 . T.unpack) outputs)
    ,buildList (map B.word8 (B.unpack (TE.encodeUtf8 (T.pack reprdatafile))))]
  where
    buildList :: [Builder] -> Builder
    buildList builders = B.word64LE (fromIntegral (length builders)) <> mconcat builders

fromWord64LE :: ByteString -> Word64
fromWord64LE bs
  | B.length bs == 8 = sum (zipWith (*) (map fromIntegral (B.unpack bs))
                                        (iterate (*256) 1))
  | otherwise = error "fromWord64LE: not length 8"

-- | Returns exit code if it terminated soon enough
terminateAfterAWhile :: ProcessHandle -> IO (Maybe Int)
terminateAfterAWhile ph = do
  -- timeout is in microseconds
  ex <- timeout 300000 $ waitForProcess ph
  case ex of
    Nothing -> do  -- timeout
      terminateProcess ph
      return Nothing
    Just (ExitFailure r) -> return (Just r)
    Just ExitSuccess -> return (Just 0)
