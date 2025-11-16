// Function: sub_16E78B0
// Address: 0x16e78b0
//
__blksize_t __fastcall sub_16E78B0(__int64 a1)
{
  struct stat stat_buf; // [rsp+0h] [rbp-A0h] BYREF

  if ( __fxstat(1, *(_DWORD *)(a1 + 36), &stat_buf)
    || (stat_buf.st_mode & 0xF000) == 0x2000 && isatty(*(_DWORD *)(a1 + 36)) )
  {
    return 0;
  }
  else
  {
    return stat_buf.st_blksize;
  }
}
