// Function: sub_CB5660
// Address: 0xcb5660
//
__blksize_t __fastcall sub_CB5660(int *a1)
{
  __int64 (__fastcall *v2)(__int64); // rax
  char v3; // al
  struct stat stat_buf; // [rsp+0h] [rbp-A0h] BYREF

  if ( __fxstat(1, a1[12], &stat_buf) )
    return 0;
  if ( (stat_buf.st_mode & 0xF000) == 0x2000
    && ((v2 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL), v2 != sub_CB5560)
      ? (v3 = v2((__int64)a1))
      : (v3 = sub_C862F0((unsigned int)a1[12])),
        v3) )
  {
    return 0;
  }
  else
  {
    return stat_buf.st_blksize;
  }
}
