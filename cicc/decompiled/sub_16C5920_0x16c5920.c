// Function: sub_16C5920
// Address: 0x16c5920
//
__int64 __fastcall sub_16C5920(int fildes, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rcx
  __int64 v4; // r8
  struct stat stat_buf; // [rsp+0h] [rbp-A0h] BYREF

  v2 = __fxstat(1, fildes, &stat_buf);
  return sub_16C3100(v2, (__int64 *)&stat_buf, a2, v3, v4);
}
