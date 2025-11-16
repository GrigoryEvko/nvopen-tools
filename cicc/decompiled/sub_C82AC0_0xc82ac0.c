// Function: sub_C82AC0
// Address: 0xc82ac0
//
__int64 __fastcall sub_C82AC0(int fildes, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v3; // rcx
  __int64 v4; // r8
  struct stat stat_buf; // [rsp+0h] [rbp-A0h] BYREF

  v2 = __fxstat(1, fildes, &stat_buf);
  return sub_C7FD70(v2, (__int64 *)&stat_buf, a2, v3, v4);
}
