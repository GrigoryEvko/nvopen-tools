// Function: sub_222D0C0
// Address: 0x222d0c0
//
__int64 __fastcall sub_222D0C0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, __time_t a5, __int64 a6)
{
  unsigned int v9; // r12d
  __time_t v11; // rbx
  __int64 v12; // rbp
  struct timeval v13; // [rsp+0h] [rbp-48h] BYREF
  __time_t v14; // [rsp+10h] [rbp-38h] BYREF
  __int64 v15; // [rsp+18h] [rbp-30h]

  if ( (_BYTE)a4 )
  {
    v9 = a4;
    gettimeofday(&v13, 0);
    if ( v13.tv_sec > a5 )
      return 0;
    v11 = a5 - v13.tv_sec;
    v12 = a6 - 1000 * v13.tv_usec;
    if ( v12 < 0 )
    {
      v14 = v11 - 1;
      v15 = v12 + 1000000000;
      if ( !v11 )
        return 0;
    }
    else
    {
      v14 = v11;
      v15 = v12;
    }
    if ( syscall(202, a2, 0, a3, &v14) != -1 || *__errno_location() != 110 )
      return v9;
    return 0;
  }
  syscall(202, a2, 0, a3, 0);
  return 1;
}
