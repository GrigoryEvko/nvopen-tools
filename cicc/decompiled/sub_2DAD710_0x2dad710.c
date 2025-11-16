// Function: sub_2DAD710
// Address: 0x2dad710
//
__int64 __fastcall sub_2DAD710(_QWORD *a1, __int64 *a2)
{
  char v2; // r8
  __int64 result; // rax
  unsigned __int8 v4; // [rsp+Fh] [rbp-71h]
  unsigned __int64 v5[14]; // [rsp+10h] [rbp-70h] BYREF

  v2 = sub_BB98D0(a1, *a2);
  result = 0;
  if ( !v2 )
  {
    memset(v5, 0, 0x60u);
    v5[3] = (unsigned __int64)&v5[5];
    v5[4] = 0x600000000LL;
    result = sub_2DAD5B0((__int64)v5, (__int64)a2);
    if ( (unsigned __int64 *)v5[3] != &v5[5] )
    {
      v4 = result;
      _libc_free(v5[3]);
      return v4;
    }
  }
  return result;
}
