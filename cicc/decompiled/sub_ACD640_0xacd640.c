// Function: sub_ACD640
// Address: 0xacd640
//
__int64 __fastcall sub_ACD640(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  unsigned int v3; // eax
  unsigned __int64 v4; // rsi
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-28h]
  unsigned __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-18h]

  v3 = *(_DWORD *)(a1 + 8) >> 8;
  v8 = v3;
  if ( v3 > 0x40 )
  {
    sub_C43690(&v7, a2, a3);
  }
  else
  {
    v4 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v3) & a2;
    if ( !v3 )
      v4 = 0;
    v7 = v4;
  }
  result = sub_ACCFD0(*(__int64 **)a1, (__int64)&v7);
  if ( v8 > 0x40 )
  {
    if ( v7 )
    {
      v6 = result;
      j_j___libc_free_0_0(v7);
      return v6;
    }
  }
  return result;
}
