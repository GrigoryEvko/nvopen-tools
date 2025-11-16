// Function: sub_AB1CA0
// Address: 0xab1ca0
//
__int64 __fastcall sub_AB1CA0(__int64 a1)
{
  unsigned int v1; // r13d
  unsigned __int64 v3; // rax
  unsigned int v4; // r13d
  unsigned __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-28h]

  v1 = 0;
  if ( sub_AAF7D0(a1) )
    return v1;
  sub_AB0910((__int64)&v5, a1);
  if ( v6 <= 0x40 )
  {
    if ( v5 )
    {
      _BitScanReverse64(&v3, v5);
      return 64 - ((unsigned int)v3 ^ 0x3F);
    }
    return v1;
  }
  v4 = v6;
  v1 = v4 - sub_C444A0(&v5);
  if ( !v5 )
    return v1;
  j_j___libc_free_0_0(v5);
  return v1;
}
