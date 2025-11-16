// Function: sub_274B7F0
// Address: 0x274b7f0
//
__int64 __fastcall sub_274B7F0(__int64 a1, __int64 *a2)
{
  __int64 *v3; // rdx
  unsigned int v4; // eax
  unsigned int v5; // r13d
  unsigned __int64 v6; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v7; // [rsp+8h] [rbp-38h]
  unsigned __int64 v8; // [rsp+10h] [rbp-30h]
  unsigned int v9; // [rsp+18h] [rbp-28h]

  if ( sub_B44910(a1) )
    return 0;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a1 - 8);
  else
    v3 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  sub_22CEA30((__int64)&v6, a2, v3, 0);
  LOBYTE(v4) = sub_AB0760((__int64)&v6);
  v5 = v4;
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  if ( v7 > 0x40 )
  {
    if ( v6 )
      j_j___libc_free_0_0(v6);
  }
  if ( !(_BYTE)v5 )
    return 0;
  sub_B448D0(a1, 1);
  return v5;
}
