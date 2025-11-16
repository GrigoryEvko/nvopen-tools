// Function: sub_AB8340
// Address: 0xab8340
//
__int64 __fastcall sub_AB8340(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  unsigned __int64 v5; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-48h]
  __int64 v7; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-38h]
  __int64 v9; // [rsp+20h] [rbp-30h]
  unsigned int v10; // [rsp+28h] [rbp-28h]

  v2 = *(_DWORD *)(a2 + 8);
  v6 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43690(&v5, -1, 1);
  }
  else
  {
    v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v3 = 0;
    v5 = v3;
  }
  sub_AADBC0((__int64)&v7, (__int64 *)&v5);
  sub_AB51C0(a1, (__int64)&v7, a2);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  return a1;
}
