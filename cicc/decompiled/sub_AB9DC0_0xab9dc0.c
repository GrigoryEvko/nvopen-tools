// Function: sub_AB9DC0
// Address: 0xab9dc0
//
__int64 __fastcall sub_AB9DC0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r14d
  __int64 v6; // r12
  __int64 v7; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-68h]
  __int64 v9; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-58h]
  __int64 v11; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+38h] [rbp-48h]
  __int64 v13; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+48h] [rbp-38h]

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB0A00((__int64)&v11, a2);
    sub_AB0A00((__int64)&v13, a3);
    sub_C49B30(&v7, &v11, &v13);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    sub_AB0910((__int64)&v9, a2);
    sub_AB0910((__int64)&v11, a3);
    sub_C49B30(&v13, &v9, &v11);
    sub_C46A40(&v13, 1);
    v5 = v14;
    v6 = v13;
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    if ( v10 > 0x40 && v9 )
      j_j___libc_free_0_0(v9);
    v14 = v5;
    v12 = v8;
    v13 = v6;
    v11 = v7;
    v8 = 0;
    sub_9875E0(a1, &v11, &v13);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
    if ( v8 > 0x40 && v7 )
      j_j___libc_free_0_0(v7);
  }
  return a1;
}
