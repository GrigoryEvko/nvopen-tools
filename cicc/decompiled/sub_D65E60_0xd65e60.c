// Function: sub_D65E60
// Address: 0xd65e60
//
__int64 __fastcall sub_D65E60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-68h]
  __int64 v7; // [rsp+10h] [rbp-60h]
  unsigned int v8; // [rsp+18h] [rbp-58h]
  __int64 v9; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-48h]
  __int64 v11; // [rsp+30h] [rbp-40h]
  unsigned int v12; // [rsp+38h] [rbp-38h]

  sub_D62600((__int64)&v9, a2, *(_QWORD *)(a3 - 32));
  sub_D62600((__int64)&v5, a2, *(_QWORD *)(a3 - 64));
  sub_D5E640(a1, a2, (__int64)&v5, (__int64)&v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  return a1;
}
