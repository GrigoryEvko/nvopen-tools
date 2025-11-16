// Function: sub_C784C0
// Address: 0xc784c0
//
__int64 __fastcall sub_C784C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // [rsp+0h] [rbp-80h] BYREF
  unsigned int v5; // [rsp+8h] [rbp-78h]
  __int64 v6; // [rsp+10h] [rbp-70h]
  unsigned int v7; // [rsp+18h] [rbp-68h]
  __int64 v8; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v9; // [rsp+28h] [rbp-58h]
  __int64 v10; // [rsp+30h] [rbp-50h]
  unsigned int v11; // [rsp+38h] [rbp-48h]
  __int64 v12; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+48h] [rbp-38h]
  __int64 v14; // [rsp+50h] [rbp-30h]
  unsigned int v15; // [rsp+58h] [rbp-28h]

  sub_C70170((__int64)&v8, a3);
  sub_C70170((__int64)&v4, a2);
  sub_C78370((__int64)&v12, (__int64)&v4, (__int64)&v8);
  sub_C70170(a1, (__int64)&v12);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v7 > 0x40 && v6 )
    j_j___libc_free_0_0(v6);
  if ( v5 > 0x40 && v4 )
    j_j___libc_free_0_0(v4);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  if ( v9 > 0x40 && v8 )
    j_j___libc_free_0_0(v8);
  return a1;
}
