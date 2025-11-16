// Function: sub_1592D00
// Address: 0x1592d00
//
__int64 __fastcall sub_1592D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v6; // [rsp+18h] [rbp-98h]
  __int64 v7; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v8; // [rsp+28h] [rbp-88h]
  __int64 v9; // [rsp+30h] [rbp-80h]
  unsigned int v10; // [rsp+38h] [rbp-78h]
  __int64 v11; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+48h] [rbp-68h]
  __int64 v13; // [rsp+50h] [rbp-60h]
  unsigned int v14; // [rsp+58h] [rbp-58h]
  __int64 v15; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+68h] [rbp-48h]
  __int64 v17; // [rsp+70h] [rbp-40h]
  unsigned int v18; // [rsp+78h] [rbp-38h]

  v12 = *(_DWORD *)(a3 + 8);
  if ( v12 > 0x40 )
    sub_16A4FD0(&v11, a3);
  else
    v11 = *(_QWORD *)a3;
  sub_1589870((__int64)&v15, &v11);
  sub_1591060((__int64)&v7, 11, (__int64)&v15, 2);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  sub_158BE00((__int64)&v11, a2, (__int64)&v7);
  v6 = *(_DWORD *)(a3 + 8);
  if ( v6 > 0x40 )
    sub_16A4FD0(&v5, a3);
  else
    v5 = *(_QWORD *)a3;
  sub_1589870((__int64)&v15, &v5);
  sub_158E130(a1, (__int64)&v11, (__int64)&v15);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0(v13);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0(v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  return a1;
}
