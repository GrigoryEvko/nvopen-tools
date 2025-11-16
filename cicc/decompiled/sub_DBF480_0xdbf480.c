// Function: sub_DBF480
// Address: 0xdbf480
//
__int64 __fastcall sub_DBF480(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v7; // r14d
  _QWORD *v9; // r14
  _QWORD *v10; // r15
  _QWORD *v11; // [rsp+0h] [rbp-E0h]
  _QWORD *v12; // [rsp+8h] [rbp-D8h]
  __int64 v13; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-C8h]
  __int64 v15; // [rsp+20h] [rbp-C0h]
  unsigned int v16; // [rsp+28h] [rbp-B8h]
  __int64 v17; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-A8h]
  __int64 v19; // [rsp+40h] [rbp-A0h]
  unsigned int v20; // [rsp+48h] [rbp-98h]
  __int64 v21; // [rsp+50h] [rbp-90h] BYREF
  __int64 v22; // [rsp+58h] [rbp-88h] BYREF
  unsigned int v23; // [rsp+60h] [rbp-80h]
  __int64 v24; // [rsp+68h] [rbp-78h] BYREF
  unsigned int v25; // [rsp+70h] [rbp-70h]
  __int64 v26; // [rsp+80h] [rbp-60h] BYREF
  __int64 v27; // [rsp+88h] [rbp-58h] BYREF
  unsigned int v28; // [rsp+90h] [rbp-50h]
  __int64 v29; // [rsp+98h] [rbp-48h] BYREF
  unsigned int v30; // [rsp+A0h] [rbp-40h]

  v7 = *(_DWORD *)(a5 + 8);
  sub_D93B80((__int64)&v21, v7, a3);
  if ( v21 )
  {
    sub_D93B80((__int64)&v26, v7, a4);
    if ( v26 && v21 == v26 )
    {
      v11 = sub_DA26C0(a2, (__int64)&v22);
      v12 = sub_DA26C0(a2, (__int64)&v27);
      v9 = sub_DA26C0(a2, (__int64)&v24);
      v10 = sub_DA26C0(a2, (__int64)&v29);
      sub_DBEFC0((__int64)&v13, (__int64)a2, (__int64)v11, (__int64)v12, a5);
      sub_DBEFC0((__int64)&v17, (__int64)a2, (__int64)v9, (__int64)v10, a5);
      sub_AB3510(a1, (__int64)&v13, (__int64)&v17, 0);
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
      if ( v18 > 0x40 && v17 )
        j_j___libc_free_0_0(v17);
      if ( v16 > 0x40 && v15 )
        j_j___libc_free_0_0(v15);
      if ( v14 > 0x40 && v13 )
        j_j___libc_free_0_0(v13);
    }
    else
    {
      sub_AADB10(a1, v7, 1);
    }
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
  }
  else
  {
    sub_AADB10(a1, v7, 1);
  }
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return a1;
}
