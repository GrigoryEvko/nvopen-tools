// Function: sub_1479080
// Address: 0x1479080
//
__int64 __fastcall sub_1479080(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r13d
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v13; // [rsp+10h] [rbp-E0h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-D8h]
  __int64 v15; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v17; // [rsp+30h] [rbp-C0h]
  __int64 v18; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-A8h]
  __int64 v20; // [rsp+50h] [rbp-A0h]
  unsigned int v21; // [rsp+58h] [rbp-98h]
  __int64 v22; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-88h]
  __int64 v24; // [rsp+70h] [rbp-80h]
  unsigned int v25; // [rsp+78h] [rbp-78h]
  __int64 v26; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v27; // [rsp+88h] [rbp-68h]
  __int64 v28; // [rsp+90h] [rbp-60h]
  unsigned int v29; // [rsp+98h] [rbp-58h]
  __int64 v30; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+A8h] [rbp-48h]
  __int64 v32; // [rsp+B0h] [rbp-40h]
  unsigned int v33; // [rsp+B8h] [rbp-38h]

  sub_1478E30((__int64)&v15, a1, a3, a5);
  v8 = v17;
  if ( v17 )
  {
    v10 = *(_QWORD *)(a6 + 32);
    v27 = *(_DWORD *)(v10 + 32);
    if ( v27 > 0x40 )
      sub_16A4FD0(&v26, v10 + 24);
    else
      v26 = *(_QWORD *)(v10 + 24);
    sub_1589870(&v30, &v26);
    sub_158AE10(&v18, a2, &v30);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    if ( v27 > 0x40 && v26 )
      j_j___libc_free_0_0(v26);
    v27 = v16;
    if ( v16 > 0x40 )
      sub_16A4FD0(&v26, &v15);
    else
      v26 = v15;
    sub_1589870(&v30, &v26);
    sub_158E130(&v22, &v18, &v30);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    if ( v27 > 0x40 && v26 )
      j_j___libc_free_0_0(v26);
    v11 = *(_QWORD *)(a4 + 32);
    v14 = *(_DWORD *)(v11 + 32);
    if ( v14 > 0x40 )
      sub_16A4FD0(&v13, v11 + 24);
    else
      v13 = *(_QWORD *)(v11 + 24);
    sub_1589870(&v30, &v13);
    sub_1590F80(&v26, a2, &v30);
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    if ( v14 > 0x40 && v13 )
      j_j___libc_free_0_0(v13);
    v8 = sub_158BB40(&v26, &v22);
    if ( v29 > 0x40 && v28 )
      j_j___libc_free_0_0(v28);
    if ( v27 > 0x40 && v26 )
      j_j___libc_free_0_0(v26);
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    if ( v23 > 0x40 && v22 )
      j_j___libc_free_0_0(v22);
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
    if ( v19 > 0x40 && v18 )
      j_j___libc_free_0_0(v18);
    if ( v17 && v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
  }
  return v8;
}
