// Function: sub_28CD1F0
// Address: 0x28cd1f0
//
_QWORD *__fastcall sub_28CD1F0(_QWORD *a1, _QWORD *a2, _QWORD *a3)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  char v19[8]; // [rsp+10h] [rbp-130h] BYREF
  unsigned __int64 v20; // [rsp+18h] [rbp-128h]
  char v21; // [rsp+2Ch] [rbp-114h]
  _BYTE v22[64]; // [rsp+30h] [rbp-110h] BYREF
  unsigned __int64 v23; // [rsp+70h] [rbp-D0h]
  __int64 v24; // [rsp+78h] [rbp-C8h]
  __int64 v25; // [rsp+80h] [rbp-C0h]
  char v26[8]; // [rsp+90h] [rbp-B0h] BYREF
  unsigned __int64 v27; // [rsp+98h] [rbp-A8h]
  char v28; // [rsp+ACh] [rbp-94h]
  _BYTE v29[64]; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int64 v30; // [rsp+F0h] [rbp-50h]
  __int64 v31; // [rsp+F8h] [rbp-48h]
  __int64 v32; // [rsp+100h] [rbp-40h]

  sub_C8CF70((__int64)v26, v29, 8, (__int64)(a3 + 4), (__int64)a3);
  v5 = a3[12];
  a3[12] = 0;
  v30 = v5;
  v6 = a3[13];
  a3[13] = 0;
  v31 = v6;
  v7 = a3[14];
  a3[14] = 0;
  v32 = v7;
  sub_C8CF70((__int64)v19, v22, 8, (__int64)(a2 + 4), (__int64)a2);
  v8 = a2[12];
  a2[12] = 0;
  v23 = v8;
  v9 = a2[13];
  a2[13] = 0;
  v24 = v9;
  v10 = a2[14];
  a2[14] = 0;
  v25 = v10;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v22, (__int64)v19);
  v11 = v23;
  v23 = 0;
  a1[12] = v11;
  v12 = v24;
  v24 = 0;
  a1[13] = v12;
  v13 = v25;
  v25 = 0;
  a1[14] = v13;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v29, (__int64)v26);
  v14 = v30;
  v15 = v23;
  v30 = 0;
  a1[27] = v14;
  v16 = v31;
  v31 = 0;
  a1[28] = v16;
  v17 = v32;
  v32 = 0;
  a1[29] = v17;
  if ( v15 )
    j_j___libc_free_0(v15);
  if ( !v21 )
    _libc_free(v20);
  if ( v30 )
    j_j___libc_free_0(v30);
  if ( !v28 )
    _libc_free(v27);
  return a1;
}
