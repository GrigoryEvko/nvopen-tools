// Function: sub_13BA6D0
// Address: 0x13ba6d0
//
_QWORD *__fastcall sub_13BA6D0(_QWORD *a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE v19[8]; // [rsp+0h] [rbp-120h] BYREF
  __int64 v20; // [rsp+8h] [rbp-118h]
  unsigned __int64 v21; // [rsp+10h] [rbp-110h]
  _BYTE v22[64]; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v23; // [rsp+68h] [rbp-B8h]
  __int64 v24; // [rsp+70h] [rbp-B0h]
  __int64 v25; // [rsp+78h] [rbp-A8h]
  _BYTE v26[8]; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+88h] [rbp-98h]
  unsigned __int64 v28; // [rsp+90h] [rbp-90h]
  _BYTE v29[64]; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v30; // [rsp+E8h] [rbp-38h]
  __int64 v31; // [rsp+F0h] [rbp-30h]
  __int64 v32; // [rsp+F8h] [rbp-28h]

  sub_16CCEE0(v26, v29, 8, a3);
  v5 = a3[13];
  a3[13] = 0;
  v30 = v5;
  v6 = a3[14];
  a3[14] = 0;
  v31 = v6;
  v7 = a3[15];
  a3[15] = 0;
  v32 = v7;
  sub_16CCEE0(v19, v22, 8, a2);
  v8 = a2[13];
  a2[13] = 0;
  v23 = v8;
  v9 = a2[14];
  a2[14] = 0;
  v24 = v9;
  v10 = a2[15];
  a2[15] = 0;
  v25 = v10;
  sub_16CCEE0(a1, a1 + 5, 8, v19);
  v11 = v23;
  v23 = 0;
  a1[13] = v11;
  v12 = v24;
  v24 = 0;
  a1[14] = v12;
  v13 = v25;
  v25 = 0;
  a1[15] = v13;
  sub_16CCEE0(a1 + 16, a1 + 21, 8, v26);
  v14 = v30;
  v15 = v23;
  v30 = 0;
  a1[29] = v14;
  v16 = v31;
  v31 = 0;
  a1[30] = v16;
  v17 = v32;
  v32 = 0;
  a1[31] = v17;
  if ( v15 )
    j_j___libc_free_0(v15, v25 - v15);
  if ( v21 != v20 )
    _libc_free(v21);
  if ( v30 )
    j_j___libc_free_0(v30, v32 - v30);
  if ( v28 != v27 )
    _libc_free(v28);
  return a1;
}
