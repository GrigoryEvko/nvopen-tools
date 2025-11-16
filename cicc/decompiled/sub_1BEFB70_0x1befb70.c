// Function: sub_1BEFB70
// Address: 0x1befb70
//
_QWORD *__fastcall sub_1BEFB70(_QWORD *a1, _QWORD *a2, _QWORD *a3)
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
  _QWORD v19[2]; // [rsp+0h] [rbp-120h] BYREF
  unsigned __int64 v20; // [rsp+10h] [rbp-110h]
  _BYTE v21[64]; // [rsp+28h] [rbp-F8h] BYREF
  __int64 v22; // [rsp+68h] [rbp-B8h]
  __int64 v23; // [rsp+70h] [rbp-B0h]
  __int64 v24; // [rsp+78h] [rbp-A8h]
  _QWORD v25[2]; // [rsp+80h] [rbp-A0h] BYREF
  unsigned __int64 v26; // [rsp+90h] [rbp-90h]
  _BYTE v27[64]; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v28; // [rsp+E8h] [rbp-38h]
  __int64 v29; // [rsp+F0h] [rbp-30h]
  __int64 v30; // [rsp+F8h] [rbp-28h]

  sub_16CCEE0(v25, (__int64)v27, 8, (__int64)a3);
  v5 = a3[13];
  a3[13] = 0;
  v28 = v5;
  v6 = a3[14];
  a3[14] = 0;
  v29 = v6;
  v7 = a3[15];
  a3[15] = 0;
  v30 = v7;
  sub_16CCEE0(v19, (__int64)v21, 8, (__int64)a2);
  v8 = a2[13];
  a2[13] = 0;
  v22 = v8;
  v9 = a2[14];
  a2[14] = 0;
  v23 = v9;
  v10 = a2[15];
  a2[15] = 0;
  v24 = v10;
  sub_16CCEE0(a1, (__int64)(a1 + 5), 8, (__int64)v19);
  v11 = v22;
  v22 = 0;
  a1[13] = v11;
  v12 = v23;
  v23 = 0;
  a1[14] = v12;
  v13 = v24;
  v24 = 0;
  a1[15] = v13;
  sub_16CCEE0(a1 + 16, (__int64)(a1 + 21), 8, (__int64)v25);
  v14 = v28;
  v15 = v22;
  v28 = 0;
  a1[29] = v14;
  v16 = v29;
  v29 = 0;
  a1[30] = v16;
  v17 = v30;
  v30 = 0;
  a1[31] = v17;
  if ( v15 )
    j_j___libc_free_0(v15, v24 - v15);
  if ( v20 != v19[1] )
    _libc_free(v20);
  if ( v28 )
    j_j___libc_free_0(v28, v30 - v28);
  if ( v26 != v25[1] )
    _libc_free(v26);
  return a1;
}
