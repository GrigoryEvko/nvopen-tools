// Function: sub_1E648D0
// Address: 0x1e648d0
//
_QWORD *__fastcall sub_1E648D0(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE v17[8]; // [rsp+0h] [rbp-220h] BYREF
  __int64 v18; // [rsp+8h] [rbp-218h]
  unsigned __int64 v19; // [rsp+10h] [rbp-210h]
  __int64 v20; // [rsp+68h] [rbp-1B8h]
  __int64 v21; // [rsp+70h] [rbp-1B0h]
  __int64 v22; // [rsp+78h] [rbp-1A8h]
  _BYTE v23[8]; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v24; // [rsp+88h] [rbp-198h]
  unsigned __int64 v25; // [rsp+90h] [rbp-190h]
  __int64 v26; // [rsp+E8h] [rbp-138h]
  __int64 v27; // [rsp+F0h] [rbp-130h]
  __int64 v28; // [rsp+F8h] [rbp-128h]
  _QWORD v29[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v30; // [rsp+110h] [rbp-110h]
  _BYTE v31[64]; // [rsp+128h] [rbp-F8h] BYREF
  __int64 v32; // [rsp+168h] [rbp-B8h]
  __int64 v33; // [rsp+170h] [rbp-B0h]
  __int64 v34; // [rsp+178h] [rbp-A8h]
  _QWORD v35[2]; // [rsp+180h] [rbp-A0h] BYREF
  unsigned __int64 v36; // [rsp+190h] [rbp-90h]
  _BYTE v37[64]; // [rsp+1A8h] [rbp-78h] BYREF
  __int64 v38; // [rsp+1E8h] [rbp-38h]
  __int64 v39; // [rsp+1F0h] [rbp-30h]
  __int64 v40; // [rsp+1F8h] [rbp-28h]

  sub_1E64160((__int64)v23, a2);
  sub_1E64840((__int64)v17, a2);
  sub_16CCEE0(v35, (__int64)v37, 8, (__int64)v23);
  v3 = v26;
  v26 = 0;
  v38 = v3;
  v4 = v27;
  v27 = 0;
  v39 = v4;
  v5 = v28;
  v28 = 0;
  v40 = v5;
  sub_16CCEE0(v29, (__int64)v31, 8, (__int64)v17);
  v6 = v20;
  v20 = 0;
  v32 = v6;
  v7 = v21;
  v21 = 0;
  v33 = v7;
  v8 = v22;
  v22 = 0;
  v34 = v8;
  sub_16CCEE0(a1, (__int64)(a1 + 5), 8, (__int64)v29);
  v9 = v32;
  v32 = 0;
  a1[13] = v9;
  v10 = v33;
  v33 = 0;
  a1[14] = v10;
  v11 = v34;
  v34 = 0;
  a1[15] = v11;
  sub_16CCEE0(a1 + 16, (__int64)(a1 + 21), 8, (__int64)v35);
  v12 = v38;
  v13 = v32;
  v38 = 0;
  a1[29] = v12;
  v14 = v39;
  v39 = 0;
  a1[30] = v14;
  v15 = v40;
  v40 = 0;
  a1[31] = v15;
  if ( v13 )
    j_j___libc_free_0(v13, v34 - v13);
  if ( v30 != v29[1] )
    _libc_free(v30);
  if ( v38 )
    j_j___libc_free_0(v38, v40 - v38);
  if ( v36 != v35[1] )
    _libc_free(v36);
  if ( v20 )
    j_j___libc_free_0(v20, v22 - v20);
  if ( v19 != v18 )
    _libc_free(v19);
  if ( v26 )
    j_j___libc_free_0(v26, v28 - v26);
  if ( v25 != v24 )
    _libc_free(v25);
  return a1;
}
