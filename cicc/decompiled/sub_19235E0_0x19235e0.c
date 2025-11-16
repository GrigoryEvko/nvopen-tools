// Function: sub_19235E0
// Address: 0x19235e0
//
_QWORD *__fastcall sub_19235E0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-220h] BYREF
  _QWORD *v16; // [rsp+8h] [rbp-218h]
  _QWORD *v17; // [rsp+10h] [rbp-210h]
  __int64 v18; // [rsp+18h] [rbp-208h]
  int v19; // [rsp+20h] [rbp-200h]
  _QWORD v20[8]; // [rsp+28h] [rbp-1F8h] BYREF
  __int64 v21; // [rsp+68h] [rbp-1B8h] BYREF
  __int64 v22; // [rsp+70h] [rbp-1B0h]
  __int64 v23; // [rsp+78h] [rbp-1A8h]
  _QWORD v24[16]; // [rsp+80h] [rbp-1A0h] BYREF
  _QWORD v25[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v26; // [rsp+110h] [rbp-110h]
  char v27[64]; // [rsp+128h] [rbp-F8h] BYREF
  __int64 v28; // [rsp+168h] [rbp-B8h]
  __int64 v29; // [rsp+170h] [rbp-B0h]
  __int64 v30; // [rsp+178h] [rbp-A8h]
  _QWORD v31[2]; // [rsp+180h] [rbp-A0h] BYREF
  unsigned __int64 v32; // [rsp+190h] [rbp-90h]
  char v33; // [rsp+198h] [rbp-88h]
  char v34[64]; // [rsp+1A8h] [rbp-78h] BYREF
  __int64 v35; // [rsp+1E8h] [rbp-38h]
  __int64 v36; // [rsp+1F0h] [rbp-30h]
  __int64 v37; // [rsp+1F8h] [rbp-28h]

  v20[0] = a2;
  memset(v24, 0, sizeof(v24));
  v31[0] = a2;
  v24[1] = &v24[5];
  v24[2] = &v24[5];
  v16 = v20;
  v17 = v20;
  v18 = 0x100000008LL;
  LODWORD(v24[3]) = 8;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v19 = 0;
  v15 = 1;
  v33 = 0;
  sub_144A690(&v21, (__int64)v31);
  sub_16CCEE0(v31, (__int64)v34, 8, (__int64)v24);
  v3 = v24[13];
  memset(&v24[13], 0, 24);
  v35 = v3;
  v36 = v24[14];
  v37 = v24[15];
  sub_16CCEE0(v25, (__int64)v27, 8, (__int64)&v15);
  v4 = v21;
  v21 = 0;
  v28 = v4;
  v5 = v22;
  v22 = 0;
  v29 = v5;
  v6 = v23;
  v23 = 0;
  v30 = v6;
  sub_16CCEE0(a1, (__int64)(a1 + 5), 8, (__int64)v25);
  v7 = v28;
  v28 = 0;
  a1[13] = v7;
  v8 = v29;
  v29 = 0;
  a1[14] = v8;
  v9 = v30;
  v30 = 0;
  a1[15] = v9;
  sub_16CCEE0(a1 + 16, (__int64)(a1 + 21), 8, (__int64)v31);
  v10 = v35;
  v11 = v28;
  v35 = 0;
  a1[29] = v10;
  v12 = v36;
  v36 = 0;
  a1[30] = v12;
  v13 = v37;
  v37 = 0;
  a1[31] = v13;
  if ( v11 )
    j_j___libc_free_0(v11, v30 - v11);
  if ( v26 != v25[1] )
    _libc_free(v26);
  if ( v35 )
    j_j___libc_free_0(v35, v37 - v35);
  if ( v32 != v31[1] )
    _libc_free(v32);
  if ( v21 )
    j_j___libc_free_0(v21, v23 - v21);
  if ( v17 != v16 )
    _libc_free((unsigned __int64)v17);
  if ( v24[13] )
    j_j___libc_free_0(v24[13], v24[15] - v24[13]);
  if ( v24[2] != v24[1] )
    _libc_free(v24[2]);
  return a1;
}
