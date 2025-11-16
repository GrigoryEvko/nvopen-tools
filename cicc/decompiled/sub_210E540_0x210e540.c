// Function: sub_210E540
// Address: 0x210e540
//
_QWORD *__fastcall sub_210E540(_QWORD *a1, __int64 a2)
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
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v16; // [rsp+0h] [rbp-220h] BYREF
  _QWORD *v17; // [rsp+8h] [rbp-218h]
  _QWORD *v18; // [rsp+10h] [rbp-210h]
  __int64 v19; // [rsp+18h] [rbp-208h]
  int v20; // [rsp+20h] [rbp-200h]
  _QWORD v21[8]; // [rsp+28h] [rbp-1F8h] BYREF
  __int64 v22; // [rsp+68h] [rbp-1B8h] BYREF
  __int64 v23; // [rsp+70h] [rbp-1B0h]
  __int64 v24; // [rsp+78h] [rbp-1A8h]
  _QWORD v25[16]; // [rsp+80h] [rbp-1A0h] BYREF
  _QWORD v26[2]; // [rsp+100h] [rbp-120h] BYREF
  unsigned __int64 v27; // [rsp+110h] [rbp-110h]
  char v28[64]; // [rsp+128h] [rbp-F8h] BYREF
  __int64 v29; // [rsp+168h] [rbp-B8h]
  __int64 v30; // [rsp+170h] [rbp-B0h]
  __int64 v31; // [rsp+178h] [rbp-A8h]
  _QWORD v32[2]; // [rsp+180h] [rbp-A0h] BYREF
  unsigned __int64 v33; // [rsp+190h] [rbp-90h]
  char v34; // [rsp+198h] [rbp-88h]
  char v35[64]; // [rsp+1A8h] [rbp-78h] BYREF
  __int64 v36; // [rsp+1E8h] [rbp-38h]
  __int64 v37; // [rsp+1F0h] [rbp-30h]
  __int64 v38; // [rsp+1F8h] [rbp-28h]

  v34 = 0;
  memset(v25, 0, sizeof(v25));
  v25[1] = &v25[5];
  v25[2] = &v25[5];
  v3 = *(_QWORD *)(a2 + 80);
  v19 = 0x100000008LL;
  LODWORD(v25[3]) = 8;
  v22 = 0;
  if ( v3 )
    v3 -= 24;
  v17 = v21;
  v21[0] = v3;
  v32[0] = v3;
  v18 = v21;
  v23 = 0;
  v24 = 0;
  v20 = 0;
  v16 = 1;
  sub_144A690(&v22, (__int64)v32);
  sub_16CCEE0(v32, (__int64)v35, 8, (__int64)v25);
  v4 = v25[13];
  memset(&v25[13], 0, 24);
  v36 = v4;
  v37 = v25[14];
  v38 = v25[15];
  sub_16CCEE0(v26, (__int64)v28, 8, (__int64)&v16);
  v5 = v22;
  v22 = 0;
  v29 = v5;
  v6 = v23;
  v23 = 0;
  v30 = v6;
  v7 = v24;
  v24 = 0;
  v31 = v7;
  sub_16CCEE0(a1, (__int64)(a1 + 5), 8, (__int64)v26);
  v8 = v29;
  v29 = 0;
  a1[13] = v8;
  v9 = v30;
  v30 = 0;
  a1[14] = v9;
  v10 = v31;
  v31 = 0;
  a1[15] = v10;
  sub_16CCEE0(a1 + 16, (__int64)(a1 + 21), 8, (__int64)v32);
  v11 = v36;
  v12 = v29;
  v36 = 0;
  a1[29] = v11;
  v13 = v37;
  v37 = 0;
  a1[30] = v13;
  v14 = v38;
  v38 = 0;
  a1[31] = v14;
  if ( v12 )
    j_j___libc_free_0(v12, v31 - v12);
  if ( v27 != v26[1] )
    _libc_free(v27);
  if ( v36 )
    j_j___libc_free_0(v36, v38 - v36);
  if ( v33 != v32[1] )
    _libc_free(v33);
  if ( v22 )
    j_j___libc_free_0(v22, v24 - v22);
  if ( v18 != v17 )
    _libc_free((unsigned __int64)v18);
  if ( v25[13] )
    j_j___libc_free_0(v25[13], v25[15] - v25[13]);
  if ( v25[2] != v25[1] )
    _libc_free(v25[2]);
  return a1;
}
