// Function: sub_2FC1490
// Address: 0x2fc1490
//
_QWORD *__fastcall sub_2FC1490(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v16; // [rsp+10h] [rbp-230h] BYREF
  _QWORD *v17; // [rsp+18h] [rbp-228h]
  __int64 v18; // [rsp+20h] [rbp-220h]
  int v19; // [rsp+28h] [rbp-218h]
  char v20; // [rsp+2Ch] [rbp-214h]
  _QWORD v21[8]; // [rsp+30h] [rbp-210h] BYREF
  unsigned __int64 v22; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v23; // [rsp+78h] [rbp-1C8h]
  __int64 v24; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v25[16]; // [rsp+90h] [rbp-1B0h] BYREF
  char v26[8]; // [rsp+110h] [rbp-130h] BYREF
  unsigned __int64 v27; // [rsp+118h] [rbp-128h]
  char v28; // [rsp+12Ch] [rbp-114h]
  _BYTE v29[64]; // [rsp+130h] [rbp-110h] BYREF
  unsigned __int64 v30; // [rsp+170h] [rbp-D0h]
  __int64 v31; // [rsp+178h] [rbp-C8h]
  __int64 v32; // [rsp+180h] [rbp-C0h]
  __m128i v33; // [rsp+190h] [rbp-B0h] BYREF
  char v34; // [rsp+1A0h] [rbp-A0h]
  char v35; // [rsp+1ACh] [rbp-94h]
  _BYTE v36[64]; // [rsp+1B0h] [rbp-90h] BYREF
  unsigned __int64 v37; // [rsp+1F0h] [rbp-50h]
  unsigned __int64 v38; // [rsp+1F8h] [rbp-48h]
  unsigned __int64 v39; // [rsp+200h] [rbp-40h]

  memset(v25, 0, 0x78u);
  v3 = *(_QWORD *)(a2 + 328);
  v17 = v21;
  v21[0] = v3;
  v33.m128i_i64[0] = v3;
  v25[1] = (unsigned __int64)&v25[4];
  v18 = 0x100000008LL;
  LODWORD(v25[2]) = 8;
  BYTE4(v25[3]) = 1;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v19 = 0;
  v20 = 1;
  v16 = 1;
  v34 = 0;
  sub_2FC1350(&v22, &v33);
  sub_C8CF70((__int64)&v33, v36, 8, (__int64)&v25[4], (__int64)v25);
  v4 = v25[12];
  memset(&v25[12], 0, 24);
  v37 = v4;
  v38 = v25[13];
  v39 = v25[14];
  sub_C8CF70((__int64)v26, v29, 8, (__int64)v21, (__int64)&v16);
  v5 = v22;
  v22 = 0;
  v30 = v5;
  v6 = v23;
  v23 = 0;
  v31 = v6;
  v7 = v24;
  v24 = 0;
  v32 = v7;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v29, (__int64)v26);
  v8 = v30;
  v30 = 0;
  a1[12] = v8;
  v9 = v31;
  v31 = 0;
  a1[13] = v9;
  v10 = v32;
  v32 = 0;
  a1[14] = v10;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v36, (__int64)&v33);
  v11 = v37;
  v12 = v30;
  v37 = 0;
  a1[27] = v11;
  v13 = v38;
  v38 = 0;
  a1[28] = v13;
  v14 = v39;
  v39 = 0;
  a1[29] = v14;
  if ( v12 )
    j_j___libc_free_0(v12);
  if ( !v28 )
    _libc_free(v27);
  if ( v37 )
    j_j___libc_free_0(v37);
  if ( !v35 )
    _libc_free(v33.m128i_u64[1]);
  if ( v22 )
    j_j___libc_free_0(v22);
  if ( !v20 )
    _libc_free((unsigned __int64)v17);
  if ( v25[12] )
    j_j___libc_free_0(v25[12]);
  if ( !BYTE4(v25[3]) )
    _libc_free(v25[1]);
  return a1;
}
