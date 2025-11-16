// Function: sub_104E270
// Address: 0x104e270
//
_QWORD *__fastcall sub_104E270(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v17; // [rsp+10h] [rbp-230h] BYREF
  _QWORD *v18; // [rsp+18h] [rbp-228h]
  __int64 v19; // [rsp+20h] [rbp-220h]
  int v20; // [rsp+28h] [rbp-218h]
  char v21; // [rsp+2Ch] [rbp-214h]
  _QWORD v22[8]; // [rsp+30h] [rbp-210h] BYREF
  __int64 v23; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v24; // [rsp+78h] [rbp-1C8h]
  __int64 v25; // [rsp+80h] [rbp-1C0h]
  _QWORD v26[16]; // [rsp+90h] [rbp-1B0h] BYREF
  _BYTE v27[8]; // [rsp+110h] [rbp-130h] BYREF
  __int64 v28; // [rsp+118h] [rbp-128h]
  char v29; // [rsp+12Ch] [rbp-114h]
  _BYTE v30[64]; // [rsp+130h] [rbp-110h] BYREF
  __int64 v31; // [rsp+170h] [rbp-D0h]
  __int64 v32; // [rsp+178h] [rbp-C8h]
  __int64 v33; // [rsp+180h] [rbp-C0h]
  __m128i v34; // [rsp+190h] [rbp-B0h] BYREF
  char v35; // [rsp+1A8h] [rbp-98h]
  char v36; // [rsp+1ACh] [rbp-94h]
  _BYTE v37[64]; // [rsp+1B0h] [rbp-90h] BYREF
  __int64 v38; // [rsp+1F0h] [rbp-50h]
  __int64 v39; // [rsp+1F8h] [rbp-48h]
  __int64 v40; // [rsp+200h] [rbp-40h]

  v19 = 0x100000008LL;
  memset(v26, 0, 0x78u);
  v3 = *(_QWORD *)(a2 + 80);
  v18 = v22;
  if ( v3 )
    v3 -= 24;
  v26[1] = &v26[4];
  v22[0] = v3;
  v34.m128i_i64[0] = v3;
  LODWORD(v26[2]) = 8;
  BYTE4(v26[3]) = 1;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v20 = 0;
  v21 = 1;
  v17 = 1;
  v35 = 0;
  sub_104E0A0((__int64)&v23, &v34);
  sub_C8CF70((__int64)&v34, v37, 8, (__int64)&v26[4], (__int64)v26);
  v4 = v26[12];
  memset(&v26[12], 0, 24);
  v38 = v4;
  v39 = v26[13];
  v40 = v26[14];
  sub_C8CF70((__int64)v27, v30, 8, (__int64)v22, (__int64)&v17);
  v5 = v23;
  v23 = 0;
  v31 = v5;
  v6 = v24;
  v24 = 0;
  v32 = v6;
  v7 = v25;
  v25 = 0;
  v33 = v7;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v30, (__int64)v27);
  v8 = v31;
  v9 = a1 + 19;
  v31 = 0;
  a1[12] = v8;
  v10 = v32;
  v32 = 0;
  a1[13] = v10;
  v11 = v33;
  v33 = 0;
  a1[14] = v11;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v37, (__int64)&v34);
  v12 = v38;
  v13 = v31;
  v38 = 0;
  a1[27] = v12;
  v14 = v39;
  v39 = 0;
  a1[28] = v14;
  v15 = v40;
  v40 = 0;
  a1[29] = v15;
  if ( v13 )
  {
    v9 = (_QWORD *)(v33 - v13);
    j_j___libc_free_0(v13, v33 - v13);
  }
  if ( !v29 )
    _libc_free(v28, v9);
  if ( v38 )
  {
    v9 = (_QWORD *)(v40 - v38);
    j_j___libc_free_0(v38, v40 - v38);
  }
  if ( !v36 )
    _libc_free(v34.m128i_i64[1], v9);
  if ( v23 )
  {
    v9 = (_QWORD *)(v25 - v23);
    j_j___libc_free_0(v23, v25 - v23);
  }
  if ( !v21 )
    _libc_free(v18, v9);
  if ( v26[12] )
  {
    v9 = (_QWORD *)(v26[14] - v26[12]);
    j_j___libc_free_0(v26[12], v26[14] - v26[12]);
  }
  if ( !BYTE4(v26[3]) )
    _libc_free(v26[1], v9);
  return a1;
}
