// Function: sub_27A4830
// Address: 0x27a4830
//
_QWORD *__fastcall sub_27A4830(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v15; // [rsp+10h] [rbp-230h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-228h]
  __int64 v17; // [rsp+20h] [rbp-220h]
  int v18; // [rsp+28h] [rbp-218h]
  char v19; // [rsp+2Ch] [rbp-214h]
  _QWORD v20[8]; // [rsp+30h] [rbp-210h] BYREF
  unsigned __int64 v21; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v22; // [rsp+78h] [rbp-1C8h]
  __int64 v23; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v24[16]; // [rsp+90h] [rbp-1B0h] BYREF
  char v25[8]; // [rsp+110h] [rbp-130h] BYREF
  unsigned __int64 v26; // [rsp+118h] [rbp-128h]
  char v27; // [rsp+12Ch] [rbp-114h]
  _BYTE v28[64]; // [rsp+130h] [rbp-110h] BYREF
  unsigned __int64 v29; // [rsp+170h] [rbp-D0h]
  __int64 v30; // [rsp+178h] [rbp-C8h]
  __int64 v31; // [rsp+180h] [rbp-C0h]
  __m128i v32; // [rsp+190h] [rbp-B0h] BYREF
  char v33; // [rsp+1A8h] [rbp-98h]
  char v34; // [rsp+1ACh] [rbp-94h]
  _BYTE v35[64]; // [rsp+1B0h] [rbp-90h] BYREF
  unsigned __int64 v36; // [rsp+1F0h] [rbp-50h]
  unsigned __int64 v37; // [rsp+1F8h] [rbp-48h]
  unsigned __int64 v38; // [rsp+200h] [rbp-40h]

  v20[0] = a2;
  memset(v24, 0, 0x78u);
  v32.m128i_i64[0] = a2;
  v16 = v20;
  v17 = 0x100000008LL;
  v24[1] = (unsigned __int64)&v24[4];
  LODWORD(v24[2]) = 8;
  BYTE4(v24[3]) = 1;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v18 = 0;
  v19 = 1;
  v15 = 1;
  v33 = 0;
  sub_27A47F0((__int64)&v21, &v32);
  sub_C8CF70((__int64)&v32, v35, 8, (__int64)&v24[4], (__int64)v24);
  v3 = v24[12];
  memset(&v24[12], 0, 24);
  v36 = v3;
  v37 = v24[13];
  v38 = v24[14];
  sub_C8CF70((__int64)v25, v28, 8, (__int64)v20, (__int64)&v15);
  v4 = v21;
  v21 = 0;
  v29 = v4;
  v5 = v22;
  v22 = 0;
  v30 = v5;
  v6 = v23;
  v23 = 0;
  v31 = v6;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v28, (__int64)v25);
  v7 = v29;
  v29 = 0;
  a1[12] = v7;
  v8 = v30;
  v30 = 0;
  a1[13] = v8;
  v9 = v31;
  v31 = 0;
  a1[14] = v9;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v35, (__int64)&v32);
  v10 = v36;
  v11 = v29;
  v36 = 0;
  a1[27] = v10;
  v12 = v37;
  v37 = 0;
  a1[28] = v12;
  v13 = v38;
  v38 = 0;
  a1[29] = v13;
  if ( v11 )
    j_j___libc_free_0(v11);
  if ( !v27 )
    _libc_free(v26);
  if ( v36 )
    j_j___libc_free_0(v36);
  if ( !v34 )
    _libc_free(v32.m128i_u64[1]);
  if ( v21 )
    j_j___libc_free_0(v21);
  if ( !v19 )
    _libc_free((unsigned __int64)v16);
  if ( v24[12] )
    j_j___libc_free_0(v24[12]);
  if ( !BYTE4(v24[3]) )
    _libc_free(v24[1]);
  return a1;
}
