// Function: sub_22E71B0
// Address: 0x22e71b0
//
_QWORD *__fastcall sub_22E71B0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rax
  _QWORD *v4; // rdi
  _QWORD *v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v18; // [rsp+10h] [rbp-230h] BYREF
  _QWORD *v19; // [rsp+18h] [rbp-228h]
  __int64 v20; // [rsp+20h] [rbp-220h]
  int v21; // [rsp+28h] [rbp-218h]
  char v22; // [rsp+2Ch] [rbp-214h]
  _QWORD v23[8]; // [rsp+30h] [rbp-210h] BYREF
  unsigned __int64 v24; // [rsp+70h] [rbp-1D0h] BYREF
  __int64 v25; // [rsp+78h] [rbp-1C8h]
  __int64 v26; // [rsp+80h] [rbp-1C0h]
  unsigned __int64 v27[16]; // [rsp+90h] [rbp-1B0h] BYREF
  char v28[8]; // [rsp+110h] [rbp-130h] BYREF
  unsigned __int64 v29; // [rsp+118h] [rbp-128h]
  char v30; // [rsp+12Ch] [rbp-114h]
  _BYTE v31[64]; // [rsp+130h] [rbp-110h] BYREF
  unsigned __int64 v32; // [rsp+170h] [rbp-D0h]
  __int64 v33; // [rsp+178h] [rbp-C8h]
  __int64 v34; // [rsp+180h] [rbp-C0h]
  __m128i v35; // [rsp+190h] [rbp-B0h] BYREF
  char v36; // [rsp+1ACh] [rbp-94h]
  _BYTE v37[64]; // [rsp+1B0h] [rbp-90h] BYREF
  unsigned __int64 v38; // [rsp+1F0h] [rbp-50h]
  unsigned __int64 v39; // [rsp+1F8h] [rbp-48h]
  unsigned __int64 v40; // [rsp+200h] [rbp-40h]

  sub_22DDF00(*(_QWORD **)(*a2 + 32), **(_QWORD **)(*a2 + 32) & 0xFFFFFFFFFFFFFFF8LL);
  memset(v27, 0, 0x78u);
  v3 = *a2;
  v27[1] = (unsigned __int64)&v27[4];
  LODWORD(v27[2]) = 8;
  v4 = *(_QWORD **)(v3 + 32);
  BYTE4(v27[3]) = 1;
  v5 = sub_22DDF00(v4, *v4 & 0xFFFFFFFFFFFFFFF8LL);
  v19 = v23;
  v23[0] = v5;
  v35.m128i_i64[0] = (__int64)v5;
  v20 = 0x100000008LL;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v21 = 0;
  v22 = 1;
  v18 = 1;
  v37[0] = 0;
  sub_22E6150(&v24, &v35);
  sub_C8CF70((__int64)&v35, v37, 8, (__int64)&v27[4], (__int64)v27);
  v6 = v27[12];
  memset(&v27[12], 0, 24);
  v38 = v6;
  v39 = v27[13];
  v40 = v27[14];
  sub_C8CF70((__int64)v28, v31, 8, (__int64)v23, (__int64)&v18);
  v7 = v24;
  v24 = 0;
  v32 = v7;
  v8 = v25;
  v25 = 0;
  v33 = v8;
  v9 = v26;
  v26 = 0;
  v34 = v9;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v31, (__int64)v28);
  v10 = v32;
  v32 = 0;
  a1[12] = v10;
  v11 = v33;
  v33 = 0;
  a1[13] = v11;
  v12 = v34;
  v34 = 0;
  a1[14] = v12;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v37, (__int64)&v35);
  v13 = v38;
  v14 = v32;
  v38 = 0;
  a1[27] = v13;
  v15 = v39;
  v39 = 0;
  a1[28] = v15;
  v16 = v40;
  v40 = 0;
  a1[29] = v16;
  if ( v14 )
    j_j___libc_free_0(v14);
  if ( !v30 )
    _libc_free(v29);
  if ( v38 )
    j_j___libc_free_0(v38);
  if ( !v36 )
    _libc_free(v35.m128i_u64[1]);
  if ( v24 )
    j_j___libc_free_0(v24);
  if ( !v22 )
    _libc_free((unsigned __int64)v19);
  if ( v27[12] )
    j_j___libc_free_0(v27[12]);
  if ( !BYTE4(v27[3]) )
    _libc_free(v27[1]);
  return a1;
}
