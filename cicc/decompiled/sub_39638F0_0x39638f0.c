// Function: sub_39638F0
// Address: 0x39638f0
//
_QWORD *__fastcall sub_39638F0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // rax
  __m128i *v5; // rsi
  unsigned __int64 v6; // rax
  const __m128i *v7; // rax
  __m128i *v8; // rax
  const __m128i *v9; // rax
  const __m128i *v10; // rax
  __m128i *v11; // rax
  const __m128i *v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v18; // [rsp+0h] [rbp-230h] BYREF
  _BYTE *v19; // [rsp+8h] [rbp-228h]
  _BYTE *v20; // [rsp+10h] [rbp-220h]
  __int64 v21; // [rsp+18h] [rbp-218h]
  int v22; // [rsp+20h] [rbp-210h]
  _BYTE v23[64]; // [rsp+28h] [rbp-208h] BYREF
  const __m128i *v24; // [rsp+68h] [rbp-1C8h] BYREF
  __m128i *v25; // [rsp+70h] [rbp-1C0h]
  const __m128i *v26; // [rsp+78h] [rbp-1B8h]
  _QWORD v27[16]; // [rsp+80h] [rbp-1B0h] BYREF
  _QWORD v28[2]; // [rsp+100h] [rbp-130h] BYREF
  unsigned __int64 v29; // [rsp+110h] [rbp-120h]
  char v30[64]; // [rsp+128h] [rbp-108h] BYREF
  const __m128i *v31; // [rsp+168h] [rbp-C8h]
  __m128i *v32; // [rsp+170h] [rbp-C0h]
  const __m128i *v33; // [rsp+178h] [rbp-B8h]
  __m128i v34; // [rsp+180h] [rbp-B0h] BYREF
  unsigned __int64 v35; // [rsp+190h] [rbp-A0h]
  char v36[64]; // [rsp+1A8h] [rbp-88h] BYREF
  unsigned __int64 v37; // [rsp+1E8h] [rbp-48h]
  __int64 v38; // [rsp+1F0h] [rbp-40h]
  __int64 v39; // [rsp+1F8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 80);
  v18 = 0;
  memset(v27, 0, sizeof(v27));
  LODWORD(v27[3]) = 8;
  v27[1] = &v27[5];
  v27[2] = &v27[5];
  v21 = 8;
  if ( v3 )
    v3 -= 24;
  v19 = v23;
  v20 = v23;
  v22 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  sub_1412190((__int64)&v18, v3);
  v4 = sub_157EBA0(v3);
  v34.m128i_i64[0] = v3;
  v5 = v25;
  v34.m128i_i64[1] = v4;
  LODWORD(v35) = 0;
  if ( v25 == v26 )
  {
    sub_13FDF40(&v24, v25, &v34);
  }
  else
  {
    if ( v25 )
    {
      *v25 = _mm_loadu_si128(&v34);
      v5[1].m128i_i64[0] = v35;
      v5 = v25;
    }
    v25 = (__m128i *)((char *)v5 + 24);
  }
  sub_1B88860((__int64)&v18);
  sub_16CCEE0(&v34, (__int64)v36, 8, (__int64)v27);
  v6 = v27[13];
  memset(&v27[13], 0, 24);
  v37 = v6;
  v38 = v27[14];
  v39 = v27[15];
  sub_16CCEE0(v28, (__int64)v30, 8, (__int64)&v18);
  v7 = v24;
  v24 = 0;
  v31 = v7;
  v8 = v25;
  v25 = 0;
  v32 = v8;
  v9 = v26;
  v26 = 0;
  v33 = v9;
  sub_16CCEE0(a1, (__int64)(a1 + 5), 8, (__int64)v28);
  v10 = v31;
  v31 = 0;
  a1[13] = v10;
  v11 = v32;
  v32 = 0;
  a1[14] = v11;
  v12 = v33;
  v33 = 0;
  a1[15] = v12;
  sub_16CCEE0(a1 + 16, (__int64)(a1 + 21), 8, (__int64)&v34);
  v13 = v37;
  v14 = (unsigned __int64)v31;
  v37 = 0;
  a1[29] = v13;
  v15 = v38;
  v38 = 0;
  a1[30] = v15;
  v16 = v39;
  v39 = 0;
  a1[31] = v16;
  if ( v14 )
    j_j___libc_free_0(v14);
  if ( v29 != v28[1] )
    _libc_free(v29);
  if ( v37 )
    j_j___libc_free_0(v37);
  if ( v35 != v34.m128i_i64[1] )
    _libc_free(v35);
  if ( v24 )
    j_j___libc_free_0((unsigned __int64)v24);
  if ( v20 != v19 )
    _libc_free((unsigned __int64)v20);
  if ( v27[13] )
    j_j___libc_free_0(v27[13]);
  if ( v27[2] != v27[1] )
    _libc_free(v27[2]);
  return a1;
}
