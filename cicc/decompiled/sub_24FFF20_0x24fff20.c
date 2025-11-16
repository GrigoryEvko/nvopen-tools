// Function: sub_24FFF20
// Address: 0x24fff20
//
_QWORD *__fastcall sub_24FFF20(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v8; // rsi
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  __m128i *v15; // rdx
  const __m128i *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  const __m128i *v21; // rcx
  __int64 v22; // rax
  unsigned __int64 v23; // r8
  __m128i *v24; // rdx
  const __m128i *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  const __m128i *v33; // rax
  unsigned __int64 v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned __int64 v40; // [rsp+8h] [rbp-258h]
  char v41[8]; // [rsp+30h] [rbp-230h] BYREF
  unsigned __int64 v42; // [rsp+38h] [rbp-228h]
  char v43; // [rsp+4Ch] [rbp-214h]
  _BYTE v44[64]; // [rsp+50h] [rbp-210h] BYREF
  unsigned __int64 v45; // [rsp+90h] [rbp-1D0h]
  unsigned __int64 v46; // [rsp+98h] [rbp-1C8h]
  unsigned __int64 v47; // [rsp+A0h] [rbp-1C0h]
  char v48[8]; // [rsp+B0h] [rbp-1B0h] BYREF
  unsigned __int64 v49; // [rsp+B8h] [rbp-1A8h]
  char v50; // [rsp+CCh] [rbp-194h]
  _BYTE v51[64]; // [rsp+D0h] [rbp-190h] BYREF
  unsigned __int64 v52; // [rsp+110h] [rbp-150h]
  unsigned __int64 v53; // [rsp+118h] [rbp-148h]
  __int64 v54; // [rsp+120h] [rbp-140h]
  __int64 v55; // [rsp+130h] [rbp-130h] BYREF
  _QWORD *v56; // [rsp+138h] [rbp-128h]
  __int64 v57; // [rsp+140h] [rbp-120h]
  int v58; // [rsp+148h] [rbp-118h]
  char v59; // [rsp+14Ch] [rbp-114h]
  _QWORD v60[8]; // [rsp+150h] [rbp-110h] BYREF
  const __m128i *v61; // [rsp+190h] [rbp-D0h] BYREF
  const __m128i *v62; // [rsp+198h] [rbp-C8h]
  unsigned __int64 v63; // [rsp+1A0h] [rbp-C0h]
  __m128i v64[11]; // [rsp+1B0h] [rbp-B0h] BYREF

  memset(v64, 0, 0x78u);
  v64[0].m128i_i64[1] = (__int64)v64[2].m128i_i64;
  v8 = v51;
  v64[1].m128i_i32[0] = 8;
  v64[1].m128i_i8[12] = 1;
  sub_C8CD80((__int64)v48, (__int64)v51, (__int64)v64, 0, a5, a6);
  v10 = v64[6].m128i_i64[1];
  v11 = v64[6].m128i_u64[0];
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v12 = v64[6].m128i_i64[1] - v64[6].m128i_i64[0];
  if ( v64[6].m128i_i64[1] == v64[6].m128i_i64[0] )
  {
    v14 = 0;
  }
  else
  {
    if ( v12 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_46;
    v13 = sub_22077B0(v64[6].m128i_i64[1] - v64[6].m128i_i64[0]);
    v10 = v64[6].m128i_i64[1];
    v11 = v64[6].m128i_u64[0];
    v14 = v13;
  }
  v52 = v14;
  v53 = v14;
  v54 = v14 + v12;
  if ( v10 != v11 )
  {
    v15 = (__m128i *)v14;
    v16 = (const __m128i *)v11;
    do
    {
      if ( v15 )
      {
        *v15 = _mm_loadu_si128(v16);
        v15[1].m128i_i64[0] = v16[1].m128i_i64[0];
      }
      v16 = (const __m128i *)((char *)v16 + 24);
      v15 = (__m128i *)((char *)v15 + 24);
    }
    while ( v16 != (const __m128i *)v10 );
    v14 += 8 * (((unsigned __int64)&v16[-2].m128i_u64[1] - v11) >> 3) + 24;
  }
  v53 = v14;
  if ( v11 )
    j_j___libc_free_0(v11);
  if ( !v64[1].m128i_i8[12] )
    _libc_free(v64[0].m128i_u64[1]);
  v17 = *a2;
  v57 = 0x100000008LL;
  v60[0] = v17;
  v64[0].m128i_i64[0] = v17;
  v56 = v60;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v58 = 0;
  v59 = 1;
  v55 = 1;
  v64[1].m128i_i8[0] = 0;
  sub_24FFD80((unsigned __int64 *)&v61, v64);
  v8 = v44;
  sub_C8CD80((__int64)v41, (__int64)v44, (__int64)&v55, v18, v19, v20);
  v21 = v62;
  v11 = (unsigned __int64)v61;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v9 = (char *)v62 - (char *)v61;
  if ( v62 != v61 )
  {
    if ( v9 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v40 = (char *)v62 - (char *)v61;
      v22 = sub_22077B0((char *)v62 - (char *)v61);
      v21 = v62;
      v11 = (unsigned __int64)v61;
      v9 = v40;
      v23 = v22;
      goto LABEL_17;
    }
LABEL_46:
    sub_4261EA(v11, v8, v9);
  }
  v23 = 0;
LABEL_17:
  v45 = v23;
  v46 = v23;
  v47 = v23 + v9;
  if ( v21 != (const __m128i *)v11 )
  {
    v24 = (__m128i *)v23;
    v25 = (const __m128i *)v11;
    do
    {
      if ( v24 )
      {
        *v24 = _mm_loadu_si128(v25);
        v24[1].m128i_i64[0] = v25[1].m128i_i64[0];
      }
      v25 = (const __m128i *)((char *)v25 + 24);
      v24 = (__m128i *)((char *)v24 + 24);
    }
    while ( v21 != v25 );
    v23 += 8 * (((unsigned __int64)&v21[-2].m128i_u64[1] - v11) >> 3) + 24;
  }
  v46 = v23;
  if ( v11 )
    j_j___libc_free_0(v11);
  if ( !v59 )
    _libc_free((unsigned __int64)v56);
  sub_C8CF70((__int64)v64, &v64[2], 8, (__int64)v51, (__int64)v48);
  v26 = v52;
  v52 = 0;
  v64[6].m128i_i64[0] = v26;
  v27 = v53;
  v53 = 0;
  v64[6].m128i_i64[1] = v27;
  v28 = v54;
  v54 = 0;
  v64[7].m128i_i64[0] = v28;
  sub_C8CF70((__int64)&v55, v60, 8, (__int64)v44, (__int64)v41);
  v29 = v45;
  v45 = 0;
  v61 = (const __m128i *)v29;
  v30 = v46;
  v46 = 0;
  v62 = (const __m128i *)v30;
  v31 = v47;
  v47 = 0;
  v63 = v31;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v60, (__int64)&v55);
  v32 = (unsigned __int64)v61;
  v61 = 0;
  a1[12] = v32;
  v33 = v62;
  v62 = 0;
  a1[13] = v33;
  v34 = v63;
  v63 = 0;
  a1[14] = v34;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v64[2].m128i_i64, (__int64)v64);
  v35 = v64[6].m128i_i64[0];
  v36 = (unsigned __int64)v61;
  v64[6].m128i_i64[0] = 0;
  a1[27] = v35;
  v37 = v64[6].m128i_i64[1];
  v64[6].m128i_i64[1] = 0;
  a1[28] = v37;
  v38 = v64[7].m128i_i64[0];
  v64[7].m128i_i64[0] = 0;
  a1[29] = v38;
  if ( v36 )
    j_j___libc_free_0(v36);
  if ( !v59 )
    _libc_free((unsigned __int64)v56);
  if ( v64[6].m128i_i64[0] )
    j_j___libc_free_0(v64[6].m128i_u64[0]);
  if ( !v64[1].m128i_i8[12] )
    _libc_free(v64[0].m128i_u64[1]);
  if ( v45 )
    j_j___libc_free_0(v45);
  if ( !v43 )
    _libc_free(v42);
  if ( v52 )
    j_j___libc_free_0(v52);
  if ( !v50 )
    _libc_free(v49);
  return a1;
}
