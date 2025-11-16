// Function: sub_3715FB0
// Address: 0x3715fb0
//
__m128i *__fastcall sub_3715FB0(
        __m128i *a1,
        _QWORD *a2,
        unsigned __int16 a3,
        const __m128i *a4,
        __int64 a5,
        __int64 a6)
{
  __m128i *v8; // rdi
  const __m128i *v9; // rbx
  const __m128i *v10; // r15
  __int64 v11; // rcx
  int v12; // r13d
  int v13; // eax
  __int64 v14; // rax
  const __m128i *v15; // r14
  unsigned __int64 v16; // rdx
  __m128i *v17; // rax
  __int64 v18; // rbx
  __m128i *v19; // r13
  unsigned __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __m128i *v24; // rbx
  const __m128i *v25; // rdi
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rax
  __m128i *v28; // rsi
  _BYTE *v29; // rsi
  __int64 v30; // rdx
  __m128i *v31; // rax
  size_t v32; // rcx
  __m128i *v33; // r9
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rdi
  __m128i *v36; // rax
  unsigned __int64 v37; // rcx
  __m128i *v38; // rdx
  unsigned __int64 *v39; // rax
  __int64 v40; // rdx
  __m128i *v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rax
  signed __int64 v44; // r14
  __m128i *v45; // [rsp+18h] [rbp-2C8h]
  __m128i *v46; // [rsp+50h] [rbp-290h] BYREF
  __int64 v47; // [rsp+58h] [rbp-288h]
  __m128i v48; // [rsp+60h] [rbp-280h] BYREF
  _QWORD *v49; // [rsp+70h] [rbp-270h] BYREF
  __int64 v50; // [rsp+78h] [rbp-268h]
  _BYTE v51[16]; // [rsp+80h] [rbp-260h] BYREF
  __m128i *v52; // [rsp+90h] [rbp-250h] BYREF
  size_t v53; // [rsp+98h] [rbp-248h]
  __m128i v54; // [rsp+A0h] [rbp-240h] BYREF
  char *v55; // [rsp+B0h] [rbp-230h] BYREF
  size_t v56; // [rsp+B8h] [rbp-228h]
  _QWORD v57[2]; // [rsp+C0h] [rbp-220h] BYREF
  __m128i *v58; // [rsp+D0h] [rbp-210h] BYREF
  __int64 v59; // [rsp+D8h] [rbp-208h]
  __m128i v60; // [rsp+E0h] [rbp-200h] BYREF
  __m128i *v61; // [rsp+F0h] [rbp-1F0h] BYREF
  size_t v62; // [rsp+F8h] [rbp-1E8h]
  __m128i v63; // [rsp+100h] [rbp-1E0h] BYREF
  __m128i *v64; // [rsp+110h] [rbp-1D0h] BYREF
  __int64 v65; // [rsp+118h] [rbp-1C8h]
  _BYTE v66[448]; // [rsp+120h] [rbp-1C0h] BYREF

  if ( !a2[7] || a2[5] || a2[6] )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_370CD40(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v8 = (__m128i *)v66;
  v9 = a4;
  v65 = 0xA00000000LL;
  v10 = (const __m128i *)((char *)a4 + 40 * a5);
  v64 = (__m128i *)v66;
  if ( a4 == v10 )
    goto LABEL_51;
  v11 = 0;
  v12 = a3;
  do
  {
    v13 = v9[2].m128i_u8[0];
    if ( (_BYTE)v13 && v13 == (v13 & v12) )
    {
      v14 = (unsigned int)v11;
      v15 = v9;
      v16 = (unsigned int)v11 + 1LL;
      if ( v16 > HIDWORD(v65) )
      {
        if ( v8 > v9 || (__m128i *)((char *)v8 + 40 * (unsigned int)v11) <= v9 )
        {
          sub_C8D5F0((__int64)&v64, v66, v16, 0x28u, a5, a6);
          v8 = v64;
          v14 = (unsigned int)v65;
        }
        else
        {
          v44 = (char *)v9 - (char *)v8;
          sub_C8D5F0((__int64)&v64, v66, v16, 0x28u, a5, a6);
          v8 = v64;
          v14 = (unsigned int)v65;
          v15 = (__m128i *)((char *)v64 + v44);
        }
      }
      v17 = (__m128i *)((char *)v8 + 40 * v14);
      *v17 = _mm_loadu_si128(v15);
      v8 = v64;
      v17[1] = _mm_loadu_si128(v15 + 1);
      v17[2].m128i_i64[0] = v15[2].m128i_i64[0];
      v11 = (unsigned int)(v65 + 1);
      LODWORD(v65) = v65 + 1;
    }
    v9 = (const __m128i *)((char *)v9 + 40);
  }
  while ( v9 != v10 );
  v18 = 40 * v11;
  v19 = (__m128i *)((char *)v8 + 40 * v11);
  if ( v8 == v19 )
  {
LABEL_51:
    v48.m128i_i8[0] = 0;
LABEL_52:
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    goto LABEL_53;
  }
  _BitScanReverse64(&v20, 0xCCCCCCCCCCCCCCCDLL * (v18 >> 3));
  sub_3715DD0(
    (__int64)v8,
    v19,
    2LL * (int)(63 - (v20 ^ 0x3F)),
    (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))sub_370CC80,
    a5,
    a6);
  if ( (unsigned __int64)v18 <= 0x280 )
  {
    sub_3712270(v8, v19, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CC80, v21, v22, v23);
  }
  else
  {
    v24 = v8 + 40;
    sub_3712270(v8, v8 + 40, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CC80, v21, v22, v23);
    if ( &v8[40] != v19 )
    {
      do
      {
        v25 = v24;
        v24 = (__m128i *)((char *)v24 + 40);
        sub_37121E0(v25, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CC80);
      }
      while ( v24 != v19 );
    }
  }
  v26 = (unsigned __int64)v64;
  v47 = 0;
  v46 = &v48;
  v48.m128i_i8[0] = 0;
  v45 = (__m128i *)((char *)v64 + 40 * (unsigned int)v65);
  if ( v64 == v45 )
    goto LABEL_52;
  v27 = v64[2].m128i_u8[0];
  if ( v64[2].m128i_i8[0] )
    goto LABEL_48;
LABEL_19:
  v63.m128i_i8[0] = 48;
  v28 = &v63;
  while ( 1 )
  {
    v55 = (char *)v57;
    sub_370CBD0((__int64 *)&v55, v28, (__int64)v63.m128i_i64 + 1);
    v29 = *(_BYTE **)v26;
    if ( *(_QWORD *)v26 )
    {
      v30 = (__int64)&v29[*(_QWORD *)(v26 + 8)];
      v49 = v51;
      sub_370CD40((__int64 *)&v49, v29, v30);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v50) <= 3 )
        goto LABEL_75;
    }
    else
    {
      v51[0] = 0;
      v50 = 0;
      v49 = v51;
    }
    v31 = (__m128i *)sub_2241490((unsigned __int64 *)&v49, " (0x", 4u);
    v52 = &v54;
    if ( (__m128i *)v31->m128i_i64[0] == &v31[1] )
    {
      v54 = _mm_loadu_si128(v31 + 1);
    }
    else
    {
      v52 = (__m128i *)v31->m128i_i64[0];
      v54.m128i_i64[0] = v31[1].m128i_i64[0];
    }
    v32 = v31->m128i_u64[1];
    v31[1].m128i_i8[0] = 0;
    v53 = v32;
    v31->m128i_i64[0] = (__int64)v31[1].m128i_i64;
    v33 = v52;
    v31->m128i_i64[1] = 0;
    v34 = 15;
    v35 = 15;
    if ( v33 != &v54 )
      v35 = v54.m128i_i64[0];
    if ( v53 + v56 <= v35 )
      goto LABEL_30;
    if ( v55 != (char *)v57 )
      v34 = v57[0];
    if ( v53 + v56 <= v34 )
    {
      v36 = (__m128i *)sub_2241130((unsigned __int64 *)&v55, 0, 0, v33, v53);
      v38 = v36 + 1;
      v58 = &v60;
      v37 = v36->m128i_i64[0];
      if ( (__m128i *)v36->m128i_i64[0] != &v36[1] )
      {
LABEL_31:
        v58 = (__m128i *)v37;
        v60.m128i_i64[0] = v36[1].m128i_i64[0];
        goto LABEL_32;
      }
    }
    else
    {
LABEL_30:
      v36 = (__m128i *)sub_2241490((unsigned __int64 *)&v52, v55, v56);
      v58 = &v60;
      v37 = v36->m128i_i64[0];
      v38 = v36 + 1;
      if ( (__m128i *)v36->m128i_i64[0] != &v36[1] )
        goto LABEL_31;
    }
    v60 = _mm_loadu_si128(v36 + 1);
LABEL_32:
    v59 = v36->m128i_i64[1];
    v36->m128i_i64[0] = (__int64)v38;
    v36->m128i_i64[1] = 0;
    v36[1].m128i_i8[0] = 0;
    if ( v59 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_75;
    v39 = sub_2241490((unsigned __int64 *)&v58, ")", 1u);
    v61 = &v63;
    if ( (unsigned __int64 *)*v39 == v39 + 2 )
    {
      v63 = _mm_loadu_si128((const __m128i *)v39 + 1);
    }
    else
    {
      v61 = (__m128i *)*v39;
      v63.m128i_i64[0] = v39[2];
    }
    v62 = v39[1];
    *v39 = (unsigned __int64)(v39 + 2);
    v39[1] = 0;
    *((_BYTE *)v39 + 16) = 0;
    sub_2241490((unsigned __int64 *)&v46, v61->m128i_i8, v62);
    if ( v61 != &v63 )
      j_j___libc_free_0((unsigned __int64)v61);
    if ( v58 != &v60 )
      j_j___libc_free_0((unsigned __int64)v58);
    if ( v52 != &v54 )
      j_j___libc_free_0((unsigned __int64)v52);
    if ( v49 != (_QWORD *)v51 )
      j_j___libc_free_0((unsigned __int64)v49);
    if ( v55 != (char *)v57 )
      j_j___libc_free_0((unsigned __int64)v55);
    v26 += 40LL;
    if ( (__m128i *)v26 == v45 )
      break;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v47) <= 2 )
LABEL_75:
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v46, " | ", 3u);
    v27 = *(unsigned __int8 *)(v26 + 32);
    if ( !*(_BYTE *)(v26 + 32) )
      goto LABEL_19;
LABEL_48:
    v28 = (__m128i *)&v63.m128i_i8[1];
    do
    {
      v28 = (__m128i *)((char *)v28 - 1);
      v40 = v27 & 0xF;
      v27 >>= 4;
      v28->m128i_i8[0] = a0123456789abcd_10[v40];
    }
    while ( v27 );
  }
  v41 = v46;
  if ( v47 )
  {
    v61 = &v63;
    v60.m128i_i32[0] = 2107424;
    v58 = &v60;
    v59 = 3;
    sub_370CBD0((__int64 *)&v61, v46, (__int64)v46->m128i_i64 + v47);
    sub_2241520((unsigned __int64 *)&v61, " )");
    sub_2241490((unsigned __int64 *)&v58, v61->m128i_i8, v62);
    sub_2240A30((unsigned __int64 *)&v61);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v58 == &v60 )
    {
      a1[1] = _mm_load_si128(&v60);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v58;
      a1[1].m128i_i64[0] = v60.m128i_i64[0];
    }
    v43 = v59;
    v59 = 0;
    v60.m128i_i8[0] = 0;
    a1->m128i_i64[1] = v43;
    v58 = &v60;
    sub_2240A30((unsigned __int64 *)&v58);
    goto LABEL_55;
  }
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v41 == &v48 )
  {
LABEL_53:
    a1[1] = _mm_load_si128(&v48);
    goto LABEL_54;
  }
  v42 = v48.m128i_i64[0];
  a1->m128i_i64[0] = (__int64)v41;
  a1[1].m128i_i64[0] = v42;
LABEL_54:
  v48.m128i_i8[0] = 0;
  a1->m128i_i64[1] = 0;
  v46 = &v48;
  v47 = 0;
LABEL_55:
  sub_2240A30((unsigned __int64 *)&v46);
  if ( v64 != (__m128i *)v66 )
    _libc_free((unsigned __int64)v64);
  return a1;
}
