// Function: sub_37145A0
// Address: 0x37145a0
//
__m128i *__fastcall sub_37145A0(
        __m128i *a1,
        _QWORD *a2,
        unsigned __int16 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __m128i *v8; // rdi
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r15
  __m128i **v12; // r8
  __int64 v13; // rdx
  unsigned __int16 v14; // ax
  __int64 v15; // rax
  const __m128i *v16; // r14
  unsigned __int64 v17; // rdx
  __m128i *v18; // rax
  __int64 v19; // rbx
  __m128i *v20; // r13
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __m128i *v25; // rbx
  const __m128i *v26; // rdi
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rax
  __m128i *v29; // rsi
  _BYTE *v30; // rsi
  __int64 v31; // rdx
  __m128i *v32; // rax
  size_t v33; // rcx
  __m128i *v34; // r9
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  __m128i *v37; // rax
  unsigned __int64 v38; // rcx
  __m128i *v39; // rdx
  unsigned __int64 *v40; // rax
  __int64 v41; // rdx
  __m128i *v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 v45; // r14
  __m128i *v46; // [rsp+18h] [rbp-2C8h]
  __m128i **src; // [rsp+48h] [rbp-298h]
  __m128i **srca; // [rsp+48h] [rbp-298h]
  __m128i *v49; // [rsp+50h] [rbp-290h] BYREF
  __int64 v50; // [rsp+58h] [rbp-288h]
  __m128i v51; // [rsp+60h] [rbp-280h] BYREF
  _QWORD *v52; // [rsp+70h] [rbp-270h] BYREF
  __int64 v53; // [rsp+78h] [rbp-268h]
  _BYTE v54[16]; // [rsp+80h] [rbp-260h] BYREF
  __m128i *v55; // [rsp+90h] [rbp-250h] BYREF
  size_t v56; // [rsp+98h] [rbp-248h]
  __m128i v57; // [rsp+A0h] [rbp-240h] BYREF
  char *v58; // [rsp+B0h] [rbp-230h] BYREF
  size_t v59; // [rsp+B8h] [rbp-228h]
  _QWORD v60[2]; // [rsp+C0h] [rbp-220h] BYREF
  __m128i *v61; // [rsp+D0h] [rbp-210h] BYREF
  __int64 v62; // [rsp+D8h] [rbp-208h]
  __m128i v63; // [rsp+E0h] [rbp-200h] BYREF
  __m128i *v64; // [rsp+F0h] [rbp-1F0h] BYREF
  size_t v65; // [rsp+F8h] [rbp-1E8h]
  __m128i v66; // [rsp+100h] [rbp-1E0h] BYREF
  __m128i *v67; // [rsp+110h] [rbp-1D0h] BYREF
  __int64 v68; // [rsp+118h] [rbp-1C8h]
  _BYTE v69[448]; // [rsp+120h] [rbp-1C0h] BYREF

  if ( !a2[7] || a2[5] || a2[6] )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_370CD40(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v8 = (__m128i *)v69;
  v9 = a4;
  v68 = 0xA00000000LL;
  v10 = a4 + 40 * a5;
  v67 = (__m128i *)v69;
  if ( a4 == v10 )
    goto LABEL_51;
  v12 = &v67;
  v13 = 0;
  do
  {
    v14 = *(_WORD *)(v9 + 32);
    if ( v14 && v14 == (a3 & v14) )
    {
      v15 = (unsigned int)v13;
      v16 = (const __m128i *)v9;
      v17 = (unsigned int)v13 + 1LL;
      if ( v17 > HIDWORD(v68) )
      {
        if ( (unsigned __int64)v8 > v9 || (unsigned __int64)v8 + 40 * v15 <= v9 )
        {
          srca = v12;
          sub_C8D5F0((__int64)v12, v69, v17, 0x28u, (__int64)v12, a6);
          v8 = v67;
          v15 = (unsigned int)v68;
          v12 = srca;
        }
        else
        {
          v45 = v9 - (_QWORD)v8;
          src = v12;
          sub_C8D5F0((__int64)v12, v69, v17, 0x28u, (__int64)v12, a6);
          v8 = v67;
          v15 = (unsigned int)v68;
          v12 = src;
          v16 = (__m128i *)((char *)v67 + v45);
        }
      }
      v18 = (__m128i *)((char *)v8 + 40 * v15);
      *v18 = _mm_loadu_si128(v16);
      v8 = v67;
      v18[1] = _mm_loadu_si128(v16 + 1);
      v18[2].m128i_i64[0] = v16[2].m128i_i64[0];
      v13 = (unsigned int)(v68 + 1);
      LODWORD(v68) = v68 + 1;
    }
    v9 += 40LL;
  }
  while ( v9 != v10 );
  v19 = 40 * v13;
  v20 = (__m128i *)((char *)v8 + 40 * v13);
  if ( v20 == v8 )
  {
LABEL_51:
    v51.m128i_i8[0] = 0;
LABEL_52:
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    goto LABEL_53;
  }
  _BitScanReverse64(&v21, 0xCCCCCCCCCCCCCCCDLL * (v19 >> 3));
  sub_3712700(
    (__int64)v8,
    (__m128i *)((char *)v8 + 40 * v13),
    2LL * (int)(63 - (v21 ^ 0x3F)),
    (unsigned __int8 (__fastcall *)(__m128i *, __m128i *))sub_370CDF0,
    (__int64)v12,
    a6);
  if ( (unsigned __int64)v19 <= 0x280 )
  {
    sub_3711F50(v8, v20, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CDF0, v22, v23, v24);
  }
  else
  {
    v25 = v8 + 40;
    sub_3711F50(v8, v8 + 40, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CDF0, v22, v23, v24);
    if ( &v8[40] != v20 )
    {
      do
      {
        v26 = v25;
        v25 = (__m128i *)((char *)v25 + 40);
        sub_3711EC0(v26, (unsigned __int8 (__fastcall *)(__m128i *, __int8 *))sub_370CDF0);
      }
      while ( v25 != v20 );
    }
  }
  v27 = (unsigned __int64)v67;
  v50 = 0;
  v49 = &v51;
  v51.m128i_i8[0] = 0;
  v46 = (__m128i *)((char *)v67 + 40 * (unsigned int)v68);
  if ( v67 == v46 )
    goto LABEL_52;
  v28 = v67[2].m128i_u16[0];
  if ( v67[2].m128i_i16[0] )
    goto LABEL_48;
LABEL_19:
  v66.m128i_i8[0] = 48;
  v29 = &v66;
  while ( 1 )
  {
    v58 = (char *)v60;
    sub_370CBD0((__int64 *)&v58, v29, (__int64)v66.m128i_i64 + 1);
    v30 = *(_BYTE **)v27;
    if ( *(_QWORD *)v27 )
    {
      v31 = (__int64)&v30[*(_QWORD *)(v27 + 8)];
      v52 = v54;
      sub_370CD40((__int64 *)&v52, v30, v31);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v53) <= 3 )
        goto LABEL_75;
    }
    else
    {
      v54[0] = 0;
      v53 = 0;
      v52 = v54;
    }
    v32 = (__m128i *)sub_2241490((unsigned __int64 *)&v52, " (0x", 4u);
    v55 = &v57;
    if ( (__m128i *)v32->m128i_i64[0] == &v32[1] )
    {
      v57 = _mm_loadu_si128(v32 + 1);
    }
    else
    {
      v55 = (__m128i *)v32->m128i_i64[0];
      v57.m128i_i64[0] = v32[1].m128i_i64[0];
    }
    v33 = v32->m128i_u64[1];
    v32[1].m128i_i8[0] = 0;
    v56 = v33;
    v32->m128i_i64[0] = (__int64)v32[1].m128i_i64;
    v34 = v55;
    v32->m128i_i64[1] = 0;
    v35 = 15;
    v36 = 15;
    if ( v34 != &v57 )
      v36 = v57.m128i_i64[0];
    if ( v56 + v59 <= v36 )
      goto LABEL_30;
    if ( v58 != (char *)v60 )
      v35 = v60[0];
    if ( v56 + v59 <= v35 )
    {
      v37 = (__m128i *)sub_2241130((unsigned __int64 *)&v58, 0, 0, v34, v56);
      v39 = v37 + 1;
      v61 = &v63;
      v38 = v37->m128i_i64[0];
      if ( (__m128i *)v37->m128i_i64[0] != &v37[1] )
      {
LABEL_31:
        v61 = (__m128i *)v38;
        v63.m128i_i64[0] = v37[1].m128i_i64[0];
        goto LABEL_32;
      }
    }
    else
    {
LABEL_30:
      v37 = (__m128i *)sub_2241490((unsigned __int64 *)&v55, v58, v59);
      v61 = &v63;
      v38 = v37->m128i_i64[0];
      v39 = v37 + 1;
      if ( (__m128i *)v37->m128i_i64[0] != &v37[1] )
        goto LABEL_31;
    }
    v63 = _mm_loadu_si128(v37 + 1);
LABEL_32:
    v62 = v37->m128i_i64[1];
    v37->m128i_i64[0] = (__int64)v39;
    v37->m128i_i64[1] = 0;
    v37[1].m128i_i8[0] = 0;
    if ( v62 == 0x3FFFFFFFFFFFFFFFLL )
      goto LABEL_75;
    v40 = sub_2241490((unsigned __int64 *)&v61, ")", 1u);
    v64 = &v66;
    if ( (unsigned __int64 *)*v40 == v40 + 2 )
    {
      v66 = _mm_loadu_si128((const __m128i *)v40 + 1);
    }
    else
    {
      v64 = (__m128i *)*v40;
      v66.m128i_i64[0] = v40[2];
    }
    v65 = v40[1];
    *v40 = (unsigned __int64)(v40 + 2);
    v40[1] = 0;
    *((_BYTE *)v40 + 16) = 0;
    sub_2241490((unsigned __int64 *)&v49, v64->m128i_i8, v65);
    if ( v64 != &v66 )
      j_j___libc_free_0((unsigned __int64)v64);
    if ( v61 != &v63 )
      j_j___libc_free_0((unsigned __int64)v61);
    if ( v55 != &v57 )
      j_j___libc_free_0((unsigned __int64)v55);
    if ( v52 != (_QWORD *)v54 )
      j_j___libc_free_0((unsigned __int64)v52);
    if ( v58 != (char *)v60 )
      j_j___libc_free_0((unsigned __int64)v58);
    v27 += 40LL;
    if ( (__m128i *)v27 == v46 )
      break;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v50) <= 2 )
LABEL_75:
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v49, " | ", 3u);
    v28 = *(unsigned __int16 *)(v27 + 32);
    if ( !*(_WORD *)(v27 + 32) )
      goto LABEL_19;
LABEL_48:
    v29 = (__m128i *)&v66.m128i_i8[1];
    do
    {
      v29 = (__m128i *)((char *)v29 - 1);
      v41 = v28 & 0xF;
      v28 >>= 4;
      v29->m128i_i8[0] = a0123456789abcd_10[v41];
    }
    while ( v28 );
  }
  v42 = v49;
  if ( v50 )
  {
    v64 = &v66;
    v63.m128i_i32[0] = 2107424;
    v61 = &v63;
    v62 = 3;
    sub_370CBD0((__int64 *)&v64, v49, (__int64)v49->m128i_i64 + v50);
    sub_2241520((unsigned __int64 *)&v64, " )");
    sub_2241490((unsigned __int64 *)&v61, v64->m128i_i8, v65);
    sub_2240A30((unsigned __int64 *)&v64);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v61 == &v63 )
    {
      a1[1] = _mm_load_si128(&v63);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v61;
      a1[1].m128i_i64[0] = v63.m128i_i64[0];
    }
    v44 = v62;
    v62 = 0;
    v63.m128i_i8[0] = 0;
    a1->m128i_i64[1] = v44;
    v61 = &v63;
    sub_2240A30((unsigned __int64 *)&v61);
    goto LABEL_55;
  }
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v42 == &v51 )
  {
LABEL_53:
    a1[1] = _mm_load_si128(&v51);
    goto LABEL_54;
  }
  v43 = v51.m128i_i64[0];
  a1->m128i_i64[0] = (__int64)v42;
  a1[1].m128i_i64[0] = v43;
LABEL_54:
  v51.m128i_i8[0] = 0;
  a1->m128i_i64[1] = 0;
  v49 = &v51;
  v50 = 0;
LABEL_55:
  sub_2240A30((unsigned __int64 *)&v49);
  if ( v67 != (__m128i *)v69 )
    _libc_free((unsigned __int64)v67);
  return a1;
}
