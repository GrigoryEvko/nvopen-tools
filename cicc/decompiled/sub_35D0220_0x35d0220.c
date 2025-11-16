// Function: sub_35D0220
// Address: 0x35d0220
//
signed __int64 __fastcall sub_35D0220(unsigned int *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  signed __int64 result; // rax
  __int64 v7; // r15
  __m128i *v8; // r14
  __m128i *v10; // r10
  __m128i *v11; // r12
  signed int v12; // r11d
  __m128i *v13; // rax
  __int32 v14; // r9d
  __int64 v15; // rsi
  unsigned __int8 v16; // di
  __int64 v17; // rdx
  unsigned __int8 v18; // cl
  unsigned __int8 v19; // r8
  signed int v20; // r14d
  __int64 v21; // rbx
  __m128i v22; // xmm1
  unsigned __int64 v23; // rdx
  __m128i v24; // xmm0
  unsigned __int64 v25; // rcx
  __m128i v26; // xmm1
  unsigned __int64 v27; // rax
  __m128i v28; // xmm0
  unsigned int v29; // edx
  __m128i *v30; // rbx
  __m128i *v31; // rsi
  __int64 v32; // r8
  __int64 v33; // rdi
  __int64 v34; // r9
  __int64 v35; // rdx
  __m128i v36; // xmm1
  unsigned __int64 v37; // rdx
  __m128i v38; // xmm0
  bool v39; // al
  __int64 v40; // rax
  __m128i *i; // rax
  unsigned __int8 v42; // dl
  unsigned __int8 v43; // r8
  signed int v44; // r14d
  __int64 v45; // rbx
  unsigned __int64 v46; // rax
  __m128i v47; // xmm1
  __m128i v48; // xmm0
  __m128i v49; // xmm6
  __m128i v50; // xmm7
  __int64 v51; // rbx
  __int64 j; // r12
  __int8 *v53; // r14
  __int64 v54; // rax
  __int128 v55; // xmm4
  __int64 v56; // r12
  __int64 v57; // rdx
  __int128 v58; // xmm5
  __m128i v59; // [rsp-68h] [rbp-68h]
  __m128i v60; // [rsp-58h] [rbp-58h]
  unsigned __int64 v61; // [rsp-48h] [rbp-48h]

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 640 )
    return result;
  v7 = a3;
  v8 = a2;
  if ( !a3 )
    goto LABEL_48;
  v10 = a2;
  v11 = (__m128i *)(a1 + 10);
  while ( 2 )
  {
    --v7;
    v12 = a1[10];
    v13 = (__m128i *)&a1[10 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((char *)v10 - (char *)a1) >> 3)) / 2)];
    v14 = v13->m128i_i32[0];
    v15 = v13[1].m128i_i64[1] + v13[1].m128i_i64[0];
    v16 = v13[2].m128i_i32[0] != 2;
    v17 = *((_QWORD *)a1 + 7) + *((_QWORD *)a1 + 8);
    v18 = a1[18] != 2;
    if ( v16 >= v18 && (v16 != v18 || v15 >= v17 && (v15 != v17 || v14 >= v12)) )
    {
      v43 = v10[-1].m128i_i32[2] != 2;
      if ( v18 > v43 )
        goto LABEL_41;
      v44 = v10[-3].m128i_i32[2];
      v45 = v10[-2].m128i_i64[1] + v10[-1].m128i_i64[0];
      if ( v18 == v43 && (v17 > v45 || v17 == v45 && v12 > v44) )
        goto LABEL_41;
      if ( v16 <= v43 && (v16 != v43 || v15 <= v45 && (v15 != v45 || v14 <= v44)) )
        goto LABEL_12;
LABEL_14:
      v26 = _mm_loadu_si128((const __m128i *)a1);
      v27 = *((_QWORD *)a1 + 4);
      v28 = _mm_loadu_si128((const __m128i *)a1 + 1);
      *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v10 - 40));
      v61 = v27;
      *((__m128i *)a1 + 1) = _mm_loadu_si128((__m128i *)((char *)v10 - 24));
      v59 = v26;
      a1[8] = v10[-1].m128i_u32[2];
      v25 = v10[-1].m128i_u8[12];
      v60 = v28;
      *((_BYTE *)a1 + 36) = v25;
      v10[-1].m128i_i32[2] = v27;
      v10[-1].m128i_i8[12] = BYTE4(v27);
      *(__m128i *)((char *)v10 - 40) = v26;
      *(__m128i *)((char *)v10 - 24) = v28;
      goto LABEL_15;
    }
    v19 = v10[-1].m128i_i32[2] != 2;
    if ( v16 <= v19 )
    {
      v20 = v10[-3].m128i_i32[2];
      v21 = v10[-2].m128i_i64[1] + v10[-1].m128i_i64[0];
      if ( v16 != v19 || v15 <= v21 && (v15 != v21 || v14 <= v20) )
      {
        if ( v18 <= v19 && (v18 != v19 || v17 <= v21 && (v17 != v21 || v12 <= v20)) )
        {
LABEL_41:
          v46 = *((_QWORD *)a1 + 4);
          v47 = _mm_loadu_si128((const __m128i *)a1);
          v48 = _mm_loadu_si128((const __m128i *)a1 + 1);
          v49 = _mm_loadu_si128((const __m128i *)(a1 + 10));
          v50 = _mm_loadu_si128((const __m128i *)(a1 + 14));
          a1[8] = a1[18];
          v25 = *((unsigned __int8 *)a1 + 76);
          v61 = v46;
          a1[18] = v46;
          *((_BYTE *)a1 + 36) = v25;
          *((_BYTE *)a1 + 76) = BYTE4(v46);
          v59 = v47;
          v60 = v48;
          *(__m128i *)a1 = v49;
          *((__m128i *)a1 + 1) = v50;
          *(__m128i *)(a1 + 10) = v47;
          *(__m128i *)(a1 + 14) = v48;
          goto LABEL_15;
        }
        goto LABEL_14;
      }
    }
LABEL_12:
    v22 = _mm_loadu_si128((const __m128i *)a1);
    v23 = *((_QWORD *)a1 + 4);
    v24 = _mm_loadu_si128((const __m128i *)a1 + 1);
    *(__m128i *)a1 = _mm_loadu_si128(v13);
    v61 = v23;
    *((__m128i *)a1 + 1) = _mm_loadu_si128(v13 + 1);
    v25 = HIDWORD(v23);
    v59 = v22;
    a1[8] = v13[2].m128i_u32[0];
    v60 = v24;
    *((_BYTE *)a1 + 36) = v13[2].m128i_i8[4];
    v13[2].m128i_i32[0] = v23;
    v13[2].m128i_i8[4] = BYTE4(v23);
    *v13 = v22;
    v13[1] = v24;
LABEL_15:
    v29 = a1[8];
    v30 = v11;
    v31 = v10;
    v32 = *a1;
    v33 = *((_QWORD *)a1 + 2) + *((_QWORD *)a1 + 3);
    while ( 1 )
    {
      v8 = v30;
      LOBYTE(v25) = v29 != 2;
      v39 = v30[2].m128i_i32[0] != 2;
      if ( (unsigned __int8)(v29 != 2) >= (unsigned __int8)v39 )
      {
        if ( (v29 != 2) != v39 )
          break;
        v40 = v30[1].m128i_i64[1] + v30[1].m128i_i64[0];
        if ( v33 >= v40 && (v33 != v40 || (int)v32 >= v30->m128i_i32[0]) )
          break;
      }
LABEL_22:
      v30 = (__m128i *)((char *)v30 + 40);
    }
    for ( i = (__m128i *)((char *)v31 - 40); ; i = (__m128i *)((char *)i - 40) )
    {
      v31 = i;
      v42 = i[2].m128i_i32[0] != 2;
      if ( (unsigned __int8)v25 <= v42 )
      {
        v34 = i->m128i_u32[0];
        if ( (_BYTE)v25 != v42 )
          break;
        v35 = i[1].m128i_i64[1] + i[1].m128i_i64[0];
        if ( v33 <= v35 && (v33 != v35 || (int)v32 <= (int)v34) )
          break;
      }
    }
    if ( v30 < i )
    {
      v36 = _mm_loadu_si128(v30);
      v37 = v30[2].m128i_u64[0];
      v38 = _mm_loadu_si128(v30 + 1);
      *v30 = _mm_loadu_si128(i);
      v61 = v37;
      v30[1] = _mm_loadu_si128(i + 1);
      v25 = HIDWORD(v37);
      v59 = v36;
      v30[2].m128i_i32[0] = i[2].m128i_i32[0];
      v60 = v38;
      v30[2].m128i_i8[4] = i[2].m128i_i8[4];
      i[2].m128i_i32[0] = v37;
      i[2].m128i_i8[4] = BYTE4(v37);
      *i = v36;
      i[1] = v38;
      v29 = a1[8];
      v32 = *a1;
      v33 = *((_QWORD *)a1 + 2) + *((_QWORD *)a1 + 3);
      goto LABEL_22;
    }
    sub_35D0220(
      v30,
      v10,
      v7,
      v25,
      v32,
      v34,
      v59.m128i_i64[0],
      v59.m128i_i64[1],
      v60.m128i_i64[0],
      v60.m128i_i64[1],
      v61);
    result = (char *)v30 - (char *)a1;
    if ( (char *)v30 - (char *)a1 > 640 )
    {
      if ( v7 )
      {
        v10 = v30;
        continue;
      }
LABEL_48:
      v51 = 0xCCCCCCCCCCCCCCCDLL * (result >> 3);
      for ( j = (v51 - 2) >> 1; ; --j )
      {
        sub_35CFFD0(
          (__int64)a1,
          j,
          v51,
          a4,
          a5,
          a6,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)&a1[10 * j]),
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)&a1[10 * j + 4]),
          *(_QWORD *)&a1[10 * j + 8]);
        if ( !j )
          break;
      }
      v53 = &v8[-3].m128i_i8[8];
      do
      {
        v54 = *((_QWORD *)v53 + 4);
        v55 = (__int128)_mm_loadu_si128((const __m128i *)v53);
        v56 = v53 - (__int8 *)a1;
        *(__m128i *)v53 = _mm_loadu_si128((const __m128i *)a1);
        v57 = v53 - (__int8 *)a1;
        v53 -= 40;
        v58 = (__int128)_mm_loadu_si128((const __m128i *)(v53 + 56));
        *(__m128i *)(v53 + 56) = _mm_loadu_si128((const __m128i *)a1 + 1);
        *((_DWORD *)v53 + 18) = a1[8];
        v53[76] = *((_BYTE *)a1 + 36);
        result = (signed __int64)sub_35CFFD0(
                                   (__int64)a1,
                                   0,
                                   0xCCCCCCCCCCCCCCCDLL * (v57 >> 3),
                                   a4,
                                   a5,
                                   a6,
                                   v55,
                                   v58,
                                   v54);
      }
      while ( v56 > 40 );
    }
    return result;
  }
}
