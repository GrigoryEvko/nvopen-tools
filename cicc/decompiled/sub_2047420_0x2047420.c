// Function: sub_2047420
// Address: 0x2047420
//
__int64 __fastcall sub_2047420(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __m128i a7)
{
  __int64 result; // rax
  __m128i *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r12
  int v11; // eax
  __int64 v12; // r15
  int v13; // eax
  __m128i v14; // xmm1
  bool v15; // sf
  __int64 v16; // rax
  __m128i *v17; // r13
  __m128i *v18; // rbx
  __int64 v19; // r12
  __m128i *v20; // r15
  __int64 v21; // rax
  __m128i v22; // xmm1
  __int64 v23; // rax
  int v24; // eax
  __m128i v25; // xmm7
  int v26; // edx
  __int64 v27; // rbx
  __int64 i; // r12
  __int8 *v29; // r15
  __int128 v30; // xmm4
  __int64 v31; // r12
  __int64 v32; // xmm5_8
  __int64 v33; // rdx
  __m128i v34; // xmm5
  __int64 v35; // [rsp+10h] [rbp-80h]
  __m128i *v36; // [rsp+18h] [rbp-78h]
  __m128i *v37; // [rsp+20h] [rbp-70h]

  result = (__int64)a2->m128i_i64 - a1;
  v35 = a3;
  v36 = a2;
  if ( (__int64)a2->m128i_i64 - a1 <= 640 )
    return result;
  if ( !a3 )
  {
    v37 = a2;
    goto LABEL_22;
  }
  while ( 2 )
  {
    --v35;
    v8 = (__m128i *)(a1 + 40 * ((__int64)(0xCCCCCCCCCCCCCCCDLL * (((__int64)v36->m128i_i64 - a1) >> 3)) / 2));
    v9 = *(_QWORD *)(a1 + 48) + 24LL;
    v10 = v8->m128i_i64[1] + 24;
    v11 = sub_16AEA10(v9, v10);
    v12 = v36[-2].m128i_i64[0] + 24;
    if ( v11 < 0 )
    {
      if ( (int)sub_16AEA10(v10, v12) < 0 )
      {
        v14 = _mm_loadu_si128((const __m128i *)a1);
        a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
        v16 = *(_QWORD *)(a1 + 32);
        goto LABEL_6;
      }
      v24 = sub_16AEA10(v9, v12);
      v14 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v15 = v24 < 0;
      v16 = *(_QWORD *)(a1 + 32);
      if ( v15 )
      {
LABEL_28:
        *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v36 - 40));
        *(__m128i *)(a1 + 16) = _mm_loadu_si128((__m128i *)((char *)v36 - 24));
        *(_DWORD *)(a1 + 32) = v36[-1].m128i_i32[2];
        v36[-1].m128i_i32[2] = v16;
        *(__m128i *)((char *)v36 - 40) = v14;
        *(__m128i *)((char *)v36 - 24) = a7;
        goto LABEL_7;
      }
      v25 = _mm_loadu_si128((const __m128i *)(a1 + 56));
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      *(__m128i *)(a1 + 16) = v25;
LABEL_20:
      v26 = *(_DWORD *)(a1 + 72);
      *(__m128i *)(a1 + 40) = v14;
      *(_DWORD *)(a1 + 72) = v16;
      *(_DWORD *)(a1 + 32) = v26;
      *(__m128i *)(a1 + 56) = a7;
      goto LABEL_7;
    }
    if ( (int)sub_16AEA10(v9, v12) < 0 )
    {
      v14 = _mm_loadu_si128((const __m128i *)a1);
      a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v16 = *(_QWORD *)(a1 + 32);
      v34 = _mm_loadu_si128((const __m128i *)(a1 + 56));
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 40));
      *(__m128i *)(a1 + 16) = v34;
      goto LABEL_20;
    }
    v13 = sub_16AEA10(v10, v12);
    v14 = _mm_loadu_si128((const __m128i *)a1);
    a7 = _mm_loadu_si128((const __m128i *)(a1 + 16));
    v15 = v13 < 0;
    v16 = *(_QWORD *)(a1 + 32);
    if ( v15 )
      goto LABEL_28;
LABEL_6:
    *(__m128i *)a1 = _mm_loadu_si128(v8);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(v8 + 1);
    *(_DWORD *)(a1 + 32) = v8[2].m128i_i32[0];
    v8[2].m128i_i32[0] = v16;
    *v8 = v14;
    v8[1] = a7;
LABEL_7:
    v17 = (__m128i *)(a1 + 40);
    v18 = v36;
    v19 = *(_QWORD *)(a1 + 8) + 24LL;
    while ( 1 )
    {
      v37 = v17;
      if ( (int)sub_16AEA10(v17->m128i_i64[1] + 24, v19) < 0 )
        goto LABEL_13;
      v20 = (__m128i *)((char *)v18 - 40);
      do
      {
        v21 = v20->m128i_i64[1];
        v18 = v20;
        v20 = (__m128i *)((char *)v20 - 40);
      }
      while ( (int)sub_16AEA10(v19, v21 + 24) < 0 );
      if ( v17 >= v18 )
        break;
      v22 = _mm_loadu_si128(v17);
      v23 = v17[2].m128i_i64[0];
      a7 = _mm_loadu_si128(v17 + 1);
      *v17 = _mm_loadu_si128(v18);
      v17[1] = _mm_loadu_si128(v18 + 1);
      v17[2].m128i_i32[0] = v18[2].m128i_i32[0];
      v18[2].m128i_i32[0] = v23;
      *v18 = v22;
      v18[1] = a7;
      v19 = *(_QWORD *)(a1 + 8) + 24LL;
LABEL_13:
      v17 = (__m128i *)((char *)v17 + 40);
    }
    sub_2047420(v17, v36, v35);
    result = (__int64)v17->m128i_i64 - a1;
    if ( (__int64)v17->m128i_i64 - a1 > 640 )
    {
      if ( v35 )
      {
        v36 = v17;
        continue;
      }
LABEL_22:
      v27 = 0xCCCCCCCCCCCCCCCDLL * (result >> 3);
      for ( i = (v27 - 2) >> 1; ; --i )
      {
        sub_20442B0(
          a1,
          i,
          v27,
          a4,
          a5,
          a6,
          a7,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 40 * i)),
          _mm_loadu_si128((const __m128i *)(a1 + 40 * i + 16)).m128i_i64[0]);
        if ( !i )
          break;
      }
      v29 = &v37[-3].m128i_i8[8];
      do
      {
        v30 = (__int128)_mm_loadu_si128((const __m128i *)v29);
        v31 = (__int64)&v29[-a1];
        v32 = _mm_loadu_si128((const __m128i *)v29 + 1).m128i_u64[0];
        *(__m128i *)v29 = _mm_loadu_si128((const __m128i *)a1);
        v33 = (__int64)&v29[-a1] >> 3;
        v29 -= 40;
        *(__m128i *)(v29 + 56) = _mm_loadu_si128((const __m128i *)(a1 + 16));
        *((_DWORD *)v29 + 18) = *(_DWORD *)(a1 + 32);
        result = sub_20442B0(a1, 0, 0xCCCCCCCCCCCCCCCDLL * v33, a4, a5, a6, a7, v30, v32);
      }
      while ( v31 > 40 );
    }
    return result;
  }
}
