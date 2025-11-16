// Function: sub_35CFFD0
// Address: 0x35cffd0
//
__m128i *__fastcall sub_35CFFD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  __int64 v11; // r10
  __int64 i; // rcx
  __int64 v13; // r12
  __int64 v14; // rbx
  __m128i *result; // rax
  __int64 v16; // rdx
  __m128i *v17; // rcx
  __int64 v18; // r9
  unsigned __int8 v19; // r12
  unsigned __int8 v20; // bl
  __int32 v21; // r13d
  __int128 v22; // kr00_16
  __int32 v23; // r11d
  __int64 v24; // rsi
  __int64 v25; // r10
  unsigned __int8 v26; // r9
  unsigned __int8 v27; // cl
  __int64 v28; // rcx
  __m128i *v29; // rdx
  __int8 v30; // dl
  __m128i v31; // xmm6
  __m128i v32; // xmm7
  const __m128i *v33; // rcx
  __m128i v34; // [rsp+0h] [rbp-50h] BYREF
  __int128 v35; // [rsp+10h] [rbp-40h] BYREF
  __int64 v36; // [rsp+20h] [rbp-30h]

  v11 = (a3 - 1) / 2;
  if ( a2 < v11 )
  {
    for ( i = a2; ; i = v16 )
    {
      v16 = 2 * (i + 1) - 1;
      v18 = a1 + 80 * (i + 1);
      result = (__m128i *)(a1 + 40 * v16);
      v19 = result[2].m128i_i32[0] != 2;
      v20 = *(_DWORD *)(v18 + 32) != 2;
      if ( v19 >= v20 )
      {
        if ( v19 != v20 )
          goto LABEL_11;
        v13 = result[1].m128i_i64[0] + result[1].m128i_i64[1];
        v14 = *(_QWORD *)(v18 + 16) + *(_QWORD *)(v18 + 24);
        if ( v13 < v14 )
          goto LABEL_7;
        if ( v13 == v14 )
        {
          if ( result->m128i_i32[0] >= *(_DWORD *)v18 )
          {
            result = (__m128i *)(a1 + 80 * (i + 1));
            v16 = 2 * (i + 1);
          }
        }
        else
        {
LABEL_11:
          result = (__m128i *)(a1 + 80 * (i + 1));
          v16 = 2 * (i + 1);
        }
      }
LABEL_7:
      v17 = (__m128i *)(a1 + 40 * i);
      *v17 = _mm_loadu_si128(result);
      v17[1] = _mm_loadu_si128(result + 1);
      v17[2].m128i_i32[0] = result[2].m128i_i32[0];
      v17[2].m128i_i8[4] = result[2].m128i_i8[4];
      if ( v16 >= v11 )
        goto LABEL_13;
    }
  }
  v16 = a2;
  result = (__m128i *)(a1 + 40 * a2);
LABEL_13:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v16 )
  {
    v16 = 2 * v16 + 1;
    v33 = (const __m128i *)(a1 + 40 * v16);
    *result = _mm_loadu_si128(v33);
    result[1] = _mm_loadu_si128(v33 + 1);
    result[2].m128i_i32[0] = v33[2].m128i_i32[0];
    result[2].m128i_i8[4] = v33[2].m128i_i8[4];
    result = (__m128i *)v33;
  }
  v21 = a7;
  v22 = a8;
  v36 = a9;
  v23 = a9;
  v34 = _mm_loadu_si128((const __m128i *)&a7);
  v35 = (__int128)_mm_loadu_si128((const __m128i *)&a8);
  v24 = (v16 - 1) / 2;
  if ( v16 > a2 )
  {
    v25 = a8 + *((_QWORD *)&a8 + 1);
    v26 = (_DWORD)a9 != 2;
    while ( 1 )
    {
      result = (__m128i *)(a1 + 40 * v24);
      v27 = result[2].m128i_i32[0] != 2;
      if ( v27 <= v26 )
      {
        if ( v27 != v26 )
          break;
        v28 = result[1].m128i_i64[0] + result[1].m128i_i64[1];
        if ( v28 <= v25 && (v28 != v25 || v21 >= result->m128i_i32[0]) )
          break;
      }
      v29 = (__m128i *)(a1 + 40 * v16);
      *v29 = _mm_loadu_si128(result);
      v29[1] = _mm_loadu_si128(result + 1);
      v29[2].m128i_i32[0] = result[2].m128i_i32[0];
      v29[2].m128i_i8[4] = result[2].m128i_i8[4];
      v16 = v24;
      if ( a2 >= v24 )
        goto LABEL_26;
      v24 = (v24 - 1) / 2;
    }
    result = (__m128i *)(a1 + 40 * v16);
  }
LABEL_26:
  v30 = BYTE4(v36);
  result[2].m128i_i32[0] = v23;
  v34.m128i_i32[0] = v21;
  v31 = _mm_loadu_si128(&v34);
  result[2].m128i_i8[4] = v30;
  v35 = v22;
  v32 = _mm_loadu_si128((const __m128i *)&v35);
  *result = v31;
  result[1] = v32;
  return result;
}
