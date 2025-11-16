// Function: sub_13821F0
// Address: 0x13821f0
//
__m128i *__fastcall sub_13821F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v9; // r13
  unsigned __int32 v10; // r15d
  __int64 i; // rbx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r14
  __m128i *result; // rax
  unsigned __int32 v16; // ecx
  unsigned __int32 v17; // esi
  unsigned __int32 v18; // r9d
  unsigned __int32 v19; // r11d
  unsigned __int32 v20; // r13d
  unsigned __int32 v21; // r8d
  unsigned __int32 v22; // r10d
  unsigned __int32 v23; // r12d
  __int64 v24; // r14
  __m128i *v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rcx
  unsigned __int32 v28; // edx
  unsigned __int32 v29; // r9d
  unsigned int v30; // edx
  unsigned __int32 v31; // r9d
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rcx
  __m128i *v36; // rdx
  __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v40; // [rsp+38h] [rbp-38h]

  v9 = a2;
  v39 = a3 & 1;
  v10 = a7;
  v40 = (a3 - 1) / 2;
  if ( a2 >= v40 )
  {
    result = (__m128i *)(a1 + 24 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_33;
    v12 = a2;
    goto LABEL_36;
  }
  for ( i = a2; ; i = v12 )
  {
    v12 = 2 * (i + 1);
    v13 = 48 * (i + 1);
    v14 = a1 + v13 - 24;
    result = (__m128i *)(a1 + v13);
    v16 = *(_DWORD *)v14;
    v17 = result->m128i_i32[0];
    v18 = *(_DWORD *)(v14 + 4);
    v19 = *(_DWORD *)(v14 + 8);
    v20 = *(_DWORD *)(v14 + 12);
    v21 = result->m128i_u32[1];
    v22 = result->m128i_u32[2];
    v23 = result->m128i_u32[3];
    v24 = *(_QWORD *)(v14 + 16);
    if ( result->m128i_i32[0] < v16
      || v18 > v21 && v17 == v16
      || v17 <= v16
      && (v18 >= v21 || v17 != v16)
      && (v19 > v22
       || v20 > v23 && v19 == v22
       || v19 >= v22 && (v20 >= v23 || v19 != v22) && v24 > result[1].m128i_i64[0]) )
    {
      --v12;
      result = (__m128i *)(a1 + 24 * v12);
    }
    v25 = (__m128i *)(a1 + 24 * i);
    *v25 = _mm_loadu_si128(result);
    v25[1].m128i_i64[0] = result[1].m128i_i64[0];
    if ( v12 >= v40 )
      break;
  }
  v10 = a7;
  v9 = a2;
  if ( !v39 )
  {
LABEL_36:
    if ( (a3 - 2) / 2 == v12 )
    {
      v32 = v12 + 1;
      v33 = 2 * (v12 + 1);
      v34 = v33 + 4 * v32;
      v12 = v33 - 1;
      v35 = a1 + 8 * v34;
      *result = _mm_loadu_si128((const __m128i *)(v35 - 24));
      result[1].m128i_i64[0] = *(_QWORD *)(v35 - 8);
      result = (__m128i *)(a1 + 24 * v12);
    }
  }
  v26 = (v12 - 1) / 2;
  if ( v12 > v9 )
  {
    v27 = v12;
    while ( 1 )
    {
      result = (__m128i *)(a1 + 24 * v26);
      v28 = result->m128i_i32[0];
      if ( result->m128i_i32[0] >= v10 )
      {
        v29 = result->m128i_u32[1];
        if ( v29 >= HIDWORD(a7) || v28 != v10 )
        {
          if ( v28 > v10 || v29 > HIDWORD(a7) && v28 == v10 )
            break;
          v30 = result->m128i_u32[2];
          if ( v30 >= (unsigned int)a8 )
          {
            v31 = result->m128i_u32[3];
            if ( (v31 >= HIDWORD(a8) || v30 != (_DWORD)a8)
              && (v30 > (unsigned int)a8 || v31 > HIDWORD(a8) && v30 == (_DWORD)a8 || result[1].m128i_i64[0] >= a9) )
            {
              break;
            }
          }
        }
      }
      v36 = (__m128i *)(a1 + 24 * v27);
      *v36 = _mm_loadu_si128(result);
      v36[1].m128i_i64[0] = result[1].m128i_i64[0];
      v27 = v26;
      if ( v9 >= v26 )
        goto LABEL_33;
      v26 = (v26 - 1) / 2;
    }
    result = (__m128i *)(a1 + 24 * v27);
  }
LABEL_33:
  result->m128i_i32[0] = v10;
  result->m128i_i32[1] = HIDWORD(a7);
  result->m128i_i64[1] = a8;
  result[1].m128i_i64[0] = a9;
  return result;
}
