// Function: sub_1384420
// Address: 0x1384420
//
__m128i *__fastcall sub_1384420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i *a6)
{
  __m128i *result; // rax
  __int64 v10; // rsi
  int v11; // r9d
  unsigned int v12; // edx
  __m128i *v13; // r10
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rdx
  unsigned int v18; // r11d
  __m128i *v19; // r10
  __int64 v20; // r14
  __int64 v21; // rsi
  __int64 v22; // r14
  __int64 v23; // rsi
  __int64 v24; // rsi
  int v25; // r10d
  int v26; // r10d
  int v27; // r15d
  int v28; // r11d
  __m128i v29; // [rsp+0h] [rbp-50h] BYREF
  __m128i *v30; // [rsp+10h] [rbp-40h]

  result = (__m128i *)*(unsigned int *)(a1 + 24);
  v10 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)result )
  {
    v16 = 0;
    goto LABEL_20;
  }
  v11 = (_DWORD)result - 1;
  v12 = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__m128i *)(v10 + 32LL * v12);
  v14 = v13->m128i_i64[0];
  if ( a2 != v13->m128i_i64[0] )
  {
    v25 = 1;
    while ( v14 != -8 )
    {
      v28 = v25 + 1;
      v12 = v11 & (v25 + v12);
      v13 = (__m128i *)(v10 + 32LL * v12);
      v14 = v13->m128i_i64[0];
      if ( a2 == v13->m128i_i64[0] )
        goto LABEL_3;
      v25 = v28;
    }
    result = (__m128i *)(v10 + 32LL * (_QWORD)result);
    goto LABEL_26;
  }
LABEL_3:
  result = (__m128i *)(v10 + 32LL * (_QWORD)result);
  if ( v13 != result )
  {
    v15 = v13->m128i_i64[1];
    if ( -1227133513 * (unsigned int)((v13[1].m128i_i64[0] - v15) >> 3) > (unsigned int)a3 )
    {
      v16 = v15 + 56LL * (unsigned int)a3;
LABEL_6:
      v17 = (unsigned int)a5;
      goto LABEL_7;
    }
LABEL_26:
    v16 = 0;
    goto LABEL_6;
  }
  v17 = (unsigned int)a5;
  v16 = 0;
LABEL_7:
  v18 = v11 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
  v19 = (__m128i *)(v10 + 32LL * v18);
  v20 = v19->m128i_i64[0];
  if ( v19->m128i_i64[0] == a4 )
  {
LABEL_8:
    if ( v19 != result )
    {
      v21 = v19->m128i_i64[1];
      result = (__m128i *)(-1227133513 * (unsigned int)((v19[1].m128i_i64[0] - v21) >> 3));
      if ( (unsigned int)result > (unsigned int)v17 )
      {
        result = (__m128i *)(7 * v17);
        v22 = v21 + 56 * v17;
        goto LABEL_11;
      }
    }
  }
  else
  {
    v26 = 1;
    while ( v20 != -8 )
    {
      v27 = v26 + 1;
      v18 = v11 & (v18 + v26);
      v19 = (__m128i *)(v10 + 32LL * v18);
      v20 = v19->m128i_i64[0];
      if ( v19->m128i_i64[0] == a4 )
        goto LABEL_8;
      v26 = v27;
    }
  }
LABEL_20:
  v22 = 0;
LABEL_11:
  v29.m128i_i64[0] = a4;
  v29.m128i_i64[1] = a5;
  v30 = a6;
  v23 = *(_QWORD *)(v16 + 8);
  if ( v23 == *(_QWORD *)(v16 + 16) )
  {
    result = sub_1384280(v16, (_BYTE *)v23, &v29);
  }
  else
  {
    if ( v23 )
    {
      *(__m128i *)v23 = _mm_loadu_si128(&v29);
      result = v30;
      *(_QWORD *)(v23 + 16) = v30;
      v23 = *(_QWORD *)(v16 + 8);
    }
    *(_QWORD *)(v16 + 8) = v23 + 24;
  }
  v29.m128i_i64[0] = a2;
  v29.m128i_i64[1] = a3;
  v30 = a6;
  v24 = *(_QWORD *)(v22 + 32);
  if ( v24 == *(_QWORD *)(v22 + 40) )
    return sub_1384280(v22 + 24, (_BYTE *)v24, &v29);
  if ( v24 )
  {
    *(__m128i *)v24 = _mm_loadu_si128(&v29);
    result = v30;
    *(_QWORD *)(v24 + 16) = v30;
    v24 = *(_QWORD *)(v22 + 32);
  }
  *(_QWORD *)(v22 + 32) = v24 + 24;
  return result;
}
