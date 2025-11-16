// Function: sub_A3B750
// Address: 0xa3b750
//
const void *__fastcall sub_A3B750(__int64 a1, __int64 a2, __int64 a3, const void *a4, size_t a5)
{
  size_t v5; // r15
  __int64 v6; // r13
  __int64 i; // r15
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r9
  size_t v11; // r11
  __int64 v12; // r13
  size_t v13; // r10
  size_t v14; // rdx
  int v15; // eax
  size_t v16; // rbx
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 v19; // r13
  size_t v20; // r9
  size_t v21; // rdx
  int v22; // eax
  __int64 v26; // [rsp+18h] [rbp-68h]
  __int64 v28; // [rsp+28h] [rbp-58h]
  size_t v29; // [rsp+38h] [rbp-48h]
  size_t v30; // [rsp+40h] [rbp-40h]
  size_t v31; // [rsp+48h] [rbp-38h]

  v5 = a5;
  v6 = a1;
  v26 = a3 & 1;
  v28 = (a3 - 1) / 2;
  if ( a2 >= v28 )
  {
    v8 = a1 + 16 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_25;
    v9 = a2;
    goto LABEL_28;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v10 = v9 - 1;
    v8 = a1 + 32 * (i + 1);
    v11 = *(_QWORD *)(v8 + 8);
    v12 = a1 + 16 * (v9 - 1);
    v13 = *(_QWORD *)(v12 + 8);
    v14 = v13;
    if ( v11 <= v13 )
      v14 = *(_QWORD *)(v8 + 8);
    if ( v14
      && (v29 = *(_QWORD *)(v12 + 8),
          v30 = *(_QWORD *)(v8 + 8),
          v15 = memcmp(*(const void **)v8, *(const void **)v12, v14),
          v10 = v9 - 1,
          v11 = v30,
          v13 = v29,
          v15) )
    {
      if ( v15 < 0 )
        v8 = a1 + 16 * --v9;
    }
    else if ( v11 != v13 && v11 < v13 )
    {
      v8 = a1 + 16 * (v9 - 1);
      v9 = v10;
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128((const __m128i *)v8);
    if ( v9 >= v28 )
      break;
  }
  v5 = a5;
  v6 = a1;
  if ( !v26 )
  {
LABEL_28:
    if ( (a3 - 2) / 2 == v9 )
    {
      v9 = 2 * v9 + 1;
      *(__m128i *)v8 = _mm_loadu_si128((const __m128i *)(v6 + 16 * v9));
      v8 = v6 + 16 * v9;
    }
  }
  if ( v9 > a2 )
  {
    v16 = v5;
    v17 = (v9 - 1) / 2;
    v18 = v6;
    while ( 1 )
    {
      v19 = v18 + 16 * v17;
      v20 = *(_QWORD *)(v19 + 8);
      v21 = v20;
      if ( v16 <= v20 )
        v21 = v16;
      if ( v21 && (v31 = *(_QWORD *)(v19 + 8), v22 = memcmp(*(const void **)v19, a4, v21), v20 = v31, v22) )
      {
        if ( v22 >= 0 )
          goto LABEL_24;
      }
      else if ( v16 == v20 || v16 <= v20 )
      {
LABEL_24:
        v5 = v16;
        v8 = v18 + 16 * v9;
        goto LABEL_25;
      }
      *(__m128i *)(v18 + 16 * v9) = _mm_loadu_si128((const __m128i *)v19);
      v9 = v17;
      if ( a2 >= v17 )
        break;
      v17 = (v17 - 1) / 2;
    }
    v5 = v16;
    v8 = v19;
  }
LABEL_25:
  *(_QWORD *)(v8 + 8) = v5;
  *(_QWORD *)v8 = a4;
  return a4;
}
