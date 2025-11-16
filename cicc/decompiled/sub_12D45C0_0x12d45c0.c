// Function: sub_12D45C0
// Address: 0x12d45c0
//
__int64 __fastcall sub_12D45C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 v9; // r14
  const __m128i *v10; // r15
  const __m128i *v11; // r12
  int v12; // eax
  __int64 v13; // rax
  int v14; // eax
  __m128i *v15; // r8
  __int64 v16; // r15
  __int64 v17; // r12
  __int64 v18; // rax
  int v19; // eax
  __int64 v20; // rax
  __m128i *v21; // r15
  const __m128i *v22; // r14
  int v23; // eax
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  unsigned int v30; // [rsp+44h] [rbp-3Ch]
  __int64 i; // [rsp+48h] [rbp-38h]
  unsigned int v32; // [rsp+48h] [rbp-38h]

  v26 = a3 & 1;
  v27 = (a3 - 1) / 2;
  if ( a2 >= v27 )
  {
    v16 = a2;
    v15 = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_32;
    goto LABEL_28;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v10 = (const __m128i *)(a1 + 32 * (i + 1));
    v11 = (const __m128i *)(a1 + 16 * (v9 - 1));
    v12 = sub_16D1B30(a6, v10->m128i_i64[0], v10->m128i_i64[1]);
    if ( v12 == -1 || (v13 = *(_QWORD *)a6 + 8LL * v12, v13 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
      v30 = 0;
    else
      v30 = *(_DWORD *)(*(_QWORD *)v13 + 8LL);
    v14 = sub_16D1B30(a6, v11->m128i_i64[0], v11->m128i_i64[1]);
    if ( v14 == -1 || (v7 = *(_QWORD *)a6 + 8LL * v14, v7 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
      v8 = 0;
    else
      v8 = *(_DWORD *)(*(_QWORD *)v7 + 8LL);
    if ( v8 < v30 )
    {
      --v9;
      v10 = v11;
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(v10);
    if ( v9 >= v27 )
      break;
  }
  v15 = (__m128i *)v10;
  v16 = v9;
  if ( !v26 )
  {
LABEL_28:
    if ( (a3 - 2) / 2 == v16 )
    {
      v16 = 2 * v16 + 1;
      *v15 = _mm_loadu_si128((const __m128i *)(a1 + 16 * v16));
      v15 = (__m128i *)(a1 + 16 * v16);
    }
  }
  v17 = (v16 - 1) / 2;
  if ( v16 > a2 )
  {
    while ( 1 )
    {
      v22 = (const __m128i *)(a1 + 16 * v17);
      v23 = sub_16D1B30(a6, v22->m128i_i64[0], v22->m128i_i64[1]);
      if ( v23 == -1 || (v18 = *(_QWORD *)a6 + 8LL * v23, v18 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
        v32 = 0;
      else
        v32 = *(_DWORD *)(*(_QWORD *)v18 + 8LL);
      v19 = sub_16D1B30(a6, a4, a5);
      if ( v19 == -1 || (v20 = *(_QWORD *)a6 + 8LL * v19, v20 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
      {
        v21 = (__m128i *)(a1 + 16 * v16);
        if ( !v32 )
        {
LABEL_31:
          v15 = v21;
          goto LABEL_32;
        }
      }
      else
      {
        v21 = (__m128i *)(a1 + 16 * v16);
        if ( *(_DWORD *)(*(_QWORD *)v20 + 8LL) >= v32 )
          goto LABEL_31;
      }
      *v21 = _mm_loadu_si128(v22);
      v16 = v17;
      if ( a2 >= v17 )
        break;
      v17 = (v17 - 1) / 2;
    }
    v15 = (__m128i *)(a1 + 16 * v17);
  }
LABEL_32:
  v15->m128i_i64[0] = a4;
  v15->m128i_i64[1] = a5;
  return a5;
}
