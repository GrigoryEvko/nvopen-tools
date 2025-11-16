// Function: sub_2260950
// Address: 0x2260950
//
size_t __fastcall sub_2260950(__int64 a1, __int64 a2, __int64 a3, const void *a4, size_t a5, __int64 a6)
{
  __int64 i; // r14
  __int64 v8; // rax
  unsigned int v9; // eax
  const __m128i *v10; // r15
  __int64 v11; // r8
  __int64 v12; // r12
  size_t v13; // r13
  int v14; // eax
  int v15; // eax
  __int64 v16; // rax
  const void *v17; // r13
  int v18; // eax
  int v19; // eax
  __m128i *v20; // r9
  __int64 v21; // r13
  __int64 v22; // r12
  __int64 v23; // rax
  int v24; // eax
  int v25; // eax
  __int64 v26; // rax
  __m128i *v27; // r13
  const __m128i *v28; // r14
  int v29; // eax
  int v30; // eax
  __int64 v33; // [rsp+10h] [rbp-80h]
  __int64 v35; // [rsp+20h] [rbp-70h]
  size_t v36; // [rsp+38h] [rbp-58h]
  __int64 v37; // [rsp+40h] [rbp-50h]
  const void *v38; // [rsp+40h] [rbp-50h]
  const void *v39; // [rsp+48h] [rbp-48h]
  unsigned int v40; // [rsp+48h] [rbp-48h]
  unsigned int v41; // [rsp+48h] [rbp-48h]
  size_t v42; // [rsp+48h] [rbp-48h]

  v33 = a3 & 1;
  v35 = (a3 - 1) / 2;
  if ( a2 >= v35 )
  {
    v21 = a2;
    v20 = (__m128i *)(a1 + 16 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_32;
    goto LABEL_28;
  }
  for ( i = a2; ; i = v11 )
  {
    v37 = 2 * (i + 1) - 1;
    v10 = (const __m128i *)(a1 + 32 * (i + 1));
    v12 = a1 + 16 * v37;
    v13 = v10->m128i_u64[1];
    v39 = (const void *)v10->m128i_i64[0];
    v14 = sub_C92610();
    v15 = sub_C92860((__int64 *)a6, v39, v13, v14);
    if ( v15 == -1 || (v16 = *(_QWORD *)a6 + 8LL * v15, v16 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
      v40 = 0;
    else
      v40 = *(_DWORD *)(*(_QWORD *)v16 + 8LL);
    v17 = *(const void **)v12;
    v36 = *(_QWORD *)(v12 + 8);
    v18 = sub_C92610();
    v19 = sub_C92860((__int64 *)a6, v17, v36, v18);
    v11 = 2 * (i + 1);
    if ( v19 == -1 || (v8 = *(_QWORD *)a6 + 8LL * v19, v8 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
      v9 = 0;
    else
      v9 = *(_DWORD *)(*(_QWORD *)v8 + 8LL);
    if ( v9 < v40 )
    {
      v10 = (const __m128i *)(a1 + 16 * v37);
      v11 = 2 * (i + 1) - 1;
    }
    *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(v10);
    if ( v11 >= v35 )
      break;
  }
  v20 = (__m128i *)v10;
  v21 = v11;
  if ( !v33 )
  {
LABEL_28:
    if ( (a3 - 2) / 2 == v21 )
    {
      v21 = 2 * v21 + 1;
      *v20 = _mm_loadu_si128((const __m128i *)(a1 + 16 * v21));
      v20 = (__m128i *)(a1 + 16 * v21);
    }
  }
  v22 = (v21 - 1) / 2;
  if ( v21 > a2 )
  {
    while ( 1 )
    {
      v28 = (const __m128i *)(a1 + 16 * v22);
      v38 = (const void *)v28->m128i_i64[0];
      v42 = v28->m128i_u64[1];
      v29 = sub_C92610();
      v30 = sub_C92860((__int64 *)a6, v38, v42, v29);
      if ( v30 == -1 || (v23 = *(_QWORD *)a6 + 8LL * v30, v23 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
        v41 = 0;
      else
        v41 = *(_DWORD *)(*(_QWORD *)v23 + 8LL);
      v24 = sub_C92610();
      v25 = sub_C92860((__int64 *)a6, a4, a5, v24);
      if ( v25 == -1 || (v26 = *(_QWORD *)a6 + 8LL * v25, v26 == *(_QWORD *)a6 + 8LL * *(unsigned int *)(a6 + 8)) )
      {
        v27 = (__m128i *)(a1 + 16 * v21);
        if ( !v41 )
        {
LABEL_31:
          v20 = v27;
          goto LABEL_32;
        }
      }
      else
      {
        v27 = (__m128i *)(a1 + 16 * v21);
        if ( *(_DWORD *)(*(_QWORD *)v26 + 8LL) >= v41 )
          goto LABEL_31;
      }
      *v27 = _mm_loadu_si128(v28);
      v21 = v22;
      if ( a2 >= v22 )
        break;
      v22 = (v22 - 1) / 2;
    }
    v20 = (__m128i *)(a1 + 16 * v22);
  }
LABEL_32:
  v20->m128i_i64[0] = (__int64)a4;
  v20->m128i_i64[1] = a5;
  return a5;
}
