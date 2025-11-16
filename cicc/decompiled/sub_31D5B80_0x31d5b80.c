// Function: sub_31D5B80
// Address: 0x31d5b80
//
const void *__fastcall sub_31D5B80(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const void *a7,
        unsigned __int64 a8,
        int a9)
{
  __int64 v9; // r15
  __int64 v10; // r13
  int v11; // r8d
  unsigned __int64 v12; // r9
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // rax
  __m128i *v16; // rax
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // esi
  size_t v22; // r10
  size_t v23; // r11
  size_t v24; // rdx
  int v25; // eax
  __int64 v26; // r14
  __int64 v27; // rcx
  unsigned int v28; // r15d
  __int64 v30; // r12
  __int64 v31; // rdx
  size_t v32; // r8
  size_t v33; // rdx
  int v34; // eax
  __int64 v35; // rax
  __m128i *v36; // rax
  __int64 v39; // [rsp+20h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  size_t v41; // [rsp+28h] [rbp-48h]
  unsigned __int64 v42; // [rsp+28h] [rbp-48h]
  size_t v43; // [rsp+30h] [rbp-40h]
  size_t v44; // [rsp+30h] [rbp-40h]

  v9 = a2;
  v10 = a1;
  v11 = a9;
  v39 = a3 & 1;
  v12 = a8;
  if ( a2 >= (a3 - 1) / 2 )
  {
    v20 = a1 + 24 * a2;
    if ( (a3 & 1) != 0 )
      goto LABEL_19;
    v13 = a2;
LABEL_22:
    if ( (a3 - 2) / 2 == v13 )
    {
      v30 = v13 + 1;
      v31 = 6 * v30;
      v13 = 2 * v30 - 1;
      *(__m128i *)v20 = _mm_loadu_si128((const __m128i *)(v10 + 8 * v31 - 24));
      *(_DWORD *)(v20 + 16) = *(_DWORD *)(v10 + 8 * v31 - 24 + 16);
      v20 = v10 + 24 * v13;
    }
    goto LABEL_14;
  }
  v13 = a2;
  v14 = (a3 - 1) / 2;
  do
  {
    v17 = 2 * (v13 + 1);
    v18 = 48 * (v13 + 1);
    v19 = a1 + v18 - 24;
    v20 = a1 + v18;
    v21 = *(_DWORD *)(v19 + 16);
    if ( *(_DWORD *)(v20 + 16) > v21 )
    {
LABEL_6:
      --v17;
      v20 = a1 + 24 * v17;
      goto LABEL_4;
    }
    if ( *(_DWORD *)(v20 + 16) != v21 )
      goto LABEL_4;
    v22 = *(_QWORD *)(v20 + 8);
    v23 = *(_QWORD *)(v19 + 8);
    v24 = v22;
    if ( v23 <= v22 )
      v24 = *(_QWORD *)(v19 + 8);
    if ( v24
      && (v41 = *(_QWORD *)(v20 + 8),
          v43 = *(_QWORD *)(v19 + 8),
          v25 = memcmp(*(const void **)v20, *(const void **)v19, v24),
          v23 = v43,
          v22 = v41,
          v25) )
    {
      if ( v25 < 0 )
      {
        --v17;
        v20 = a1 + 24 * v17;
      }
    }
    else if ( v23 > v22 )
    {
      goto LABEL_6;
    }
LABEL_4:
    v15 = 3 * v13;
    v13 = v17;
    v16 = (__m128i *)(a1 + 8 * v15);
    *v16 = _mm_loadu_si128((const __m128i *)v20);
    v16[1].m128i_i32[0] = *(_DWORD *)(v20 + 16);
  }
  while ( v17 < v14 );
  v12 = a8;
  v10 = a1;
  v11 = a9;
  v9 = a2;
  if ( !v39 )
    goto LABEL_22;
LABEL_14:
  v26 = (v13 - 1) / 2;
  if ( v13 > v9 )
  {
    v27 = v9;
    v28 = v11;
    while ( 1 )
    {
      v20 = v10 + 24 * v26;
      if ( v28 >= *(_DWORD *)(v20 + 16) )
      {
        if ( v28 != *(_DWORD *)(v20 + 16) )
          goto LABEL_18;
        v32 = *(_QWORD *)(v20 + 8);
        v33 = v32;
        if ( v12 <= v32 )
          v33 = v12;
        if ( v33
          && (v40 = v27,
              v42 = v12,
              v44 = *(_QWORD *)(v20 + 8),
              v34 = memcmp(*(const void **)v20, a7, v33),
              v32 = v44,
              v12 = v42,
              v27 = v40,
              v34) )
        {
          if ( v34 >= 0 )
          {
LABEL_18:
            v11 = v28;
            v20 = v10 + 24 * v13;
            goto LABEL_19;
          }
        }
        else if ( v12 <= v32 )
        {
          goto LABEL_18;
        }
      }
      v35 = 3 * v13;
      v13 = v26;
      v36 = (__m128i *)(v10 + 8 * v35);
      *v36 = _mm_loadu_si128((const __m128i *)v20);
      v36[1].m128i_i32[0] = *(_DWORD *)(v20 + 16);
      if ( v27 >= v26 )
        break;
      v26 = (v26 - 1) / 2;
    }
    v11 = v28;
  }
LABEL_19:
  *(_QWORD *)(v20 + 8) = v12;
  *(_DWORD *)(v20 + 16) = v11;
  *(_QWORD *)v20 = a7;
  return a7;
}
