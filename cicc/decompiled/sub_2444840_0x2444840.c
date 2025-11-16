// Function: sub_2444840
// Address: 0x2444840
//
__m128i *__fastcall sub_2444840(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r10
  bool v11; // zf
  __int64 v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 j; // rax
  __int64 v17; // r9
  int v18; // edi
  int v19; // r15d
  __int64 *v20; // r14
  unsigned int v21; // ecx
  __int64 *v22; // rsi
  __int64 v23; // r11
  __int64 v24; // rdx
  int v25; // edi
  __m128i *result; // rax
  _QWORD *v27; // r14
  _QWORD *v28; // rcx
  _QWORD *v29; // rax
  __m128i *v30; // r12
  __int64 v31; // rdx
  __m128i v32; // xmm5
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rdx
  _QWORD *k; // rdx
  _QWORD *v37; // r8
  int v38; // edi
  int v39; // r11d
  __int64 *v40; // r10
  unsigned int v41; // ecx
  __int64 *v42; // rsi
  __int64 v43; // r9
  __m128i v44; // xmm2
  __m128i v45; // xmm3
  int v46; // ecx
  _BYTE v47[208]; // [rsp+10h] [rbp-D0h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_6;
    }
    v27 = (_QWORD *)(a1 + 16);
    v28 = (_QWORD *)(a1 + 176);
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v27 = (_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(a1 + 176);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 40LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 2560;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 40LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v4 + v10;
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 5LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 20;
        }
        for ( i = &v13[v14]; i != v13; v13 += 5 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = v4; v12 != j; j += 40 )
        {
          v24 = *(_QWORD *)j;
          if ( *(_QWORD *)j != -4096 && v24 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = a1 + 16;
              v18 = 3;
            }
            else
            {
              v25 = *(_DWORD *)(a1 + 24);
              v17 = *(_QWORD *)(a1 + 16);
              if ( !v25 )
                goto LABEL_77;
              v18 = v25 - 1;
            }
            v19 = 1;
            v20 = 0;
            v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v22 = (__int64 *)(v17 + 40LL * v21);
            v23 = *v22;
            if ( *v22 != v24 )
            {
              while ( v23 != -4096 )
              {
                if ( v23 == -8192 && !v20 )
                  v20 = v22;
                v21 = v18 & (v19 + v21);
                v22 = (__int64 *)(v17 + 40LL * v21);
                v23 = *v22;
                if ( v24 == *v22 )
                  goto LABEL_16;
                ++v19;
              }
              if ( v20 )
                v22 = v20;
            }
LABEL_16:
            *v22 = v24;
            *(__m128i *)(v22 + 1) = _mm_loadu_si128((const __m128i *)(j + 8));
            *(__m128i *)(v22 + 3) = _mm_loadu_si128((const __m128i *)(j + 24));
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return (__m128i *)sub_C7D6A0(v4, v10, 8);
      }
      v27 = (_QWORD *)(a1 + 16);
      v28 = (_QWORD *)(a1 + 176);
      v2 = 64;
    }
  }
  v29 = v27;
  v30 = (__m128i *)v47;
  do
  {
    v31 = *v29;
    if ( *v29 != -4096 && v31 != -8192 )
    {
      if ( v30 )
        v30->m128i_i64[0] = v31;
      v32 = _mm_loadu_si128((const __m128i *)(v29 + 3));
      v30 = (__m128i *)((char *)v30 + 40);
      v30[-2] = _mm_loadu_si128((const __m128i *)(v29 + 1));
      v30[-1] = v32;
    }
    v29 += 5;
  }
  while ( v29 != v28 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(40LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(_QWORD **)(a1 + 16);
    v35 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v27;
    v35 = 20;
  }
  for ( k = &v34[v35]; k != v34; v34 += 5 )
  {
    if ( v34 )
      *v34 = -4096;
  }
  result = (__m128i *)v47;
  if ( v30 != (__m128i *)v47 )
  {
    do
    {
      v24 = result->m128i_i64[0];
      if ( result->m128i_i64[0] != -4096 && v24 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v37 = v27;
          v38 = 3;
        }
        else
        {
          v46 = *(_DWORD *)(a1 + 24);
          v37 = *(_QWORD **)(a1 + 16);
          if ( !v46 )
          {
LABEL_77:
            MEMORY[0] = v24;
            BUG();
          }
          v38 = v46 - 1;
        }
        v39 = 1;
        v40 = 0;
        v41 = v38 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v42 = &v37[5 * v41];
        v43 = *v42;
        if ( v24 != *v42 )
        {
          while ( v43 != -4096 )
          {
            if ( v43 == -8192 && !v40 )
              v40 = v42;
            v41 = v38 & (v39 + v41);
            v42 = &v37[5 * v41];
            v43 = *v42;
            if ( v24 == *v42 )
              goto LABEL_47;
            ++v39;
          }
          if ( v40 )
            v42 = v40;
        }
LABEL_47:
        v44 = _mm_loadu_si128((const __m128i *)&result->m128i_u64[1]);
        v45 = _mm_loadu_si128((__m128i *)((char *)result + 24));
        *v42 = v24;
        *(__m128i *)(v42 + 1) = v44;
        *(__m128i *)(v42 + 3) = v45;
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      }
      result = (__m128i *)((char *)result + 40);
    }
    while ( v30 != result );
  }
  return result;
}
