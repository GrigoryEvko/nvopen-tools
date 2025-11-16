// Function: sub_37EBC90
// Address: 0x37ebc90
//
__m128i *__fastcall sub_37EBC90(__int64 a1, unsigned int a2)
{
  char v4; // si
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r10
  bool v11; // zf
  __int64 v12; // rsi
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *i; // rdx
  __int64 j; // rax
  int v17; // edx
  __int64 v18; // r9
  int v19; // ecx
  int v20; // r15d
  int *v21; // r14
  unsigned int v22; // r8d
  int *v23; // rdi
  int v24; // r11d
  __m128i v25; // xmm1
  __m128i *result; // rax
  int v27; // ecx
  _DWORD *v28; // r15
  _DWORD *v29; // rcx
  _DWORD *v30; // rax
  __m128i *v31; // r12
  __m128i v32; // xmm0
  __int64 v33; // rax
  _DWORD *v34; // rax
  __int64 v35; // rdx
  _DWORD *k; // rdx
  __int32 v37; // edx
  _DWORD *v38; // r9
  int v39; // edi
  int v40; // r11d
  int *v41; // r10
  unsigned int v42; // esi
  __int32 *v43; // rcx
  int v44; // r8d
  __m128i v45; // xmm2
  int v46; // ecx
  _BYTE v47[368]; // [rsp+10h] [rbp-170h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x10 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v28 = (_DWORD *)(a1 + 16);
    v29 = (_DWORD *)(a1 + 336);
  }
  else
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    a2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v28 = (_DWORD *)(a1 + 16);
      v29 = (_DWORD *)(a1 + 336);
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 20LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        a2 = 64;
        v8 = 1280;
LABEL_5:
        v9 = sub_C7D670(v8, 4);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 20LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = v6 + v10;
        if ( v11 )
        {
          v13 = *(_DWORD **)(a1 + 16);
          v14 = 5LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_DWORD *)(a1 + 16);
          v14 = 80;
        }
        for ( i = &v13[v14]; i != v13; v13 += 5 )
        {
          if ( v13 )
            *v13 = -1;
        }
        for ( j = v6; v12 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
        {
          while ( 1 )
          {
            v17 = *(_DWORD *)j;
            if ( *(_DWORD *)j <= 0xFFFFFFFD )
              break;
            j += 20;
            if ( v12 == j )
              return (__m128i *)sub_C7D6A0(v6, v10, 4);
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = a1 + 16;
            v19 = 15;
          }
          else
          {
            v27 = *(_DWORD *)(a1 + 24);
            v18 = *(_QWORD *)(a1 + 16);
            if ( !v27 )
              goto LABEL_74;
            v19 = v27 - 1;
          }
          v20 = 1;
          v21 = 0;
          v22 = v19 & (37 * v17);
          v23 = (int *)(v18 + 20LL * v22);
          v24 = *v23;
          if ( v17 != *v23 )
          {
            while ( v24 != -1 )
            {
              if ( v24 == -2 && !v21 )
                v21 = v23;
              v22 = v19 & (v20 + v22);
              v23 = (int *)(v18 + 20LL * v22);
              v24 = *v23;
              if ( v17 == *v23 )
                goto LABEL_21;
              ++v20;
            }
            if ( v21 )
              v23 = v21;
          }
LABEL_21:
          *v23 = v17;
          v25 = _mm_loadu_si128((const __m128i *)(j + 4));
          j += 20;
          *(__m128i *)(v23 + 1) = v25;
        }
        return (__m128i *)sub_C7D6A0(v6, v10, 4);
      }
      v28 = (_DWORD *)(a1 + 16);
      v29 = (_DWORD *)(a1 + 336);
      a2 = 64;
    }
  }
  v30 = v28;
  v31 = (__m128i *)v47;
  do
  {
    while ( *v30 > 0xFFFFFFFD )
    {
      v30 += 5;
      if ( v30 == v29 )
        goto LABEL_33;
    }
    if ( v31 )
      v31->m128i_i32[0] = *v30;
    v32 = _mm_loadu_si128((const __m128i *)(v30 + 1));
    v30 += 5;
    v31 = (__m128i *)((char *)v31 + 20);
    v31[-1] = v32;
  }
  while ( v30 != v29 );
LABEL_33:
  if ( a2 > 0x10 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v33 = sub_C7D670(20LL * a2, 4);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v34 = *(_DWORD **)(a1 + 16);
    v35 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v34 = v28;
    v35 = 80;
  }
  for ( k = &v34[v35]; k != v34; v34 += 5 )
  {
    if ( v34 )
      *v34 = -1;
  }
  result = (__m128i *)v47;
  if ( v31 != (__m128i *)v47 )
  {
    do
    {
      while ( 1 )
      {
        v37 = result->m128i_i32[0];
        if ( result->m128i_i32[0] <= 0xFFFFFFFD )
          break;
        result = (__m128i *)((char *)result + 20);
        if ( v31 == result )
          return result;
      }
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v38 = v28;
        v39 = 15;
      }
      else
      {
        v46 = *(_DWORD *)(a1 + 24);
        v38 = *(_DWORD **)(a1 + 16);
        if ( !v46 )
        {
LABEL_74:
          MEMORY[0] = 0;
          BUG();
        }
        v39 = v46 - 1;
      }
      v40 = 1;
      v41 = 0;
      v42 = v39 & (37 * v37);
      v43 = &v38[5 * v42];
      v44 = *v43;
      if ( v37 != *v43 )
      {
        while ( v44 != -1 )
        {
          if ( v44 == -2 && !v41 )
            v41 = v43;
          v42 = v39 & (v40 + v42);
          v43 = &v38[5 * v42];
          v44 = *v43;
          if ( v37 == *v43 )
            goto LABEL_48;
          ++v40;
        }
        if ( v41 )
          v43 = v41;
      }
LABEL_48:
      v45 = _mm_loadu_si128((const __m128i *)((char *)result->m128i_i64 + 4));
      *v43 = v37;
      result = (__m128i *)((char *)result + 20);
      *(__m128i *)(v43 + 1) = v45;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
    while ( v31 != result );
  }
  return result;
}
