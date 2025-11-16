// Function: sub_34C9F90
// Address: 0x34c9f90
//
__int64 __fastcall sub_34C9F90(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // dl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rcx
  bool v12; // zf
  __m128i *v13; // r11
  _DWORD *v14; // rax
  __int64 v15; // rdx
  _DWORD *i; // rdx
  __m128i *j; // rdx
  __int32 v18; // ecx
  __int64 v19; // r10
  int v20; // esi
  __int32 v21; // r8d
  int v22; // r15d
  __int64 v23; // r14
  unsigned int k; // eax
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 v27; // rax
  __m128i *v28; // r12
  __int64 v29; // rax
  __int64 result; // rax
  __int64 v31; // rcx
  __int64 m; // rcx
  __int32 v33; // ecx
  __int64 v34; // r8
  int v35; // esi
  __int32 v36; // edi
  int v37; // r14d
  __int64 v38; // r11
  unsigned int n; // eax
  __int64 v40; // r10
  unsigned int v41; // eax
  int v42; // esi
  int v43; // esi
  __m128i v44; // xmm1
  int v45; // r9d
  int v46; // r15d
  __int64 v47; // [rsp+0h] [rbp-A0h]
  _BYTE v48[144]; // [rsp+10h] [rbp-90h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = a1 + 16;
    v11 = a1 + 112;
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
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
    v2 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v10 = a1 + 16;
      v11 = a1 + 112;
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 24LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(_QWORD *)(a1 + 16);
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 1536;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v47 = 24LL * v7;
        v13 = (__m128i *)(v6 + v47);
        if ( v12 )
        {
          v14 = *(_DWORD **)(a1 + 16);
          v15 = 6LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v14 = (_DWORD *)(a1 + 16);
          v15 = 24;
        }
        for ( i = &v14[v15]; i != v14; v14 += 6 )
        {
          if ( v14 )
          {
            *v14 = 0;
            v14[1] = -1;
          }
        }
        for ( j = (__m128i *)v6; v13 != j; j = (__m128i *)((char *)j + 24) )
        {
          v18 = j->m128i_i32[0];
          if ( j->m128i_i32[0] || j->m128i_i32[1] <= 0xFFFFFFFD )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v19 = a1 + 16;
              v20 = 3;
            }
            else
            {
              v42 = *(_DWORD *)(a1 + 24);
              v19 = *(_QWORD *)(a1 + 16);
              if ( !v42 )
                goto LABEL_81;
              v20 = v42 - 1;
            }
            v21 = j->m128i_i32[1];
            v22 = 1;
            v23 = 0;
            for ( k = v20
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)(37 * v21) | ((unsigned __int64)(unsigned int)(37 * v18) << 32))) >> 31)
                     ^ (756364221 * v21)); ; k = v20 & v26 )
            {
              v25 = v19 + 24LL * k;
              if ( v18 == *(_DWORD *)v25 && v21 == *(_DWORD *)(v25 + 4) )
                break;
              if ( !*(_DWORD *)v25 )
              {
                v45 = *(_DWORD *)(v25 + 4);
                if ( v45 == -1 )
                {
                  if ( v23 )
                    v25 = v23;
                  break;
                }
                if ( v45 == -2 && !v23 )
                  v23 = v19 + 24LL * k;
              }
              v26 = v22 + k;
              ++v22;
            }
            *(_QWORD *)v25 = j->m128i_i64[0];
            *(__m128i *)(v25 + 8) = _mm_loadu_si128((const __m128i *)&j->m128i_u64[1]);
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return sub_C7D6A0(v6, v47, 8);
      }
      v10 = a1 + 16;
      v11 = a1 + 112;
      v2 = 64;
    }
  }
  v27 = v10;
  v28 = (__m128i *)v48;
  do
  {
    if ( *(_DWORD *)v27 || *(_DWORD *)(v27 + 4) <= 0xFFFFFFFD )
    {
      if ( v28 )
        v28->m128i_i64[0] = *(_QWORD *)v27;
      v28 = (__m128i *)((char *)v28 + 24);
      v28[-1] = _mm_loadu_si128((const __m128i *)(v27 + 8));
    }
    v27 += 24;
  }
  while ( v27 != v11 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v29 = sub_C7D670(24LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v29;
  }
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v12 )
  {
    result = *(_QWORD *)(a1 + 16);
    v31 = 24LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = v10;
    v31 = 96;
  }
  for ( m = result + v31; m != result; result += 24 )
  {
    if ( result )
    {
      *(_DWORD *)result = 0;
      *(_DWORD *)(result + 4) = -1;
    }
  }
  j = (__m128i *)v48;
  if ( v28 != (__m128i *)v48 )
  {
    do
    {
      v33 = j->m128i_i32[0];
      if ( j->m128i_i32[0] || j->m128i_i32[1] <= 0xFFFFFFFD )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v34 = v10;
          v35 = 3;
        }
        else
        {
          v43 = *(_DWORD *)(a1 + 24);
          v34 = *(_QWORD *)(a1 + 16);
          if ( !v43 )
          {
LABEL_81:
            MEMORY[0] = j->m128i_i64[0];
            BUG();
          }
          v35 = v43 - 1;
        }
        v36 = j->m128i_i32[1];
        v37 = 1;
        v38 = 0;
        for ( n = v35
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v36) | ((unsigned __int64)(unsigned int)(37 * v33) << 32))) >> 31)
                 ^ (756364221 * v36)); ; n = v35 & v41 )
        {
          v40 = v34 + 24LL * n;
          if ( v33 == *(_DWORD *)v40 && v36 == *(_DWORD *)(v40 + 4) )
            break;
          if ( !*(_DWORD *)v40 )
          {
            v46 = *(_DWORD *)(v40 + 4);
            if ( v46 == -1 )
            {
              if ( v38 )
                v40 = v38;
              break;
            }
            if ( v46 == -2 && !v38 )
              v38 = v34 + 24LL * n;
          }
          v41 = v37 + n;
          ++v37;
        }
        v44 = _mm_loadu_si128((const __m128i *)&j->m128i_u64[1]);
        *(_QWORD *)v40 = j->m128i_i64[0];
        *(__m128i *)(v40 + 8) = v44;
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
      }
      j = (__m128i *)((char *)j + 24);
    }
    while ( v28 != j );
  }
  return result;
}
