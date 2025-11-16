// Function: sub_2D33C00
// Address: 0x2d33c00
//
__int64 __fastcall sub_2D33C00(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  char v4; // r13
  unsigned int v5; // eax
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  __m128i *v10; // rbx
  const __m128i *v11; // rcx
  bool v12; // zf
  __m128i *v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  __m128i *v17; // rbx
  __int64 result; // rax
  const __m128i *v19; // rax
  __m128i *v20; // r14
  __int64 v21; // rax
  __int64 v22; // rax
  __m128i *v23; // rbx
  __m128i *v24; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v25[14]; // [rsp+20h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = (__m128i *)(a1 + 16);
    v11 = (const __m128i *)(a1 + 80);
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    goto LABEL_25;
  }
  v5 = sub_AF1560(a2 - 1);
  v2 = v5;
  if ( v5 > 0x40 )
  {
    v10 = (__m128i *)(a1 + 16);
    v11 = (const __m128i *)(a1 + 80);
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      v8 = 16LL * v5;
      goto LABEL_5;
    }
    goto LABEL_25;
  }
  if ( v4 )
  {
    v10 = (__m128i *)(a1 + 16);
    v11 = (const __m128i *)(a1 + 80);
    v2 = 64;
LABEL_25:
    v19 = v10;
    v20 = (__m128i *)v25;
    while ( 1 )
    {
      if ( v19->m128i_i64[0] == -1 )
      {
        if ( v19->m128i_i64[1] != -1 )
          goto LABEL_29;
      }
      else if ( v19->m128i_i64[0] != -2 || v19->m128i_i64[1] != -2 )
      {
LABEL_29:
        if ( v20 )
          *v20 = _mm_loadu_si128(v19);
        ++v20;
      }
      if ( ++v19 == v11 )
      {
        if ( v2 > 4 )
        {
          *(_BYTE *)(a1 + 8) &= ~1u;
          v21 = sub_C7D670(16LL * v2, 8);
          *(_DWORD *)(a1 + 24) = v2;
          *(_QWORD *)(a1 + 16) = v21;
        }
        v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v22 = 4;
        if ( v12 )
        {
          v10 = *(__m128i **)(a1 + 16);
          v22 = *(unsigned int *)(a1 + 24);
        }
        for ( result = (__int64)v10[v22].m128i_i64; (__m128i *)result != v10; ++v10 )
        {
          if ( v10 )
          {
            v10->m128i_i64[0] = -1;
            v10->m128i_i64[1] = -1;
          }
        }
        if ( v20 == (__m128i *)v25 )
          return result;
        v23 = (__m128i *)v25;
        while ( 2 )
        {
          while ( 1 )
          {
            result = v23->m128i_i64[0];
            if ( v23->m128i_i64[0] != -1 )
              break;
            if ( v23->m128i_i64[1] != -1 )
              goto LABEL_45;
            if ( v20 == ++v23 )
              return result;
          }
          if ( result != -2 || v23->m128i_i64[1] != -2 )
          {
LABEL_45:
            sub_2D2B6B0(a1, v23->m128i_i64, &v24);
            *v24 = _mm_loadu_si128(v23);
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
          }
          if ( v20 == ++v23 )
            return result;
          continue;
        }
      }
    }
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(unsigned int *)(a1 + 24);
  v8 = 1024;
  v2 = 64;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v13 = (__m128i *)(v6 + 16 * v7);
  if ( v12 )
  {
    v14 = *(_QWORD **)(a1 + 16);
    v15 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v14 = (_QWORD *)(a1 + 16);
    v15 = 8;
  }
  for ( i = &v14[v15]; i != v14; v14 += 2 )
  {
    if ( v14 )
    {
      *v14 = -1;
      v14[1] = -1;
    }
  }
  if ( v13 != (__m128i *)v6 )
  {
    v17 = (__m128i *)v6;
    while ( v17->m128i_i64[0] == -1 )
    {
      if ( v17->m128i_i64[1] == -1 )
      {
        if ( v13 == ++v17 )
          return sub_C7D6A0(v6, 16 * v7, 8);
      }
      else
      {
LABEL_18:
        sub_2D2B6B0(a1, v17->m128i_i64, v25);
        *(__m128i *)v25[0] = _mm_loadu_si128(v17);
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
LABEL_19:
        if ( v13 == ++v17 )
          return sub_C7D6A0(v6, 16 * v7, 8);
      }
    }
    if ( v17->m128i_i64[0] == -2 && v17->m128i_i64[1] == -2 )
      goto LABEL_19;
    goto LABEL_18;
  }
  return sub_C7D6A0(v6, 16 * v7, 8);
}
