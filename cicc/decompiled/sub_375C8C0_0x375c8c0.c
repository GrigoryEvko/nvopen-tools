// Function: sub_375C8C0
// Address: 0x375c8c0
//
__m128i *__fastcall sub_375C8C0(const __m128i *a1, unsigned int a2)
{
  unsigned int v2; // r14d
  __int64 v4; // r12
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  const __m128i *v10; // r13
  const __m128i *v11; // rcx
  bool v12; // zf
  __m128i *v13; // r11
  const __m128i *v14; // rax
  __int64 v15; // rdx
  const __m128i *i; // rdx
  __m128i *result; // rax
  unsigned __int64 v18; // rcx
  const __m128i *v19; // r8
  int v20; // edi
  __int32 v21; // r9d
  int v22; // r10d
  __int64 *v23; // r13
  unsigned int j; // edx
  __int64 *v25; // rsi
  unsigned int v26; // edx
  const __m128i *v27; // rax
  __m128i *v28; // r12
  __int32 v29; // edx
  __int64 v30; // rax
  const __m128i *v31; // rax
  __int64 v32; // rdx
  const __m128i *k; // rdx
  unsigned __int64 v34; // rcx
  const __m128i *v35; // r8
  int v36; // edi
  __int32 v37; // r9d
  int v38; // r14d
  __int64 *v39; // r11
  unsigned int m; // edx
  __int64 *v41; // rsi
  unsigned int v42; // edx
  __int32 v43; // edi
  __int32 v44; // edi
  int v45; // r15d
  int v46; // r10d
  __int64 v47; // [rsp+8h] [rbp-F8h]
  _BYTE v48[240]; // [rsp+10h] [rbp-F0h] BYREF

  v2 = a2;
  v4 = a1[1].m128i_i64[0];
  v5 = a1->m128i_i8[8] & 1;
  if ( a2 <= 8 )
  {
    v10 = a1 + 1;
    v11 = a1 + 13;
    if ( !v5 )
    {
      v7 = a1[1].m128i_u32[2];
      a1->m128i_i8[8] |= 1u;
      goto LABEL_8;
    }
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
      v10 = a1 + 1;
      v11 = a1 + 13;
      if ( !v5 )
      {
        v7 = a1[1].m128i_u32[2];
        v8 = 24LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = a1[1].m128i_u32[2];
        v2 = 64;
        v8 = 1536;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        a1[1].m128i_i32[2] = v2;
        a1[1].m128i_i64[0] = v9;
LABEL_8:
        v12 = (a1->m128i_i64[1] & 1) == 0;
        a1->m128i_i64[1] &= 1uLL;
        v47 = 24LL * v7;
        v13 = (__m128i *)(v4 + v47);
        if ( v12 )
        {
          v14 = (const __m128i *)a1[1].m128i_i64[0];
          v15 = 24LL * a1[1].m128i_u32[2];
        }
        else
        {
          v14 = a1 + 1;
          v15 = 192;
        }
        for ( i = (const __m128i *)((char *)v14 + v15); i != v14; v14 = (const __m128i *)((char *)v14 + 24) )
        {
          if ( v14 )
          {
            v14->m128i_i64[0] = 0;
            v14->m128i_i32[2] = -1;
          }
        }
        for ( result = (__m128i *)v4; v13 != result; result = (__m128i *)((char *)result + 24) )
        {
          v18 = result->m128i_i64[0];
          if ( result->m128i_i64[0] || result->m128i_i32[2] <= 0xFFFFFFFD )
          {
            if ( (a1->m128i_i8[8] & 1) != 0 )
            {
              v19 = a1 + 1;
              v20 = 7;
            }
            else
            {
              v43 = a1[1].m128i_i32[2];
              v19 = (const __m128i *)a1[1].m128i_i64[0];
              if ( !v43 )
                goto LABEL_80;
              v20 = v43 - 1;
            }
            v21 = result->m128i_i32[2];
            v22 = 1;
            v23 = 0;
            for ( j = v20 & (v21 + ((v18 >> 9) ^ (v18 >> 4))); ; j = v20 & v26 )
            {
              v25 = &v19->m128i_i64[3 * j];
              if ( v18 == *v25 && v21 == *((_DWORD *)v25 + 2) )
                break;
              if ( !*v25 )
              {
                v45 = *((_DWORD *)v25 + 2);
                if ( v45 == -1 )
                {
                  if ( v23 )
                    v25 = v23;
                  break;
                }
                if ( v45 == -2 && !v23 )
                  v23 = &v19->m128i_i64[3 * j];
              }
              v26 = v22 + j;
              ++v22;
            }
            *v25 = result->m128i_i64[0];
            *((_DWORD *)v25 + 2) = result->m128i_i32[2];
            *((_DWORD *)v25 + 4) = result[1].m128i_i32[0];
            a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          }
        }
        return (__m128i *)sub_C7D6A0(v4, v47, 8);
      }
      v10 = a1 + 1;
      v11 = a1 + 13;
      v2 = 64;
    }
  }
  v27 = v10;
  v28 = (__m128i *)v48;
  do
  {
    while ( !v27->m128i_i64[0] && v27->m128i_i32[2] > 0xFFFFFFFD )
    {
      v27 = (const __m128i *)((char *)v27 + 24);
      if ( v27 == v11 )
        goto LABEL_30;
    }
    if ( v28 )
      *v28 = _mm_loadu_si128(v27);
    v29 = v27[1].m128i_i32[0];
    v27 = (const __m128i *)((char *)v27 + 24);
    v28 = (__m128i *)((char *)v28 + 24);
    v28[-1].m128i_i32[2] = v29;
  }
  while ( v27 != v11 );
LABEL_30:
  if ( v2 > 8 )
  {
    a1->m128i_i8[8] &= ~1u;
    v30 = sub_C7D670(24LL * v2, 8);
    a1[1].m128i_i32[2] = v2;
    a1[1].m128i_i64[0] = v30;
  }
  v12 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  if ( v12 )
  {
    v31 = (const __m128i *)a1[1].m128i_i64[0];
    v32 = 24LL * a1[1].m128i_u32[2];
  }
  else
  {
    v31 = v10;
    v32 = 192;
  }
  for ( k = (const __m128i *)((char *)v31 + v32); k != v31; v31 = (const __m128i *)((char *)v31 + 24) )
  {
    if ( v31 )
    {
      v31->m128i_i64[0] = 0;
      v31->m128i_i32[2] = -1;
    }
  }
  result = (__m128i *)v48;
  if ( v28 != (__m128i *)v48 )
  {
    do
    {
      v34 = result->m128i_i64[0];
      if ( result->m128i_i64[0] || result->m128i_i32[2] <= 0xFFFFFFFD )
      {
        if ( (a1->m128i_i8[8] & 1) != 0 )
        {
          v35 = v10;
          v36 = 7;
        }
        else
        {
          v44 = a1[1].m128i_i32[2];
          v35 = (const __m128i *)a1[1].m128i_i64[0];
          if ( !v44 )
          {
LABEL_80:
            MEMORY[0] = result->m128i_i64[0];
            MEMORY[8] = result->m128i_i32[2];
            BUG();
          }
          v36 = v44 - 1;
        }
        v37 = result->m128i_i32[2];
        v38 = 1;
        v39 = 0;
        for ( m = v36 & (v37 + ((v34 >> 9) ^ (v34 >> 4))); ; m = v36 & v42 )
        {
          v41 = &v35->m128i_i64[3 * m];
          if ( v34 == *v41 && v37 == *((_DWORD *)v41 + 2) )
            break;
          if ( !*v41 )
          {
            v46 = *((_DWORD *)v41 + 2);
            if ( v46 == -1 )
            {
              if ( v39 )
                v41 = v39;
              break;
            }
            if ( v46 == -2 && !v39 )
              v39 = &v35->m128i_i64[3 * m];
          }
          v42 = v38 + m;
          ++v38;
        }
        *v41 = result->m128i_i64[0];
        *((_DWORD *)v41 + 2) = result->m128i_i32[2];
        *((_DWORD *)v41 + 4) = result[1].m128i_i32[0];
        a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
      }
      result = (__m128i *)((char *)result + 24);
    }
    while ( v28 != result );
  }
  return result;
}
