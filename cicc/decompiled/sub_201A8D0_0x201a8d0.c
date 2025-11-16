// Function: sub_201A8D0
// Address: 0x201a8d0
//
__int64 __fastcall sub_201A8D0(const __m128i *a1, unsigned int a2)
{
  __int64 result; // rax
  const __m128i *v4; // r13
  unsigned __int64 v5; // rbx
  int v6; // r15d
  __int64 v7; // r13
  const __m128i *v8; // rdx
  const __m128i *v9; // rax
  __m128i *v10; // rbx
  __m128i v11; // xmm0
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rdx
  __int64 m128i_i64; // rcx
  const __m128i *v16; // rdx
  unsigned __int64 v17; // rsi
  const __m128i *v18; // r8
  int v19; // edi
  __int32 v20; // r9d
  int v21; // r13d
  __m128i *v22; // r11
  unsigned int i; // r10d
  __m128i *v24; // rax
  unsigned int v25; // eax
  unsigned __int32 v26; // r14d
  __int64 v27; // rax
  const __m128i *v28; // rdi
  const __m128i *v29; // rax
  __int64 v30; // rdx
  const __m128i *j; // rdx
  const __m128i *k; // rax
  unsigned __int64 v33; // rsi
  const __m128i *v34; // r10
  int v35; // r9d
  __int32 v36; // r11d
  int v37; // r15d
  __m128i *v38; // r14
  unsigned int m; // ebx
  __m128i *v40; // rdx
  unsigned int v41; // edx
  __int32 v42; // r9d
  __int32 v43; // edi
  __m128i v44; // xmm3
  __int32 v45; // ecx
  __int32 v46; // ecx
  __int32 v47; // ecx
  _BYTE v48[2096]; // [rsp+10h] [rbp-830h] BYREF

  result = a1->m128i_i8[8] & 1;
  if ( a2 <= 0x3F )
  {
    if ( (_BYTE)result )
      return result;
    v4 = (const __m128i *)a1[1].m128i_i64[0];
    v26 = a1[1].m128i_u32[2];
LABEL_46:
    a1->m128i_i8[8] |= 1u;
    goto LABEL_31;
  }
  v4 = (const __m128i *)a1[1].m128i_i64[0];
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
  v6 = v5;
  if ( (unsigned int)v5 <= 0x40 )
  {
    if ( (_BYTE)result )
    {
      v7 = 2048;
      v6 = 64;
      goto LABEL_5;
    }
    v26 = a1[1].m128i_u32[2];
    goto LABEL_46;
  }
  if ( (_BYTE)result )
  {
    v7 = 32LL * (unsigned int)v5;
LABEL_5:
    v8 = a1 + 129;
    v9 = a1 + 1;
    v10 = (__m128i *)v48;
    do
    {
      while ( !v9->m128i_i64[0] && v9->m128i_i32[2] > 0xFFFFFFFD )
      {
        v9 += 2;
        if ( v9 == v8 )
          goto LABEL_12;
      }
      if ( v10 )
        *v10 = _mm_loadu_si128(v9);
      v11 = _mm_loadu_si128(v9 + 1);
      v9 += 2;
      v10 += 2;
      v10[-1] = v11;
    }
    while ( v9 != v8 );
LABEL_12:
    a1->m128i_i8[8] &= ~1u;
    v12 = sub_22077B0(v7);
    v13 = (a1->m128i_i64[1] & 1) == 0;
    a1->m128i_i64[1] &= 1uLL;
    a1[1].m128i_i64[0] = v12;
    v14 = v12;
    m128i_i64 = v12;
    a1[1].m128i_i32[2] = v6;
    if ( !v13 )
    {
      m128i_i64 = (__int64)a1[1].m128i_i64;
      v14 = (__int64)a1[1].m128i_i64;
      v7 = 2048;
    }
    result = v14 + v7;
    while ( 1 )
    {
      if ( m128i_i64 )
      {
        *(_QWORD *)v14 = 0;
        *(_DWORD *)(v14 + 8) = -1;
      }
      v14 += 32;
      if ( result == v14 )
        break;
      m128i_i64 = v14;
    }
    v16 = (const __m128i *)v48;
    if ( v10 != (__m128i *)v48 )
    {
      do
      {
        v17 = v16->m128i_i64[0];
        if ( v16->m128i_i64[0] || v16->m128i_i32[2] <= 0xFFFFFFFD )
        {
          if ( (a1->m128i_i8[8] & 1) != 0 )
          {
            v18 = a1 + 1;
            v19 = 63;
          }
          else
          {
            v43 = a1[1].m128i_i32[2];
            v18 = (const __m128i *)a1[1].m128i_i64[0];
            if ( !v43 )
            {
              MEMORY[0] = v16->m128i_i64[0];
              MEMORY[8] = v16->m128i_i32[2];
              BUG();
            }
            v19 = v43 - 1;
          }
          v20 = v16->m128i_i32[2];
          v21 = 1;
          v22 = 0;
          for ( i = v19 & (v20 + ((v17 >> 9) ^ (v17 >> 4))); ; i = v19 & v25 )
          {
            v24 = (__m128i *)&v18[2 * i];
            if ( v17 == v24->m128i_i64[0] && v20 == v24->m128i_i32[2] )
              break;
            if ( !v24->m128i_i64[0] )
            {
              v47 = v24->m128i_i32[2];
              if ( v47 == -1 )
              {
                if ( v22 )
                  v24 = v22;
                break;
              }
              if ( v47 == -2 && !v22 )
                v22 = (__m128i *)&v18[2 * i];
            }
            v25 = i + v21++;
          }
          v44 = _mm_loadu_si128(v16 + 1);
          v24->m128i_i64[0] = v16->m128i_i64[0];
          v45 = v16->m128i_i32[2];
          v24[1] = v44;
          v24->m128i_i32[2] = v45;
          result = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
          a1->m128i_i32[2] = result;
        }
        v16 += 2;
      }
      while ( v10 != v16 );
    }
    return result;
  }
  v26 = a1[1].m128i_u32[2];
  v27 = sub_22077B0(32LL * (unsigned int)v5);
  a1[1].m128i_i32[2] = v5;
  a1[1].m128i_i64[0] = v27;
LABEL_31:
  v28 = &v4[2 * v26];
  v13 = (a1->m128i_i64[1] & 1) == 0;
  a1->m128i_i64[1] &= 1uLL;
  if ( v13 )
  {
    v29 = (const __m128i *)a1[1].m128i_i64[0];
    v30 = 2LL * a1[1].m128i_u32[2];
  }
  else
  {
    v29 = a1 + 1;
    v30 = 128;
  }
  for ( j = &v29[v30]; j != v29; v29 += 2 )
  {
    if ( v29 )
    {
      v29->m128i_i64[0] = 0;
      v29->m128i_i32[2] = -1;
    }
  }
  for ( k = v4; v28 != k; k += 2 )
  {
    v33 = k->m128i_i64[0];
    if ( k->m128i_i64[0] || k->m128i_i32[2] <= 0xFFFFFFFD )
    {
      if ( (a1->m128i_i8[8] & 1) != 0 )
      {
        v34 = a1 + 1;
        v35 = 63;
      }
      else
      {
        v42 = a1[1].m128i_i32[2];
        v34 = (const __m128i *)a1[1].m128i_i64[0];
        if ( !v42 )
        {
          MEMORY[0] = k->m128i_i64[0];
          MEMORY[8] = k->m128i_i32[2];
          BUG();
        }
        v35 = v42 - 1;
      }
      v36 = k->m128i_i32[2];
      v37 = 1;
      v38 = 0;
      for ( m = v35 & (v36 + ((v33 >> 9) ^ (v33 >> 4))); ; m = v35 & v41 )
      {
        v40 = (__m128i *)&v34[2 * m];
        if ( v33 == v40->m128i_i64[0] && v36 == v40->m128i_i32[2] )
          break;
        if ( !v40->m128i_i64[0] )
        {
          v46 = v40->m128i_i32[2];
          if ( v46 == -1 )
          {
            if ( v38 )
              v40 = v38;
            break;
          }
          if ( !v38 && v46 == -2 )
            v38 = (__m128i *)&v34[2 * m];
        }
        v41 = m + v37++;
      }
      v40->m128i_i64[0] = k->m128i_i64[0];
      v40->m128i_i32[2] = k->m128i_i32[2];
      v40[1] = _mm_loadu_si128(k + 1);
      a1->m128i_i32[2] = (2 * ((unsigned __int32)a1->m128i_i32[2] >> 1) + 2) | a1->m128i_i32[2] & 1;
    }
  }
  return j___libc_free_0(v4);
}
