// Function: sub_1DC3680
// Address: 0x1dc3680
//
unsigned __int64 __fastcall sub_1DC3680(const __m128i *a1, __int64 a2, __int64 a3, int a4, unsigned int a5, int a6)
{
  unsigned __int64 v7; // rbx
  __int64 v8; // r13
  __int64 v9; // r13
  __int32 v10; // esi
  _QWORD *v11; // rbx
  _QWORD *v12; // r15
  unsigned int v13; // eax
  unsigned __int64 result; // rax
  __int64 v15; // r14
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  __int32 v18; // ecx
  unsigned __int64 v19; // rsi
  int v20; // ecx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // r15
  __int64 v23; // rax
  __int32 v24; // ecx
  int v25; // ecx
  __int64 v26; // rdx
  unsigned int v27; // edx
  unsigned __int64 v28; // rbx
  unsigned int v29; // r15d
  __int64 v30; // rax
  __int32 v31; // edx
  __int64 v32; // rbx
  unsigned int v33; // eax
  _QWORD *v34; // rdi
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rcx
  _QWORD *i; // rdx
  __int64 v40; // rax
  _QWORD *v41; // rax
  unsigned int v42; // [rsp+8h] [rbp-38h]

  v7 = a1[3].m128i_u64[0];
  v8 = *(_QWORD *)(a1->m128i_i64[0] + 104) - *(_QWORD *)(a1->m128i_i64[0] + 96);
  a1[3].m128i_i32[2] = 0;
  v9 = v8 >> 3;
  if ( (unsigned int)v9 > v7 << 6 )
  {
    v21 = a1[2].m128i_u64[1];
    v22 = (unsigned int)(v9 + 63) >> 6;
    if ( v22 < 2 * v7 )
      v22 = 2 * v7;
    v23 = (__int64)realloc(v21, 8 * v22, 8 * (int)v22, a4, a5, a6);
    if ( !v23 )
    {
      if ( 8 * v22 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v23 = 0;
      }
      else
      {
        v23 = sub_13A3880(1u);
      }
    }
    v24 = a1[3].m128i_i32[2];
    a1[2].m128i_i64[1] = v23;
    a1[3].m128i_i64[0] = v22;
    a5 = (unsigned int)(v24 + 63) >> 6;
    if ( a5 < v22 )
    {
      v42 = (unsigned int)(v24 + 63) >> 6;
      memset((void *)(v23 + 8LL * a5), 0, 8 * (v22 - a5));
      v24 = a1[3].m128i_i32[2];
      v23 = a1[2].m128i_i64[1];
      a5 = v42;
    }
    v25 = v24 & 0x3F;
    if ( v25 )
    {
      v7 = (unsigned int)v7;
      *(_QWORD *)(v23 + 8LL * (a5 - 1)) &= ~(-1LL << v25);
      v23 = a1[2].m128i_i64[1];
      v26 = a1[3].m128i_i64[0] - (unsigned int)v7;
      if ( !v26 )
        goto LABEL_43;
    }
    else
    {
      v7 = (unsigned int)v7;
      v26 = a1[3].m128i_i64[0] - (unsigned int)v7;
      if ( !v26 )
        goto LABEL_43;
    }
    memset((void *)(v23 + 8 * v7), 0, 8 * v26);
LABEL_43:
    v27 = a1[3].m128i_u32[2];
    v16 = v27;
    if ( (unsigned int)v9 <= v27 )
      goto LABEL_32;
    v28 = a1[3].m128i_u64[0];
    v29 = (v27 + 63) >> 6;
    v30 = v29;
    if ( v28 <= v29 || (v7 = v28 - v29) == 0 )
    {
LABEL_45:
      v16 = v27;
      if ( (v27 & 0x3F) != 0 )
      {
        *(_QWORD *)(a1[2].m128i_i64[1] + 8LL * (v29 - 1)) &= ~(-1LL << (v27 & 0x3F));
        v16 = a1[3].m128i_u32[2];
      }
      goto LABEL_32;
    }
LABEL_50:
    memset((void *)(a1[2].m128i_i64[1] + 8 * v30), 0, 8 * v7);
    v27 = a1[3].m128i_u32[2];
    goto LABEL_45;
  }
  if ( !(_DWORD)v9 )
    goto LABEL_3;
  v16 = 0;
  if ( v7 )
  {
    v30 = 0;
    v29 = 0;
    goto LABEL_50;
  }
LABEL_32:
  a1[3].m128i_i32[2] = v9;
  if ( (unsigned int)v9 < v16 )
  {
    v17 = a1[3].m128i_u64[0];
    LOBYTE(v18) = v9;
    v19 = (unsigned int)(v9 + 63) >> 6;
    if ( v17 > v19 )
    {
      v40 = v17 - v19;
      if ( v40 )
      {
        memset((void *)(a1[2].m128i_i64[1] + 8 * v19), 0, 8 * v40);
        v18 = a1[3].m128i_i32[2];
      }
    }
    v20 = v18 & 0x3F;
    if ( v20 )
      *(_QWORD *)(a1[2].m128i_i64[1] + 8LL * (((unsigned int)(v9 + 63) >> 6) - 1)) &= ~(-1LL << v20);
  }
LABEL_3:
  v10 = a1[5].m128i_i32[0];
  ++a1[4].m128i_i64[0];
  if ( !v10 && !a1[5].m128i_i32[1] )
    goto LABEL_16;
  v11 = (_QWORD *)a1[4].m128i_i64[1];
  v12 = &v11[7 * a1[5].m128i_u32[2]];
  v13 = 4 * v10;
  if ( (unsigned int)(4 * v10) < 0x40 )
    v13 = 64;
  if ( a1[5].m128i_i32[2] <= v13 )
  {
    while ( v11 != v12 )
    {
      if ( *v11 != -8 )
      {
        if ( *v11 != -16 )
        {
          _libc_free(v11[4]);
          _libc_free(v11[1]);
        }
        *v11 = -8;
      }
      v11 += 7;
    }
    goto LABEL_15;
  }
  do
  {
    if ( *v11 != -16 && *v11 != -8 )
    {
      _libc_free(v11[4]);
      _libc_free(v11[1]);
    }
    v11 += 7;
  }
  while ( v11 != v12 );
  v31 = a1[5].m128i_i32[2];
  if ( !v10 )
  {
    if ( v31 )
    {
      j___libc_free_0(a1[4].m128i_i64[1]);
      a1[4].m128i_i64[1] = 0;
      a1[5].m128i_i64[0] = 0;
      a1[5].m128i_i32[2] = 0;
      goto LABEL_16;
    }
LABEL_15:
    a1[5].m128i_i64[0] = 0;
    goto LABEL_16;
  }
  v32 = 64;
  if ( v10 != 1 )
  {
    _BitScanReverse(&v33, v10 - 1);
    v32 = (unsigned int)(1 << (33 - (v33 ^ 0x1F)));
    if ( (int)v32 < 64 )
      v32 = 64;
  }
  v34 = (_QWORD *)a1[4].m128i_i64[1];
  if ( (_DWORD)v32 == v31 )
  {
    a1[5].m128i_i64[0] = 0;
    v41 = &v34[7 * v32];
    do
    {
      if ( v34 )
        *v34 = -8;
      v34 += 7;
    }
    while ( v41 != v34 );
  }
  else
  {
    j___libc_free_0(v34);
    v35 = ((((((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v32 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v32 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 16;
    v36 = (v35
         | (((((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v32 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v32 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v32 / 3u + 1) | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v32 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v32 / 3u + 1) >> 1))
        + 1;
    a1[5].m128i_i32[2] = v36;
    v37 = (_QWORD *)sub_22077B0(56 * v36);
    v38 = a1[5].m128i_u32[2];
    a1[5].m128i_i64[0] = 0;
    a1[4].m128i_i64[1] = (__int64)v37;
    for ( i = &v37[7 * v38]; i != v37; v37 += 7 )
    {
      if ( v37 )
        *v37 = -8;
    }
  }
LABEL_16:
  result = a1[6].m128i_u32[2];
  if ( result <= (unsigned int)v9 )
  {
    if ( result >= (unsigned int)v9 )
      return result;
    if ( (unsigned int)v9 > (unsigned __int64)a1[6].m128i_u32[3] )
    {
      sub_16CD150((__int64)a1[6].m128i_i64, &a1[7], (unsigned int)v9, 16, a5, a6);
      result = a1[6].m128i_u32[2];
    }
    v15 = a1[6].m128i_i64[0] + 16LL * (unsigned int)v9;
    for ( result = a1[6].m128i_i64[0] + 16 * result; v15 != result; result += 16LL )
    {
      if ( result )
        *(__m128i *)result = _mm_loadu_si128(a1 + 7);
    }
  }
  a1[6].m128i_i32[2] = v9;
  return result;
}
