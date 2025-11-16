// Function: sub_B92A60
// Address: 0xb92a60
//
__int64 __fastcall sub_B92A60(const __m128i *a1, __int64 m128i_i64, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int32 v6; // edx
  unsigned __int32 v7; // eax
  const __m128i *v8; // r12
  const __m128i *v9; // r14
  unsigned __int32 v10; // edx
  unsigned int v11; // eax
  __m128i *v12; // rax
  __int64 v13; // rdx
  __m128i *i; // rdx
  char *v15; // r12
  __int64 result; // rax
  char *v17; // r13
  unsigned __int8 *v18; // rdi
  const __m128i *v19; // rax
  __int64 v20; // r15
  const __m128i *v21; // rdx
  __m128i *v22; // rcx
  const __m128i *v23; // rax
  __m128i *v24; // r14
  __int64 v25; // r12
  __m128i *v26; // r15
  unsigned __int64 v27; // rax
  __m128i *v28; // r12
  __m128i *v29; // rdi
  unsigned __int32 v30; // edx
  bool v31; // zf
  __m128i *v32; // rax
  __int64 v33; // rdx
  __m128i *j; // rdx
  unsigned int v35; // edx
  unsigned int v36; // eax
  unsigned int v37; // r12d
  __int8 v38; // al
  __int64 v39; // rdi
  __int64 v40; // rax
  __m128i *v41; // [rsp+0h] [rbp-100h] BYREF
  __int64 v42; // [rsp+8h] [rbp-F8h]
  _BYTE v43[240]; // [rsp+10h] [rbp-F0h] BYREF

  v6 = a1[1].m128i_u32[2];
  v7 = v6 >> 1;
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
    v8 = a1 + 8;
    v9 = a1 + 2;
    if ( !v7 )
    {
LABEL_7:
      HIDWORD(v42) = 8;
      v41 = (__m128i *)v43;
      goto LABEL_8;
    }
  }
  else
  {
    v9 = (const __m128i *)a1[2].m128i_i64[0];
    v8 = (const __m128i *)((char *)v9 + 24 * a1[2].m128i_u32[2]);
    if ( !v7 )
      goto LABEL_7;
  }
  if ( v9 == v8 )
    goto LABEL_7;
  while ( v9->m128i_i64[0] == -4096 || v9->m128i_i64[0] == -8192 )
  {
    v9 = (const __m128i *)((char *)v9 + 24);
    if ( v9 == v8 )
      goto LABEL_7;
  }
  v41 = (__m128i *)v43;
  v42 = 0x800000000LL;
  if ( v9 == v8 )
  {
LABEL_8:
    ++a1[1].m128i_i64[0];
    v10 = v6 >> 1;
    LODWORD(v42) = 0;
    if ( !v10 )
      goto LABEL_48;
    goto LABEL_9;
  }
  v19 = v9;
  v20 = 0;
  while ( 1 )
  {
    v21 = (const __m128i *)((char *)v19 + 24);
    if ( &v19[1].m128i_u64[1] == (unsigned __int64 *)v8 )
      break;
    while ( 1 )
    {
      v19 = v21;
      if ( v21->m128i_i64[0] != -4096 && v21->m128i_i64[0] != -8192 )
        break;
      v21 = (const __m128i *)((char *)v21 + 24);
      if ( v8 == v21 )
        goto LABEL_34;
    }
    ++v20;
    if ( v21 == v8 )
      goto LABEL_35;
  }
LABEL_34:
  ++v20;
LABEL_35:
  v22 = (__m128i *)v43;
  if ( v20 > 8 )
  {
    m128i_i64 = (__int64)v43;
    sub_C8D5F0(&v41, v43, v20, 24);
    v22 = (__m128i *)((char *)v41 + 24 * (unsigned int)v42);
  }
  do
  {
    if ( v22 )
    {
      *v22 = _mm_loadu_si128(v9);
      v22[1].m128i_i64[0] = v9[1].m128i_i64[0];
    }
    v23 = (const __m128i *)((char *)v9 + 24);
    if ( &v9[1].m128i_u64[1] == (unsigned __int64 *)v8 )
      break;
    while ( 1 )
    {
      v9 = v23;
      if ( v23->m128i_i64[0] != -8192 && v23->m128i_i64[0] != -4096 )
        break;
      v23 = (const __m128i *)((char *)v23 + 24);
      if ( v8 == v23 )
        goto LABEL_43;
    }
    v22 = (__m128i *)((char *)v22 + 24);
  }
  while ( v23 != v8 );
LABEL_43:
  v24 = v41;
  LODWORD(v42) = v42 + v20;
  v25 = 24LL * (unsigned int)v42;
  v26 = (__m128i *)((char *)v41 + v25);
  if ( &v41->m128i_i8[v25] != (__int8 *)v41 )
  {
    _BitScanReverse64(&v27, 0xAAAAAAAAAAAAAAABLL * (v25 >> 3));
    sub_B8F370(v41->m128i_i8, (__m128i *)((char *)v41 + v25), 2LL * (int)(63 - (v27 ^ 0x3F)), (__int64)v22, a5);
    if ( (unsigned __int64)v25 <= 0x180 )
    {
      m128i_i64 = (__int64)v26;
      sub_B8F720(v24, v26);
    }
    else
    {
      v28 = v24 + 24;
      m128i_i64 = (__int64)v24[24].m128i_i64;
      sub_B8F720(v24, v24 + 24);
      if ( v26 != &v24[24] )
      {
        do
        {
          v29 = v28;
          v28 = (__m128i *)((char *)v28 + 24);
          sub_B8E150(v29);
        }
        while ( v26 != v28 );
      }
    }
  }
  v30 = a1[1].m128i_u32[2];
  ++a1[1].m128i_i64[0];
  v10 = v30 >> 1;
  if ( !v10 )
  {
LABEL_48:
    if ( !a1[1].m128i_i32[3] )
      goto LABEL_17;
    v11 = 0;
    if ( (a1[1].m128i_i8[8] & 1) != 0 )
      goto LABEL_50;
    goto LABEL_11;
  }
LABEL_9:
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
LABEL_50:
    v12 = (__m128i *)&a1[2];
    v13 = 96;
    goto LABEL_14;
  }
  v11 = 4 * v10;
LABEL_11:
  m128i_i64 = a1[2].m128i_u32[2];
  if ( (unsigned int)m128i_i64 <= v11 || (unsigned int)m128i_i64 <= 0x40 )
  {
    v12 = (__m128i *)a1[2].m128i_i64[0];
    v13 = 24 * m128i_i64;
LABEL_14:
    for ( i = (__m128i *)((char *)v12 + v13); i != v12; v12 = (__m128i *)((char *)v12 + 24) )
      v12->m128i_i64[0] = -4096;
    a1[1].m128i_i64[1] &= 1uLL;
    goto LABEL_17;
  }
  if ( v10 && (v35 = v10 - 1) != 0 )
  {
    _BitScanReverse(&v36, v35);
    v37 = 1 << (33 - (v36 ^ 0x1F));
    if ( v37 - 5 > 0x3A )
    {
      if ( (_DWORD)m128i_i64 == v37 )
      {
        sub_B92A00((__int64)a1[1].m128i_i64);
        goto LABEL_17;
      }
      m128i_i64 *= 24;
      sub_C7D6A0(a1[2].m128i_i64[0], m128i_i64, 8);
      v38 = a1[1].m128i_i8[8] | 1;
      a1[1].m128i_i8[8] = v38;
      if ( v37 <= 4 )
        goto LABEL_62;
      v39 = 24LL * v37;
    }
    else
    {
      v37 = 64;
      sub_C7D6A0(a1[2].m128i_i64[0], 24 * m128i_i64, 8);
      v38 = a1[1].m128i_i8[8];
      v39 = 1536;
    }
    m128i_i64 = 8;
    a1[1].m128i_i8[8] = v38 & 0xFE;
    v40 = sub_C7D670(v39, 8);
    a1[2].m128i_i32[2] = v37;
    a1[2].m128i_i64[0] = v40;
  }
  else
  {
    m128i_i64 *= 24;
    sub_C7D6A0(a1[2].m128i_i64[0], m128i_i64, 8);
    a1[1].m128i_i8[8] |= 1u;
  }
LABEL_62:
  v31 = (a1[1].m128i_i64[1] & 1) == 0;
  a1[1].m128i_i64[1] &= 1uLL;
  if ( v31 )
  {
    v32 = (__m128i *)a1[2].m128i_i64[0];
    v33 = 24LL * a1[2].m128i_u32[2];
  }
  else
  {
    v32 = (__m128i *)&a1[2];
    v33 = 96;
  }
  for ( j = (__m128i *)((char *)v32 + v33); j != v32; v32 = (__m128i *)((char *)v32 + 24) )
  {
    if ( v32 )
      v32->m128i_i64[0] = -4096;
  }
LABEL_17:
  v15 = (char *)v41;
  result = 3LL * (unsigned int)v42;
  v17 = &v41->m128i_i8[24 * (unsigned int)v42];
  if ( v41 != (__m128i *)v17 )
  {
    do
    {
      while ( 1 )
      {
        result = *((_QWORD *)v15 + 1);
        v18 = (unsigned __int8 *)(result & 0xFFFFFFFFFFFFFFFCLL);
        if ( (result & 0xFFFFFFFFFFFFFFFCLL) != 0 )
        {
          result &= 3u;
          if ( result == 1 )
          {
            result = (unsigned int)*v18 - 5;
            if ( (unsigned __int8)(*v18 - 5) <= 0x1Fu )
            {
              if ( (v18[1] & 0x7F) == 2 )
                break;
              result = *((unsigned int *)v18 - 2);
              if ( (_DWORD)result )
                break;
            }
          }
        }
        v15 += 24;
        if ( v17 == v15 )
          goto LABEL_24;
      }
      result = sub_B93250();
      v15 += 24;
    }
    while ( v17 != v15 );
LABEL_24:
    v17 = (char *)v41;
  }
  if ( v17 != v43 )
    return _libc_free(v17, m128i_i64);
  return result;
}
