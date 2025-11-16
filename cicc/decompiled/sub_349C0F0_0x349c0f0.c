// Function: sub_349C0F0
// Address: 0x349c0f0
//
unsigned __int64 __fastcall sub_349C0F0(
        const __m128i *a1,
        const __m128i *a2,
        __m128i *a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        __int64 a7)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // r8
  const __m128i *v11; // rcx
  const __m128i *v12; // rax
  __m128i v13; // xmm0
  signed __int64 v14; // r10
  __m128i v15; // xmm1
  const __m128i *v16; // rax
  signed __int64 v17; // rdx
  __m128i v18; // xmm2
  __int64 v19; // rdx
  __int64 v21; // rbx
  __int64 v22; // r12
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // r8
  const __m128i *v25; // rcx
  const __m128i *v26; // rax
  __m128i v27; // xmm3
  signed __int64 v28; // r10
  __m128i v29; // xmm4
  __m128i *v30; // rax
  signed __int64 v31; // rcx
  __m128i v32; // xmm5
  __int64 v33; // rdi

  if ( a4 > a5 && a5 <= a7 )
  {
    if ( !a5 )
      return (unsigned __int64)a1;
    v7 = (char *)a3 - (char *)a2;
    v8 = (char *)a2 - (char *)a1;
    v9 = 0xAAAAAAAAAAAAAAABLL * (((char *)a3 - (char *)a2) >> 3);
    v10 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
    if ( (char *)a3 - (char *)a2 <= 0 )
    {
      if ( v8 <= 0 )
        return (unsigned __int64)a1;
      v14 = 0;
      v7 = 0;
    }
    else
    {
      v11 = a6;
      v12 = a2;
      do
      {
        v13 = _mm_loadu_si128(v12);
        v11 = (const __m128i *)((char *)v11 + 24);
        v12 = (const __m128i *)((char *)v12 + 24);
        *(__m128i *)((char *)v11 - 24) = v13;
        v11[-1].m128i_i32[2] = v12[-1].m128i_i32[2];
        --v9;
      }
      while ( v9 );
      if ( v7 <= 0 )
        v7 = 24;
      v14 = 0xAAAAAAAAAAAAAAABLL * (v7 >> 3);
      if ( v8 <= 0 )
      {
LABEL_11:
        if ( v7 > 0 )
        {
          v16 = a1;
          v17 = v14;
          do
          {
            v18 = _mm_loadu_si128(a6);
            v16 = (const __m128i *)((char *)v16 + 24);
            a6 = (const __m128i *)((char *)a6 + 24);
            *(__m128i *)((char *)v16 - 24) = v18;
            v16[-1].m128i_i32[2] = a6[-1].m128i_i32[2];
            --v17;
          }
          while ( v17 );
          v19 = 24;
          if ( v14 > 0 )
            v19 = 24 * v14;
          return (unsigned __int64)a1 + v19;
        }
        return (unsigned __int64)a1;
      }
    }
    do
    {
      v15 = _mm_loadu_si128((const __m128i *)((char *)a2 - 24));
      a2 = (const __m128i *)((char *)a2 - 24);
      a3 = (__m128i *)((char *)a3 - 24);
      *a3 = v15;
      a3[1].m128i_i32[0] = a2[1].m128i_i32[0];
      --v10;
    }
    while ( v10 );
    goto LABEL_11;
  }
  if ( a4 > a7 )
    return sub_343F700(a1, a2, a3);
  if ( !a4 )
    return (unsigned __int64)a3;
  v21 = (char *)a2 - (char *)a1;
  v22 = (char *)a3 - (char *)a2;
  v23 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  v24 = 0xAAAAAAAAAAAAAAABLL * (((char *)a3 - (char *)a2) >> 3);
  if ( (char *)a2 - (char *)a1 <= 0 )
  {
    if ( v22 <= 0 )
      return (unsigned __int64)a3;
    v28 = 0;
    v21 = 0;
    goto LABEL_25;
  }
  v25 = a6;
  v26 = a1;
  do
  {
    v27 = _mm_loadu_si128(v26);
    v25 = (const __m128i *)((char *)v25 + 24);
    v26 = (const __m128i *)((char *)v26 + 24);
    *(__m128i *)((char *)v25 - 24) = v27;
    v25[-1].m128i_i32[2] = v26[-1].m128i_i32[2];
    --v23;
  }
  while ( v23 );
  if ( v21 <= 0 )
    v21 = 24;
  a6 = (const __m128i *)((char *)a6 + v21);
  v28 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 3);
  if ( v22 > 0 )
  {
    do
    {
LABEL_25:
      v29 = _mm_loadu_si128(a2);
      a1 = (const __m128i *)((char *)a1 + 24);
      a2 = (const __m128i *)((char *)a2 + 24);
      *(__m128i *)((char *)a1 - 24) = v29;
      a1[-1].m128i_i32[2] = a2[-1].m128i_i32[2];
      --v24;
    }
    while ( v24 );
  }
  if ( v21 <= 0 )
    return (unsigned __int64)a3;
  v30 = a3;
  v31 = v28;
  do
  {
    v32 = _mm_loadu_si128((const __m128i *)((char *)a6 - 24));
    a6 = (const __m128i *)((char *)a6 - 24);
    v30 = (__m128i *)((char *)v30 - 24);
    *v30 = v32;
    v30[1].m128i_i32[0] = a6[1].m128i_i32[0];
    --v31;
  }
  while ( v31 );
  v33 = -24 * v28;
  if ( v28 <= 0 )
    v33 = -24;
  return (unsigned __int64)a3 + v33;
}
