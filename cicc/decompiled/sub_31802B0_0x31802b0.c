// Function: sub_31802B0
// Address: 0x31802b0
//
__int64 __fastcall sub_31802B0(_QWORD *a1, const __m128i *a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rax
  const __m128i *v5; // rdi
  __m128i *v6; // rax
  __m128i *v7; // rcx
  __m128i *v8; // rdx
  __m128i *v9; // rax
  __m128i *v10; // rdx
  __int64 v11; // rax
  __m128i v12; // xmm0
  __int64 v14; // rax
  const __m128i *v15; // rdi
  __int64 v16; // rax
  __m128i *v17; // rax
  __m128i *v18; // rcx
  __m128i *v19; // rdx
  __m128i *v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rdx
  __int64 i; // rax
  __int64 v26; // rax

  v2 = a1[1];
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 8);
    a1[1] = v3;
    if ( v3 )
    {
      if ( v2 == *(_QWORD *)(v3 + 24) )
      {
        *(_QWORD *)(v3 + 24) = 0;
        v24 = *(_QWORD *)(a1[1] + 16LL);
        if ( v24 )
        {
          a1[1] = v24;
          for ( i = *(_QWORD *)(v24 + 24); i; i = *(_QWORD *)(i + 24) )
          {
            a1[1] = i;
            v24 = i;
          }
          v26 = *(_QWORD *)(v24 + 16);
          if ( v26 )
            a1[1] = v26;
        }
      }
      else
      {
        *(_QWORD *)(v3 + 16) = 0;
      }
    }
    else
    {
      *a1 = 0;
    }
    sub_317D930(*(_QWORD **)(v2 + 56));
    v4 = a2->m128i_i64[0];
    *(_DWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 56) = 0;
    *(_QWORD *)(v2 + 32) = v4;
    *(_QWORD *)(v2 + 64) = v2 + 48;
    *(_QWORD *)(v2 + 72) = v2 + 48;
    *(_QWORD *)(v2 + 80) = 0;
    v5 = (const __m128i *)a2[1].m128i_i64[1];
    if ( v5 )
    {
      v6 = sub_317D720(v5, v2 + 48);
      v7 = v6;
      do
      {
        v8 = v6;
        v6 = (__m128i *)v6[1].m128i_i64[0];
      }
      while ( v6 );
      *(_QWORD *)(v2 + 64) = v8;
      v9 = v7;
      do
      {
        v10 = v9;
        v9 = (__m128i *)v9[1].m128i_i64[1];
      }
      while ( v9 );
      *(_QWORD *)(v2 + 72) = v10;
      v11 = a2[3].m128i_i64[0];
      *(_QWORD *)(v2 + 56) = v7;
      *(_QWORD *)(v2 + 80) = v11;
    }
    v12 = _mm_loadu_si128(a2 + 4);
    *(_QWORD *)(v2 + 88) = a2[3].m128i_i64[1];
    *(__m128i *)(v2 + 96) = v12;
  }
  else
  {
    v14 = sub_22077B0(0x88u);
    v15 = (const __m128i *)a2[1].m128i_i64[1];
    v2 = v14;
    v16 = a2->m128i_i64[0];
    *(_DWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 32) = v16;
    *(_QWORD *)(v2 + 56) = 0;
    *(_QWORD *)(v2 + 64) = v2 + 48;
    *(_QWORD *)(v2 + 72) = v2 + 48;
    *(_QWORD *)(v2 + 80) = 0;
    if ( v15 )
    {
      v17 = sub_317D720(v15, v2 + 48);
      v18 = v17;
      do
      {
        v19 = v17;
        v17 = (__m128i *)v17[1].m128i_i64[0];
      }
      while ( v17 );
      *(_QWORD *)(v2 + 64) = v19;
      v20 = v18;
      do
      {
        v21 = v20;
        v20 = (__m128i *)v20[1].m128i_i64[1];
      }
      while ( v20 );
      v22 = a2[3].m128i_i64[0];
      *(_QWORD *)(v2 + 72) = v21;
      *(_QWORD *)(v2 + 56) = v18;
      *(_QWORD *)(v2 + 80) = v22;
    }
    v23 = _mm_loadu_si128(a2 + 4);
    *(_QWORD *)(v2 + 88) = a2[3].m128i_i64[1];
    *(__m128i *)(v2 + 96) = v23;
  }
  *(_QWORD *)(v2 + 112) = a2[5].m128i_i64[0];
  *(_QWORD *)(v2 + 120) = a2[5].m128i_i64[1];
  *(_QWORD *)(v2 + 128) = a2[6].m128i_i64[0];
  return v2;
}
