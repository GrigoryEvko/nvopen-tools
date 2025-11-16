// Function: sub_BC3E80
// Address: 0xbc3e80
//
void __fastcall sub_BC3E80(__int64 a1, __m128i *a2)
{
  const __m128i *v2; // rbx
  __int64 v3; // r13
  __int64 v5; // rax
  const __m128i *v6; // r13
  const __m128i *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 *v10; // rdi

  v2 = *(const __m128i **)a1;
  v3 = *(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        a2[1].m128i_i64[1] = 0;
        v5 = v2[1].m128i_i64[1];
        a2[1].m128i_i64[1] = v5;
        if ( (v2[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (v5 & 2) != 0 && (v5 & 4) != 0 )
          {
            (*(void (__fastcall **)(__m128i *, const __m128i *))((v5 & 0xFFFFFFFFFFFFFFF8LL) + 8))(a2, v2);
            (*(void (__fastcall **)(const __m128i *))((a2[1].m128i_i64[1] & 0xFFFFFFFFFFFFFFF8LL) + 16))(v2);
          }
          else
          {
            *a2 = _mm_loadu_si128(v2);
            a2[1].m128i_i64[0] = v2[1].m128i_i64[0];
          }
          v2[1].m128i_i64[1] = 0;
        }
      }
      v2 += 2;
      a2 += 2;
    }
    while ( (const __m128i *)v3 != v2 );
    v6 = *(const __m128i **)a1;
    v7 = (const __m128i *)(*(_QWORD *)a1 + 32LL * *(unsigned int *)(a1 + 8));
    if ( *(const __m128i **)a1 != v7 )
    {
      do
      {
        v8 = v7[-1].m128i_i64[1];
        v7 -= 2;
        if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v9 = (v8 >> 1) & 1;
          if ( (v8 & 4) != 0 )
          {
            v10 = (__int64 *)v7;
            if ( !(_BYTE)v9 )
              v10 = (__int64 *)v7->m128i_i64[0];
            (*(void (__fastcall **)(__int64 *))((v8 & 0xFFFFFFFFFFFFFFF8LL) + 16))(v10);
          }
          if ( !(_BYTE)v9 )
            sub_C7D6A0(v7->m128i_i64[0], v7->m128i_i64[1], v7[1].m128i_i64[0]);
        }
      }
      while ( v7 != v6 );
    }
  }
}
