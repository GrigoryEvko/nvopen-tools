// Function: sub_2FA1270
// Address: 0x2fa1270
//
void __fastcall sub_2FA1270(__int64 a1, __int64 *a2)
{
  __m128i *v2; // r12
  __int64 v3; // r13
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r11
  __int64 v15; // r10
  __m128i *v16; // r12
  unsigned __int64 *v17; // rbx

  v2 = *(__m128i **)a1;
  v3 = *(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v3 )
  {
    do
    {
      if ( a2 )
      {
        *a2 = 0;
        a2[1] = 0;
        a2[2] = 0;
        a2[3] = 0;
        a2[4] = 0;
        a2[5] = 0;
        a2[6] = 0;
        a2[7] = 0;
        a2[8] = 0;
        a2[9] = 0;
        sub_2785050(a2, 0);
        if ( v2->m128i_i64[0] )
        {
          v6 = a2[3];
          v7 = a2[4];
          a2[3] = 0;
          v8 = a2[5];
          v9 = a2[6];
          a2[4] = 0;
          v10 = a2[7];
          v11 = a2[8];
          a2[5] = 0;
          v12 = a2[9];
          v13 = *a2;
          a2[6] = 0;
          *a2 = 0;
          v14 = a2[1];
          a2[7] = 0;
          v15 = a2[2];
          a2[1] = 0;
          a2[2] = 0;
          a2[8] = 0;
          a2[9] = 0;
          *(__m128i *)a2 = _mm_loadu_si128(v2);
          *((__m128i *)a2 + 1) = _mm_loadu_si128(v2 + 1);
          *((__m128i *)a2 + 2) = _mm_loadu_si128(v2 + 2);
          *((__m128i *)a2 + 3) = _mm_loadu_si128(v2 + 3);
          *((__m128i *)a2 + 4) = _mm_loadu_si128(v2 + 4);
          v2->m128i_i64[0] = v13;
          v2->m128i_i64[1] = v14;
          v2[1].m128i_i64[0] = v15;
          v2[1].m128i_i64[1] = v6;
          v2[2].m128i_i64[0] = v7;
          v2[2].m128i_i64[1] = v8;
          v2[3].m128i_i64[0] = v9;
          v2[3].m128i_i64[1] = v10;
          v2[4].m128i_i64[0] = v11;
          v2[4].m128i_i64[1] = v12;
        }
      }
      v2 += 5;
      a2 += 10;
    }
    while ( (__m128i *)v3 != v2 );
    v16 = *(__m128i **)a1;
    v17 = (unsigned __int64 *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
    if ( v17 != *(unsigned __int64 **)a1 )
    {
      do
      {
        v17 -= 10;
        sub_2784FD0(v17);
      }
      while ( v17 != (unsigned __int64 *)v16 );
    }
  }
}
