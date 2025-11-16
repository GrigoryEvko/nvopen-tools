// Function: sub_2EBEBD0
// Address: 0x2ebebd0
//
void __fastcall sub_2EBEBD0(__int64 a1, __m128i *a2, const __m128i *a3, __int64 a4)
{
  __int64 v5; // rdi
  __m128i **v6; // rax
  const __m128i *v7; // r9
  __int64 v8; // rax

  if ( a2 >= a3 && a2 < (const __m128i *)((char *)a3 + 40 * (unsigned int)a4) )
  {
    v5 = -40;
    a4 = (unsigned int)(a4 - 1);
    a2 = (__m128i *)((char *)a2 + 40 * a4);
    a3 = (const __m128i *)((char *)a3 + 40 * a4);
    goto LABEL_9;
  }
  v5 = 40;
  do
  {
    LODWORD(a4) = a4 - 1;
LABEL_9:
    if ( a2 )
    {
      *a2 = _mm_loadu_si128(a3);
      a2[1] = _mm_loadu_si128(a3 + 1);
      a2[2].m128i_i64[0] = a3[2].m128i_i64[0];
    }
    if ( !a3->m128i_i8[0] )
    {
      v8 = a3->m128i_u32[2];
      if ( (int)v8 >= 0 )
      {
        v6 = (__m128i **)(*(_QWORD *)(a1 + 304) + 8 * v8);
        v7 = (const __m128i *)a3[2].m128i_i64[0];
        if ( *v6 == a3 )
          goto LABEL_14;
      }
      else
      {
        v7 = (const __m128i *)a3[2].m128i_i64[0];
        v6 = (__m128i **)(*(_QWORD *)(a1 + 56) + 16 * (v8 & 0x7FFFFFFF) + 8);
        if ( *v6 == a3 )
        {
LABEL_14:
          *v6 = a2;
          if ( !v7 )
            goto LABEL_15;
          goto LABEL_6;
        }
      }
      *(_QWORD *)(a3[1].m128i_i64[1] + 32) = a2;
      if ( !v7 )
LABEL_15:
        v7 = *v6;
LABEL_6:
      v7[1].m128i_i64[1] = (__int64)a2;
    }
    a2 = (__m128i *)((char *)a2 + v5);
    a3 = (const __m128i *)((char *)a3 + v5);
  }
  while ( (_DWORD)a4 );
}
