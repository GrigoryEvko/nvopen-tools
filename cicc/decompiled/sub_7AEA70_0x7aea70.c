// Function: sub_7AEA70
// Address: 0x7aea70
//
void __fastcall sub_7AEA70(const __m128i *a1)
{
  __m128i *v1; // rax
  __int64 v2; // rax
  __int64 v3; // rsi
  char v4; // cl
  __int64 v5; // rdx

  v1 = (__m128i *)qword_4F08538;
  if ( qword_4F08538 )
  {
    while ( (const __m128i *)v1[1].m128i_i64[1] != a1 )
    {
      v1 = (__m128i *)v1->m128i_i64[0];
      if ( !v1 )
        goto LABEL_6;
    }
    v1[4].m128i_i8[4] = 1;
    v1[2] = _mm_loadu_si128(a1);
    v1[3] = _mm_loadu_si128(a1 + 1);
  }
  else
  {
LABEL_6:
    v2 = a1->m128i_i64[1];
    v3 = qword_4F08558;
    if ( v2 )
    {
      while ( 1 )
      {
        v4 = *(_BYTE *)(v2 + 26);
        v5 = *(_QWORD *)v2;
        if ( v4 == 2 )
        {
          *(_QWORD *)(*(_QWORD *)(v2 + 48) + 120LL) = qword_4F08550;
          qword_4F08550 = *(_QWORD *)(v2 + 48);
        }
        else if ( v4 == 8 )
        {
          *(_QWORD *)(*(_QWORD *)(v2 + 48) + 120LL) = qword_4F08550;
          *(_QWORD *)(*(_QWORD *)(v2 + 56) + 120LL) = *(_QWORD *)(v2 + 48);
          qword_4F08550 = *(_QWORD *)(v2 + 56);
        }
        *(_QWORD *)v2 = v3;
        v3 = v2;
        qword_4F08558 = v2;
        if ( !v5 )
          break;
        v2 = v5;
      }
    }
    sub_7ADF70((__int64)a1, a1[1].m128i_i8[8]);
  }
}
