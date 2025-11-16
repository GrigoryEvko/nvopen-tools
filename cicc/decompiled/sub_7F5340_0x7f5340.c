// Function: sub_7F5340
// Address: 0x7f5340
//
__int64 __fastcall sub_7F5340(const __m128i *a1)
{
  __int64 v1; // r12

  v1 = (__int64)qword_4D03F78;
  if ( qword_4D03F78 )
    qword_4D03F78 = (_QWORD *)*qword_4D03F78;
  else
    v1 = sub_823970(48);
  *(_QWORD *)v1 = 0;
  *(_QWORD *)(v1 + 8) = 0;
  *(_QWORD *)(v1 + 16) = 0;
  *(_QWORD *)(v1 + 24) = 0;
  *(_QWORD *)(v1 + 32) = 0;
  *(_BYTE *)(v1 + 40) = 0;
  *(__m128i *)v1 = _mm_loadu_si128(a1);
  *(__m128i *)(v1 + 16) = _mm_loadu_si128(a1 + 1);
  *(__m128i *)(v1 + 32) = _mm_loadu_si128(a1 + 2);
  if ( a1->m128i_i64[0] )
    *(_QWORD *)v1 = sub_7F5340();
  return v1;
}
