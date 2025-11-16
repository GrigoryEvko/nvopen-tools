// Function: sub_24DAD90
// Address: 0x24dad90
//
void __fastcall sub_24DAD90(__int64 a1, const __m128i *a2)
{
  __int64 v3; // rsi

  v3 = *(_QWORD *)(a1 + 8);
  if ( v3 == *(_QWORD *)(a1 + 16) )
  {
    sub_24DABF0(a1, (_BYTE *)v3, a2);
  }
  else
  {
    if ( v3 )
    {
      *(__m128i *)v3 = _mm_loadu_si128(a2);
      *(_QWORD *)(v3 + 16) = a2[1].m128i_i64[0];
      v3 = *(_QWORD *)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v3 + 24;
  }
}
