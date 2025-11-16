// Function: sub_A564B0
// Address: 0xa564b0
//
const __m128i *__fastcall sub_A564B0(__int64 a1, __int64 a2)
{
  const __m128i *result; // rax

  result = sub_A56340(a1, a2);
  if ( result )
  {
    result = *(const __m128i **)(a1 + 32);
    if ( result != (const __m128i *)a2 )
    {
      if ( result )
        sub_A56010(*(_QWORD *)(a1 + 40));
      result = *(const __m128i **)(a1 + 40);
      result[1].m128i_i64[0] = a2;
      result[1].m128i_i8[8] = 0;
      *(_QWORD *)(a1 + 32) = a2;
    }
  }
  return result;
}
