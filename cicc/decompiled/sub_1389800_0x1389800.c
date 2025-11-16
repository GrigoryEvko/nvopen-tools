// Function: sub_1389800
// Address: 0x1389800
//
__m128i *__fastcall sub_1389800(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  __m128i *result; // rax

  result = *(__m128i **)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
  {
    result = *(__m128i **)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 15 )
    {
      result = (__m128i *)sub_1389430(a1, a2, 0);
      if ( a2 != a3 )
        return sub_1389510(a1, a2, a3, a4);
    }
  }
  return result;
}
