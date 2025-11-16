// Function: sub_1389080
// Address: 0x1389080
//
__m128i *__fastcall sub_1389080(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __m128i *result; // rax

  result = *(__m128i **)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
  {
    result = *(__m128i **)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 15 )
    {
      sub_1389430(a1, a2, 0);
      sub_1389430(a1, a3, 0);
      if ( a4 )
      {
        return sub_1384C40(a1, a2, a3);
      }
      else
      {
        sub_13848E0(*(_QWORD *)(a1 + 24), a3, 1u, 0);
        return sub_1384420(*(_QWORD *)(a1 + 24), a2, 0, a3, 1, 0);
      }
    }
  }
  return result;
}
