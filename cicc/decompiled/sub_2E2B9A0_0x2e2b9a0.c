// Function: sub_2E2B9A0
// Address: 0x2e2b9a0
//
__int64 *__fastcall sub_2E2B9A0(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 *result; // rax

  v8 = sub_2E29D60(a1, a2, a3, a4, a5, a6);
  v9 = *(__int64 **)(v8 + 40);
  for ( result = *(__int64 **)(v8 + 32); result != v9; ++result )
  {
    while ( a3 != *result )
    {
      if ( ++result == v9 )
        return result;
    }
    *result = a4;
  }
  return result;
}
