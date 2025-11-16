// Function: sub_147DD40
// Address: 0x147dd40
//
__int64 *__fastcall sub_147DD40(__int64 a1, __int64 *a2, unsigned int a3, unsigned int a4, __m128i a5, __m128i a6)
{
  if ( *((_DWORD *)a2 + 2) == 1 )
    return *(__int64 **)*a2;
  else
    return sub_147C070(a1, a2, a3, a4, a5, a6);
}
