// Function: sub_386AFB0
// Address: 0x386afb0
//
char __fastcall sub_386AFB0(__int64 a1, __int64 *a2, __m128i a3, __m128i a4)
{
  char result; // al

  result = sub_386A280(a1, a2, a3, a4);
  if ( !result )
    return **(_QWORD **)(*(_QWORD *)(a1 + 56) + 32LL) == a2[5];
  return result;
}
