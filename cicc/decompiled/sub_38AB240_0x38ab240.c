// Function: sub_38AB240
// Address: 0x38ab240
//
__int64 __fastcall sub_38AB240(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 result; // rax

  while ( *(_DWORD *)(a1 + 64) == 376 )
  {
    result = sub_38AA540(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    if ( (_BYTE)result )
      return result;
  }
  return 0;
}
