// Function: sub_1630830
// Address: 0x1630830
//
unsigned __int64 __fastcall sub_1630830(
        __int64 a1,
        unsigned int a2,
        unsigned __int8 *a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned __int64 result; // rax

  result = a2 - (unsigned __int64)*(unsigned int *)(a1 + 8);
  if ( a3 != *(unsigned __int8 **)(a1 + 8 * result) )
  {
    if ( *(_BYTE *)(a1 + 1) )
      return sub_1623D00(a1, a2, (__int64)a3);
    else
      return (unsigned __int64)sub_162FCD0(a1, a1 + 8 * result, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  }
  return result;
}
