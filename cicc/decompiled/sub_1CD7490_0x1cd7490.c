// Function: sub_1CD7490
// Address: 0x1cd7490
//
unsigned __int64 __fastcall sub_1CD7490(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  unsigned __int64 result; // rax

  result = a1[1] - *a1;
  if ( result > 8 )
    return sub_1CD6ED0(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
  return result;
}
