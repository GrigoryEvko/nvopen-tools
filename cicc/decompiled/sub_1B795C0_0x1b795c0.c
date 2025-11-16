// Function: sub_1B795C0
// Address: 0x1b795c0
//
__int64 __fastcall sub_1B795C0(
        _DWORD **a1,
        int a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _DWORD *v10; // rdi

  v10 = *a1;
  *v10 |= a2;
  return sub_1B78FF0((__int64)v10, a3, a4, a5, a6, a7, a8, a9, a10);
}
