// Function: sub_204D450
// Address: 0x204d450
//
__int64 *__fastcall sub_204D450(
        __int64 *a1,
        unsigned int a2,
        const void **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9)
{
  __int128 v10; // [rsp-10h] [rbp-10h]

  *((_QWORD *)&v10 + 1) = a6;
  *(_QWORD *)&v10 = a5;
  return sub_1D359D0(a1, 104, a4, a2, a3, 0, a7, a8, a9, v10);
}
