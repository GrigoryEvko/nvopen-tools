// Function: sub_1FEB510
// Address: 0x1feb510
//
__int64 *__fastcall sub_1FEB510(
        double a1,
        double a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 *a10)
{
  __int128 v11; // [rsp-10h] [rbp-10h]

  *((_QWORD *)&v11 + 1) = a9;
  *(_QWORD *)&v11 = a8;
  return sub_1D332F0(a10, 189, a5, 1, 0, 0, a1, a2, a3, a6, a7, v11);
}
