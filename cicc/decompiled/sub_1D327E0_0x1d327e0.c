// Function: sub_1D327E0
// Address: 0x1d327e0
//
__int64 __fastcall sub_1D327E0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned int a5,
        const void **a6,
        double a7,
        double a8,
        double a9)
{
  __int128 v10; // [rsp-10h] [rbp-10h]

  *((_QWORD *)&v10 + 1) = a3;
  *(_QWORD *)&v10 = a2;
  return sub_1D309E0(a1, 150, a4, a5, a6, 0, a7, a8, a9, v10);
}
