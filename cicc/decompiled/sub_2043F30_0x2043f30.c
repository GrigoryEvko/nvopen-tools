// Function: sub_2043F30
// Address: 0x2043f30
//
__int64 __fastcall sub_2043F30(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int128 v9; // rax
  __int64 *v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r15
  __int64 v13; // r14
  __int128 v14; // rax
  __int128 v15; // rax

  *(_QWORD *)&v9 = sub_1D38BB0((__int64)a1, 0x7FFFFF, a4, 5, 0, 0, a5, a6, a7, 0);
  v10 = sub_1D332F0(a1, 118, a4, 5, 0, 0, *(double *)a5.m128i_i64, a6, a7, a2, a3, v9);
  v12 = v11;
  v13 = (__int64)v10;
  *(_QWORD *)&v14 = sub_1D38BB0((__int64)a1, 1065353216, a4, 5, 0, 0, a5, a6, a7, 0);
  *(_QWORD *)&v15 = sub_1D332F0(a1, 119, a4, 5, 0, 0, *(double *)a5.m128i_i64, a6, a7, v13, v12, v14);
  return sub_1D309E0(a1, 158, a4, 9, 0, 0, *(double *)a5.m128i_i64, a6, *(double *)a7.m128i_i64, v15);
}
