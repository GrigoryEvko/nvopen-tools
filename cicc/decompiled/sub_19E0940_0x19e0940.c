// Function: sub_19E0940
// Address: 0x19e0940
//
_BOOL8 __fastcall sub_19E0940(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15)
{
  double v15; // xmm4_8
  double v16; // xmm5_8
  _BOOL4 v17; // eax
  _BOOL4 v18; // r13d

  *a1 = a3;
  a1[2] = a4;
  a1[3] = a5;
  a1[4] = a6;
  a1[5] = a15;
  a1[1] = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v17 = 0;
  do
  {
    v18 = v17;
    v17 = sub_19DF5F0((__int64)a1, a7, a8, a9, a10, v15, v16, a13, a14);
  }
  while ( v17 );
  return v18;
}
