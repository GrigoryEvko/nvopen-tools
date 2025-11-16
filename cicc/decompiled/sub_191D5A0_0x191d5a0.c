// Function: sub_191D5A0
// Address: 0x191d5a0
//
__int64 __fastcall sub_191D5A0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  double v14; // xmm4_8
  double v15; // xmm5_8

  if ( (unsigned __int8)sub_1560180(*(_QWORD *)(a2[5] + 56) + 112LL, 42)
    || (unsigned __int8)sub_1560180(*(_QWORD *)(a2[5] + 56) + 112LL, 43) )
  {
    return 0;
  }
  else
  {
    return sub_191D150(a1, a2, a3, a4, a5, a6, v14, v15, a9, a10, v11, v12, v13);
  }
}
