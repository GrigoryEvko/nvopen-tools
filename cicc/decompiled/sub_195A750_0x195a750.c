// Function: sub_195A750
// Address: 0x195a750
//
__int64 __fastcall sub_195A750(
        __int64 a1,
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
  unsigned __int64 v13; // rax
  double v14; // xmm4_8
  double v15; // xmm5_8

  if ( sub_1377F70(a1 + 56, a2) )
    return 0;
  v13 = sub_157EBA0(a2);
  if ( *(_DWORD *)(a1 + 256) < (unsigned int)sub_1951E40(a2, v13, *(_DWORD *)(a1 + 256)) )
    return 0;
  else
    return sub_19596B0((__int64 *)a1, a2, a3, a4, a5, a6, a7, v14, v15, a10, a11);
}
