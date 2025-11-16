// Function: sub_170E100
// Address: 0x170e100
//
__int64 __fastcall sub_170E100(
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
  __int64 v11; // r12
  __int64 v12; // rbx
  _QWORD *v14; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8

  v11 = *(_QWORD *)(a2 + 8);
  if ( v11 )
  {
    v12 = *a1;
    do
    {
      v14 = sub_1648700(v11);
      sub_170B990(v12, (__int64)v14);
      v11 = *(_QWORD *)(v11 + 8);
    }
    while ( v11 );
    if ( a2 == a3 )
      a3 = sub_1599EF0(*(__int64 ***)a2);
    v11 = a2;
    sub_164D160(a2, a3, a4, a5, a6, a7, v15, v16, a10, a11);
  }
  return v11;
}
