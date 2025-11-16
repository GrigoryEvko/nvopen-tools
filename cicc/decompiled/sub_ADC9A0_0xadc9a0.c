// Function: sub_ADC9A0
// Address: 0xadc9a0
//
__int64 __fastcall sub_ADC9A0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6, int a7)
{
  int v7; // r10d
  __int64 v11; // r14

  v7 = 0;
  v11 = *(_QWORD *)(a1 + 8);
  if ( a3 )
    v7 = sub_B9B140(*(_QWORD *)(a1 + 8), a2, a3);
  return sub_B04E90(v11, 36, v7, a4, 0, a5, a7, a6, 0, 1);
}
