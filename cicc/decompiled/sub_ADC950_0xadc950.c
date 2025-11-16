// Function: sub_ADC950
// Address: 0xadc950
//
__int64 __fastcall sub_ADC950(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // r10d
  __int64 v4; // r12

  v3 = 0;
  v4 = *(_QWORD *)(a1 + 8);
  if ( a3 )
    v3 = sub_B9B140(*(_QWORD *)(a1 + 8), a2, a3);
  return sub_B04E90(v4, 59, v3, 0, 0, 0, 0, 0, 0, 1);
}
