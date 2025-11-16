// Function: sub_18B4900
// Address: 0x18b4900
//
__int64 __fastcall sub_18B4900(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rbx
  __int64 result; // rax
  double v11; // xmm4_8
  double v12; // xmm5_8
  __int64 v13; // r12
  __int64 v14; // r13

  v9 = a1 + 168;
  result = sub_159C4F0(**(__int64 ***)a1);
  v13 = *(_QWORD *)(a1 + 184);
  if ( v13 != a1 + 168 )
  {
    v14 = result;
    do
    {
      while ( *(_DWORD *)(v13 + 40) )
      {
        result = sub_220EEE0(v13);
        v13 = result;
        if ( v9 == result )
          return result;
      }
      sub_164D160(*(_QWORD *)(v13 + 32), v14, a2, a3, a4, a5, v11, v12, a8, a9);
      sub_15F20C0(*(_QWORD **)(v13 + 32));
      result = sub_220EEE0(v13);
      v13 = result;
    }
    while ( v9 != result );
  }
  return result;
}
