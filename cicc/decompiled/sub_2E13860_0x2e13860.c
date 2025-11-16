// Function: sub_2E13860
// Address: 0x2e13860
//
float __fastcall sub_2E13860(unsigned __int8 a1, unsigned __int8 a2, __int64 a3, __int64 a4, __int64 a5)
{
  float result; // xmm0_4
  __int64 v8; // rax
  double v9; // r13
  __int64 v10; // rax
  double v11; // xmm2_8

  result = (float)(a2 + a1);
  if ( !a5 || !(unsigned __int8)sub_2EE6520(*(_QWORD *)(a4 + 32), a5, a3, 2) )
  {
    v8 = sub_2E39EA0(a3, a4);
    if ( v8 < 0 )
      v9 = (double)(int)(v8 & 1 | ((unsigned __int64)v8 >> 1)) + (double)(int)(v8 & 1 | ((unsigned __int64)v8 >> 1));
    else
      v9 = (double)(int)v8;
    v10 = sub_2E3A080(a3);
    if ( v10 < 0 )
      v11 = (double)(int)(v10 & 1 | ((unsigned __int64)v10 >> 1))
          + (double)(int)(v10 & 1 | ((unsigned __int64)v10 >> 1));
    else
      v11 = (double)(int)v10;
    return result * (v9 / v11);
  }
  return result;
}
