// Function: sub_1DBCC30
// Address: 0x1dbcc30
//
float __fastcall sub_1DBCC30(unsigned __int8 a1, unsigned __int8 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v6; // rax
  float v7; // xmm0_4
  float v8; // xmm2_4

  v5 = sub_1DDC3C0(a3, a4);
  v6 = sub_1DDC5F0(a3);
  if ( v5 < 0 )
  {
    v7 = (float)(v5 & 1 | (unsigned int)((unsigned __int64)v5 >> 1))
       + (float)(v5 & 1 | (unsigned int)((unsigned __int64)v5 >> 1));
    if ( v6 >= 0 )
      goto LABEL_3;
LABEL_6:
    v8 = (float)(v6 & 1 | (unsigned int)((unsigned __int64)v6 >> 1))
       + (float)(v6 & 1 | (unsigned int)((unsigned __int64)v6 >> 1));
    return (float)(v7 * (float)(1.0 / v8)) * (float)(a2 + a1);
  }
  v7 = (float)(int)v5;
  if ( v6 < 0 )
    goto LABEL_6;
LABEL_3:
  v8 = (float)(int)v6;
  return (float)(v7 * (float)(1.0 / v8)) * (float)(a2 + a1);
}
