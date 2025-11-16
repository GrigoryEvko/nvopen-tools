// Function: sub_26CAFB0
// Address: 0x26cafb0
//
bool __fastcall sub_26CAFB0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  _QWORD *v5; // r13
  float v6; // xmm1_4
  __int64 v7; // rax
  float v8; // xmm0_4
  float v9; // xmm0_4
  unsigned __int64 v10; // rax
  bool result; // al
  __int64 v12; // rdx
  _BYTE v13[16]; // [rsp+10h] [rbp-40h] BYREF
  float v14; // [rsp+20h] [rbp-30h]
  char v15; // [rsp+24h] [rbp-2Ch]

  v5 = sub_26CAE90(a1, a3);
  if ( v5 )
  {
    sub_3143F80(v13, a3, v4);
    if ( v15 )
      v6 = v14;
    else
      v6 = 1.0;
    v7 = sub_EF9210(v5);
    if ( v7 < 0 )
      v8 = (float)(v7 & 1 | (unsigned int)((unsigned __int64)v7 >> 1))
         + (float)(v7 & 1 | (unsigned int)((unsigned __int64)v7 >> 1));
    else
      v8 = (float)(int)v7;
    v9 = v8 * v6;
    if ( v9 >= 9.223372e18 )
      v10 = (unsigned int)(int)(float)(v9 - 9.223372e18) ^ 0x8000000000000000LL;
    else
      v10 = (unsigned int)(int)v9;
LABEL_8:
    *(_QWORD *)a2 = a3;
    *(_QWORD *)(a2 + 8) = v5;
    *(_QWORD *)(a2 + 16) = v10;
    result = 1;
    *(float *)(a2 + 24) = v6;
    return result;
  }
  result = sub_26C3E80((__int64)a1, a3);
  if ( result )
  {
    sub_3143F80(v13, a3, v12);
    if ( v15 )
      v6 = v14;
    else
      v6 = 1.0;
    v10 = 0;
    goto LABEL_8;
  }
  return result;
}
