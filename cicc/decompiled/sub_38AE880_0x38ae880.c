// Function: sub_38AE880
// Address: 0x38ae880
//
__int64 __fastcall sub_38AE880(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v7; // r14
  unsigned int v8; // r12d
  _QWORD *v10; // rax
  _QWORD *v11; // rbx
  _QWORD *v12; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v13; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v14; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v15[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v16; // [rsp+30h] [rbp-30h]

  v7 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v12, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after shuffle mask") )
    return 1;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v13, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after shuffle value") )
    return 1;
  v8 = sub_38AB270((__int64 **)a1, &v14, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
  {
    return 1;
  }
  else if ( (unsigned __int8)sub_15FA830((__int64)v12, v13, (__int64)v14) )
  {
    v16 = 257;
    v10 = sub_1648A60(56, 3u);
    v11 = v10;
    if ( v10 )
      sub_15FA660((__int64)v10, v12, (__int64)v13, v14, (__int64)v15, 0);
    *a2 = v11;
  }
  else
  {
    v16 = 259;
    v15[0] = "invalid shufflevector operands";
    return (unsigned int)sub_38814C0(a1 + 8, v7, (__int64)v15);
  }
  return v8;
}
