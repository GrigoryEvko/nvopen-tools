// Function: sub_38AE640
// Address: 0x38ae640
//
__int64 __fastcall sub_38AE640(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v7; // r14
  unsigned int v8; // r12d
  __int64 v10; // r15
  _QWORD *v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v17; // [rsp+20h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v14, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after extract value") )
    return 1;
  v8 = sub_38AB270((__int64 **)a1, &v15, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
  {
    return 1;
  }
  else if ( sub_15FA460((__int64)v14, v15) )
  {
    v10 = v15;
    v11 = v14;
    v17 = 257;
    v12 = sub_1648A60(56, 2u);
    v13 = v12;
    if ( v12 )
      sub_15FA320((__int64)v12, v11, v10, (__int64)v16, 0);
    *a2 = v13;
  }
  else
  {
    v17 = 259;
    v16[0] = "invalid extractelement operands";
    return (unsigned int)sub_38814C0(a1 + 8, v7, (__int64)v16);
  }
  return v8;
}
