// Function: sub_38AE740
// Address: 0x38ae740
//
__int64 __fastcall sub_38AE740(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v7; // r14
  unsigned int v8; // r12d
  _QWORD *v10; // r15
  __int64 *v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // rbx
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 *v15; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v16; // [rsp+20h] [rbp-60h] BYREF
  __int64 v17; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v18[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v15, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after insertelement value") )
    return 1;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v16, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after insertelement value") )
    return 1;
  v8 = sub_38AB270((__int64 **)a1, &v17, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
  {
    return 1;
  }
  else if ( (unsigned __int8)sub_15FA630((__int64)v15, v16, v17) )
  {
    v10 = v16;
    v11 = v15;
    v19 = 257;
    v14 = v17;
    v12 = sub_1648A60(56, 3u);
    v13 = v12;
    if ( v12 )
      sub_15FA480((__int64)v12, v11, (__int64)v10, v14, (__int64)v18, 0);
    *a2 = v13;
  }
  else
  {
    v19 = 259;
    v18[0] = "invalid insertelement operands";
    return (unsigned int)sub_38814C0(a1 + 8, v7, (__int64)v18);
  }
  return v8;
}
