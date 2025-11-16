// Function: sub_1233580
// Address: 0x1233580
//
__int64 __fastcall sub_1233580(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int64 v4; // r14
  unsigned int v5; // r12d
  _QWORD *v7; // rax
  __int64 v8; // r9
  _QWORD *v9; // rbx
  __int64 v10; // [rsp+8h] [rbp-68h] BYREF
  __int64 v11; // [rsp+10h] [rbp-60h] BYREF
  __int64 v12; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v13[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v14; // [rsp+40h] [rbp-30h]

  v4 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v10, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after shuffle mask") )
    return 1;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v11, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after shuffle value") )
    return 1;
  v5 = sub_122FE20((__int64 **)a1, &v12, a3);
  if ( (_BYTE)v5 )
    return 1;
  if ( !(unsigned __int8)sub_B4E200(v10, v11, v12) )
  {
    v14 = 259;
    v13[0] = "invalid shufflevector operands";
    sub_11FD800(a1 + 176, v4, (__int64)v13, 1);
    return 1;
  }
  v14 = 257;
  v7 = sub_BD2C40(112, unk_3F1FE60);
  v9 = v7;
  if ( v7 )
    sub_B4EBA0((__int64)v7, v10, v11, v12, (__int64)v13, v8, 0, 0);
  *a2 = v9;
  return v5;
}
