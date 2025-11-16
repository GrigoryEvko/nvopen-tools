// Function: sub_1233410
// Address: 0x1233410
//
__int64 __fastcall sub_1233410(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int64 v4; // r14
  unsigned int v5; // r12d
  _QWORD *v7; // rax
  _QWORD *v8; // rbx
  __int64 v9; // [rsp+8h] [rbp-98h]
  __int64 v10; // [rsp+10h] [rbp-90h]
  __int64 v11; // [rsp+18h] [rbp-88h]
  __int64 v12; // [rsp+28h] [rbp-78h] BYREF
  __int64 v13; // [rsp+30h] [rbp-70h] BYREF
  __int64 v14; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v15[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v16; // [rsp+60h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v12, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after insertelement value") )
    return 1;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v13, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after insertelement value") )
    return 1;
  v5 = sub_122FE20((__int64 **)a1, &v14, a3);
  if ( (_BYTE)v5 )
    return 1;
  if ( !(unsigned __int8)sub_B4E100(v12, v13, v14) )
  {
    v16 = 259;
    v15[0] = "invalid insertelement operands";
    sub_11FD800(a1 + 176, v4, (__int64)v15, 1);
    return 1;
  }
  v16 = 257;
  v10 = v13;
  v9 = v14;
  v11 = v12;
  v7 = sub_BD2C40(72, 3u);
  v8 = v7;
  if ( v7 )
    sub_B4DFA0((__int64)v7, v11, v10, v9, (__int64)v15, v11, 0, 0);
  *a2 = v8;
  return v5;
}
