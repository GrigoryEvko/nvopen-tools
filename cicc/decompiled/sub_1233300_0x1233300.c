// Function: sub_1233300
// Address: 0x1233300
//
__int64 __fastcall sub_1233300(__int64 a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int64 v4; // r14
  unsigned int v5; // r12d
  __int64 v7; // r15
  __int64 v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  __int64 v12; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v13[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v14; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v11, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after extract value") )
    return 1;
  v5 = sub_122FE20((__int64 **)a1, &v12, a3);
  if ( (_BYTE)v5 )
    return 1;
  if ( !(unsigned __int8)sub_B4DF70(v11, v12) )
  {
    v14 = 259;
    v13[0] = "invalid extractelement operands";
    sub_11FD800(a1 + 176, v4, (__int64)v13, 1);
    return 1;
  }
  v7 = v12;
  v8 = v11;
  v14 = 257;
  v9 = sub_BD2C40(72, 2u);
  v10 = v9;
  if ( v9 )
    sub_B4DE80((__int64)v9, v8, v7, (__int64)v13, 0, 0);
  *a2 = v10;
  return v5;
}
