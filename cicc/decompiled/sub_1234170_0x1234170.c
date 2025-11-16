// Function: sub_1234170
// Address: 0x1234170
//
__int64 __fastcall sub_1234170(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v4; // r13d
  __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // r14
  __int64 v9; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  unsigned int v12; // [rsp+Ch] [rbp-54h]
  __int64 v13; // [rsp+18h] [rbp-48h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v13 = 0;
  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 59, "expected 'from' after cleanupret") )
    return 1;
  v6 = sub_BCB190(*a1);
  if ( (unsigned __int8)sub_1224B80(a1, v6, &v13, a3) )
    return 1;
  v4 = sub_120AFE0((__int64)a1, 66, "expected 'unwind' in cleanupret");
  if ( (_BYTE)v4 )
    return 1;
  v7 = *((_DWORD *)a1 + 60) == 56;
  v14 = 0;
  if ( v7 )
  {
    *((_DWORD *)a1 + 60) = sub_1205200((__int64)(a1 + 22));
    if ( !(unsigned __int8)sub_120AFE0((__int64)a1, 57, "expected 'caller' in cleanupret") )
      goto LABEL_8;
    return 1;
  }
  v15[0] = 0;
  if ( (unsigned __int8)sub_122FEA0((__int64)a1, &v14, v15, a3) )
    return 1;
LABEL_8:
  v8 = v14;
  v9 = v13;
  v12 = (2 - (v14 == 0)) & 0x1FFFFFFF;
  v10 = sub_BD2C40(72, 2 - (unsigned int)(v14 == 0));
  v11 = v10;
  if ( v10 )
    sub_B4BF70((__int64)v10, v9, v8, v12, 0, 0);
  *a2 = v11;
  return v4;
}
