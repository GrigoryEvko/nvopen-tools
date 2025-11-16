// Function: sub_12342C0
// Address: 0x12342c0
//
__int64 __fastcall sub_12342C0(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v4; // r12d
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // r13
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  __int64 v12; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v11 = 0;
  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 59, "expected 'from' after catchret") )
    return 1;
  v6 = sub_BCB190(*a1);
  if ( (unsigned __int8)sub_1224B80(a1, v6, &v11, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0((__int64)a1, 56, "expected 'to' in catchret") )
    return 1;
  v13[0] = 0;
  v4 = sub_122FEA0((__int64)a1, &v12, v13, a3);
  if ( (_BYTE)v4 )
  {
    return 1;
  }
  else
  {
    v7 = v12;
    v8 = v11;
    v9 = sub_BD2C40(72, 2u);
    v10 = v9;
    if ( v9 )
      sub_B4C170((__int64)v9, v8, v7, 0, 0);
    *a2 = v10;
  }
  return v4;
}
