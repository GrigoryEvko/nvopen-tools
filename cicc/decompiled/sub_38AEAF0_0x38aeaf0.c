// Function: sub_38AEAF0
// Address: 0x38aeaf0
//
__int64 __fastcall sub_38AEAF0(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned int v7; // r12d
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r14
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v14 = 0;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 56, "expected 'from' after catchret") )
    return 1;
  v9 = sub_16432D0(*a1);
  if ( (unsigned __int8)sub_38A1070(a1, v9, &v14, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 53, "expected 'to' in catchret") )
    return 1;
  v16[0] = 0;
  v7 = sub_38AB2F0((__int64)a1, &v15, v16, a3, a4, a5, a6);
  if ( (_BYTE)v7 )
  {
    return 1;
  }
  else
  {
    v10 = v15;
    v11 = v14;
    v12 = sub_1648A60(56, 2u);
    v13 = v12;
    if ( v12 )
      sub_15F7960((__int64)v12, v11, v10, 0);
    *a2 = v13;
  }
  return v7;
}
