// Function: sub_38AE9B0
// Address: 0x38ae9b0
//
__int64 __fastcall sub_38AE9B0(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned int v7; // r13d
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r15
  unsigned int v12; // r14d
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // [rsp+8h] [rbp-58h]
  __int64 v16; // [rsp+18h] [rbp-48h] BYREF
  __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  unsigned __int64 v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v16 = 0;
  if ( (unsigned __int8)sub_388AF10((__int64)a1, 56, "expected 'from' after cleanupret") )
    return 1;
  v9 = sub_16432D0(*a1);
  if ( (unsigned __int8)sub_38A1070(a1, v9, &v16, a3, a4, a5, a6) )
    return 1;
  v7 = sub_388AF10((__int64)a1, 63, "expected 'unwind' in cleanupret");
  if ( (_BYTE)v7 )
    return 1;
  v10 = *((_DWORD *)a1 + 16) == 53;
  v17 = 0;
  if ( v10 )
  {
    *((_DWORD *)a1 + 16) = sub_3887100((__int64)(a1 + 1));
    if ( !(unsigned __int8)sub_388AF10((__int64)a1, 54, "expected 'caller' in cleanupret") )
      goto LABEL_8;
    return 1;
  }
  v18[0] = 0;
  if ( (unsigned __int8)sub_38AB2F0((__int64)a1, &v17, v18, a3, a4, a5, a6) )
    return 1;
LABEL_8:
  v11 = v17;
  v15 = v16;
  v12 = 1 - ((v17 == 0) - 1);
  v13 = sub_1648A60(56, v12);
  v14 = v13;
  if ( v13 )
    sub_15F76D0((__int64)v13, v15, v11, v12, 0);
  *a2 = v14;
  return v7;
}
