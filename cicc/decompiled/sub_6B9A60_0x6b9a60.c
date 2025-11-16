// Function: sub_6B9A60
// Address: 0x6b9a60
//
__int64 __fastcall sub_6B9A60(int a1, __int64 a2, int a3)
{
  int v3; // r12d
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-228h] BYREF
  _BYTE v11[160]; // [rsp+10h] [rbp-220h] BYREF
  _QWORD v12[48]; // [rsp+B0h] [rbp-180h] BYREF

  v3 = a2;
  sub_6E1DD0(&v10);
  sub_6E1E00(4, v11, 0, 0);
  sub_6E2170(v10);
  sub_69ED20((__int64)v12, 0, 0, 0);
  if ( !a2 || v12[0] != a2 && !(unsigned int)sub_8D97D0(a2, v12[0], 0, v5, v6) )
    v3 = a1;
  sub_843C40((unsigned int)v12, v3, 0, 0, 0, 0, a3);
  v7 = sub_6F6F40(v12, 0);
  v8 = sub_6E2700(v7);
  sub_6E2B30(v7, 0);
  sub_6E1DF0(v10);
  unk_4F061D8 = *(_QWORD *)((char *)&v12[9] + 4);
  return v8;
}
