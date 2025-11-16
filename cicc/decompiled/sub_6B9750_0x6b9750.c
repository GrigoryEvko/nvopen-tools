// Function: sub_6B9750
// Address: 0x6b9750
//
__int64 __fastcall sub_6B9750(int a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // r12
  __int64 v5; // [rsp+8h] [rbp-228h] BYREF
  _BYTE v6[160]; // [rsp+10h] [rbp-220h] BYREF
  _QWORD v7[48]; // [rsp+B0h] [rbp-180h] BYREF

  sub_6E1DD0(&v5);
  sub_6E1E00(4, v6, 1, 0);
  sub_6E2170(v5);
  if ( a2 )
    sub_6E6610(a2, v7, 1);
  else
    sub_69ED20((__int64)v7, 0, 0, 0);
  sub_689050(v7, a1);
  v2 = sub_6F6F40(v7, 0);
  v3 = sub_6E2700(v2);
  sub_6E2B30(v2, 0);
  sub_6E1DF0(v5);
  unk_4F061D8 = *(_QWORD *)((char *)&v7[9] + 4);
  return v3;
}
