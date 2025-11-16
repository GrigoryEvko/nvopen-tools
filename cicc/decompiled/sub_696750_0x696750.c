// Function: sub_696750
// Address: 0x696750
//
__int64 __fastcall sub_696750(_QWORD *a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v6; // [rsp+8h] [rbp-228h] BYREF
  _BYTE v7[160]; // [rsp+10h] [rbp-220h] BYREF
  _QWORD v8[48]; // [rsp+B0h] [rbp-180h] BYREF

  sub_6E1DD0(&v6);
  sub_6E1E00(4, v7, 0, 0);
  v3 = sub_6E50B0(*a1, a1 + 8);
  sub_6F8E70(a1, a1 + 8, a1[9] + 8LL, v8, v3);
  sub_6F69D0(v8, 0);
  if ( a2 )
    sub_689050(v8, 1);
  else
    sub_688FA0(v8);
  v4 = sub_6F6F40(v8, 0);
  sub_6E2B30(v8, 0);
  sub_6E1DF0(v6);
  return v4;
}
