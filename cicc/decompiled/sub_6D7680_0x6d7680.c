// Function: sub_6D7680
// Address: 0x6d7680
//
__int64 __fastcall sub_6D7680(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v2; // r12
  __int64 v4; // [rsp+8h] [rbp-218h] BYREF
  _BYTE v5[160]; // [rsp+10h] [rbp-210h] BYREF
  _QWORD v6[46]; // [rsp+B0h] [rbp-170h] BYREF

  sub_6E1DD0(&v4);
  sub_6E1E00(4, v5, 1, 0);
  sub_6E2170(v4);
  if ( a1 )
    sub_6E6610(a1, v6, 1);
  else
    sub_69ED20((__int64)v6, 0, 0, 0);
  sub_688FA0(v6);
  v1 = sub_6F6F40(v6, 0);
  v2 = sub_6E2700(v1);
  sub_6E2B30(v1, 0);
  sub_6E1DF0(v4);
  *(_QWORD *)&dword_4F061D8 = *(_QWORD *)((char *)&v6[9] + 4);
  return v2;
}
