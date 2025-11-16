// Function: sub_6B9B50
// Address: 0x6b9b50
//
void *__fastcall sub_6B9B50(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v6; // [rsp+8h] [rbp-218h] BYREF
  _BYTE v7[160]; // [rsp+10h] [rbp-210h] BYREF
  _QWORD v8[46]; // [rsp+B0h] [rbp-170h] BYREF

  sub_6E1DD0(&v6);
  sub_6E1E00(1, v7, 0, 0);
  sub_6E2170(v6);
  sub_69ED20((__int64)v8, 0, 0, 1);
  sub_688FA0(v8);
  sub_6F4950(v8, a1, v1, v2, v3, v4);
  sub_6E2AC0(a1);
  sub_6E2B30(a1, a1);
  sub_6E1DF0(v6);
  unk_4F061D8 = *(_QWORD *)((char *)&v8[9] + 4);
  return &unk_4F061D8;
}
