// Function: sub_6B9C00
// Address: 0x6b9c00
//
void *__fastcall sub_6B9C00(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v6; // [rsp+8h] [rbp-218h] BYREF
  _BYTE v7[160]; // [rsp+10h] [rbp-210h] BYREF
  _BYTE v8[76]; // [rsp+B0h] [rbp-170h] BYREF
  __int64 v9; // [rsp+FCh] [rbp-124h]

  sub_6E1DD0(&v6);
  sub_6E1E00(0, v7, 0, 0);
  sub_69ED20((__int64)v8, 0, 0, 1);
  sub_6F69D0(v8, 0);
  sub_6F4950(v8, a1, v1, v2, v3, v4);
  sub_6E2AC0(a1);
  sub_6E2B30(a1, a1);
  sub_6E1DF0(v6);
  unk_4F061D8 = v9;
  return &unk_4F061D8;
}
