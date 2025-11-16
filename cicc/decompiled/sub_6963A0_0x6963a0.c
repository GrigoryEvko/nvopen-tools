// Function: sub_6963A0
// Address: 0x6963a0
//
__int64 __fastcall sub_6963A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rdi
  __int64 v5; // r12
  __int64 v7; // [rsp+8h] [rbp-228h] BYREF
  _BYTE v8[160]; // [rsp+10h] [rbp-220h] BYREF
  _BYTE v9[68]; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v10; // [rsp+F4h] [rbp-13Ch]

  sub_6E1DD0(&v7);
  sub_6E1E00(4, v8, 0, 0);
  sub_6E7170(a1, v9);
  v10 = *a3;
  sub_843D70(v9, a2, 0, 167);
  v4 = sub_6F6F40(v9, 0);
  v5 = sub_6E2700(v4);
  sub_6E2B30(v4, 0);
  sub_6E1DF0(v7);
  return v5;
}
