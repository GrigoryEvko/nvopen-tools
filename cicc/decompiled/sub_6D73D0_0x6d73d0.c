// Function: sub_6D73D0
// Address: 0x6d73d0
//
__int64 __fastcall sub_6D73D0(__int64 a1)
{
  int v1; // r13d
  __int64 v3; // [rsp-10h] [rbp-240h]
  __int64 v4; // [rsp+8h] [rbp-228h] BYREF
  _BYTE v5[160]; // [rsp+10h] [rbp-220h] BYREF
  _BYTE v6[76]; // [rsp+B0h] [rbp-180h] BYREF
  __int64 v7; // [rsp+FCh] [rbp-134h]

  sub_6E2250(v5, &v4, 4, 1, a1, 0);
  v1 = -((*(_BYTE *)(qword_4D03C50 + 19LL) & 4) == 0);
  sub_69ED20((__int64)v6, 0, 0, 1);
  sub_8470D0((unsigned int)v6, *(_QWORD *)(a1 + 288), 1, (v1 & 0xFFFFFE00) + 513, 144, 0, a1 + 144);
  sub_68A7B0(a1 + 136);
  sub_6E2C70(v4, 1, a1, 0);
  *(_QWORD *)&dword_4F061D8 = v7;
  return v3;
}
