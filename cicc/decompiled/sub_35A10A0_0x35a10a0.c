// Function: sub_35A10A0
// Address: 0x35a10a0
//
__int64 __fastcall sub_35A10A0(__int64 a1)
{
  __int64 v2[10]; // [rsp+0h] [rbp-B0h] BYREF
  unsigned int v3; // [rsp+50h] [rbp-60h]
  __int64 v4; // [rsp+60h] [rbp-50h]
  unsigned int v5; // [rsp+70h] [rbp-40h]
  __int64 v6; // [rsp+80h] [rbp-30h]
  unsigned int v7; // [rsp+90h] [rbp-20h]

  sub_3598460((__int64)v2, **(_QWORD **)a1, *(_QWORD *)a1, *(_QWORD *)(a1 + 48));
  sub_35A06B0(v2);
  sub_C7D6A0(v6, 8LL * v7, 4);
  sub_C7D6A0(v4, 12LL * v5, 4);
  return sub_C7D6A0(v2[8], 16LL * v3, 8);
}
