// Function: sub_B32A80
// Address: 0xb32a80
//
__int64 __fastcall sub_B32A80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdi
  _QWORD v5[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  __int64 v7; // [rsp+18h] [rbp-18h]
  __int64 v8; // [rsp+20h] [rbp-10h]
  __int64 v9; // [rsp+28h] [rbp-8h]

  v3 = *(_QWORD *)(a1 - 32);
  v5[0] = a2;
  v5[1] = a3;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  sub_B32750(v3, (__int64)&v6, (__int64)v5);
  return sub_C7D6A0(v7, 8LL * (unsigned int)v9, 8);
}
