// Function: sub_B325F0
// Address: 0xb325f0
//
__int64 __fastcall sub_B325F0(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-30h] BYREF
  __int64 v5; // [rsp+8h] [rbp-28h]
  __int64 v6; // [rsp+10h] [rbp-20h]
  __int64 v7; // [rsp+18h] [rbp-18h]

  v1 = *(_QWORD *)(a1 - 32);
  v4 = 0;
  v5 = 0;
  v6 = 0;
  v7 = 0;
  v2 = sub_B32260(v1, (__int64)&v4);
  sub_C7D6A0(v5, 8LL * (unsigned int)v7, 8);
  return v2;
}
