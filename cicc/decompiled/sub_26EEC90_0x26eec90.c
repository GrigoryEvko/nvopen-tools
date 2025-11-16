// Function: sub_26EEC90
// Address: 0x26eec90
//
__int64 __fastcall sub_26EEC90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // rdi
  __int64 v6; // [rsp+0h] [rbp-60h] BYREF
  __int64 v7; // [rsp+8h] [rbp-58h]
  __int64 v8; // [rsp+10h] [rbp-50h]
  __int64 v9; // [rsp+18h] [rbp-48h]
  __int64 v10; // [rsp+20h] [rbp-40h] BYREF
  __int64 v11; // [rsp+28h] [rbp-38h]
  __int64 v12; // [rsp+30h] [rbp-30h]
  __int64 v13; // [rsp+38h] [rbp-28h]

  v4 = (__int64 *)(a1 + 8);
  *(v4 - 1) = a2;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  sub_26E91F0(v4, *(_BYTE **)a3, *(_QWORD *)a3 + *(_QWORD *)(a3 + 8));
  *(_QWORD *)(a1 + 48) = a1 + 96;
  *(_QWORD *)(a1 + 104) = a1 + 152;
  *(_QWORD *)(a1 + 56) = 1;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 112) = 1;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 80) = 1065353216;
  *(_DWORD *)(a1 + 136) = 1065353216;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  sub_26EDE20((__int64 *)a1, (__int64)&v6, (__int64)&v10);
  sub_26E9DC0((_QWORD *)a1, (__int64)&v6, (__int64)&v10);
  sub_26EBE40((__int64 *)a1, (__int64)&v6);
  sub_C7D6A0(v11, 8LL * (unsigned int)v13, 8);
  return sub_C7D6A0(v7, 8LL * (unsigned int)v9, 8);
}
