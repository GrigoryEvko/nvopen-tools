// Function: sub_213D130
// Address: 0x213d130
//
__int64 __fastcall sub_213D130(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rcx
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v12; // [rsp+0h] [rbp-50h]
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14; // [rsp+10h] [rbp-40h] BYREF
  int v15; // [rsp+18h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 32);
  v4 = *(_QWORD *)(a2 + 72);
  v5 = *v3;
  v6 = v3[10];
  v14 = v4;
  v7 = v3[11];
  v13 = v5;
  v12 = v3[1];
  if ( v4 )
  {
    sub_1623A60((__int64)&v14, v4, 2);
    v3 = *(__int64 **)(a2 + 32);
  }
  v15 = *(_DWORD *)(a2 + 64);
  v8 = sub_2138AD0(a1, v3[5], v3[6]);
  v10 = sub_1D2C2D0(
          *(_QWORD **)(a1 + 8),
          v13,
          v12,
          (__int64)&v14,
          v8,
          v9,
          v6,
          v7,
          *(unsigned __int8 *)(a2 + 88),
          *(_QWORD *)(a2 + 96),
          *(_QWORD *)(a2 + 104));
  if ( v14 )
    sub_161E7C0((__int64)&v14, v14);
  return v10;
}
