// Function: sub_2136B20
// Address: 0x2136b20
//
__int64 *__fastcall sub_2136B20(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _QWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // r9
  __int64 *v7; // r12
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  int v10; // [rsp+8h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 72);
  v9 = v3;
  if ( v3 )
    sub_1623A60((__int64)&v9, v3, 2);
  v4 = *(_QWORD **)(a1 + 8);
  v5 = *(unsigned __int8 *)(a2 + 88);
  v6 = *(_QWORD *)(a2 + 104);
  v10 = *(_DWORD *)(a2 + 64);
  v7 = sub_1D2B8F0(
         v4,
         223,
         (__int64)&v9,
         v5,
         *(_QWORD *)(a2 + 96),
         v6,
         *(_OWORD *)*(_QWORD *)(a2 + 32),
         *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  if ( v9 )
    sub_161E7C0((__int64)&v9, v9);
  return v7;
}
