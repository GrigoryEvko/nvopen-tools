// Function: sub_2127D20
// Address: 0x2127d20
//
__int64 *__fastcall sub_2127D20(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r12
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  int v12; // [rsp+8h] [rbp-48h]
  _BYTE v13[16]; // [rsp+10h] [rbp-40h] BYREF
  const void **v14; // [rsp+20h] [rbp-30h]

  v6 = *(_QWORD *)(a2 + 72);
  v11 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v11, v6, 2);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v12 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)v13,
    v8,
    *(_QWORD *)(v7 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v9 = sub_1D332F0(
         *(__int64 **)(a1 + 8),
         106,
         (__int64)&v11,
         v13[8],
         v14,
         0,
         a3,
         a4,
         a5,
         **(_QWORD **)(a2 + 32),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
         *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL));
  if ( v11 )
    sub_161E7C0((__int64)&v11, v11);
  return v9;
}
