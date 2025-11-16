// Function: sub_20368C0
// Address: 0x20368c0
//
__int64 *__fastcall sub_20368C0(__int64 a1, __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int16 *v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // r12
  __int128 v17; // [rsp-20h] [rbp-A0h]
  const void **v18; // [rsp+0h] [rbp-80h]
  unsigned __int64 v19; // [rsp+8h] [rbp-78h]
  unsigned __int64 v20; // [rsp+10h] [rbp-70h]
  __int16 *v21; // [rsp+18h] [rbp-68h]
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  int v23; // [rsp+28h] [rbp-58h]
  _BYTE v24[16]; // [rsp+30h] [rbp-50h] BYREF
  const void **v25; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 72);
  v22 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v22, v6, 2);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v23 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)v24,
    v8,
    *(_QWORD *)(v7 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v18 = v25;
  v19 = v24[8];
  v20 = sub_20363F0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v21 = v9;
  v10 = sub_20363F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v12 = v11;
  v13 = sub_20363F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  *((_QWORD *)&v17 + 1) = v12;
  *(_QWORD *)&v17 = v10;
  v15 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v22,
          v19,
          v18,
          0,
          a3,
          a4,
          a5,
          v20,
          v21,
          v17,
          v13,
          v14);
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v15;
}
