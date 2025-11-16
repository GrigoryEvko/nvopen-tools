// Function: sub_20369E0
// Address: 0x20369e0
//
__int64 *__fastcall sub_20369E0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rsi
  unsigned int v9; // r13d
  const void **v10; // r15
  unsigned __int64 v11; // rdx
  __int128 v12; // rax
  __int64 *v13; // r12
  __int64 v15; // [rsp+0h] [rbp-70h]
  unsigned __int64 v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+10h] [rbp-60h] BYREF
  int v18; // [rsp+18h] [rbp-58h]
  _BYTE v19[16]; // [rsp+20h] [rbp-50h] BYREF
  const void **v20; // [rsp+30h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 72);
  v17 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v17, v6, 2);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)a1;
  v18 = *(_DWORD *)(a2 + 64);
  sub_1F40D10(
    (__int64)v19,
    v8,
    *(_QWORD *)(v7 + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v9 = v19[8];
  v10 = v20;
  v15 = sub_20363F0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v16 = v11;
  *(_QWORD *)&v12 = sub_20363F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v13 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v17,
          v9,
          v10,
          *(unsigned __int16 *)(a2 + 80),
          a3,
          a4,
          a5,
          v15,
          v16,
          v12);
  if ( v17 )
    sub_161E7C0((__int64)&v17, v17);
  return v13;
}
