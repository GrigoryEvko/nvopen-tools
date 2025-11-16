// Function: sub_380CB00
// Address: 0x380cb00
//
unsigned __int8 *__fastcall sub_380CB00(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // r13
  unsigned int v6; // r12d
  __int64 v7; // r8
  __int64 v8; // rsi
  unsigned __int8 *v9; // r12
  __int64 v11; // [rsp+0h] [rbp-50h]
  __int64 v12; // [rsp+10h] [rbp-40h] BYREF
  int v13; // [rsp+18h] [rbp-38h]

  sub_380AAE0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = **(unsigned __int16 **)(a2 + 48);
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v12 = v4;
  if ( v4 )
  {
    v11 = v7;
    sub_B96E90((__int64)&v12, v4, 1);
    v7 = v11;
  }
  v8 = *(unsigned int *)(a2 + 24);
  v13 = *(_DWORD *)(a2 + 72);
  v9 = sub_33FAF80(v5, v8, (__int64)&v12, v6, v7, (unsigned int)&v12, a3);
  if ( v12 )
    sub_B91220((__int64)&v12, v12);
  return v9;
}
