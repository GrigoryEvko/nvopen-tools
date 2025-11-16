// Function: sub_3798610
// Address: 0x3798610
//
unsigned __int8 *__fastcall sub_3798610(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // r13
  unsigned int v6; // r12d
  __int64 v7; // r8
  unsigned __int8 *v8; // r12
  __int64 v10; // [rsp+0h] [rbp-50h]
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  int v12; // [rsp+18h] [rbp-38h]

  sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = **(unsigned __int16 **)(a2 + 48);
  v7 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v11 = v4;
  if ( v4 )
  {
    v10 = v7;
    sub_B96E90((__int64)&v11, v4, 1);
    v7 = v10;
  }
  v12 = *(_DWORD *)(a2 + 72);
  v8 = sub_33FAF80(v5, 234, (__int64)&v11, v6, v7, (unsigned int)&v11, a3);
  if ( v11 )
    sub_B91220((__int64)&v11, v11);
  return v8;
}
