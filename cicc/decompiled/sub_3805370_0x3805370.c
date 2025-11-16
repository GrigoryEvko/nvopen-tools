// Function: sub_3805370
// Address: 0x3805370
//
unsigned __int8 *__fastcall sub_3805370(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // r12
  int v5; // edx
  __int64 v6; // rsi
  int v7; // r9d
  unsigned __int8 *v8; // r12
  int v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h] BYREF
  int v12; // [rsp+18h] [rbp-28h]

  v4 = *(_QWORD *)(a1 + 8);
  sub_375A6A0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = v5;
  v11 = v6;
  if ( v6 )
  {
    v10 = v5;
    sub_B96E90((__int64)&v11, v6, 1);
    v7 = v10;
  }
  v12 = *(_DWORD *)(a2 + 72);
  v8 = sub_33FAF80(v4, 335, (__int64)&v11, 6, 0, v7, a3);
  if ( v11 )
    sub_B91220((__int64)&v11, v11);
  return v8;
}
