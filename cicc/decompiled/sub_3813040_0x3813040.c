// Function: sub_3813040
// Address: 0x3813040
//
__m128i *__fastcall sub_3813040(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned __int64 v5; // r15
  __int64 v6; // r13
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __m128i *v9; // r12
  __int64 v11; // [rsp+0h] [rbp-40h] BYREF
  int v12; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD *)(v3 + 40);
  v6 = *(_QWORD *)(v3 + 48);
  v11 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v11, v4, 1);
  v12 = *(_DWORD *)(a2 + 72);
  v7 = sub_380F170(a1, v5, v6);
  v9 = sub_33F3F90(
         *(_QWORD **)(a1 + 8),
         **(_QWORD **)(a2 + 40),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
         (__int64)&v11,
         v7,
         v8,
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
         *(const __m128i **)(a2 + 112));
  if ( v11 )
    sub_B91220((__int64)&v11, v11);
  return v9;
}
