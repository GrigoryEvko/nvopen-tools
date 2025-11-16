// Function: sub_382A090
// Address: 0x382a090
//
__int64 *__fastcall sub_382A090(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 *v4; // rdi
  unsigned __int16 v5; // cx
  const __m128i *v6; // r9
  __int64 *v7; // r12
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  int v10; // [rsp+8h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 80);
  v9 = v3;
  if ( v3 )
    sub_B96E90((__int64)&v9, v3, 1);
  v4 = *(__int64 **)(a1 + 8);
  v5 = *(_WORD *)(a2 + 96);
  v6 = *(const __m128i **)(a2 + 112);
  v10 = *(_DWORD *)(a2 + 72);
  v7 = sub_33F34C0(
         v4,
         342,
         (__int64)&v9,
         v5,
         *(_QWORD *)(a2 + 104),
         v6,
         *(_OWORD *)*(_QWORD *)(a2 + 40),
         *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  if ( v9 )
    sub_B91220((__int64)&v9, v9);
  return v7;
}
