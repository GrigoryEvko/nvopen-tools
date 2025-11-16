// Function: sub_3841890
// Address: 0x3841890
//
unsigned __int8 *__fastcall sub_3841890(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  unsigned __int8 *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rdi
  unsigned __int16 *v9; // rdx
  unsigned __int8 *v10; // r14
  __int128 v12; // [rsp-40h] [rbp-80h]
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  int v14; // [rsp+8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v13 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v13, v5, 1);
  v14 = *(_DWORD *)(a2 + 72);
  v6 = sub_3841260(a1, a2, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
  v8 = v7;
  v9 = (unsigned __int16 *)(*((_QWORD *)v6 + 6) + 16LL * (unsigned int)v7);
  *((_QWORD *)&v12 + 1) = v8;
  *(_QWORD *)&v12 = v6;
  v10 = sub_33FC130(
          *(_QWORD **)(a1 + 8),
          *(unsigned int *)(a2 + 24),
          (__int64)&v13,
          *v9,
          *((_QWORD *)v9 + 1),
          *(_QWORD *)(a1 + 8),
          v12,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 120LL));
  if ( v13 )
    sub_B91220((__int64)&v13, v13);
  return v10;
}
