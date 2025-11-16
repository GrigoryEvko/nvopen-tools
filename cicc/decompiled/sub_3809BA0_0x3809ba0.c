// Function: sub_3809BA0
// Address: 0x3809ba0
//
__m128i *__fastcall sub_3809BA0(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned __int64 v6; // r12
  __int64 v7; // r13
  __int128 v8; // rax
  __int64 v9; // r9
  unsigned __int8 *v10; // rax
  unsigned int v11; // edx
  unsigned int v12; // edx
  unsigned __int64 v13; // r12
  __m128i *v14; // r12
  __int128 v16; // [rsp-20h] [rbp-90h]
  _QWORD *v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+30h] [rbp-40h] BYREF
  int v19; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *(_QWORD *)(v4 + 40);
  v7 = *(_QWORD *)(v4 + 48);
  v18 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v18, v5, 1);
  v19 = *(_DWORD *)(a2 + 72);
  if ( (*(_BYTE *)(a2 + 33) & 4) != 0 )
  {
    v17 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v8 = sub_3400D50((__int64)v17, 0, (__int64)&v18, 1u, a3);
    *((_QWORD *)&v16 + 1) = v7;
    *(_QWORD *)&v16 = v6;
    v10 = sub_3406EB0(v17, 0xE6u, (__int64)&v18, *(unsigned __int16 *)(a2 + 96), *(_QWORD *)(a2 + 104), v9, v16, v8);
    v13 = (unsigned __int64)sub_375A6A0(a1, (__int64)v10, v11, a3);
  }
  else
  {
    v13 = sub_3805E70(a1, v6, v7);
  }
  v14 = sub_33F3F90(
          *(_QWORD **)(a1 + 8),
          **(_QWORD **)(a2 + 40),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
          (__int64)&v18,
          v13,
          v12 | v7 & 0xFFFFFFFF00000000LL,
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
          *(const __m128i **)(a2 + 112));
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v14;
}
