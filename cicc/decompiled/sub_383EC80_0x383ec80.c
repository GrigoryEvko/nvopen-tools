// Function: sub_383EC80
// Address: 0x383ec80
//
unsigned __int8 *__fastcall sub_383EC80(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __m128i v4; // xmm1
  __int64 v5; // r9
  __int64 v6; // rsi
  _QWORD *v7; // r13
  unsigned int v8; // r12d
  __int64 v9; // r15
  unsigned int v10; // esi
  unsigned __int8 *v11; // r12
  __m128i v13; // [rsp+0h] [rbp-60h] BYREF
  __int128 v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+20h] [rbp-40h] BYREF
  int v16; // [rsp+28h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 40);
  v4 = _mm_loadu_si128((const __m128i *)(v3 + 40));
  v13 = _mm_loadu_si128((const __m128i *)v3);
  v14 = (__int128)v4;
  sub_383E4F0(a1, (__int64)&v13, (__int64)&v14, v13);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = (_QWORD *)a1[1];
  v8 = *(unsigned __int16 *)(*(_QWORD *)(v13.m128i_i64[0] + 48) + 16LL * v13.m128i_u32[2]);
  v9 = *(_QWORD *)(*(_QWORD *)(v13.m128i_i64[0] + 48) + 16LL * v13.m128i_u32[2] + 8);
  v15 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v15, v6, 1);
  v10 = *(_DWORD *)(a2 + 24);
  v16 = *(_DWORD *)(a2 + 72);
  v11 = sub_3406EB0(v7, v10, (__int64)&v15, v8, v9, v5, *(_OWORD *)&v13, v14);
  if ( v15 )
    sub_B91220((__int64)&v15, v15);
  return v11;
}
