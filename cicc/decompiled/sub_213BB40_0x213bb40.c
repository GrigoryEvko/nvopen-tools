// Function: sub_213BB40
// Address: 0x213bb40
//
__int64 *__fastcall sub_213BB40(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  const __m128i *v6; // rax
  __m128 v7; // xmm0
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r10
  __int64 v15; // r11
  __int64 *v16; // r15
  unsigned __int64 v17; // rcx
  const void **v18; // r8
  __int64 *v19; // r12
  __int128 v21; // [rsp-20h] [rbp-90h]
  __int64 v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  unsigned __int64 v24; // [rsp+10h] [rbp-60h]
  const void **v25; // [rsp+18h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  int v27; // [rsp+38h] [rbp-38h]

  v6 = *(const __m128i **)(a2 + 32);
  v7 = (__m128)_mm_loadu_si128(v6);
  v8 = sub_2138AD0(a1, v6[2].m128i_u64[1], v6[3].m128i_i64[0]);
  v10 = v9;
  v11 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v13 = *(_QWORD *)(a2 + 72);
  v14 = v11;
  v15 = v12;
  v16 = *(__int64 **)(a1 + 8);
  v17 = *(unsigned __int8 *)(*(_QWORD *)(v8 + 40) + 16LL * (unsigned int)v10);
  v18 = *(const void ***)(*(_QWORD *)(v8 + 40) + 16LL * (unsigned int)v10 + 8);
  v26 = v13;
  if ( v13 )
  {
    v23 = v12;
    v24 = v17;
    v22 = v11;
    v25 = v18;
    sub_1623A60((__int64)&v26, v13, 2);
    v17 = v24;
    v14 = v22;
    v15 = v23;
    v18 = v25;
  }
  *((_QWORD *)&v21 + 1) = v10;
  *(_QWORD *)&v21 = v8;
  v27 = *(_DWORD *)(a2 + 64);
  v19 = sub_1D3A900(
          v16,
          0x87u,
          (__int64)&v26,
          v17,
          v18,
          0,
          v7,
          a4,
          a5,
          v7.m128_u64[0],
          (__int16 *)v7.m128_u64[1],
          v21,
          v14,
          v15);
  if ( v26 )
    sub_161E7C0((__int64)&v26, v26);
  return v19;
}
