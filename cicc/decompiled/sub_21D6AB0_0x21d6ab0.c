// Function: sub_21D6AB0
// Address: 0x21d6ab0
//
__int64 __fastcall sub_21D6AB0(double a1, double a2, double a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v9; // rsi
  const __m128i *v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  unsigned int v15; // edx
  unsigned __int64 v16; // r13
  __int64 v17; // r12
  int v19; // [rsp-28h] [rbp-C8h]
  __int128 v20; // [rsp-10h] [rbp-B0h]
  __int64 v21; // [rsp+8h] [rbp-98h]
  __int16 v22; // [rsp+14h] [rbp-8Ch]
  __int64 v23; // [rsp+18h] [rbp-88h]
  __m128i v24; // [rsp+20h] [rbp-80h]
  __int64 v25; // [rsp+40h] [rbp-60h] BYREF
  int v26; // [rsp+48h] [rbp-58h]
  _QWORD v27[10]; // [rsp+50h] [rbp-50h] BYREF

  v9 = *(_QWORD *)(a5 + 72);
  v25 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v25, v9, 2);
  v26 = *(_DWORD *)(a5 + 64);
  v10 = *(const __m128i **)(a5 + 32);
  v11 = v10[3].m128i_i64[0];
  *((_QWORD *)&v20 + 1) = v11;
  *(_QWORD *)&v20 = v10[2].m128i_i64[1];
  v23 = v10->m128i_i64[1];
  v21 = v10->m128i_i64[0];
  v24 = _mm_loadu_si128(v10 + 5);
  v12 = sub_1D309E0(a7, 143, (__int64)&v25, 4, 0, 0, *(double *)v24.m128i_i64, a2, a3, v20);
  memset(v27, 0, 24);
  v13 = v12;
  v14 = *(_QWORD *)(a5 + 104);
  v16 = v15 | v11 & 0xFFFFFFFF00000000LL;
  v22 = *(_WORD *)(v14 + 32);
  v19 = sub_1E34390(v14);
  v17 = sub_1D2C750(
          a7,
          v21,
          v23,
          (__int64)&v25,
          v13,
          v16,
          v24.m128i_i64[0],
          v24.m128i_i64[1],
          *(_OWORD *)*(_QWORD *)(a5 + 104),
          *(_QWORD *)(*(_QWORD *)(a5 + 104) + 16LL),
          3,
          0,
          v19,
          v22,
          (__int64)v27);
  if ( v25 )
    sub_161E7C0((__int64)&v25, v25);
  return v17;
}
