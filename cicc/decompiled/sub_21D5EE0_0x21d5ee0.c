// Function: sub_21D5EE0
// Address: 0x21d5ee0
//
__int64 __fastcall sub_21D5EE0(double a1, double a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6, __int64 *a7)
{
  __int64 v7; // r8
  __int64 *v8; // r10
  const __m128i *v9; // rax
  __int64 v10; // rsi
  __m128 v11; // xmm0
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // r15
  unsigned int v16; // edx
  unsigned __int64 v17; // r13
  __int64 v18; // rax
  unsigned int v19; // edx
  __int128 v20; // rax
  __int64 v21; // r12
  __int128 v23; // [rsp-20h] [rbp-A0h]
  __int128 v24; // [rsp-20h] [rbp-A0h]
  __int128 v25; // [rsp-10h] [rbp-90h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 *v28; // [rsp+8h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-50h]
  __int64 v30; // [rsp+40h] [rbp-40h] BYREF
  int v31; // [rsp+48h] [rbp-38h]

  v7 = a5;
  v8 = a7;
  v9 = *(const __m128i **)(a5 + 32);
  v10 = *(_QWORD *)(a5 + 72);
  v11 = (__m128)_mm_loadu_si128(v9);
  v12 = v9[2].m128i_i64[1];
  v30 = v10;
  v13 = v9[3].m128i_i64[0];
  v14 = v9[5].m128i_i64[0];
  v15 = v9[5].m128i_i64[1];
  if ( v10 )
  {
    v27 = v7;
    sub_1623A60((__int64)&v30, v10, 2);
    v8 = a7;
    v7 = v27;
  }
  *((_QWORD *)&v25 + 1) = v13;
  *(_QWORD *)&v25 = v12;
  v28 = v8;
  v31 = *(_DWORD *)(v7 + 64);
  *((_QWORD *)&v23 + 1) = v15;
  *(_QWORD *)&v23 = v14;
  v29 = sub_1D309E0(v8, 144, (__int64)&v30, 5, 0, 0, *(double *)v11.m128_u64, a2, *(double *)a3.m128i_i64, v25);
  v17 = v16 | v13 & 0xFFFFFFFF00000000LL;
  v18 = sub_1D309E0(v28, 144, (__int64)&v30, 5, 0, 0, *(double *)v11.m128_u64, a2, *(double *)a3.m128i_i64, v23);
  *((_QWORD *)&v24 + 1) = v17;
  *(_QWORD *)&v24 = v29;
  *(_QWORD *)&v20 = sub_1D3A900(
                      v28,
                      0x86u,
                      (__int64)&v30,
                      5u,
                      0,
                      0,
                      v11,
                      a2,
                      a3,
                      v11.m128_u64[0],
                      (__int16 *)v11.m128_u64[1],
                      v24,
                      v18,
                      v19 | v15 & 0xFFFFFFFF00000000LL);
  v21 = sub_1D309E0(v28, 145, (__int64)&v30, 2, 0, 0, *(double *)v11.m128_u64, a2, *(double *)a3.m128i_i64, v20);
  if ( v30 )
    sub_161E7C0((__int64)&v30, v30);
  return v21;
}
