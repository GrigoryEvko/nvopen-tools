// Function: sub_2124550
// Address: 0x2124550
//
__int64 *__fastcall sub_2124550(__m128i **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  char v13; // r15
  __int32 v14; // edx
  unsigned __int64 v15; // rax
  __int64 v16; // rsi
  __m128i *v17; // r10
  __int32 v18; // edx
  __int64 *v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r9
  __int64 v23; // r8
  __m128i *v24; // r13
  __int128 v25; // rax
  __int64 v27; // [rsp-8h] [rbp-A8h]
  __m128i *v28; // [rsp+8h] [rbp-98h]
  unsigned int v29; // [rsp+3Ch] [rbp-64h] BYREF
  __m128i v30; // [rsp+40h] [rbp-60h] BYREF
  __m128i v31; // [rsp+50h] [rbp-50h] BYREF
  __int64 v32; // [rsp+60h] [rbp-40h] BYREF
  int v33; // [rsp+68h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)v6;
  v8 = _mm_loadu_si128((const __m128i *)v6);
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v10 = *(_QWORD *)(v6 + 80);
  v30 = v8;
  LODWORD(v10) = *(_DWORD *)(v10 + 84);
  v31 = v9;
  v29 = v10;
  v11 = *(_QWORD *)(v7 + 40) + 16LL * v8.m128i_u32[2];
  v12 = *(_QWORD *)(v11 + 8);
  v13 = *(_BYTE *)v11;
  v30.m128i_i64[0] = sub_2120330((__int64)a1, v7, v8.m128i_i64[1]);
  v30.m128i_i32[2] = v14;
  v15 = sub_2120330((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
  v16 = *(_QWORD *)(a2 + 72);
  v17 = *a1;
  v31.m128i_i64[0] = v15;
  v32 = v16;
  v31.m128i_i32[2] = v18;
  if ( v16 )
  {
    v28 = v17;
    sub_1623A60((__int64)&v32, v16, 2);
    v17 = v28;
  }
  v19 = (__int64 *)a1[1];
  v33 = *(_DWORD *)(a2 + 64);
  sub_20BED60(
    v17,
    v19,
    v13,
    *(double *)v8.m128i_i64,
    *(double *)v9.m128i_i64,
    a5,
    v12,
    (__int64)&v30,
    &v31,
    &v29,
    (__int64)&v32);
  v23 = v27;
  if ( v32 )
    sub_161E7C0((__int64)&v32, v32);
  if ( !v31.m128i_i64[0] )
    return (__int64 *)v30.m128i_i64[0];
  v24 = a1[1];
  *(_QWORD *)&v25 = sub_1D28D50(v24, v29, v20, v21, v23, v22);
  return sub_1D2E2F0(v24, (__int64 *)a2, v30.m128i_i64[0], v30.m128i_i64[1], v31.m128i_i64[0], v31.m128i_i64[1], v25);
}
