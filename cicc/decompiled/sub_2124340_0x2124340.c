// Function: sub_2124340
// Address: 0x2124340
//
__int64 *__fastcall sub_2124340(__m128i **a1, __int64 a2, double a3, double a4, __m128i a5)
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
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rcx
  unsigned int v24; // esi
  __m128i *v25; // r13
  __int128 v26; // rax
  __int64 v28; // rsi
  __int64 v29; // r14
  const void **v30; // r8
  unsigned int v31; // r15d
  __int32 v32; // edx
  __int64 v33; // [rsp-10h] [rbp-C0h]
  __int64 v34; // [rsp-8h] [rbp-B8h]
  __m128i *v35; // [rsp+8h] [rbp-A8h]
  const void **v36; // [rsp+8h] [rbp-A8h]
  unsigned int v37; // [rsp+4Ch] [rbp-64h] BYREF
  __m128i v38; // [rsp+50h] [rbp-60h] BYREF
  __m128i v39; // [rsp+60h] [rbp-50h] BYREF
  __int64 v40; // [rsp+70h] [rbp-40h] BYREF
  int v41; // [rsp+78h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)v6;
  v8 = _mm_loadu_si128((const __m128i *)v6);
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 40));
  v10 = *(_QWORD *)(v6 + 160);
  v38 = v8;
  LODWORD(v10) = *(_DWORD *)(v10 + 84);
  v39 = v9;
  v37 = v10;
  v11 = *(_QWORD *)(v7 + 40) + 16LL * v8.m128i_u32[2];
  v12 = *(_QWORD *)(v11 + 8);
  v13 = *(_BYTE *)v11;
  v38.m128i_i64[0] = sub_2120330((__int64)a1, v7, v8.m128i_i64[1]);
  v38.m128i_i32[2] = v14;
  v15 = sub_2120330((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
  v16 = *(_QWORD *)(a2 + 72);
  v17 = *a1;
  v39.m128i_i64[0] = v15;
  v40 = v16;
  v39.m128i_i32[2] = v18;
  if ( v16 )
  {
    v35 = v17;
    sub_1623A60((__int64)&v40, v16, 2);
    v17 = v35;
  }
  v19 = (__int64 *)a1[1];
  v41 = *(_DWORD *)(a2 + 64);
  sub_20BED60(
    v17,
    v19,
    v13,
    *(double *)v8.m128i_i64,
    *(double *)v9.m128i_i64,
    a5,
    v12,
    (__int64)&v38,
    &v39,
    &v37,
    (__int64)&v40);
  v23 = v33;
  if ( v40 )
    sub_161E7C0((__int64)&v40, v40);
  if ( v39.m128i_i64[0] )
  {
    v24 = v37;
  }
  else
  {
    v28 = *(_QWORD *)(a2 + 72);
    v29 = (__int64)a1[1];
    v30 = *(const void ***)(*(_QWORD *)(v38.m128i_i64[0] + 40) + 16LL * v38.m128i_u32[2] + 8);
    v31 = *(unsigned __int8 *)(*(_QWORD *)(v38.m128i_i64[0] + 40) + 16LL * v38.m128i_u32[2]);
    v40 = v28;
    if ( v28 )
    {
      v36 = v30;
      sub_1623A60((__int64)&v40, v28, 2);
      v30 = v36;
    }
    v41 = *(_DWORD *)(a2 + 64);
    v39.m128i_i64[0] = sub_1D38BB0(v29, 0, (__int64)&v40, v31, v30, 0, v8, *(double *)v9.m128i_i64, a5, 0);
    v39.m128i_i32[2] = v32;
    v20 = v34;
    if ( v40 )
      sub_161E7C0((__int64)&v40, v40);
    v37 = 22;
    v24 = 22;
  }
  v25 = a1[1];
  *(_QWORD *)&v26 = sub_1D28D50(v25, v24, v20, v23, v21, v22);
  return sub_1D2E370(
           v25,
           (__int64 *)a2,
           v38.m128i_i64[0],
           v38.m128i_i64[1],
           v39.m128i_i64[0],
           v39.m128i_i64[1],
           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 120LL),
           v26);
}
