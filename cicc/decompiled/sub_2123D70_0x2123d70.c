// Function: sub_2123D70
// Address: 0x2123d70
//
__int64 *__fastcall sub_2123D70(__m128i **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rsi
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r14
  char v13; // r15
  int v14; // edx
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
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v30; // rsi
  __int64 v31; // r14
  const void **v32; // r8
  unsigned int v33; // r15d
  __int32 v34; // edx
  __int64 v35; // [rsp-10h] [rbp-C0h]
  __int64 v36; // [rsp-8h] [rbp-B8h]
  __m128i *v37; // [rsp+8h] [rbp-A8h]
  const void **v38; // [rsp+8h] [rbp-A8h]
  unsigned int v39; // [rsp+4Ch] [rbp-64h] BYREF
  __int128 v40; // [rsp+50h] [rbp-60h] BYREF
  __m128i v41; // [rsp+60h] [rbp-50h] BYREF
  __int64 v42; // [rsp+70h] [rbp-40h] BYREF
  int v43; // [rsp+78h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(v6 + 80);
  v8 = _mm_loadu_si128((const __m128i *)(v6 + 80));
  v9 = _mm_loadu_si128((const __m128i *)(v6 + 120));
  v10 = *(_QWORD *)(v6 + 40);
  v40 = (__int128)v8;
  LODWORD(v10) = *(_DWORD *)(v10 + 84);
  v41 = v9;
  v39 = v10;
  v11 = *(_QWORD *)(v7 + 40) + 16LL * v8.m128i_u32[2];
  v12 = *(_QWORD *)(v11 + 8);
  v13 = *(_BYTE *)v11;
  *(_QWORD *)&v40 = sub_2120330((__int64)a1, v7, v8.m128i_i64[1]);
  DWORD2(v40) = v14;
  v15 = sub_2120330((__int64)a1, v9.m128i_u64[0], v9.m128i_i64[1]);
  v16 = *(_QWORD *)(a2 + 72);
  v17 = *a1;
  v41.m128i_i64[0] = v15;
  v42 = v16;
  v41.m128i_i32[2] = v18;
  if ( v16 )
  {
    v37 = v17;
    sub_1623A60((__int64)&v42, v16, 2);
    v17 = v37;
  }
  v19 = (__int64 *)a1[1];
  v43 = *(_DWORD *)(a2 + 64);
  sub_20BED60(
    v17,
    v19,
    v13,
    *(double *)v8.m128i_i64,
    *(double *)v9.m128i_i64,
    a5,
    v12,
    (__int64)&v40,
    &v41,
    &v39,
    (__int64)&v42);
  v23 = v35;
  if ( v42 )
    sub_161E7C0((__int64)&v42, v42);
  if ( v41.m128i_i64[0] )
  {
    v24 = v39;
  }
  else
  {
    v30 = *(_QWORD *)(a2 + 72);
    v31 = (__int64)a1[1];
    v32 = *(const void ***)(*(_QWORD *)(v40 + 40) + 16LL * DWORD2(v40) + 8);
    v33 = *(unsigned __int8 *)(*(_QWORD *)(v40 + 40) + 16LL * DWORD2(v40));
    v42 = v30;
    if ( v30 )
    {
      v38 = v32;
      sub_1623A60((__int64)&v42, v30, 2);
      v32 = v38;
    }
    v43 = *(_DWORD *)(a2 + 64);
    v41.m128i_i64[0] = sub_1D38BB0(v31, 0, (__int64)&v42, v33, v32, 0, v8, *(double *)v9.m128i_i64, a5, 0);
    v41.m128i_i32[2] = v34;
    v20 = v36;
    if ( v42 )
      sub_161E7C0((__int64)&v42, v42);
    v39 = 22;
    v24 = 22;
  }
  v25 = a1[1];
  v26 = *(_QWORD *)(a2 + 32);
  v27 = sub_1D28D50(v25, v24, v20, v23, v21, v22);
  return sub_1D2E370(
           v25,
           (__int64 *)a2,
           **(_QWORD **)(a2 + 32),
           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
           v27,
           v28,
           v40,
           *(_OWORD *)&v41,
           *(_OWORD *)(v26 + 160));
}
