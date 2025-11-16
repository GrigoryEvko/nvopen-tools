// Function: sub_3785410
// Address: 0x3785410
//
unsigned __int8 *__fastcall sub_3785410(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  unsigned int v10; // r14d
  unsigned __int64 *v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rcx
  unsigned __int16 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  const __m128i *v17; // rdx
  int v18; // r9d
  __int64 v19; // rbx
  __m128i v20; // xmm3
  __int64 v21; // rax
  __m128i v22; // xmm4
  _QWORD *v23; // rdi
  __m128i v24; // rax
  _QWORD *v25; // rdi
  __m128i v26; // xmm5
  unsigned __int8 *v27; // r14
  unsigned __int64 v29; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v30; // [rsp+8h] [rbp-D8h]
  __int64 v31; // [rsp+10h] [rbp-D0h]
  __int64 v32; // [rsp+18h] [rbp-C8h]
  __int64 v33; // [rsp+20h] [rbp-C0h]
  __int64 v34; // [rsp+28h] [rbp-B8h]
  unsigned int v35; // [rsp+34h] [rbp-ACh]
  __int64 v36; // [rsp+38h] [rbp-A8h]
  __m128i v37; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v38; // [rsp+50h] [rbp-90h] BYREF
  __int64 v39; // [rsp+60h] [rbp-80h] BYREF
  int v40; // [rsp+68h] [rbp-78h]
  __m128i v41; // [rsp+70h] [rbp-70h] BYREF
  __m128i v42; // [rsp+80h] [rbp-60h]
  __int64 v43; // [rsp+90h] [rbp-50h]
  unsigned __int64 v44; // [rsp+98h] [rbp-48h]
  __int64 v45; // [rsp+A0h] [rbp-40h]
  __int64 v46; // [rsp+A8h] [rbp-38h]

  v35 = *(_DWORD *)(a2 + 24);
  v7 = *(unsigned __int16 **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *((_QWORD *)v7 + 1);
  v10 = *v7;
  v37.m128i_i64[0] = 0;
  v37.m128i_i32[2] = 0;
  v36 = v9;
  v38.m128i_i64[0] = 0;
  v38.m128i_i32[2] = 0;
  v39 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v39, v8, 1);
  v40 = *(_DWORD *)(a2 + 72);
  v11 = (unsigned __int64 *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v12 = *v11;
  v13 = v11[1];
  v14 = (unsigned __int16 *)(*(_QWORD *)(*v11 + 48) + 16LL * *((unsigned int *)v11 + 2));
  v15 = *v14;
  v31 = *((_QWORD *)v14 + 1);
  v32 = v15;
  sub_375E8D0(a1, v12, v13, (__int64)&v37, (__int64)&v38);
  sub_3777990(
    &v41,
    (__int64 *)a1,
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL),
    a4);
  v16 = *(_QWORD *)(a2 + 40);
  v34 = v41.m128i_i64[0];
  v33 = v42.m128i_i64[0];
  v30 = _mm_cvtsi32_si128(v42.m128i_u32[2]).m128i_u64[0];
  v29 = _mm_cvtsi32_si128(v41.m128i_u32[2]).m128i_u64[0];
  sub_3408380(
    &v41,
    *(_QWORD **)(a1 + 8),
    *(_QWORD *)(v16 + 120),
    *(_QWORD *)(v16 + 128),
    (unsigned int)v32,
    v31,
    a4,
    (__int64)&v39);
  v17 = *(const __m128i **)(a2 + 40);
  v18 = *(_DWORD *)(a2 + 28);
  v19 = v34;
  v20 = _mm_loadu_si128(v17);
  LODWORD(v34) = v18;
  v21 = v41.m128i_i64[0];
  v43 = v19;
  LODWORD(v19) = v35;
  v22 = _mm_loadu_si128(&v37);
  v32 = v42.m128i_i64[0];
  v23 = *(_QWORD **)(a1 + 8);
  v46 = v41.m128i_u32[2];
  v41 = v20;
  v31 = _mm_cvtsi32_si128(v42.m128i_u32[2]).m128i_u64[0];
  v42 = v22;
  v44 = v29;
  v45 = v21;
  v24.m128i_i64[0] = (__int64)sub_33FBA10(v23, v35, (__int64)&v39, v10, v36, v18, (__int64)&v41, 4);
  v25 = *(_QWORD **)(a1 + 8);
  v26 = _mm_loadu_si128(&v38);
  v41 = v24;
  v43 = v33;
  v42 = v26;
  v45 = v32;
  v44 = v30;
  v46 = v31;
  v27 = sub_33FBA10(v25, (unsigned int)v19, (__int64)&v39, v10, v36, v34, (__int64)&v41, 4);
  if ( v39 )
    sub_B91220((__int64)&v39, v39);
  return v27;
}
