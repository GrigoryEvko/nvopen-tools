// Function: sub_1F6F630
// Address: 0x1f6f630
//
__int64 __fastcall sub_1F6F630(__int64 a1, __int64 a2, __int64 *a3, double a4, double a5, __m128i a6)
{
  __int64 v8; // rsi
  char *v9; // rax
  char v10; // dl
  __int64 v11; // rax
  __int64 v12; // rsi
  __m128i v13; // kr00_16
  unsigned int *v14; // r11
  int v15; // eax
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  __m128 v19; // xmm0
  __m128i v20; // xmm1
  __int64 v21; // rsi
  __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rax
  char v25; // dl
  __int64 v26; // rax
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __int64 v29; // r14
  __int64 v30; // r15
  int v31; // edx
  int v32; // ebx
  __int64 *v33; // rax
  __int64 v34; // rsi
  int v35; // edx
  __int128 v37; // [rsp-20h] [rbp-130h]
  __int128 v38; // [rsp-20h] [rbp-130h]
  __int64 v39; // [rsp+8h] [rbp-108h]
  const void **v40; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v41; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v42; // [rsp+50h] [rbp-C0h]
  __int16 *v43; // [rsp+58h] [rbp-B8h]
  __int64 v44; // [rsp+60h] [rbp-B0h]
  unsigned __int64 v45; // [rsp+60h] [rbp-B0h]
  __int64 *v46; // [rsp+60h] [rbp-B0h]
  __int16 *v47; // [rsp+68h] [rbp-A8h]
  __int64 v48; // [rsp+70h] [rbp-A0h] BYREF
  int v49; // [rsp+78h] [rbp-98h]
  __int64 v50; // [rsp+80h] [rbp-90h] BYREF
  int v51; // [rsp+88h] [rbp-88h]
  __m128 v52; // [rsp+90h] [rbp-80h] BYREF
  __m128i v53; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v54; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-58h]
  __m128i v56; // [rsp+C0h] [rbp-50h] BYREF
  __m128i v57; // [rsp+D0h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 72);
  v48 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v48, v8, 2);
  v49 = *(_DWORD *)(a2 + 64);
  v9 = *(char **)(a2 + 40);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOBYTE(v54) = v10;
  v55 = v11;
  sub_1D19A30((__int64)&v56, (__int64)a3, &v54);
  v12 = *(_QWORD *)(a2 + 72);
  v41 = v56.m128i_i64[0];
  v50 = v12;
  v40 = (const void **)v56.m128i_i64[1];
  v13 = v57;
  if ( v12 )
    sub_1623A60((__int64)&v50, v12, 2);
  v14 = *(unsigned int **)(a2 + 32);
  v15 = *(_DWORD *)(a2 + 64);
  v52.m128_i8[0] = 0;
  v52.m128_u64[1] = 0;
  v51 = v15;
  v53.m128i_i8[0] = 0;
  v53.m128i_i64[1] = 0;
  v44 = (__int64)v14;
  v16 = *(_QWORD *)(*(_QWORD *)v14 + 40LL) + 16LL * v14[2];
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOBYTE(v54) = v17;
  v55 = v18;
  sub_1D19A30((__int64)&v56, (__int64)a3, &v54);
  v19 = (__m128)_mm_loadu_si128(&v56);
  v20 = _mm_loadu_si128(&v57);
  v52 = v19;
  v53 = v20;
  sub_1D40600(
    (__int64)&v56,
    a3,
    v44,
    (__int64)&v50,
    (const void ***)&v52,
    (const void ***)&v53,
    (__m128i)v19,
    *(double *)v20.m128i_i64,
    a6);
  if ( v50 )
    sub_161E7C0((__int64)&v50, v50);
  v21 = *(_QWORD *)(a2 + 72);
  v45 = v56.m128i_i64[0];
  v50 = v21;
  v47 = (__int16 *)v56.m128i_u32[2];
  v42 = v57.m128i_i64[0];
  v43 = (__int16 *)v57.m128i_u32[2];
  if ( v21 )
    sub_1623A60((__int64)&v50, v21, 2);
  v22 = *(_QWORD *)(a2 + 32);
  v23 = *(_DWORD *)(a2 + 64);
  v52.m128_i8[0] = 0;
  v52.m128_u64[1] = 0;
  v51 = v23;
  v53.m128i_i8[0] = 0;
  v53.m128i_i64[1] = 0;
  v39 = v22;
  v24 = *(_QWORD *)(*(_QWORD *)(v22 + 40) + 40LL) + 16LL * *(unsigned int *)(v22 + 48);
  v25 = *(_BYTE *)v24;
  v26 = *(_QWORD *)(v24 + 8);
  LOBYTE(v54) = v25;
  v55 = v26;
  sub_1D19A30((__int64)&v56, (__int64)a3, &v54);
  v27 = _mm_loadu_si128(&v56);
  v28 = _mm_loadu_si128(&v57);
  v52 = (__m128)v27;
  v53 = v28;
  sub_1D40600(
    (__int64)&v56,
    a3,
    v39 + 40,
    (__int64)&v50,
    (const void ***)&v52,
    (const void ***)&v53,
    (__m128i)v19,
    *(double *)v20.m128i_i64,
    v27);
  if ( v50 )
    sub_161E7C0((__int64)&v50, v50);
  v29 = v57.m128i_i64[0];
  v30 = v57.m128i_u32[2];
  *((_QWORD *)&v37 + 1) = v56.m128i_u32[2];
  *(_QWORD *)&v37 = v56.m128i_i64[0];
  v46 = sub_1D3A900(
          a3,
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v48,
          v41,
          v40,
          0,
          v19,
          *(double *)v20.m128i_i64,
          v27,
          v45,
          v47,
          v37,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v32 = v31;
  *((_QWORD *)&v38 + 1) = v30;
  *(_QWORD *)&v38 = v29;
  v33 = sub_1D3A900(
          a3,
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v48,
          v13.m128i_u64[0],
          (const void **)v13.m128i_i64[1],
          0,
          v19,
          *(double *)v20.m128i_i64,
          v27,
          v42,
          v43,
          v38,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v34 = v48;
  *(_QWORD *)a1 = v46;
  *(_DWORD *)(a1 + 8) = v32;
  *(_QWORD *)(a1 + 16) = v33;
  *(_DWORD *)(a1 + 24) = v35;
  if ( v34 )
    sub_161E7C0((__int64)&v48, v34);
  return a1;
}
