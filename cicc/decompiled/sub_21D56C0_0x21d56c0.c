// Function: sub_21D56C0
// Address: 0x21d56c0
//
__int64 *__fastcall sub_21D56C0(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, double a5, double a6, __m128i a7)
{
  __int64 v9; // rdx
  char v10; // al
  const void **v11; // rdx
  __int64 v12; // r10
  __int64 v13; // rsi
  bool v14; // zf
  __int64 v15; // rax
  __m128 v16; // xmm0
  __m128i v17; // xmm1
  unsigned int v18; // r13d
  unsigned __int64 v19; // r14
  __int16 *v20; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __m128i v23; // rax
  __int64 *v24; // rax
  unsigned __int64 v25; // rdx
  __int128 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r15
  __int64 *v30; // r14
  __int128 v31; // rax
  __m128i v32; // rax
  __int128 v33; // rax
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // r15
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // rax
  __int16 *v43; // rdx
  __int16 *v44; // r15
  unsigned __int64 v45; // r14
  __m128i v46; // rax
  __int64 *v47; // rax
  int v48; // r8d
  __int64 v49; // rdx
  __int64 *v50; // r12
  __m128i v52; // rax
  __int64 *v53; // rax
  __m128i v54; // xmm3
  __int64 v55; // rdx
  __int128 v56; // [rsp-30h] [rbp-F0h]
  __int128 v57; // [rsp-10h] [rbp-D0h]
  __int64 v58; // [rsp+8h] [rbp-B8h]
  __int64 v59; // [rsp+10h] [rbp-B0h]
  __int128 v60; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v61; // [rsp+18h] [rbp-A8h]
  __m128i v62; // [rsp+20h] [rbp-A0h] BYREF
  __m128 v63; // [rsp+30h] [rbp-90h]
  __m128i v64; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v65; // [rsp+50h] [rbp-70h] BYREF
  const void **v66; // [rsp+58h] [rbp-68h]
  __int64 v67; // [rsp+60h] [rbp-60h] BYREF
  int v68; // [rsp+68h] [rbp-58h]
  __int64 *v69; // [rsp+70h] [rbp-50h] BYREF
  __int64 v70; // [rsp+78h] [rbp-48h]
  __m128i v71; // [rsp+80h] [rbp-40h]

  v9 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v62.m128i_i64[0] = a1;
  v10 = *(_BYTE *)v9;
  v11 = *(const void ***)(v9 + 8);
  LOBYTE(v65) = v10;
  v66 = v11;
  if ( v10 )
    v12 = (unsigned int)sub_1F3E310(&v65);
  else
    v12 = (unsigned int)sub_1F58D40((__int64)&v65);
  v13 = *(_QWORD *)(a2 + 72);
  v67 = v13;
  if ( v13 )
  {
    v64.m128i_i32[0] = v12;
    sub_1623A60((__int64)&v67, v13, 2);
    v12 = v64.m128i_u32[0];
  }
  v14 = *(_WORD *)(a2 + 24) == 140;
  v68 = *(_DWORD *)(a2 + 64);
  v15 = *(_QWORD *)(a2 + 32);
  v16 = (__m128)_mm_loadu_si128((const __m128i *)(v15 + 40));
  v17 = _mm_loadu_si128((const __m128i *)(v15 + 80));
  v18 = !v14 + 123;
  v19 = *(_QWORD *)v15;
  v20 = *(__int16 **)(v15 + 8);
  v63 = v16;
  v64 = v17;
  if ( (_DWORD)v12 == 32 && *(_DWORD *)(*(_QWORD *)(v62.m128i_i64[0] + 81552) + 252LL) > 0x22u )
  {
    v52.m128i_i64[0] = (__int64)sub_1D332F0(
                                  a4,
                                  v18,
                                  (__int64)&v67,
                                  v65,
                                  v66,
                                  0,
                                  *(double *)v16.m128_u64,
                                  *(double *)v17.m128i_i64,
                                  a7,
                                  v63.m128_i64[0],
                                  v63.m128_u64[1],
                                  *(_OWORD *)&v64);
    v62 = v52;
    v53 = sub_1D3A900(
            a4,
            0x126u,
            (__int64)&v67,
            v65,
            v66,
            0,
            v16,
            *(double *)v17.m128i_i64,
            a7,
            v19,
            v20,
            *(_OWORD *)&v63,
            v64.m128i_i64[0],
            v64.m128i_i64[1]);
    v54 = _mm_load_si128(&v62);
    v69 = v53;
    v70 = v55;
    v71 = v54;
  }
  else
  {
    v58 = v12;
    v21 = sub_1D38BB0((__int64)a4, v12, (__int64)&v67, 5, 0, 0, (__m128i)v16, *(double *)v17.m128i_i64, a7, 0);
    v23.m128i_i64[0] = (__int64)sub_1D332F0(
                                  a4,
                                  53,
                                  (__int64)&v67,
                                  5,
                                  0,
                                  0,
                                  *(double *)v16.m128_u64,
                                  *(double *)v17.m128i_i64,
                                  a7,
                                  v21,
                                  v22,
                                  *(_OWORD *)&v64);
    v62 = v23;
    v24 = sub_1D332F0(
            a4,
            124,
            (__int64)&v67,
            v65,
            v66,
            0,
            *(double *)v16.m128_u64,
            *(double *)v17.m128i_i64,
            a7,
            v19,
            (unsigned __int64)v20,
            *(_OWORD *)&v64);
    v61 = v25;
    v59 = (__int64)v24;
    *(_QWORD *)&v26 = sub_1D38BB0(
                        (__int64)a4,
                        v58,
                        (__int64)&v67,
                        5,
                        0,
                        0,
                        (__m128i)v16,
                        *(double *)v17.m128i_i64,
                        a7,
                        0);
    v27 = sub_1D332F0(
            a4,
            53,
            (__int64)&v67,
            5,
            0,
            0,
            *(double *)v16.m128_u64,
            *(double *)v17.m128i_i64,
            a7,
            v64.m128i_i64[0],
            v64.m128i_u64[1],
            v26);
    v29 = v28;
    v30 = v27;
    *(_QWORD *)&v31 = sub_1D332F0(
                        a4,
                        122,
                        (__int64)&v67,
                        v65,
                        v66,
                        0,
                        *(double *)v16.m128_u64,
                        *(double *)v17.m128i_i64,
                        a7,
                        v63.m128_i64[0],
                        v63.m128_u64[1],
                        *(_OWORD *)&v62);
    v32.m128i_i64[0] = (__int64)sub_1D332F0(
                                  a4,
                                  119,
                                  (__int64)&v67,
                                  v65,
                                  v66,
                                  0,
                                  *(double *)v16.m128_u64,
                                  *(double *)v17.m128i_i64,
                                  a7,
                                  v59,
                                  v61,
                                  v31);
    *((_QWORD *)&v57 + 1) = v29;
    *(_QWORD *)&v57 = v30;
    v62 = v32;
    *(_QWORD *)&v33 = sub_1D332F0(
                        a4,
                        v18,
                        (__int64)&v67,
                        v65,
                        v66,
                        0,
                        *(double *)v16.m128_u64,
                        *(double *)v17.m128i_i64,
                        a7,
                        v63.m128_i64[0],
                        v63.m128_u64[1],
                        v57);
    v60 = v33;
    v34 = sub_1D38BB0((__int64)a4, v58, (__int64)&v67, 5, 0, 0, (__m128i)v16, *(double *)v17.m128i_i64, a7, 0);
    v36 = v35;
    v40 = sub_1D28D50(a4, 0x13u, v35, v37, v38, v39);
    *((_QWORD *)&v56 + 1) = v36;
    *(_QWORD *)&v56 = v34;
    v42 = sub_1D3A900(
            a4,
            0x89u,
            (__int64)&v67,
            2u,
            0,
            0,
            v16,
            *(double *)v17.m128i_i64,
            a7,
            v64.m128i_u64[0],
            (__int16 *)v64.m128i_i64[1],
            v56,
            v40,
            v41);
    v44 = v43;
    v45 = (unsigned __int64)v42;
    v46.m128i_i64[0] = (__int64)sub_1D332F0(
                                  a4,
                                  v18,
                                  (__int64)&v67,
                                  v65,
                                  v66,
                                  0,
                                  *(double *)v16.m128_u64,
                                  *(double *)v17.m128i_i64,
                                  a7,
                                  v63.m128_i64[0],
                                  v63.m128_u64[1],
                                  *(_OWORD *)&v64);
    v64 = v46;
    v47 = sub_1D3A900(
            a4,
            0x86u,
            (__int64)&v67,
            v65,
            v66,
            0,
            v16,
            *(double *)v17.m128i_i64,
            a7,
            v45,
            v44,
            v60,
            v62.m128i_i64[0],
            v62.m128i_i64[1]);
    a7 = _mm_load_si128(&v64);
    v69 = v47;
    v70 = v49;
    v71 = a7;
  }
  v50 = sub_1D37190(
          (__int64)a4,
          (__int64)&v69,
          2u,
          (__int64)&v67,
          v48,
          *(double *)v16.m128_u64,
          *(double *)v17.m128i_i64,
          a7);
  if ( v67 )
    sub_161E7C0((__int64)&v67, v67);
  return v50;
}
