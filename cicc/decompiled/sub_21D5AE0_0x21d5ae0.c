// Function: sub_21D5AE0
// Address: 0x21d5ae0
//
__int64 *__fastcall sub_21D5AE0(__int64 a1, __int64 a2, unsigned int a3, __int64 *a4, double a5, double a6, __m128i a7)
{
  __int64 v9; // rdx
  char v10; // al
  const void **v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rsi
  const __m128i *v14; // rax
  __m128 v15; // xmm0
  __m128i v16; // xmm1
  __int64 v17; // r12
  unsigned __int64 v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int128 v21; // rax
  __int64 *v22; // rax
  unsigned __int64 v23; // rdx
  __int128 v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 *v28; // r12
  __int128 v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int128 v32; // rax
  __int64 v33; // r12
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 *v41; // rax
  __int16 *v42; // rdx
  __int16 *v43; // r13
  unsigned __int64 v44; // r12
  __m128i v45; // rax
  __int64 *v46; // rax
  int v47; // r8d
  __int64 v48; // rdx
  __int64 *v49; // r12
  __int64 *v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 *v54; // r12
  __m128i v55; // rax
  __int128 v56; // [rsp-30h] [rbp-E0h]
  __int128 v57; // [rsp-20h] [rbp-D0h]
  __int128 v58; // [rsp-10h] [rbp-C0h]
  __int128 v59; // [rsp+0h] [rbp-B0h]
  __int128 v60; // [rsp+0h] [rbp-B0h]
  __int64 v61; // [rsp+10h] [rbp-A0h]
  __int64 v62; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v63; // [rsp+18h] [rbp-98h]
  __int64 v64; // [rsp+18h] [rbp-98h]
  __m128i v65; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v66; // [rsp+40h] [rbp-70h] BYREF
  const void **v67; // [rsp+48h] [rbp-68h]
  __int64 v68; // [rsp+50h] [rbp-60h] BYREF
  int v69; // [rsp+58h] [rbp-58h]
  __m128i v70; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v71; // [rsp+70h] [rbp-40h]
  __int64 v72; // [rsp+78h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v10 = *(_BYTE *)v9;
  v11 = *(const void ***)(v9 + 8);
  LOBYTE(v66) = v10;
  v67 = v11;
  if ( v10 )
    v12 = (unsigned int)sub_1F3E310(&v66);
  else
    v12 = (unsigned int)sub_1F58D40((__int64)&v66);
  v13 = *(_QWORD *)(a2 + 72);
  v68 = v13;
  if ( v13 )
    sub_1623A60((__int64)&v68, v13, 2);
  v69 = *(_DWORD *)(a2 + 64);
  v14 = *(const __m128i **)(a2 + 32);
  v15 = (__m128)_mm_loadu_si128(v14);
  v16 = _mm_loadu_si128(v14 + 5);
  v17 = v14[2].m128i_i64[1];
  v18 = v14[3].m128i_u64[0];
  v65 = v16;
  if ( (_DWORD)v12 == 32 && *(_DWORD *)(*(_QWORD *)(a1 + 81552) + 252LL) > 0x22u )
  {
    *((_QWORD *)&v57 + 1) = v18;
    *(_QWORD *)&v57 = v17;
    v51 = sub_1D3A900(
            a4,
            0x125u,
            (__int64)&v68,
            v66,
            v67,
            0,
            v15,
            *(double *)v16.m128i_i64,
            a7,
            v15.m128_u64[0],
            (__int16 *)v15.m128_u64[1],
            v57,
            v65.m128i_i64[0],
            v65.m128i_i64[1]);
    v53 = v52;
    v54 = v51;
    v55.m128i_i64[0] = (__int64)sub_1D332F0(
                                  a4,
                                  122,
                                  (__int64)&v68,
                                  v66,
                                  v67,
                                  0,
                                  *(double *)v15.m128_u64,
                                  *(double *)v16.m128i_i64,
                                  a7,
                                  v15.m128_i64[0],
                                  v15.m128_u64[1],
                                  *(_OWORD *)&v65);
    v71 = v54;
    v70 = v55;
    v72 = v53;
  }
  else
  {
    v19 = sub_1D38BB0((__int64)a4, v12, (__int64)&v68, 5, 0, 0, (__m128i)v15, *(double *)v16.m128i_i64, a7, 0);
    *(_QWORD *)&v21 = sub_1D332F0(
                        a4,
                        53,
                        (__int64)&v68,
                        5,
                        0,
                        0,
                        *(double *)v15.m128_u64,
                        *(double *)v16.m128i_i64,
                        a7,
                        v19,
                        v20,
                        *(_OWORD *)&v65);
    v59 = v21;
    v22 = sub_1D332F0(
            a4,
            122,
            (__int64)&v68,
            v66,
            v67,
            0,
            *(double *)v15.m128_u64,
            *(double *)v16.m128i_i64,
            a7,
            v17,
            v18,
            *(_OWORD *)&v65);
    v63 = v23;
    v61 = (__int64)v22;
    *(_QWORD *)&v24 = sub_1D38BB0(
                        (__int64)a4,
                        v12,
                        (__int64)&v68,
                        5,
                        0,
                        0,
                        (__m128i)v15,
                        *(double *)v16.m128i_i64,
                        a7,
                        0);
    v25 = sub_1D332F0(
            a4,
            53,
            (__int64)&v68,
            5,
            0,
            0,
            *(double *)v15.m128_u64,
            *(double *)v16.m128i_i64,
            a7,
            v65.m128i_i64[0],
            v65.m128i_u64[1],
            v24);
    v27 = v26;
    v28 = v25;
    *(_QWORD *)&v29 = sub_1D332F0(
                        a4,
                        124,
                        (__int64)&v68,
                        v66,
                        v67,
                        0,
                        *(double *)v15.m128_u64,
                        *(double *)v16.m128i_i64,
                        a7,
                        v15.m128_i64[0],
                        v15.m128_u64[1],
                        v59);
    v30 = sub_1D332F0(
            a4,
            119,
            (__int64)&v68,
            v66,
            v67,
            0,
            *(double *)v15.m128_u64,
            *(double *)v16.m128i_i64,
            a7,
            v61,
            v63,
            v29);
    *((_QWORD *)&v58 + 1) = v27;
    *(_QWORD *)&v58 = v28;
    v64 = v31;
    v62 = (__int64)v30;
    *(_QWORD *)&v32 = sub_1D332F0(
                        a4,
                        122,
                        (__int64)&v68,
                        v66,
                        v67,
                        0,
                        *(double *)v15.m128_u64,
                        *(double *)v16.m128i_i64,
                        a7,
                        v15.m128_i64[0],
                        v15.m128_u64[1],
                        v58);
    v60 = v32;
    v33 = sub_1D38BB0((__int64)a4, v12, (__int64)&v68, 5, 0, 0, (__m128i)v15, *(double *)v16.m128i_i64, a7, 0);
    v35 = v34;
    v39 = sub_1D28D50(a4, 0x13u, v34, v36, v37, v38);
    *((_QWORD *)&v56 + 1) = v35;
    *(_QWORD *)&v56 = v33;
    v41 = sub_1D3A900(
            a4,
            0x89u,
            (__int64)&v68,
            2u,
            0,
            0,
            v15,
            *(double *)v16.m128i_i64,
            a7,
            v65.m128i_u64[0],
            (__int16 *)v65.m128i_i64[1],
            v56,
            v39,
            v40);
    v43 = v42;
    v44 = (unsigned __int64)v41;
    v45.m128i_i64[0] = (__int64)sub_1D332F0(
                                  a4,
                                  122,
                                  (__int64)&v68,
                                  v66,
                                  v67,
                                  0,
                                  *(double *)v15.m128_u64,
                                  *(double *)v16.m128i_i64,
                                  a7,
                                  v15.m128_i64[0],
                                  v15.m128_u64[1],
                                  *(_OWORD *)&v65);
    v65 = v45;
    v46 = sub_1D3A900(a4, 0x86u, (__int64)&v68, v66, v67, 0, v15, *(double *)v16.m128i_i64, a7, v44, v43, v60, v62, v64);
    a7 = _mm_load_si128(&v65);
    v71 = v46;
    v72 = v48;
    v70 = a7;
  }
  v49 = sub_1D37190(
          (__int64)a4,
          (__int64)&v70,
          2u,
          (__int64)&v68,
          v47,
          *(double *)v15.m128_u64,
          *(double *)v16.m128i_i64,
          a7);
  if ( v68 )
    sub_161E7C0((__int64)&v68, v68);
  return v49;
}
