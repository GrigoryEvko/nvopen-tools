// Function: sub_1F95190
// Address: 0x1f95190
//
__int64 *__fastcall sub_1F95190(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *result; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r13
  bool v11; // zf
  char *v12; // rax
  __int64 v13; // rsi
  char v14; // dl
  __int64 *v15; // rax
  __int64 *v16; // r13
  __m128i v17; // xmm0
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rax
  __m128i v21; // xmm2
  __m128i v22; // xmm1
  __int64 *v23; // rax
  __int64 *v24; // rsi
  char v25; // dl
  __int64 v26; // rax
  __int16 v27; // cx
  const __m128i *v28; // rax
  __int64 *v29; // r13
  __int32 v30; // edx
  __int64 v31; // rdi
  __int32 v32; // edx
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __m128i v36; // xmm5
  __int64 v37; // rcx
  __int64 v38; // r9
  __int64 v39; // r13
  int v40; // edx
  __int64 v41; // rcx
  int v42; // r9d
  __int64 v43; // rax
  __int64 *v44; // rdi
  __int64 v45; // rax
  int v46; // edx
  unsigned int v47; // edx
  __int64 v48; // rax
  int v49; // edx
  unsigned int v50; // edx
  __int64 v51; // r15
  unsigned int v52; // edx
  unsigned __int64 v53; // r13
  __m128i v54; // rax
  __int64 *v55; // rdi
  int v56; // r8d
  int v57; // eax
  __int128 v58; // [rsp-2B8h] [rbp-2B8h]
  __int128 v59; // [rsp-2B8h] [rbp-2B8h]
  int v60; // [rsp-2A0h] [rbp-2A0h]
  __int64 v61; // [rsp-298h] [rbp-298h]
  __int64 v62; // [rsp-290h] [rbp-290h]
  unsigned __int32 v63; // [rsp-288h] [rbp-288h]
  __int64 v64; // [rsp-280h] [rbp-280h]
  unsigned __int32 v65; // [rsp-278h] [rbp-278h]
  __int64 v66; // [rsp-270h] [rbp-270h]
  __int64 v67; // [rsp-268h] [rbp-268h]
  __int64 v68; // [rsp-260h] [rbp-260h]
  __int64 v69; // [rsp-258h] [rbp-258h]
  __int64 v70; // [rsp-250h] [rbp-250h]
  __int64 v71; // [rsp-248h] [rbp-248h]
  __int64 v72; // [rsp-240h] [rbp-240h]
  __int64 v73; // [rsp-238h] [rbp-238h]
  __int64 v74; // [rsp-230h] [rbp-230h]
  __int64 v75; // [rsp-228h] [rbp-228h]
  __int64 *v76; // [rsp-228h] [rbp-228h]
  unsigned __int64 v77; // [rsp-220h] [rbp-220h]
  __int64 v78; // [rsp-218h] [rbp-218h]
  __int64 v79; // [rsp-208h] [rbp-208h]
  __int64 v80; // [rsp-200h] [rbp-200h]
  unsigned int v81; // [rsp-1F8h] [rbp-1F8h]
  __int64 v82; // [rsp-1F8h] [rbp-1F8h]
  __int64 v83; // [rsp-1F0h] [rbp-1F0h]
  __int32 v84; // [rsp-1E4h] [rbp-1E4h]
  __int32 v85; // [rsp-1E0h] [rbp-1E0h]
  __int64 *v86; // [rsp-1E0h] [rbp-1E0h]
  __int64 v87; // [rsp-1D8h] [rbp-1D8h]
  __int64 *v88; // [rsp-1D8h] [rbp-1D8h]
  __int64 v89; // [rsp-1C8h] [rbp-1C8h]
  __int64 *v90; // [rsp-1C0h] [rbp-1C0h]
  __int64 *v91; // [rsp-1B8h] [rbp-1B8h]
  __int64 v92; // [rsp-188h] [rbp-188h] BYREF
  int v93; // [rsp-180h] [rbp-180h]
  __int64 v94; // [rsp-178h] [rbp-178h] BYREF
  const void **v95; // [rsp-170h] [rbp-170h]
  __m128i v96; // [rsp-168h] [rbp-168h] BYREF
  _QWORD v97[2]; // [rsp-158h] [rbp-158h] BYREF
  __m128i v98; // [rsp-148h] [rbp-148h] BYREF
  __m128i v99; // [rsp-138h] [rbp-138h] BYREF
  __m128i v100; // [rsp-128h] [rbp-128h] BYREF
  __m128i v101; // [rsp-118h] [rbp-118h] BYREF
  __int64 *v102; // [rsp-108h] [rbp-108h]
  unsigned __int64 v103; // [rsp-100h] [rbp-100h]
  __int64 v104; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v105; // [rsp-F0h] [rbp-F0h]
  __int64 v106; // [rsp-E8h] [rbp-E8h]
  __int64 v107; // [rsp-E0h] [rbp-E0h]
  __int64 v108; // [rsp-D8h] [rbp-D8h]
  __int64 v109; // [rsp-D0h] [rbp-D0h]
  __int64 v110; // [rsp-C8h] [rbp-C8h]
  __int32 v111; // [rsp-C0h] [rbp-C0h]
  __int64 v112; // [rsp-B8h] [rbp-B8h]
  __int64 v113; // [rsp-B0h] [rbp-B0h]
  __int64 v114; // [rsp-A8h] [rbp-A8h]
  __int32 v115; // [rsp-A0h] [rbp-A0h]
  __m128i v116; // [rsp-98h] [rbp-98h] BYREF
  __m128i v117; // [rsp-88h] [rbp-88h] BYREF
  __int64 v118; // [rsp-78h] [rbp-78h]
  __int64 v119; // [rsp-70h] [rbp-70h]
  __int64 v120; // [rsp-68h] [rbp-68h]
  __int32 v121; // [rsp-60h] [rbp-60h]
  __int64 v122; // [rsp-58h] [rbp-58h]
  __int64 v123; // [rsp-50h] [rbp-50h]
  __int64 v124; // [rsp-48h] [rbp-48h]
  __int32 v125; // [rsp-40h] [rbp-40h]
  __int64 v126; // [rsp-8h] [rbp-8h] BYREF

  if ( *(int *)(a1 + 16) > 0 )
    return 0;
  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = *(_QWORD *)(v8 + 80);
  v92 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v92, v9, 2);
  v11 = *(_WORD *)(v10 + 24) == 137;
  v93 = *(_DWORD *)(a2 + 64);
  if ( v11
    && (v12 = *(char **)(a2 + 40),
        v13 = *(_QWORD *)(a1 + 8),
        v14 = *v12,
        v95 = (const void **)*((_QWORD *)v12 + 1),
        v15 = *(__int64 **)a1,
        LOBYTE(v94) = v14,
        sub_1F40D10((__int64)&v116, v13, v15[6], v94, (__int64)v95),
        v116.m128i_i8[0] == 6) )
  {
    sub_1F6F630((__int64)&v116, v10, *(__int64 **)a1, a3, a4, a5);
    v16 = *(__int64 **)a1;
    v71 = v116.m128i_i64[0];
    v79 = v116.m128i_u32[2];
    v70 = v117.m128i_i64[0];
    v78 = v117.m128i_u32[2];
    v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + 40LL));
    v100.m128i_i8[0] = 0;
    v100.m128i_i64[1] = 0;
    v96 = v17;
    v101.m128i_i8[0] = 0;
    v101.m128i_i64[1] = 0;
    v18 = *(_QWORD *)(v17.m128i_i64[0] + 40) + 16LL * v17.m128i_u32[2];
    v19 = *(_BYTE *)v18;
    v20 = *(_QWORD *)(v18 + 8);
    LOBYTE(v104) = v19;
    v105 = v20;
    sub_1D19A30((__int64)&v116, (__int64)v16, &v104);
    v21 = _mm_loadu_si128(&v117);
    v22 = _mm_loadu_si128(&v116);
    v100 = v22;
    v101 = v21;
    sub_1D40600(
      (__int64)&v116,
      v16,
      (__int64)&v96,
      (__int64)&v92,
      (const void ***)&v100,
      (const void ***)&v101,
      v17,
      *(double *)v22.m128i_i64,
      v21);
    v69 = v116.m128i_i64[0];
    v75 = v116.m128i_u32[2];
    v68 = v117.m128i_i64[0];
    v74 = v117.m128i_u32[2];
    sub_1D19A30((__int64)&v116, *(_QWORD *)a1, &v94);
    v23 = *(__int64 **)(a2 + 32);
    v24 = *(__int64 **)a1;
    v65 = v116.m128i_i32[0];
    v64 = v116.m128i_i64[1];
    v63 = v117.m128i_i32[0];
    v62 = v117.m128i_i64[1];
    v87 = *v23;
    v89 = v23[1];
    v25 = *(_BYTE *)(a2 + 88);
    v97[1] = *(_QWORD *)(a2 + 96);
    v26 = *(_QWORD *)(a2 + 104);
    LOBYTE(v97[0]) = v25;
    v27 = *(_WORD *)(v26 + 34);
    v98.m128i_i8[0] = 0;
    v98.m128i_i64[1] = 0;
    v81 = (unsigned int)(1 << v27) >> 1;
    sub_1D19A30((__int64)&v116, (__int64)v24, v97);
    v28 = *(const __m128i **)(a2 + 32);
    v29 = *(__int64 **)a1;
    v30 = v28[13].m128i_i32[0];
    v31 = v28[12].m128i_i64[1];
    v98 = _mm_loadu_si128(&v116);
    v85 = v30;
    v32 = v28[8].m128i_i32[0];
    v99 = _mm_loadu_si128(v28 + 10);
    v80 = v31;
    v33 = v28[7].m128i_i64[1];
    v84 = v32;
    v100.m128i_i8[0] = 0;
    v100.m128i_i64[1] = 0;
    v101.m128i_i8[0] = 0;
    v101.m128i_i64[1] = 0;
    v34 = *(_QWORD *)(v99.m128i_i64[0] + 40) + 16LL * v99.m128i_u32[2];
    LOBYTE(v32) = *(_BYTE *)v34;
    v35 = *(_QWORD *)(v34 + 8);
    v83 = v33;
    LOBYTE(v104) = v32;
    v105 = v35;
    sub_1D19A30((__int64)&v116, (__int64)v29, &v104);
    v36 = _mm_loadu_si128(&v116);
    v101 = _mm_loadu_si128(&v117);
    v100 = v36;
    sub_1D40600(
      (__int64)&v116,
      v29,
      (__int64)&v99,
      (__int64)&v92,
      (const void ***)&v100,
      (const void ***)&v101,
      v17,
      *(double *)v22.m128i_i64,
      v21);
    v37 = *(_QWORD *)(a2 + 104);
    v38 = *(_QWORD *)(v37 + 64);
    v67 = v116.m128i_i64[0];
    v73 = v116.m128i_u32[2];
    v66 = v117.m128i_i64[0];
    v72 = v117.m128i_u32[2];
    v39 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    v116 = _mm_loadu_si128((const __m128i *)(v37 + 40));
    v117.m128i_i64[0] = *(_QWORD *)(v37 + 56);
    if ( v98.m128i_i8[0] )
    {
      v40 = sub_1F6C8D0(v98.m128i_i8[0]);
    }
    else
    {
      v60 = v38;
      v61 = v37;
      v57 = sub_1F58D40((__int64)&v98);
      v42 = v60;
      v41 = v61;
      v40 = v57;
    }
    v43 = sub_1E0B8E0(
            v39,
            1u,
            (unsigned int)(v40 + 7) >> 3,
            v81,
            (unsigned int)&v126 - 144,
            v42,
            *(_OWORD *)v41,
            *(_QWORD *)(v41 + 16),
            1u,
            0,
            0);
    v44 = *(__int64 **)a1;
    v82 = v43;
    v106 = v69;
    v112 = v67;
    v107 = v75;
    v104 = v87;
    v108 = v71;
    v113 = v73;
    v109 = v79;
    v105 = v89;
    v110 = v83;
    v111 = v84;
    v114 = v80;
    v115 = v85;
    v45 = sub_1D252B0((__int64)v44, v65, v64, 1, 0);
    v76 = sub_1D24AE0(v44, v45, v46, v65, v64, (__int64)&v92, &v104, 6, v82);
    v117.m128i_i64[0] = v68;
    v124 = v80;
    v120 = v83;
    v117.m128i_i64[1] = v74;
    v116.m128i_i64[0] = v87;
    v118 = v70;
    v122 = v66;
    v119 = v78;
    v77 = v47;
    v116.m128i_i64[1] = v89;
    v121 = v84;
    v123 = v72;
    v125 = v85;
    v86 = *(__int64 **)a1;
    v48 = sub_1D252B0(*(_QWORD *)a1, v63, v62, 1, 0);
    v88 = sub_1D24AE0(v86, v48, v49, v63, v62, (__int64)&v92, v116.m128i_i64, 6, v82);
    v51 = v50;
    sub_1F81BC0(a1, (__int64)v76);
    sub_1F81BC0(a1, (__int64)v88);
    *((_QWORD *)&v58 + 1) = 1;
    *(_QWORD *)&v58 = v88;
    v91 = sub_1D332F0(
            *(__int64 **)a1,
            2,
            (__int64)&v92,
            1,
            0,
            0,
            *(double *)v17.m128i_i64,
            *(double *)v22.m128i_i64,
            v21,
            (__int64)v76,
            1u,
            v58);
    v53 = v52 | v89 & 0xFFFFFFFF00000000LL;
    sub_1D44C70(*(_QWORD *)a1, a2, 1, (__int64)v91, v52);
    *((_QWORD *)&v59 + 1) = v51;
    *(_QWORD *)&v59 = v88;
    v54.m128i_i64[0] = (__int64)sub_1D332F0(
                                  *(__int64 **)a1,
                                  107,
                                  (__int64)&v92,
                                  (unsigned int)v94,
                                  v95,
                                  0,
                                  *(double *)v17.m128i_i64,
                                  *(double *)v22.m128i_i64,
                                  v21,
                                  (__int64)v76,
                                  v77,
                                  v59);
    v55 = *(__int64 **)a1;
    v101 = v54;
    v102 = v91;
    v103 = v53;
    result = sub_1D37190(
               (__int64)v55,
               (__int64)&v101,
               2u,
               (__int64)&v92,
               v56,
               *(double *)v17.m128i_i64,
               *(double *)v22.m128i_i64,
               v21);
  }
  else
  {
    result = 0;
  }
  if ( v92 )
  {
    v90 = result;
    sub_161E7C0((__int64)&v92, v92);
    return v90;
  }
  return result;
}
