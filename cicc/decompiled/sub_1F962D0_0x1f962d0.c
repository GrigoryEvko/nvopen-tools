// Function: sub_1F962D0
// Address: 0x1f962d0
//
__int64 *__fastcall sub_1F962D0(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *result; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __m128i v10; // xmm0
  __int64 v11; // r13
  bool v12; // zf
  unsigned __int8 *v13; // rax
  __int64 *v14; // rsi
  char *v15; // rax
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rsi
  char v20; // dl
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 *v23; // r14
  __m128i v24; // xmm1
  __int64 v25; // rax
  char v26; // dl
  __int64 v27; // rax
  __m128i v28; // xmm3
  __m128i v29; // xmm2
  __int64 v30; // r14
  int v31; // edi
  __int64 v32; // rdx
  char v33; // cl
  __m128i v34; // xmm5
  __int64 v35; // r15
  __int64 v36; // r9
  __int64 v37; // r10
  int v38; // edx
  int v39; // r9d
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 *v42; // r15
  __int64 v43; // rax
  unsigned __int8 v44; // r14
  __int64 v45; // r13
  __int64 v46; // r8
  __int64 v47; // rax
  int v48; // edx
  __int64 *v49; // r13
  unsigned int v50; // edx
  unsigned __int64 v51; // r15
  unsigned __int8 *v52; // rax
  __int64 v53; // r12
  __int64 v54; // r8
  __int64 v55; // rax
  int v56; // edx
  unsigned int v57; // edx
  int v58; // eax
  __int64 v59; // [rsp-230h] [rbp-230h]
  int v60; // [rsp-228h] [rbp-228h]
  __int64 v61; // [rsp-220h] [rbp-220h]
  __int64 v62; // [rsp-218h] [rbp-218h]
  __int64 v63; // [rsp-210h] [rbp-210h]
  __int64 v64; // [rsp-208h] [rbp-208h]
  __int64 v65; // [rsp-200h] [rbp-200h]
  __int64 v66; // [rsp-200h] [rbp-200h]
  __int64 *v67; // [rsp-1F8h] [rbp-1F8h]
  __int64 v68; // [rsp-1F8h] [rbp-1F8h]
  __int64 v69; // [rsp-1F0h] [rbp-1F0h]
  unsigned __int32 v70; // [rsp-1E8h] [rbp-1E8h]
  __int64 v71; // [rsp-1D8h] [rbp-1D8h]
  unsigned __int32 v72; // [rsp-1CCh] [rbp-1CCh]
  __int64 v73; // [rsp-1C8h] [rbp-1C8h]
  int v74; // [rsp-1C0h] [rbp-1C0h]
  unsigned int v75; // [rsp-1B8h] [rbp-1B8h]
  __int64 v76; // [rsp-1B8h] [rbp-1B8h]
  __int64 v77; // [rsp-1B0h] [rbp-1B0h]
  __int64 v78; // [rsp-1A8h] [rbp-1A8h]
  __int32 v79; // [rsp-198h] [rbp-198h]
  __int64 *v80; // [rsp-198h] [rbp-198h]
  __int128 v81; // [rsp-198h] [rbp-198h]
  __int64 v82; // [rsp-188h] [rbp-188h]
  __int64 v83; // [rsp-188h] [rbp-188h]
  __int64 *v84; // [rsp-180h] [rbp-180h]
  __m128i v85; // [rsp-158h] [rbp-158h] BYREF
  __int64 v86; // [rsp-148h] [rbp-148h] BYREF
  int v87; // [rsp-140h] [rbp-140h]
  _QWORD v88[2]; // [rsp-138h] [rbp-138h] BYREF
  __m128i v89; // [rsp-128h] [rbp-128h] BYREF
  __m128i v90; // [rsp-118h] [rbp-118h] BYREF
  __m128i v91; // [rsp-108h] [rbp-108h] BYREF
  __int64 v92; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v93; // [rsp-F0h] [rbp-F0h]
  __int64 v94; // [rsp-E8h] [rbp-E8h]
  unsigned __int32 v95; // [rsp-E0h] [rbp-E0h]
  __int64 v96; // [rsp-D8h] [rbp-D8h]
  __int64 v97; // [rsp-D0h] [rbp-D0h]
  __int64 v98; // [rsp-C8h] [rbp-C8h]
  int v99; // [rsp-C0h] [rbp-C0h]
  __int64 v100; // [rsp-B8h] [rbp-B8h]
  __int64 v101; // [rsp-B0h] [rbp-B0h]
  __int64 v102; // [rsp-A8h] [rbp-A8h]
  int v103; // [rsp-A0h] [rbp-A0h]
  __m128i v104; // [rsp-98h] [rbp-98h] BYREF
  __m128i v105; // [rsp-88h] [rbp-88h] BYREF
  __int64 v106; // [rsp-78h] [rbp-78h]
  __int64 v107; // [rsp-70h] [rbp-70h]
  __int64 v108; // [rsp-68h] [rbp-68h]
  int v109; // [rsp-60h] [rbp-60h]
  __int64 v110; // [rsp-58h] [rbp-58h]
  __int64 v111; // [rsp-50h] [rbp-50h]
  __int64 v112; // [rsp-48h] [rbp-48h]
  int v113; // [rsp-40h] [rbp-40h]
  __int64 v114; // [rsp-8h] [rbp-8h] BYREF

  if ( *(int *)(a1 + 16) > 0 )
    return 0;
  v8 = *(_QWORD *)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = _mm_loadu_si128((const __m128i *)(v8 + 40));
  v11 = *(_QWORD *)(v8 + 80);
  v86 = v9;
  v85 = v10;
  if ( v9 )
    sub_1623A60((__int64)&v86, v9, 2);
  v12 = *(_WORD *)(v11 + 24) == 137;
  v87 = *(_DWORD *)(a2 + 64);
  if ( v12
    && (v13 = (unsigned __int8 *)(*(_QWORD *)(v85.m128i_i64[0] + 40) + 16LL * v85.m128i_u32[2]),
        sub_1F40D10((__int64)&v104, *(_QWORD *)(a1 + 8), *(_QWORD *)(*(_QWORD *)a1 + 48LL), *v13, *((_QWORD *)v13 + 1)),
        v104.m128i_i8[0] == 6) )
  {
    sub_1F6F630((__int64)&v104, v11, *(__int64 **)a1, *(double *)v10.m128i_i64, a4, a5);
    v14 = *(__int64 **)a1;
    v64 = v104.m128i_i64[0];
    v71 = v104.m128i_u32[2];
    v63 = v105.m128i_i64[0];
    v69 = v105.m128i_u32[2];
    v15 = *(char **)(a2 + 40);
    v16 = *v15;
    v17 = *((_QWORD *)v15 + 1);
    LOBYTE(v92) = v16;
    v93 = v17;
    sub_1D19A30((__int64)&v104, (__int64)v14, &v92);
    v18 = *(_QWORD *)(a2 + 32);
    v19 = *(__int64 **)a1;
    v20 = *(_BYTE *)(a2 + 88);
    v21 = *(_QWORD *)v18;
    LODWORD(v18) = *(_DWORD *)(v18 + 8);
    v89.m128i_i8[0] = 0;
    LOBYTE(v88[0]) = v20;
    v79 = v18;
    v77 = v21;
    v88[1] = *(_QWORD *)(a2 + 96);
    v22 = *(_QWORD *)(a2 + 104);
    v89.m128i_i64[1] = 0;
    v75 = (unsigned int)(1 << *(_WORD *)(v22 + 34)) >> 1;
    sub_1D19A30((__int64)&v104, (__int64)v19, v88);
    v23 = *(__int64 **)a1;
    v90.m128i_i8[0] = 0;
    v24 = _mm_loadu_si128(&v104);
    v90.m128i_i64[1] = 0;
    v91.m128i_i8[0] = 0;
    v91.m128i_i64[1] = 0;
    v25 = *(_QWORD *)(v85.m128i_i64[0] + 40) + 16LL * v85.m128i_u32[2];
    v89 = v24;
    v26 = *(_BYTE *)v25;
    v27 = *(_QWORD *)(v25 + 8);
    LOBYTE(v92) = v26;
    v93 = v27;
    sub_1D19A30((__int64)&v104, (__int64)v23, &v92);
    v28 = _mm_loadu_si128(&v105);
    v29 = _mm_loadu_si128(&v104);
    v90 = v29;
    v91 = v28;
    sub_1D40600(
      (__int64)&v104,
      v23,
      (__int64)&v85,
      (__int64)&v86,
      (const void ***)&v90,
      (const void ***)&v91,
      v10,
      *(double *)v24.m128i_i64,
      v29);
    v30 = v104.m128i_i64[0];
    v70 = v104.m128i_u32[2];
    v65 = *(_QWORD *)(a2 + 32);
    v73 = v105.m128i_i64[0];
    v67 = *(__int64 **)a1;
    v72 = v105.m128i_u32[2];
    v82 = *(_QWORD *)(v65 + 200);
    v74 = *(_DWORD *)(v65 + 208);
    v78 = *(_QWORD *)(v65 + 120);
    v31 = *(_DWORD *)(v65 + 128);
    v90.m128i_i8[0] = 0;
    v90.m128i_i64[1] = 0;
    v91.m128i_i8[0] = 0;
    v91.m128i_i64[1] = 0;
    v32 = *(_QWORD *)(*(_QWORD *)(v65 + 160) + 40LL) + 16LL * *(unsigned int *)(v65 + 168);
    v33 = *(_BYTE *)v32;
    v93 = *(_QWORD *)(v32 + 8);
    LOBYTE(v92) = v33;
    sub_1D19A30((__int64)&v104, (__int64)v67, &v92);
    v34 = _mm_loadu_si128(&v105);
    v90 = _mm_loadu_si128(&v104);
    v91 = v34;
    sub_1D40600(
      (__int64)&v104,
      v67,
      v65 + 160,
      (__int64)&v86,
      (const void ***)&v90,
      (const void ***)&v91,
      v10,
      *(double *)v24.m128i_i64,
      v29);
    v35 = *(_QWORD *)(a2 + 104);
    v62 = v104.m128i_i64[0];
    v36 = *(_QWORD *)(v35 + 64);
    v68 = v104.m128i_u32[2];
    v61 = v105.m128i_i64[0];
    v66 = v105.m128i_u32[2];
    v37 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    v104 = _mm_loadu_si128((const __m128i *)(v35 + 40));
    v105.m128i_i64[0] = *(_QWORD *)(v35 + 56);
    if ( v89.m128i_i8[0] )
    {
      v38 = sub_1F6C8D0(v89.m128i_i8[0]);
    }
    else
    {
      v59 = v37;
      v60 = v36;
      v58 = sub_1F58D40((__int64)&v89);
      v40 = v59;
      v39 = v60;
      v38 = v58;
    }
    v41 = sub_1E0B8E0(
            v40,
            2u,
            (unsigned int)(v38 + 7) >> 3,
            v75,
            (unsigned int)&v114 - 144,
            v39,
            *(_OWORD *)v35,
            *(_QWORD *)(v35 + 16),
            1u,
            0,
            0);
    v42 = *(__int64 **)a1;
    v76 = v41;
    LODWORD(v93) = v79;
    v96 = v64;
    v100 = v62;
    v97 = v71;
    v101 = v68;
    v92 = v77;
    v94 = v30;
    v95 = v70;
    v98 = v78;
    v99 = v31;
    v102 = v82;
    v103 = v74;
    v43 = *(_QWORD *)(v30 + 40) + 16LL * v70;
    v44 = *(_BYTE *)v43;
    v45 = *(_QWORD *)(v43 + 8);
    v47 = sub_1D29190((__int64)v42, 1u, 0, v77, v46, v82);
    v49 = sub_1D24800(v42, v47, v48, v44, v45, (__int64)&v86, &v92, 6, v76);
    v104.m128i_i32[2] = v79;
    v106 = v63;
    v51 = v50;
    v104.m128i_i64[0] = v77;
    v107 = v69;
    v112 = v82;
    v110 = v61;
    v109 = v31;
    v113 = v74;
    v105.m128i_i64[0] = v73;
    v105.m128i_i32[2] = v72;
    v108 = v78;
    v111 = v66;
    v52 = (unsigned __int8 *)(*(_QWORD *)(v73 + 40) + 16LL * v72);
    v53 = *((_QWORD *)v52 + 1);
    v83 = *v52;
    v80 = *(__int64 **)a1;
    v55 = sub_1D29190(*(_QWORD *)a1, 1u, 0, v83, v54, 6);
    *(_QWORD *)&v81 = sub_1D24800(v80, v55, v56, v83, v53, (__int64)&v86, v104.m128i_i64, 6, v76);
    *((_QWORD *)&v81 + 1) = v57;
    sub_1F81BC0(a1, (__int64)v49);
    sub_1F81BC0(a1, v81);
    result = sub_1D332F0(
               *(__int64 **)a1,
               2,
               (__int64)&v86,
               1,
               0,
               0,
               *(double *)v10.m128i_i64,
               *(double *)v24.m128i_i64,
               v29,
               (__int64)v49,
               v51,
               v81);
  }
  else
  {
    result = 0;
  }
  if ( v86 )
  {
    v84 = result;
    sub_161E7C0((__int64)&v86, v86);
    return v84;
  }
  return result;
}
