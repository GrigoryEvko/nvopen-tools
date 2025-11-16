// Function: sub_1F95A00
// Address: 0x1f95a00
//
__int64 *__fastcall sub_1F95A00(int *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *result; // rax
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v10; // r14
  unsigned __int8 *v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rax
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 *v18; // rsi
  char *v19; // rax
  char v20; // dl
  __int64 v21; // rax
  __m128i v22; // kr00_16
  __m128i v23; // kr10_16
  __int64 v24; // rax
  __int64 v25; // rcx
  __int128 v26; // xmm3
  __int64 v27; // rax
  unsigned int v28; // r15d
  char v29; // di
  __int64 v30; // rax
  int v31; // r14d
  unsigned int v32; // eax
  int v33; // r15d
  __int64 *v34; // rsi
  __int64 v35; // rcx
  __m128i v36; // xmm5
  __int64 *v37; // rax
  __int64 v38; // r15
  __int64 v39; // r9
  int v40; // edx
  __int64 v41; // rcx
  int v42; // r9d
  __int64 v43; // rax
  unsigned int v44; // edx
  unsigned int v45; // edx
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // r9
  int v49; // edx
  int v50; // r9d
  unsigned int v51; // edx
  unsigned __int64 v52; // rdi
  int v53; // eax
  __int64 v54; // rax
  __int64 v55; // r14
  unsigned int v56; // edx
  __int64 v57; // r15
  unsigned int v58; // edx
  __m128i v59; // rax
  __int64 *v60; // rdi
  int v61; // r8d
  char v62; // r8
  __int64 v63; // rax
  __int64 v64; // rax
  int v65; // eax
  int v66; // eax
  __int128 v67; // [rsp-1D8h] [rbp-1D8h]
  __int128 v68; // [rsp-1D8h] [rbp-1D8h]
  int v69; // [rsp-1C0h] [rbp-1C0h]
  __int64 v70; // [rsp-1B8h] [rbp-1B8h]
  int v71; // [rsp-190h] [rbp-190h]
  __int64 v72; // [rsp-188h] [rbp-188h]
  __int64 v73; // [rsp-180h] [rbp-180h]
  __int64 v74; // [rsp-170h] [rbp-170h]
  __int128 v75; // [rsp-168h] [rbp-168h]
  __int128 v76; // [rsp-158h] [rbp-158h]
  unsigned __int64 v77; // [rsp-150h] [rbp-150h]
  __int128 v78; // [rsp-148h] [rbp-148h]
  int v79; // [rsp-138h] [rbp-138h]
  unsigned __int8 v80; // [rsp-131h] [rbp-131h]
  const void **v81; // [rsp-130h] [rbp-130h]
  __int64 *v82; // [rsp-128h] [rbp-128h]
  __int128 v83; // [rsp-128h] [rbp-128h]
  __int64 *v84; // [rsp-118h] [rbp-118h]
  __int128 v85; // [rsp-118h] [rbp-118h]
  int v86; // [rsp-118h] [rbp-118h]
  unsigned __int64 v87; // [rsp-118h] [rbp-118h]
  __int64 *v88; // [rsp-108h] [rbp-108h]
  __int64 v89; // [rsp-C8h] [rbp-C8h] BYREF
  int v90; // [rsp-C0h] [rbp-C0h]
  __m128i v91; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD v92[2]; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v93; // [rsp-98h] [rbp-98h] BYREF
  __m128i v94; // [rsp-88h] [rbp-88h] BYREF
  __int128 v95; // [rsp-78h] [rbp-78h] BYREF
  __int64 v96; // [rsp-68h] [rbp-68h]
  __m128i v97; // [rsp-58h] [rbp-58h] BYREF
  __m128i v98[4]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v99; // [rsp-8h] [rbp-8h] BYREF

  if ( a1[4] > 0 )
    return 0;
  if ( *(_WORD *)(a2 + 24) != 235 )
    BUG();
  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v89 = v8;
  v10 = *(_QWORD *)(v7 + 80);
  if ( v8 )
    sub_1623A60((__int64)&v89, v8, 2);
  v90 = *(_DWORD *)(a2 + 64);
  result = 0;
  if ( *(_WORD *)(v10 + 24) == 137 )
  {
    v11 = *(unsigned __int8 **)(a2 + 40);
    v80 = *v11;
    v81 = (const void **)*((_QWORD *)v11 + 1);
    sub_1F40D10((__int64)&v97, *((_QWORD *)a1 + 1), *(_QWORD *)(*(_QWORD *)a1 + 48LL), *v11, (__int64)v81);
    if ( v97.m128i_i8[0] == 6 )
    {
      sub_1F6F630((__int64)&v97, v10, *(__int64 **)a1, a3, a4, a5);
      *(_QWORD *)&v85 = v97.m128i_i64[0];
      v82 = *(__int64 **)a1;
      *((_QWORD *)&v85 + 1) = v97.m128i_u32[2];
      *(_QWORD *)&v78 = v98[0].m128i_i64[0];
      *((_QWORD *)&v78 + 1) = v98[0].m128i_u32[2];
      v12 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 32) + 120LL));
      v93.m128i_i8[0] = 0;
      v93.m128i_i64[1] = 0;
      v91 = v12;
      v94.m128i_i8[0] = 0;
      v94.m128i_i64[1] = 0;
      v13 = *(_QWORD *)(v12.m128i_i64[0] + 40) + 16LL * v12.m128i_u32[2];
      v14 = *(_BYTE *)v13;
      v15 = *(_QWORD *)(v13 + 8);
      LOBYTE(v95) = v14;
      *((_QWORD *)&v95 + 1) = v15;
      sub_1D19A30((__int64)&v97, (__int64)v82, &v95);
      v16 = _mm_loadu_si128(&v97);
      v17 = _mm_loadu_si128(v98);
      v93 = v16;
      v94 = v17;
      sub_1D40600(
        (__int64)&v97,
        v82,
        (__int64)&v91,
        (__int64)&v89,
        (const void ***)&v93,
        (const void ***)&v94,
        v12,
        *(double *)v16.m128i_i64,
        v17);
      v18 = *(__int64 **)a1;
      *(_QWORD *)&v76 = v97.m128i_i64[0];
      *((_QWORD *)&v76 + 1) = v97.m128i_u32[2];
      *(_QWORD *)&v75 = v98[0].m128i_i64[0];
      v19 = *(char **)(a2 + 40);
      v20 = *v19;
      v21 = *((_QWORD *)v19 + 1);
      *((_QWORD *)&v75 + 1) = v98[0].m128i_u32[2];
      LOBYTE(v95) = v20;
      *((_QWORD *)&v95 + 1) = v21;
      sub_1D19A30((__int64)&v97, (__int64)v18, &v95);
      v22 = v97;
      v23 = v98[0];
      v24 = *(_QWORD *)(a2 + 32);
      v25 = *(_QWORD *)v24;
      v26 = (__int128)_mm_loadu_si128((const __m128i *)(v24 + 40));
      LOBYTE(v92[0]) = *(_BYTE *)(a2 + 88);
      v74 = v25;
      v73 = *(_QWORD *)(v24 + 8);
      v92[1] = *(_QWORD *)(a2 + 96);
      v27 = *(_QWORD *)(a2 + 40);
      v28 = 1 << *(_WORD *)(*(_QWORD *)(a2 + 104) + 34LL);
      v29 = *(_BYTE *)v27;
      v30 = *(_QWORD *)(v27 + 8);
      v97.m128i_i8[0] = v29;
      v31 = v28 >> 1;
      v97.m128i_i64[1] = v30;
      if ( v29 )
        v32 = sub_1F6C8D0(v29);
      else
        v32 = sub_1F58D40((__int64)&v97);
      v33 = v28 >> 2;
      v34 = *(__int64 **)a1;
      v94.m128i_i8[0] = 0;
      if ( v32 >> 3 != v31 )
        v33 = v31;
      v93.m128i_i8[0] = 0;
      v93.m128i_i64[1] = 0;
      v79 = v33;
      v94.m128i_i64[1] = 0;
      sub_1D19A30((__int64)&v97, (__int64)v34, v92);
      v35 = *(_QWORD *)(a2 + 104);
      v36 = _mm_loadu_si128(v98);
      v37 = *(__int64 **)a1;
      v93 = _mm_loadu_si128(&v97);
      v94 = v36;
      v38 = v37[4];
      v39 = *(_QWORD *)(v35 + 64);
      v97 = _mm_loadu_si128((const __m128i *)(v35 + 40));
      v98[0].m128i_i64[0] = *(_QWORD *)(v35 + 56);
      if ( v93.m128i_i8[0] )
      {
        v40 = sub_1F6C8D0(v93.m128i_i8[0]);
      }
      else
      {
        v69 = v39;
        v70 = v35;
        v66 = sub_1F58D40((__int64)&v93);
        v42 = v69;
        v41 = v70;
        v40 = v66;
      }
      v43 = sub_1E0B8E0(
              v38,
              1u,
              (unsigned int)(v40 + 7) >> 3,
              v31,
              (unsigned int)&v99 - 80,
              v42,
              *(_OWORD *)v41,
              *(_QWORD *)(v41 + 16),
              1u,
              0,
              0);
      v72 = sub_1D257D0(
              *(_QWORD **)a1,
              v22.m128i_i64[0],
              v22.m128i_i64[1],
              (__int64)&v89,
              v74,
              v73,
              v26,
              v85,
              v76,
              v93.m128i_i64[0],
              v93.m128i_i64[1],
              v43,
              0,
              (*(_BYTE *)(a2 + 27) & 0x10) != 0);
      v77 = v44;
      *(_QWORD *)&v83 = sub_20BCE60(
                          *((_QWORD *)a1 + 1),
                          v26,
                          DWORD2(v26),
                          v85,
                          DWORD2(v85),
                          (unsigned int)&v89,
                          v93.m128i_i8[0],
                          v93.m128i_i64[1],
                          *(_QWORD *)a1,
                          (*(_BYTE *)(a2 + 27) & 0x10) != 0);
      *((_QWORD *)&v83 + 1) = v45 | *((_QWORD *)&v26 + 1) & 0xFFFFFFFF00000000LL;
      if ( v93.m128i_i8[0] )
        v86 = sub_1F6C8D0(v93.m128i_i8[0]);
      else
        v86 = sub_1F58D40((__int64)&v93);
      v46 = *(_QWORD *)(a2 + 104);
      v47 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      v48 = *(_QWORD *)(v46 + 64);
      v97 = _mm_loadu_si128((const __m128i *)(v46 + 40));
      v98[0].m128i_i64[0] = *(_QWORD *)(v46 + 56);
      if ( v94.m128i_i8[0] )
      {
        v49 = sub_1F6C8D0(v94.m128i_i8[0]);
      }
      else
      {
        v71 = v48;
        v65 = sub_1F58D40((__int64)&v94);
        v50 = v71;
        v49 = v65;
      }
      v51 = (unsigned int)(v49 + 7) >> 3;
      v52 = *(_QWORD *)v46 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v52 )
      {
        v62 = *(_BYTE *)(v46 + 16);
        v63 = *(_QWORD *)(v46 + 8) + ((unsigned int)(v86 + 7) >> 3);
        if ( (*(_QWORD *)v46 & 4) != 0 )
        {
          *((_QWORD *)&v95 + 1) = *(_QWORD *)(v46 + 8) + ((unsigned int)(v86 + 7) >> 3);
          LOBYTE(v96) = v62;
          *(_QWORD *)&v95 = v52 | 4;
          HIDWORD(v96) = *(_DWORD *)(v52 + 12);
        }
        else
        {
          *(_QWORD *)&v95 = *(_QWORD *)v46 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v95 + 1) = v63;
          LOBYTE(v96) = v62;
          v64 = *(_QWORD *)v52;
          if ( *(_BYTE *)(*(_QWORD *)v52 + 8LL) == 16 )
            v64 = **(_QWORD **)(v64 + 16);
          HIDWORD(v96) = *(_DWORD *)(v64 + 8) >> 8;
        }
      }
      else
      {
        v53 = *(_DWORD *)(v46 + 20);
        LODWORD(v96) = 0;
        v95 = 0u;
        HIDWORD(v96) = v53;
      }
      v54 = sub_1E0B8E0(v47, 1u, v51, v79, (unsigned int)&v99 - 80, v50, v95, v96, 1u, 0, 0);
      v55 = sub_1D257D0(
              *(_QWORD **)a1,
              v23.m128i_i64[0],
              v23.m128i_i64[1],
              (__int64)&v89,
              v74,
              v73,
              v83,
              v78,
              v75,
              v94.m128i_i64[0],
              v94.m128i_i64[1],
              v54,
              0,
              (*(_BYTE *)(a2 + 27) & 0x10) != 0);
      v57 = v56;
      sub_1F81BC0((__int64)a1, v72);
      sub_1F81BC0((__int64)a1, v55);
      *((_QWORD *)&v67 + 1) = 1;
      *(_QWORD *)&v67 = v55;
      v88 = sub_1D332F0(
              *(__int64 **)a1,
              2,
              (__int64)&v89,
              1,
              0,
              0,
              *(double *)v12.m128i_i64,
              *(double *)v16.m128i_i64,
              v17,
              v72,
              1u,
              v67);
      v87 = v58 | v73 & 0xFFFFFFFF00000000LL;
      sub_1D44C70(*(_QWORD *)a1, a2, 1, (__int64)v88, v58);
      *((_QWORD *)&v68 + 1) = v57;
      *(_QWORD *)&v68 = v55;
      v59.m128i_i64[0] = (__int64)sub_1D332F0(
                                    *(__int64 **)a1,
                                    107,
                                    (__int64)&v89,
                                    v80,
                                    v81,
                                    0,
                                    *(double *)v12.m128i_i64,
                                    *(double *)v16.m128i_i64,
                                    v17,
                                    v72,
                                    v77,
                                    v68);
      v60 = *(__int64 **)a1;
      v97 = v59;
      v98[0].m128i_i64[0] = (__int64)v88;
      v98[0].m128i_i64[1] = v87;
      result = sub_1D37190(
                 (__int64)v60,
                 (__int64)&v97,
                 2u,
                 (__int64)&v89,
                 v61,
                 *(double *)v12.m128i_i64,
                 *(double *)v16.m128i_i64,
                 v17);
    }
    else
    {
      result = 0;
    }
  }
  if ( v89 )
  {
    v84 = result;
    sub_161E7C0((__int64)&v89, v89);
    return v84;
  }
  return result;
}
