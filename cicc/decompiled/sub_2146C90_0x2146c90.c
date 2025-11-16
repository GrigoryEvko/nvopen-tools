// Function: sub_2146C90
// Address: 0x2146c90
//
unsigned __int64 __fastcall sub_2146C90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        double a5,
        __m128i a6,
        __m128i a7)
{
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // r8
  __int64 v14; // rax
  __int8 v15; // dl
  __int64 v16; // rax
  bool v17; // al
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rax
  __int8 v22; // dl
  __int64 v23; // rax
  bool v24; // al
  __m128 v25; // xmm0
  __int64 v26; // rax
  __int8 v27; // dl
  __int64 v28; // rax
  const void ***v29; // rax
  int v30; // edx
  __int64 v31; // rdx
  const void ***v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rsi
  unsigned int v35; // edx
  unsigned __int64 result; // rax
  bool v37; // al
  bool v38; // al
  __int64 *v39; // rax
  unsigned int *v40; // r10
  __int64 v41; // rsi
  __int64 v42; // rax
  char v43; // dl
  __int64 v44; // rax
  __m128 *v45; // rdx
  unsigned __int64 v46; // r10
  __int64 v47; // rsi
  __int64 v48; // rax
  char v49; // dl
  __int64 v50; // rax
  __m128i v51; // xmm4
  __int64 v52; // rsi
  char *v53; // rax
  char v54; // dl
  unsigned __int64 v55; // r10
  __int64 v56; // rsi
  __m128i v57; // kr00_16
  unsigned int *v58; // r11
  __int64 v59; // rax
  char v60; // dl
  __int64 v61; // rax
  __m128i v62; // xmm6
  unsigned __int64 v63; // r10
  __int64 v64; // rsi
  __int64 v65; // rcx
  __int64 v66; // rax
  char v67; // dl
  __int64 v68; // rax
  __m128i v69; // xmm5
  unsigned __int64 v70; // r10
  int v71; // edx
  __int64 *v72; // rax
  int v73; // edx
  __int128 v74; // [rsp-20h] [rbp-1F0h]
  unsigned __int64 v75; // [rsp+0h] [rbp-1D0h]
  unsigned __int64 v76; // [rsp+8h] [rbp-1C8h]
  __int64 v77; // [rsp+8h] [rbp-1C8h]
  const void **v78; // [rsp+20h] [rbp-1B0h]
  unsigned __int64 v79; // [rsp+28h] [rbp-1A8h]
  unsigned __int64 v80; // [rsp+40h] [rbp-190h]
  __int128 v81; // [rsp+50h] [rbp-180h]
  __int64 v82; // [rsp+60h] [rbp-170h]
  __int64 v83; // [rsp+60h] [rbp-170h]
  unsigned int *v84; // [rsp+60h] [rbp-170h]
  unsigned __int64 v85; // [rsp+60h] [rbp-170h]
  unsigned __int64 v86; // [rsp+60h] [rbp-170h]
  int v87; // [rsp+60h] [rbp-170h]
  __int16 *v88; // [rsp+68h] [rbp-168h]
  __int64 *v89; // [rsp+70h] [rbp-160h]
  unsigned __int64 v90; // [rsp+70h] [rbp-160h]
  unsigned __int64 v91; // [rsp+70h] [rbp-160h]
  __int64 v92; // [rsp+70h] [rbp-160h]
  unsigned __int64 v93; // [rsp+70h] [rbp-160h]
  __int64 *v94; // [rsp+70h] [rbp-160h]
  __int16 *v95; // [rsp+78h] [rbp-158h]
  int v96; // [rsp+80h] [rbp-150h]
  __int64 *v97; // [rsp+88h] [rbp-148h]
  __int64 *v98; // [rsp+88h] [rbp-148h]
  int v99; // [rsp+A8h] [rbp-128h]
  __int128 v100; // [rsp+B0h] [rbp-120h] BYREF
  __int128 v101; // [rsp+C0h] [rbp-110h] BYREF
  __int64 v102; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v103; // [rsp+D8h] [rbp-F8h]
  __int64 v104; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v105; // [rsp+E8h] [rbp-E8h]
  __int64 *v106; // [rsp+F0h] [rbp-E0h] BYREF
  __int16 *v107; // [rsp+F8h] [rbp-D8h]
  __int64 *v108; // [rsp+100h] [rbp-D0h] BYREF
  __int16 *v109; // [rsp+108h] [rbp-C8h]
  __int64 v110; // [rsp+110h] [rbp-C0h] BYREF
  int v111; // [rsp+118h] [rbp-B8h]
  __m128 v112; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v113; // [rsp+130h] [rbp-A0h] BYREF
  int v114; // [rsp+138h] [rbp-98h]
  __int64 v115; // [rsp+140h] [rbp-90h] BYREF
  int v116; // [rsp+148h] [rbp-88h]
  __m128i v117; // [rsp+150h] [rbp-80h] BYREF
  __m128i v118; // [rsp+160h] [rbp-70h] BYREF
  __int64 v119; // [rsp+170h] [rbp-60h] BYREF
  __int64 v120; // [rsp+178h] [rbp-58h]
  __m128i v121; // [rsp+180h] [rbp-50h] BYREF
  __m128i v122[4]; // [rsp+190h] [rbp-40h] BYREF

  v10 = *(_QWORD *)(a2 + 72);
  *(_QWORD *)&v100 = 0;
  DWORD2(v100) = 0;
  *(_QWORD *)&v101 = 0;
  DWORD2(v101) = 0;
  v102 = 0;
  LODWORD(v103) = 0;
  v104 = 0;
  LODWORD(v105) = 0;
  v106 = 0;
  LODWORD(v107) = 0;
  v108 = 0;
  LODWORD(v109) = 0;
  v110 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v110, v10, 2);
  v111 = *(_DWORD *)(a2 + 64);
  v11 = *(_QWORD *)(a2 + 32);
  v12 = *(_QWORD *)(v11 + 40);
  v13 = *(_QWORD *)(v11 + 48);
  v14 = *(_QWORD *)(v12 + 40) + 16LL * *(unsigned int *)(v11 + 48);
  v15 = *(_BYTE *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v121.m128i_i8[0] = v15;
  v121.m128i_i64[1] = v16;
  if ( v15 )
  {
    if ( (unsigned __int8)(v15 - 14) > 0x5Fu )
    {
      v17 = (unsigned __int8)(v15 - 2) <= 5u;
      goto LABEL_6;
    }
LABEL_18:
    sub_2017DE0(a1, v12, v13, &v100, &v101);
    goto LABEL_8;
  }
  v82 = v13;
  v37 = sub_1F58D20((__int64)&v121);
  v13 = v82;
  if ( v37 )
    goto LABEL_18;
  v17 = sub_1F58CF0((__int64)&v121);
  v13 = v82;
LABEL_6:
  if ( v17 )
    sub_20174B0(a1, v12, v13, &v100, &v101);
  else
    sub_2016B80(a1, v12, v13, &v100, &v101);
LABEL_8:
  v18 = *(_QWORD *)(a2 + 32);
  v19 = *(_QWORD *)(v18 + 80);
  v20 = *(_QWORD *)(v18 + 88);
  v21 = *(_QWORD *)(v19 + 40) + 16LL * *(unsigned int *)(v18 + 88);
  v22 = *(_BYTE *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  v121.m128i_i8[0] = v22;
  v121.m128i_i64[1] = v23;
  if ( v22 )
  {
    if ( (unsigned __int8)(v22 - 14) > 0x5Fu )
    {
      v24 = (unsigned __int8)(v22 - 2) <= 5u;
      goto LABEL_11;
    }
LABEL_29:
    sub_2017DE0(a1, v19, v20, &v102, &v104);
    goto LABEL_13;
  }
  v83 = v20;
  v38 = sub_1F58D20((__int64)&v121);
  v20 = v83;
  if ( v38 )
    goto LABEL_29;
  v24 = sub_1F58CF0((__int64)&v121);
  v20 = v83;
LABEL_11:
  if ( v24 )
    sub_20174B0(a1, v19, v20, &v102, &v104);
  else
    sub_2016B80(a1, v19, v20, &v102, &v104);
LABEL_13:
  v25 = (__m128)_mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 32));
  v112 = v25;
  LODWORD(v109) = v25.m128_i32[2];
  LODWORD(v107) = v25.m128_i32[2];
  v26 = *(_QWORD *)(v25.m128_u64[0] + 40) + 16LL * v25.m128_u32[2];
  v108 = (__int64 *)v25.m128_u64[0];
  v106 = (__int64 *)v25.m128_u64[0];
  v27 = *(_BYTE *)v26;
  v28 = *(_QWORD *)(v26 + 8);
  v121.m128i_i8[0] = v27;
  v121.m128i_i64[1] = v28;
  if ( v27 )
  {
    if ( (unsigned __int8)(v27 - 14) > 0x5Fu )
      goto LABEL_15;
  }
  else if ( !sub_1F58D20((__int64)&v121) )
  {
    goto LABEL_15;
  }
  v39 = sub_203AD40((__int64 *)a1, a2);
  if ( v39 )
  {
    v40 = (unsigned int *)v39[4];
    v117.m128i_i8[0] = 0;
    v117.m128i_i64[1] = 0;
    v41 = *(_QWORD *)(a1 + 8);
    v118.m128i_i8[0] = 0;
    v118.m128i_i64[1] = 0;
    v84 = v40;
    v89 = (__int64 *)v41;
    v42 = *(_QWORD *)(*(_QWORD *)v40 + 40LL) + 16LL * v40[2];
    v43 = *(_BYTE *)v42;
    v44 = *(_QWORD *)(v42 + 8);
    LOBYTE(v119) = v43;
    v120 = v44;
    sub_1D19A30((__int64)&v121, v41, &v119);
    a6 = _mm_loadu_si128(&v121);
    a7 = _mm_loadu_si128(v122);
    v117 = a6;
    v118 = a7;
    v45 = (__m128 *)v84;
LABEL_26:
    sub_1D40600(
      (__int64)&v121,
      v89,
      (__int64)v45,
      (__int64)&v110,
      (const void ***)&v117,
      (const void ***)&v118,
      (__m128i)v25,
      *(double *)a6.m128i_i64,
      a7);
    v106 = (__int64 *)v121.m128i_i64[0];
    LODWORD(v107) = v121.m128i_i32[2];
    v108 = (__int64 *)v122[0].m128i_i64[0];
    LODWORD(v109) = v122[0].m128i_i32[2];
    goto LABEL_15;
  }
  v46 = v112.m128_u64[0];
  v97 = *(__int64 **)(a1 + 8);
  if ( *(_WORD *)(v112.m128_u64[0] + 24) == 137 )
  {
    v52 = *(_QWORD *)(v112.m128_u64[0] + 72);
    v113 = v52;
    if ( v52 )
    {
      v90 = v112.m128_u64[0];
      sub_1623A60((__int64)&v113, v52, 2);
      v46 = v90;
    }
    v91 = v46;
    v114 = *(_DWORD *)(v46 + 64);
    v53 = *(char **)(v46 + 40);
    v54 = *v53;
    v120 = *((_QWORD *)v53 + 1);
    LOBYTE(v119) = v54;
    sub_1D19A30((__int64)&v121, (__int64)v97, &v119);
    v55 = v91;
    v79 = v121.m128i_i64[0];
    v56 = *(_QWORD *)(v91 + 72);
    v78 = (const void **)v121.m128i_i64[1];
    v115 = v56;
    v57 = v122[0];
    if ( v56 )
    {
      sub_1623A60((__int64)&v115, v56, 2);
      v55 = v91;
    }
    v85 = v55;
    v116 = *(_DWORD *)(v55 + 64);
    v58 = *(unsigned int **)(v55 + 32);
    v117.m128i_i8[0] = 0;
    v117.m128i_i64[1] = 0;
    v118.m128i_i8[0] = 0;
    v118.m128i_i64[1] = 0;
    v92 = (__int64)v58;
    v59 = *(_QWORD *)(*(_QWORD *)v58 + 40LL) + 16LL * v58[2];
    v60 = *(_BYTE *)v59;
    v61 = *(_QWORD *)(v59 + 8);
    LOBYTE(v119) = v60;
    v120 = v61;
    sub_1D19A30((__int64)&v121, (__int64)v97, &v119);
    v62 = _mm_loadu_si128(v122);
    v117 = _mm_loadu_si128(&v121);
    v118 = v62;
    sub_1D40600(
      (__int64)&v121,
      v97,
      v92,
      (__int64)&v115,
      (const void ***)&v117,
      (const void ***)&v118,
      (__m128i)v25,
      *(double *)a6.m128i_i64,
      a7);
    v63 = v85;
    if ( v115 )
    {
      sub_161E7C0((__int64)&v115, v115);
      v63 = v85;
    }
    v64 = *(_QWORD *)(v63 + 72);
    v93 = v121.m128i_i64[0];
    v115 = v64;
    v95 = (__int16 *)v121.m128i_u32[2];
    v86 = v122[0].m128i_i64[0];
    v88 = (__int16 *)v122[0].m128i_u32[2];
    if ( v64 )
    {
      v76 = v63;
      sub_1623A60((__int64)&v115, v64, 2);
      v63 = v76;
    }
    v75 = v63;
    v116 = *(_DWORD *)(v63 + 64);
    v65 = *(_QWORD *)(v63 + 32);
    v117.m128i_i8[0] = 0;
    v117.m128i_i64[1] = 0;
    v118.m128i_i8[0] = 0;
    v118.m128i_i64[1] = 0;
    v77 = v65;
    v66 = *(_QWORD *)(*(_QWORD *)(v65 + 40) + 40LL) + 16LL * *(unsigned int *)(v65 + 48);
    v67 = *(_BYTE *)v66;
    v68 = *(_QWORD *)(v66 + 8);
    LOBYTE(v119) = v67;
    v120 = v68;
    sub_1D19A30((__int64)&v121, (__int64)v97, &v119);
    v69 = _mm_loadu_si128(v122);
    v117 = _mm_loadu_si128(&v121);
    v118 = v69;
    sub_1D40600(
      (__int64)&v121,
      v97,
      v77 + 40,
      (__int64)&v115,
      (const void ***)&v117,
      (const void ***)&v118,
      (__m128i)v25,
      *(double *)a6.m128i_i64,
      a7);
    v70 = v75;
    if ( v115 )
    {
      sub_161E7C0((__int64)&v115, v115);
      v70 = v75;
    }
    v80 = v70;
    *(_QWORD *)&v81 = v122[0].m128i_i64[0];
    *((_QWORD *)&v74 + 1) = v121.m128i_u32[2];
    *(_QWORD *)&v74 = v121.m128i_i64[0];
    *((_QWORD *)&v81 + 1) = v122[0].m128i_u32[2];
    v94 = sub_1D3A900(
            v97,
            *(unsigned __int16 *)(v70 + 24),
            (__int64)&v113,
            v79,
            v78,
            0,
            v25,
            *(double *)a6.m128i_i64,
            a7,
            v93,
            v95,
            v74,
            *(_QWORD *)(*(_QWORD *)(v70 + 32) + 80LL),
            *(_QWORD *)(*(_QWORD *)(v70 + 32) + 88LL));
    v96 = v71;
    v72 = sub_1D3A900(
            v97,
            *(unsigned __int16 *)(v80 + 24),
            (__int64)&v113,
            v57.m128i_u64[0],
            (const void **)v57.m128i_i64[1],
            0,
            v25,
            *(double *)a6.m128i_i64,
            a7,
            v86,
            v88,
            v81,
            *(_QWORD *)(*(_QWORD *)(v80 + 32) + 80LL),
            *(_QWORD *)(*(_QWORD *)(v80 + 32) + 88LL));
    if ( v113 )
    {
      v87 = v73;
      v98 = v72;
      sub_161E7C0((__int64)&v113, v113);
      v73 = v87;
      v72 = v98;
    }
    v108 = v72;
    LODWORD(v109) = v73;
    v106 = v94;
    LODWORD(v107) = v96;
  }
  else
  {
    sub_1F40D10(
      (__int64)&v121,
      *(_QWORD *)a1,
      v97[6],
      *(unsigned __int8 *)(*(_QWORD *)(v112.m128_u64[0] + 40) + 16LL * v112.m128_u32[2]),
      *(_QWORD *)(*(_QWORD *)(v112.m128_u64[0] + 40) + 16LL * v112.m128_u32[2] + 8));
    if ( v121.m128i_i8[0] != 6 )
    {
      v117.m128i_i8[0] = 0;
      v117.m128i_i64[1] = 0;
      v47 = *(_QWORD *)(a1 + 8);
      v118.m128i_i8[0] = 0;
      v118.m128i_i64[1] = 0;
      v48 = *(_QWORD *)(v112.m128_u64[0] + 40) + 16LL * v112.m128_u32[2];
      v49 = *(_BYTE *)v48;
      v50 = *(_QWORD *)(v48 + 8);
      v89 = (__int64 *)v47;
      LOBYTE(v119) = v49;
      v120 = v50;
      sub_1D19A30((__int64)&v121, v47, &v119);
      v51 = _mm_loadu_si128(v122);
      v45 = &v112;
      v117 = _mm_loadu_si128(&v121);
      v118 = v51;
      goto LABEL_26;
    }
    sub_2017DE0(a1, v112.m128_u64[0], v112.m128_i64[1], &v106, &v108);
  }
LABEL_15:
  v29 = (const void ***)(*(_QWORD *)(v100 + 40) + 16LL * DWORD2(v100));
  *(_QWORD *)a3 = sub_1D3A900(
                    *(__int64 **)(a1 + 8),
                    *(unsigned __int16 *)(a2 + 24),
                    (__int64)&v110,
                    *(unsigned __int8 *)v29,
                    v29[1],
                    0,
                    v25,
                    *(double *)a6.m128i_i64,
                    a7,
                    (unsigned __int64)v106,
                    v107,
                    v100,
                    v102,
                    v103);
  v99 = v30;
  v31 = v101;
  *(_DWORD *)(a3 + 8) = v99;
  v32 = (const void ***)(*(_QWORD *)(v31 + 40) + 16LL * DWORD2(v101));
  v33 = sub_1D3A900(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v110,
          *(unsigned __int8 *)v32,
          v32[1],
          0,
          v25,
          *(double *)a6.m128i_i64,
          a7,
          (unsigned __int64)v108,
          v109,
          v101,
          v104,
          v105);
  v34 = v110;
  *(_QWORD *)a4 = v33;
  result = v35;
  *(_DWORD *)(a4 + 8) = v35;
  if ( v34 )
    return sub_161E7C0((__int64)&v110, v34);
  return result;
}
