// Function: sub_1F806E0
// Address: 0x1f806e0
//
__int64 *__fastcall sub_1F806E0(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        char a10)
{
  __int64 v11; // rbx
  unsigned __int8 *v12; // rdx
  __int64 v13; // rax
  const void **v14; // rdx
  __int64 v15; // rdx
  unsigned int v19; // r10d
  __int64 v20; // r11
  __int64 v21; // rdx
  char v22; // cl
  char v23; // al
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // r11
  __int64 *v27; // rax
  __int32 v28; // edx
  __int64 v29; // rsi
  __int64 *v30; // rax
  __int32 v31; // edx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // rsi
  __int64 *result; // rax
  unsigned int *v37; // rcx
  unsigned int *v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // r14
  unsigned __int8 *v44; // rbx
  const void **v45; // r8
  __int64 *v46; // r12
  __int64 v47; // rcx
  __int64 v48; // rcx
  __int64 v49; // rdx
  __m128i v50; // xmm0
  __int32 v51; // eax
  int v52; // r8d
  __m128i v53; // xmm2
  __m128i v54; // xmm1
  __int64 v55; // rax
  __int64 v56; // r14
  unsigned __int64 v57; // rbx
  __int128 v58; // rax
  char v59; // al
  __int64 v60; // r11
  char v61; // bl
  __int128 *v62; // rax
  __int64 v63; // rsi
  unsigned __int64 v64; // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // rdx
  __int64 v67; // r11
  __int64 v68; // r14
  unsigned __int64 v69; // r15
  __int128 v70; // rax
  __int64 *v71; // r13
  __int128 v72; // rax
  __int64 *v73; // rax
  unsigned int v74; // edx
  __int128 v75; // rax
  __int64 *v76; // r13
  __int128 v77; // rax
  __int64 *v78; // rax
  unsigned int v79; // edx
  __int64 v80; // r9
  int v81; // edx
  int v82; // r10d
  __int64 v83; // rbx
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // [rsp-20h] [rbp-180h]
  __int128 v87; // [rsp-10h] [rbp-170h]
  bool v88; // [rsp+4h] [rbp-15Ch]
  int v89; // [rsp+4h] [rbp-15Ch]
  int v90; // [rsp+4h] [rbp-15Ch]
  __int64 v91; // [rsp+8h] [rbp-158h]
  unsigned __int64 v92; // [rsp+8h] [rbp-158h]
  __int64 v94; // [rsp+10h] [rbp-150h]
  unsigned int v95; // [rsp+20h] [rbp-140h]
  __int64 v96; // [rsp+20h] [rbp-140h]
  __int64 v98; // [rsp+20h] [rbp-140h]
  __int64 v99; // [rsp+20h] [rbp-140h]
  __int64 v100; // [rsp+20h] [rbp-140h]
  __int64 v101; // [rsp+20h] [rbp-140h]
  unsigned __int64 v102; // [rsp+28h] [rbp-138h]
  const void **v103; // [rsp+30h] [rbp-130h]
  __int64 *v104; // [rsp+30h] [rbp-130h]
  __int64 v105; // [rsp+30h] [rbp-130h]
  __int64 v106; // [rsp+30h] [rbp-130h]
  __int64 v107; // [rsp+30h] [rbp-130h]
  __int64 v108; // [rsp+30h] [rbp-130h]
  __int64 v109; // [rsp+30h] [rbp-130h]
  unsigned __int64 v110; // [rsp+38h] [rbp-128h]
  unsigned int v111; // [rsp+A0h] [rbp-C0h] BYREF
  const void **v112; // [rsp+A8h] [rbp-B8h]
  __m128i v113; // [rsp+B0h] [rbp-B0h] BYREF
  __m128i v114; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v115; // [rsp+D0h] [rbp-90h] BYREF
  __int32 v116; // [rsp+D8h] [rbp-88h]
  __int64 v117; // [rsp+E0h] [rbp-80h] BYREF
  unsigned __int64 v118; // [rsp+E8h] [rbp-78h]
  __m128i v119; // [rsp+F0h] [rbp-70h] BYREF
  __int128 v120; // [rsp+100h] [rbp-60h] BYREF
  __int64 v121; // [rsp+110h] [rbp-50h] BYREF
  int v122; // [rsp+118h] [rbp-48h]
  __int64 (__fastcall *v123)(__int64 *, __int64 *, int); // [rsp+120h] [rbp-40h]
  __int64 (__fastcall *v124)(unsigned int *, __int64, __int64); // [rsp+128h] [rbp-38h]

  v11 = 16LL * a3;
  v12 = (unsigned __int8 *)(v11 + *(_QWORD *)(a2 + 40));
  v13 = *v12;
  v14 = (const void **)*((_QWORD *)v12 + 1);
  LOBYTE(v111) = v13;
  v112 = v14;
  if ( !(_BYTE)v13 )
    return 0;
  v15 = *(_QWORD *)(a1 + 8);
  if ( !*(_QWORD *)(v15 + 8 * v13 + 120) )
    return 0;
  v19 = a5;
  v20 = a6;
  v21 = 259LL * (unsigned __int8)v13 + v15;
  v22 = *(_BYTE *)(v21 + 2547);
  v23 = *(_BYTE *)(v21 + 2548);
  if ( *(_BYTE *)(a1 + 24) )
  {
    v88 = v22 == 0;
    if ( !v23 )
      goto LABEL_6;
  }
  else
  {
    v88 = (v22 & 0xFB) == 0;
    if ( (v23 & 0xFB) == 0 )
      goto LABEL_6;
  }
  if ( !v88 )
    return 0;
LABEL_6:
  if ( *(_WORD *)(a2 + 24) == 145 && *(_WORD *)(a4 + 24) == 145 )
  {
    v37 = *(unsigned int **)(a4 + 32);
    v38 = *(unsigned int **)(a2 + 32);
    v39 = *(_QWORD *)(*(_QWORD *)v37 + 40LL) + 16LL * v37[2];
    v40 = *(_QWORD *)(*(_QWORD *)v38 + 40LL) + 16LL * v38[2];
    if ( *(_BYTE *)v39 == *(_BYTE *)v40 && (*(_QWORD *)(v39 + 8) == *(_QWORD *)(v40 + 8) || *(_BYTE *)v40) )
    {
      v41 = sub_1F806E0(a1, *(_QWORD *)v38, *((_QWORD *)v38 + 1), *(_QWORD *)v37, *((_QWORD *)v37 + 1), a6, 0);
      v20 = a6;
      v19 = a5;
      if ( v41 )
      {
        v42 = *(_QWORD *)(a2 + 72);
        v43 = v41;
        v44 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + v11);
        v45 = (const void **)*((_QWORD *)v44 + 1);
        v46 = *(__int64 **)a1;
        v47 = *v44;
        v121 = v42;
        if ( v42 )
        {
          v98 = v47;
          v103 = v45;
          sub_1623A60((__int64)&v121, v42, 2);
          v47 = v98;
          v45 = v103;
        }
        v122 = *(_DWORD *)(a2 + 64);
        result = (__int64 *)sub_1D309E0(
                              v46,
                              145,
                              (__int64)&v121,
                              v47,
                              v45,
                              0,
                              *(double *)a7.m128i_i64,
                              a8,
                              *(double *)a9.m128i_i64,
                              (unsigned __int64)v43);
        if ( v121 )
        {
          v104 = result;
          sub_161E7C0((__int64)&v121, v121);
          return v104;
        }
        return result;
      }
    }
  }
  v24 = *(_QWORD *)a1;
  v91 = v20;
  v95 = v19;
  v113.m128i_i64[0] = 0;
  v113.m128i_i32[2] = 0;
  v114.m128i_i64[0] = 0;
  v114.m128i_i32[2] = 0;
  sub_1F6CCF0(v24, a2, a3, (__int64)&v113, (__int64)&v114);
  v25 = *(_QWORD *)a1;
  v115 = 0;
  v117 = 0;
  v116 = 0;
  LODWORD(v118) = 0;
  sub_1F6CCF0(v25, a4, v95, (__int64)&v115, (__int64)&v117);
  v26 = v91;
  if ( v113.m128i_i64[0] )
  {
    v27 = sub_1F7FF10(*(__int64 **)a1, v113.m128i_i64[0], a4, v95, (__int64)&v117, v91, *(double *)a7.m128i_i64, a8, a9);
    v26 = v91;
    if ( v27 )
    {
      v115 = (__int64)v27;
      v116 = v28;
    }
    v29 = v115;
    if ( !v115 )
      return 0;
  }
  else
  {
    v29 = v115;
    if ( !v115 )
      return 0;
  }
  v96 = v26;
  v30 = sub_1F7FF10(*(__int64 **)a1, v29, a2, a3, (__int64)&v114, v26, *(double *)a7.m128i_i64, a8, a9);
  if ( v30 )
  {
    v113.m128i_i64[0] = (__int64)v30;
    v113.m128i_i32[2] = v31;
  }
  if ( !v115 )
    return 0;
  if ( !v113.m128i_i64[0] )
    return 0;
  v34 = *(_QWORD *)(v115 + 32);
  v35 = *(_QWORD *)(v113.m128i_i64[0] + 32);
  if ( *(_QWORD *)v35 != *(_QWORD *)v34 )
    return 0;
  if ( *(_DWORD *)(v35 + 8) != *(_DWORD *)(v34 + 8) )
    return 0;
  v48 = *(unsigned __int16 *)(v115 + 24);
  v49 = *(unsigned __int16 *)(v113.m128i_i64[0] + 24);
  if ( (_DWORD)v48 == (_DWORD)v49 )
    return 0;
  if ( (_DWORD)v48 == 122 )
  {
    v50 = _mm_loadu_si128(&v113);
    v113.m128i_i64[0] = v115;
    v113.m128i_i32[2] = v116;
    v51 = v50.m128i_i32[2];
    v115 = v50.m128i_i64[0];
    a7 = _mm_loadu_si128(&v114);
    v116 = v51;
    v114.m128i_i64[0] = v117;
    v117 = a7.m128i_i64[0];
    v114.m128i_i32[2] = v118;
    LODWORD(v118) = a7.m128i_i32[2];
  }
  v52 = sub_1D159C0((__int64)&v111, v35, v49, v48, v32, v33);
  v53 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v113.m128i_i64[0] + 32));
  v54 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v113.m128i_i64[0] + 32) + 40LL));
  v119 = v54;
  v55 = *(_QWORD *)(v115 + 32);
  v56 = *(_QWORD *)v55;
  v57 = *(_QWORD *)(v55 + 8);
  v58 = *(_OWORD *)(v55 + 40);
  LODWORD(v121) = v52;
  v92 = v57;
  v124 = sub_1F6DFB0;
  v123 = (__int64 (__fastcall *)(__int64 *, __int64 *, int))sub_1F6C090;
  v120 = v58;
  v59 = sub_1D16BF0(v54.m128i_i64[0], v54.m128i_u32[2], v58, DWORD2(v58), (__int64)&v121);
  v60 = v96;
  v61 = v59;
  if ( v123 )
  {
    v123(&v121, &v121, 3);
    v60 = v96;
  }
  if ( !v61 )
  {
    if ( !v114.m128i_i64[0] && !v117 )
    {
      v80 = v119.m128i_i64[0];
      v81 = v119.m128i_i32[2];
      v82 = DWORD2(v120);
      v83 = v120;
      if ( (unsigned __int16)(*(_WORD *)(v119.m128i_i64[0] + 24) - 142) <= 3u
        && (unsigned __int16)(*(_WORD *)(v120 + 24) - 142) <= 3u )
      {
        v84 = *(_QWORD *)(v119.m128i_i64[0] + 32);
        v81 = *(_DWORD *)(v84 + 8);
        v80 = *(_QWORD *)v84;
        v85 = *(_QWORD *)(v120 + 32);
        v82 = *(_DWORD *)(v85 + 8);
        v83 = *(_QWORD *)v85;
      }
      if ( !a10 )
      {
        v90 = v82;
        v101 = v60;
        v109 = v80;
        result = sub_1F75B00(
                   (__int64 **)a1,
                   v53.m128i_i64[0],
                   v53.m128i_u64[1],
                   v119.m128i_i64[0],
                   v119.m128i_i64[1],
                   v80,
                   *(double *)a7.m128i_i64,
                   *(double *)v54.m128i_i64,
                   v53,
                   v120,
                   SDWORD2(v120),
                   v81,
                   v83,
                   0x7Du,
                   0x7Eu,
                   v60);
        v80 = v109;
        v60 = v101;
        v82 = v90;
        if ( result )
          return result;
        return sub_1F75B00(
                 (__int64 **)a1,
                 v56,
                 v92,
                 v120,
                 *((__int64 *)&v120 + 1),
                 v83,
                 *(double *)a7.m128i_i64,
                 *(double *)v54.m128i_i64,
                 v53,
                 v119.m128i_i8[0],
                 v119.m128i_i32[2],
                 v82,
                 v80,
                 0x7Eu,
                 0x7Du,
                 v60);
      }
      if ( *(_WORD *)(v83 + 24) != 118 )
      {
        v89 = v82;
        v100 = v60;
        v108 = v80;
        result = sub_1F75B00(
                   (__int64 **)a1,
                   v53.m128i_i64[0],
                   v53.m128i_u64[1],
                   v119.m128i_i64[0],
                   v119.m128i_i64[1],
                   v80,
                   *(double *)a7.m128i_i64,
                   *(double *)v54.m128i_i64,
                   v53,
                   v120,
                   SDWORD2(v120),
                   v81,
                   v83,
                   0x7Du,
                   0x7Eu,
                   v60);
        v80 = v108;
        v60 = v100;
        v82 = v89;
        if ( result )
          return result;
      }
      if ( *(_WORD *)(v80 + 24) != 118 )
        return sub_1F75B00(
                 (__int64 **)a1,
                 v56,
                 v92,
                 v120,
                 *((__int64 *)&v120 + 1),
                 v83,
                 *(double *)a7.m128i_i64,
                 *(double *)v54.m128i_i64,
                 v53,
                 v119.m128i_i8[0],
                 v119.m128i_i32[2],
                 v82,
                 v80,
                 0x7Eu,
                 0x7Du,
                 v60);
    }
    return 0;
  }
  if ( v88 )
  {
    v62 = (__int128 *)&v119;
    v63 = 125;
  }
  else
  {
    v62 = &v120;
    v63 = 126;
  }
  v105 = v60;
  result = sub_1D332F0(
             *(__int64 **)a1,
             v63,
             v60,
             v111,
             v112,
             0,
             *(double *)a7.m128i_i64,
             *(double *)v54.m128i_i64,
             v53,
             v53.m128i_i64[0],
             v53.m128i_u64[1],
             *v62);
  v99 = (__int64)result;
  v102 = v64;
  if ( v114.m128i_i64[0] || v117 )
  {
    v94 = v105;
    v65 = sub_1D389D0(*(_QWORD *)a1, v105, v111, v112, 0, 0, a7, *(double *)v54.m128i_i64, v53);
    v67 = v105;
    v106 = v65;
    v68 = v65;
    v69 = v66;
    v110 = v66;
    if ( v114.m128i_i64[0] )
    {
      *(_QWORD *)&v70 = sub_1D332F0(
                          *(__int64 **)a1,
                          124,
                          v94,
                          v111,
                          v112,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)v54.m128i_i64,
                          v53,
                          v65,
                          v66,
                          v120);
      v71 = *(__int64 **)a1;
      *(_QWORD *)&v72 = sub_1D332F0(
                          *(__int64 **)a1,
                          119,
                          v94,
                          v111,
                          v112,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)v54.m128i_i64,
                          v53,
                          v114.m128i_i64[0],
                          v114.m128i_u64[1],
                          v70);
      v73 = sub_1D332F0(
              v71,
              118,
              v94,
              v111,
              v112,
              0,
              *(double *)a7.m128i_i64,
              *(double *)v54.m128i_i64,
              v53,
              v106,
              v110,
              v72);
      v67 = v94;
      v68 = (__int64)v73;
      v69 = v74 | v69 & 0xFFFFFFFF00000000LL;
    }
    if ( v117 )
    {
      v86 = v106;
      v107 = v67;
      *(_QWORD *)&v75 = sub_1D332F0(
                          *(__int64 **)a1,
                          122,
                          v67,
                          v111,
                          v112,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)v54.m128i_i64,
                          v53,
                          v86,
                          v110,
                          *(_OWORD *)&v119);
      v76 = *(__int64 **)a1;
      *(_QWORD *)&v77 = sub_1D332F0(
                          *(__int64 **)a1,
                          119,
                          v107,
                          v111,
                          v112,
                          0,
                          *(double *)a7.m128i_i64,
                          *(double *)v54.m128i_i64,
                          v53,
                          v117,
                          v118,
                          v75);
      v78 = sub_1D332F0(
              v76,
              118,
              v107,
              v111,
              v112,
              0,
              *(double *)a7.m128i_i64,
              *(double *)v54.m128i_i64,
              v53,
              v68,
              v69,
              v77);
      v67 = v107;
      v68 = (__int64)v78;
      v69 = v79 | v69 & 0xFFFFFFFF00000000LL;
    }
    *((_QWORD *)&v87 + 1) = v69;
    *(_QWORD *)&v87 = v68;
    return sub_1D332F0(
             *(__int64 **)a1,
             118,
             v67,
             v111,
             v112,
             0,
             *(double *)a7.m128i_i64,
             *(double *)v54.m128i_i64,
             v53,
             v99,
             v102,
             v87);
  }
  return result;
}
