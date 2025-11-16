// Function: sub_2025FA0
// Address: 0x2025fa0
//
__int64 __fastcall sub_2025FA0(__int64 *a1, unsigned __int64 a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  __m128 v10; // xmm0
  __int128 v11; // xmm1
  __m128i v12; // xmm2
  __int64 v13; // r14
  int v14; // eax
  __int64 result; // rax
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int8 v21; // di
  __int64 v22; // rdx
  unsigned int v23; // eax
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int v33; // edx
  __int64 v34; // rsi
  __int64 v35; // rsi
  _QWORD *v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned int v41; // eax
  __int64 v42; // r14
  __int128 v43; // rax
  int v44; // edx
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  char v50; // r13
  const void **v51; // r14
  __int64 v52; // rax
  __int64 v53; // rax
  _QWORD *v54; // r13
  unsigned __int8 *v55; // r14
  __int8 v56; // al
  __int64 v57; // rdx
  unsigned int v58; // eax
  __int64 *v59; // r15
  unsigned int v60; // edx
  unsigned int v61; // edx
  __int64 v62; // r12
  __int64 (__fastcall *v63)(__int64, __int64); // rbx
  __int64 v64; // rax
  int v65; // eax
  unsigned int v66; // edx
  unsigned int v67; // ecx
  unsigned __int8 v68; // al
  __int64 v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rdi
  const void ***v72; // rdx
  unsigned int v73; // edx
  int v74; // edx
  int v75; // edx
  __int64 v76; // rax
  unsigned int v77; // eax
  unsigned int v78; // eax
  __int64 *v79; // rdi
  const void **v80; // rdx
  __int64 v81; // rax
  unsigned int v82; // edx
  unsigned int v83; // edi
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // r9
  __int64 v90; // rax
  unsigned int v91; // edx
  __int64 v92; // [rsp-10h] [rbp-220h]
  __int64 v93; // [rsp-10h] [rbp-220h]
  __int64 v94; // [rsp-8h] [rbp-218h]
  int v95; // [rsp+0h] [rbp-210h]
  __int64 v96; // [rsp+0h] [rbp-210h]
  __int64 v97; // [rsp+8h] [rbp-208h]
  __int64 v98; // [rsp+8h] [rbp-208h]
  int v99; // [rsp+10h] [rbp-200h]
  __int64 v100; // [rsp+10h] [rbp-200h]
  __int64 v101; // [rsp+18h] [rbp-1F8h]
  __int64 v102; // [rsp+18h] [rbp-1F8h]
  unsigned int v103; // [rsp+24h] [rbp-1ECh]
  unsigned int v104; // [rsp+24h] [rbp-1ECh]
  __int64 v105; // [rsp+28h] [rbp-1E8h]
  unsigned int v106; // [rsp+28h] [rbp-1E8h]
  __int64 v107; // [rsp+30h] [rbp-1E0h]
  __int64 *v108; // [rsp+30h] [rbp-1E0h]
  _QWORD *v109; // [rsp+30h] [rbp-1E0h]
  __int64 v110; // [rsp+38h] [rbp-1D8h]
  __int64 v111; // [rsp+40h] [rbp-1D0h]
  __int64 v112; // [rsp+40h] [rbp-1D0h]
  unsigned int v113; // [rsp+48h] [rbp-1C8h]
  _QWORD *v114; // [rsp+48h] [rbp-1C8h]
  __int64 *v115; // [rsp+48h] [rbp-1C8h]
  __int64 v116; // [rsp+50h] [rbp-1C0h]
  _QWORD *v117; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v118; // [rsp+50h] [rbp-1C0h]
  __int64 v119; // [rsp+60h] [rbp-1B0h]
  unsigned int v120; // [rsp+60h] [rbp-1B0h]
  __int64 v121; // [rsp+68h] [rbp-1A8h]
  __int64 v122; // [rsp+68h] [rbp-1A8h]
  __int64 v123; // [rsp+68h] [rbp-1A8h]
  const void **v124; // [rsp+68h] [rbp-1A8h]
  unsigned int v125; // [rsp+70h] [rbp-1A0h]
  _QWORD *v126; // [rsp+70h] [rbp-1A0h]
  unsigned int v127; // [rsp+70h] [rbp-1A0h]
  unsigned __int64 v128; // [rsp+78h] [rbp-198h]
  __int64 v129; // [rsp+80h] [rbp-190h]
  __int64 v130; // [rsp+80h] [rbp-190h]
  __int64 v132; // [rsp+A0h] [rbp-170h]
  __int64 v133; // [rsp+A0h] [rbp-170h]
  unsigned int v134; // [rsp+A0h] [rbp-170h]
  unsigned __int64 v135; // [rsp+A8h] [rbp-168h]
  __int64 v136; // [rsp+120h] [rbp-F0h] BYREF
  int v137; // [rsp+128h] [rbp-E8h]
  __int64 v138; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v139; // [rsp+138h] [rbp-D8h]
  __m128i v140; // [rsp+140h] [rbp-D0h] BYREF
  _QWORD v141[2]; // [rsp+150h] [rbp-C0h] BYREF
  __int128 v142; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v143; // [rsp+170h] [rbp-A0h]
  __int128 v144; // [rsp+180h] [rbp-90h] BYREF
  __int64 v145; // [rsp+190h] [rbp-80h]
  __int128 v146; // [rsp+1A0h] [rbp-70h] BYREF
  __int64 v147; // [rsp+1B0h] [rbp-60h]
  __m128i v148; // [rsp+1C0h] [rbp-50h] BYREF
  __int64 v149; // [rsp+1D0h] [rbp-40h]
  const void **v150; // [rsp+1D8h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(_QWORD *)v7;
  v10 = (__m128)_mm_loadu_si128((const __m128i *)v7);
  v11 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v12 = _mm_loadu_si128((const __m128i *)(v7 + 80));
  v136 = v8;
  v121 = v9;
  v13 = *(_QWORD *)(v7 + 80);
  v135 = v10.m128_u64[1];
  v125 = *(_DWORD *)(v7 + 8);
  v119 = *(_QWORD *)(v7 + 40);
  v113 = *(_DWORD *)(v7 + 48);
  if ( v8 )
    sub_1623A60((__int64)&v136, v8, 2);
  v137 = *(_DWORD *)(a2 + 64);
  sub_2017DE0((__int64)a1, v10.m128_u64[0], v10.m128_i64[1], a3, (_DWORD *)a4);
  v14 = *(unsigned __int16 *)(v13 + 24);
  if ( v14 == 10 || v14 == 32 )
  {
    v53 = *(_QWORD *)(v13 + 88);
    v54 = *(_QWORD **)(v53 + 24);
    if ( *(_DWORD *)(v53 + 32) > 0x40u )
      v54 = (_QWORD *)*v54;
    v55 = (unsigned __int8 *)(*(_QWORD *)(*a3 + 40) + 16LL * *((unsigned int *)a3 + 2));
    v56 = *v55;
    v57 = *((_QWORD *)v55 + 1);
    v148.m128i_i8[0] = v56;
    v148.m128i_i64[1] = v57;
    if ( v56 )
    {
      v59 = (__int64 *)a1[1];
      v60 = word_4305480[(unsigned __int8)(v56 - 14)];
      if ( (unsigned int)v54 < v60 )
        goto LABEL_22;
    }
    else
    {
      v58 = sub_1F58D30((__int64)&v148);
      v59 = (__int64 *)a1[1];
      v60 = v58;
      if ( (unsigned int)v54 < v58 )
      {
LABEL_22:
        *a3 = (unsigned __int64)sub_1D3A900(
                                  v59,
                                  0x69u,
                                  (__int64)&v136,
                                  *v55,
                                  *((const void ***)v55 + 1),
                                  0,
                                  v10,
                                  *(double *)&v11,
                                  v12,
                                  *a3,
                                  (__int16 *)a3[1],
                                  v11,
                                  v12.m128i_i64[0],
                                  v12.m128i_i64[1]);
        result = v61;
        *((_DWORD *)a3 + 2) = v61;
        goto LABEL_23;
      }
    }
    v62 = *a1;
    v134 = v60;
    v63 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
    v64 = sub_1E0A0C0(v59[4]);
    if ( v63 == sub_1D13A20 )
    {
      v65 = sub_15A9520(v64, 0);
      v66 = v134;
      v67 = 8 * v65;
      if ( 8 * v65 == 32 )
      {
        v68 = 5;
      }
      else if ( v67 > 0x20 )
      {
        v68 = 6;
        if ( v67 != 64 )
        {
          v68 = 0;
          if ( v67 == 128 )
            v68 = 7;
        }
      }
      else
      {
        v68 = 3;
        if ( v67 != 8 )
          v68 = 4 * (v67 == 16);
      }
    }
    else
    {
      v68 = v63(v62, v64);
      v66 = v134;
    }
    v69 = sub_1D38BB0(
            (__int64)v59,
            (unsigned int)v54 - v66,
            (__int64)&v136,
            v68,
            0,
            0,
            (__m128i)v10,
            *(double *)&v11,
            v12,
            0);
    v71 = v70;
    v72 = (const void ***)(*(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8));
    *(_QWORD *)a4 = sub_1D3A900(
                      v59,
                      0x69u,
                      (__int64)&v136,
                      *(unsigned __int8 *)v72,
                      v72[1],
                      0,
                      v10,
                      *(double *)&v11,
                      v12,
                      *(_QWORD *)a4,
                      *(__int16 **)(a4 + 8),
                      v11,
                      v69,
                      v71);
    result = v73;
    *(_DWORD *)(a4 + 8) = v73;
    goto LABEL_23;
  }
  result = sub_2016240(a1, a2, **(_BYTE **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 1u, 0, 0);
  if ( !(_BYTE)result )
  {
    v105 = v125;
    v16 = *(_QWORD *)(v121 + 40) + 16LL * v125;
    v17 = *(_BYTE *)v16;
    v18 = *(_QWORD *)(v16 + 8);
    LOBYTE(v138) = v17;
    v139 = v18;
    LOBYTE(v19) = sub_1F7E0F0((__int64)&v138);
    v107 = v19;
    v110 = v20;
    if ( (_BYTE)v138 )
    {
      if ( (unsigned __int8)(v138 - 14) > 0x5Fu )
        goto LABEL_8;
    }
    else if ( !sub_1F58D20((__int64)&v138) )
    {
LABEL_8:
      v21 = v138;
      v22 = v139;
LABEL_9:
      v148.m128i_i8[0] = v21;
      v148.m128i_i64[1] = v22;
      if ( v21 )
        v23 = sub_2021900(v21);
      else
        v23 = sub_1F58D40((__int64)&v148);
      v111 = v113;
      if ( v23 <= 7 )
      {
        v76 = v107;
        v110 = 0;
        LOBYTE(v76) = 3;
        v107 = v76;
        v77 = sub_1D15970(&v138);
        v78 = sub_1F7DEB0(*(_QWORD **)(a1[1] + 48), v107, 0, v77, 0);
        v79 = (__int64 *)a1[1];
        v139 = (__int64)v80;
        LODWORD(v138) = v78;
        v81 = sub_1D309E0(
                v79,
                144,
                (__int64)&v136,
                v78,
                v80,
                0,
                *(double *)v10.m128_u64,
                *(double *)&v11,
                *(double *)v12.m128i_i64,
                *(_OWORD *)&v10);
        v25 = v93;
        v121 = v81;
        v83 = v82;
        v127 = v82;
        v135 = v82 | v10.m128_u64[1] & 0xFFFFFFFF00000000LL;
        v84 = *(_QWORD *)(v119 + 40) + 16LL * v113;
        LOBYTE(v82) = *(_BYTE *)v84;
        v85 = *(_QWORD *)(v84 + 8);
        v148.m128i_i8[0] = v82;
        v148.m128i_i64[1] = v85;
        if ( (_BYTE)v82 == 3 )
        {
          v105 = v83;
        }
        else
        {
          v104 = sub_2021900(3);
          v105 = v127;
          if ( (unsigned int)sub_1D159A0(v148.m128i_i8, 144, v86, v87, v88, v89, v95, v97, v99, v101) < v104 )
          {
            v90 = sub_1D309E0(
                    (__int64 *)a1[1],
                    144,
                    (__int64)&v136,
                    (unsigned int)v107,
                    0,
                    0,
                    *(double *)v10.m128_u64,
                    *(double *)&v11,
                    *(double *)v12.m128i_i64,
                    v11);
            v24 = v94;
            v119 = v90;
            v111 = v91;
          }
        }
      }
      v126 = sub_1D29C20((_QWORD *)a1[1], (unsigned int)v138, v139, 1, v24, v25);
      v128 = v26;
      v103 = v26;
      v114 = v126;
      v96 = *(_QWORD *)(a1[1] + 32);
      sub_1E341E0((__int64)&v142, v96, *((_DWORD *)v126 + 21), 0);
      v27 = (_QWORD *)a1[1];
      v148 = 0u;
      v149 = 0;
      v28 = sub_1D2BF40(
              v27,
              (__int64)(v27 + 11),
              0,
              (__int64)&v136,
              v121,
              v105 | v135 & 0xFFFFFFFF00000000LL,
              (__int64)v126,
              v128,
              v142,
              v143,
              0,
              0,
              (__int64)&v148);
      v122 = v29;
      v132 = v28;
      v100 = sub_20BD400(*a1, a1[1], (_DWORD)v126, v128, v138, v139, v12.m128i_i64[0], v12.m128i_i64[1]);
      v102 = v30;
      v116 = sub_1F58E60((__int64)&v138, *(_QWORD **)(a1[1] + 48));
      v31 = sub_1E0A0C0(*(_QWORD *)(a1[1] + 32));
      v106 = sub_15AAE50(v31, v116);
      v117 = (_QWORD *)a1[1];
      v148 = 0u;
      v149 = 0;
      sub_1E34280((__int64)&v144, v96);
      v32 = sub_1D2C750(
              v117,
              v132,
              v122,
              (__int64)&v136,
              v119,
              *((_QWORD *)&v11 + 1) & 0xFFFFFFFF00000000LL | v111,
              v100,
              v102,
              v144,
              v145,
              v107,
              v110,
              0,
              0,
              (__int64)&v148);
      v140.m128i_i8[0] = 0;
      v34 = a1[1];
      v118 = v33 | v122 & 0xFFFFFFFF00000000LL;
      v133 = v32;
      v140.m128i_i64[1] = 0;
      sub_1D19A30((__int64)&v148, v34, &v138);
      v35 = v148.m128i_u32[0];
      v36 = (_QWORD *)a1[1];
      v140 = _mm_loadu_si128(&v148);
      v129 = v149;
      v123 = (__int64)v150;
      v148 = 0u;
      v149 = 0;
      *a3 = sub_1D2B730(
              v36,
              v35,
              v140.m128i_i64[1],
              (__int64)&v136,
              v133,
              v118,
              (__int64)v126,
              v128,
              v142,
              v143,
              0,
              0,
              (__int64)&v148,
              0);
      *((_DWORD *)a3 + 2) = v37;
      v41 = sub_1D159A0(v140.m128i_i8, v35, v37, v38, v39, v40, v96, (__int64)&v142, v100, v102);
      v108 = (__int64 *)a1[1];
      v42 = 16LL * v103;
      v120 = v41 >> 3;
      v112 = v41 >> 3;
      *(_QWORD *)&v43 = sub_1D38BB0(
                          (__int64)v108,
                          v112,
                          (__int64)&v136,
                          *(unsigned __int8 *)(v42 + v114[5]),
                          *(const void ***)(v42 + v114[5] + 8),
                          0,
                          (__m128i)v10,
                          *(double *)&v11,
                          v12,
                          0);
      v115 = sub_1D332F0(
               v108,
               52,
               (__int64)&v136,
               *(unsigned __int8 *)(v114[5] + v42),
               *(const void ***)(v114[5] + v42 + 8),
               0,
               *(double *)v10.m128_u64,
               *(double *)&v11,
               v12,
               (__int64)v126,
               v128,
               v43);
      LODWORD(v42) = v44;
      v109 = (_QWORD *)a1[1];
      v148 = 0u;
      v149 = 0;
      sub_1F7DDA0((__int64)&v146, v98, v112);
      *(_QWORD *)a4 = sub_1D2B730(
                        v109,
                        v129,
                        v123,
                        (__int64)&v136,
                        v133,
                        v118,
                        (__int64)v115,
                        (unsigned int)v42 | v128 & 0xFFFFFFFF00000000LL,
                        v146,
                        v147,
                        -(v120 | v106) & (v120 | v106),
                        0,
                        (__int64)&v148,
                        0);
      *(_DWORD *)(a4 + 8) = v45;
      v46 = *(_QWORD *)(a2 + 40);
      v47 = a1[1];
      LOBYTE(v45) = *(_BYTE *)v46;
      v48 = *(_QWORD *)(v46 + 8);
      LOBYTE(v141[0]) = v45;
      v141[1] = v48;
      sub_1D19A30((__int64)&v148, v47, v141);
      v49 = *a3;
      v130 = v149;
      v50 = v149;
      v140 = _mm_loadu_si128(&v148);
      v124 = v150;
      v51 = v150;
      v52 = *(_QWORD *)(v49 + 40) + 16LL * *((unsigned int *)a3 + 2);
      if ( v148.m128i_i8[0] != *(_BYTE *)v52 || !v148.m128i_i8[0] && v140.m128i_i64[1] != *(_QWORD *)(v52 + 8) )
      {
        *a3 = sub_1D309E0(
                (__int64 *)a1[1],
                145,
                (__int64)&v136,
                v140.m128i_u32[0],
                (const void **)v140.m128i_i64[1],
                0,
                *(double *)v10.m128_u64,
                *(double *)&v11,
                *(double *)v12.m128i_i64,
                *(_OWORD *)a3);
        *((_DWORD *)a3 + 2) = v75;
      }
      result = *(_QWORD *)(*(_QWORD *)a4 + 40LL) + 16LL * *(unsigned int *)(a4 + 8);
      if ( v50 != *(_BYTE *)result || !v50 && *(const void ***)(result + 8) != v51 )
      {
        *(_QWORD *)a4 = sub_1D309E0(
                          (__int64 *)a1[1],
                          145,
                          (__int64)&v136,
                          v130,
                          v124,
                          0,
                          *(double *)v10.m128_u64,
                          *(double *)&v11,
                          *(double *)v12.m128i_i64,
                          *(_OWORD *)a4);
        *(_DWORD *)(a4 + 8) = v74;
        result = v92;
      }
      if ( v136 )
        return sub_161E7C0((__int64)&v136, v136);
      return result;
    }
    v21 = sub_1F7E0F0((__int64)&v138);
    goto LABEL_9;
  }
LABEL_23:
  if ( v136 )
    return sub_161E7C0((__int64)&v136, v136);
  return result;
}
