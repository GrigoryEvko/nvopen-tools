// Function: sub_3434B20
// Address: 0x3434b20
//
void __fastcall sub_3434B20(
        unsigned __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  unsigned __int64 v7; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  int v16; // eax
  __int64 v17; // rdx
  void *v18; // rax
  __int64 *v19; // rsi
  _QWORD *v20; // rdx
  _QWORD *v21; // r15
  __int64 *v22; // rdi
  __int64 v23; // rdi
  unsigned int v24; // esi
  __int64 (__fastcall *v25)(__int64, __int64, unsigned int); // rax
  int v26; // edx
  unsigned __int16 v27; // ax
  __int64 v28; // rdx
  _QWORD *v29; // r8
  __int64 v30; // rax
  __int64 v31; // r9
  _QWORD *v32; // rax
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 *v38; // rax
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // r8
  _QWORD *v43; // rdx
  int v44; // r11d
  unsigned int i; // edx
  __int64 v46; // rax
  unsigned int v47; // edx
  __int64 v48; // rdx
  __int64 v49; // r11
  _QWORD *v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rcx
  __m128i v53; // xmm0
  __int64 v54; // rdx
  __int64 v55; // r12
  unsigned __int64 v56; // rbx
  __int64 v57; // r8
  __int64 v58; // rax
  __int64 *v59; // rdx
  unsigned int v60; // eax
  unsigned __int16 *v61; // rax
  _QWORD *v62; // rax
  __int64 v63; // r9
  __int64 v64; // r8
  __int64 *v65; // rdi
  __int64 v66; // rdi
  unsigned int v67; // esi
  __int64 (__fastcall *v68)(__int64, __int64, unsigned int); // rax
  _DWORD *v69; // rax
  _QWORD *v70; // r9
  int v71; // edx
  __int64 v72; // r11
  unsigned __int16 v73; // ax
  __int64 v74; // rdx
  __m128i v75; // xmm1
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  int v80; // ecx
  unsigned __int64 v81; // rax
  _QWORD *v82; // rdi
  int v83; // edx
  const __m128i *v84; // rcx
  __int64 v85; // rax
  unsigned __int64 v86; // r10
  unsigned __int64 v87; // r11
  _QWORD *v88; // rsi
  __m128i *v89; // rax
  unsigned __int64 v90; // r10
  __m128i *v91; // r8
  __int64 v92; // rdx
  unsigned __int64 v93; // r11
  unsigned int v94; // esi
  unsigned __int64 v95; // r10
  __int64 v96; // rdi
  __int64 v97; // r11
  __int64 v98; // r14
  unsigned __int64 v99; // rbx
  unsigned int v100; // r9d
  unsigned int v101; // r15d
  unsigned int j; // esi
  __int64 *v103; // rdx
  unsigned int v104; // r9d
  __int64 *v105; // rdx
  int v106; // edx
  __int64 *v107; // rcx
  int v108; // esi
  unsigned int v109; // r9d
  unsigned int v110; // r9d
  int v111; // r10d
  int v112; // r9d
  int v113; // ecx
  int v114; // ecx
  int v115; // r9d
  int v116; // r9d
  int v117; // ecx
  unsigned int v118; // edi
  __int64 *v119; // rsi
  unsigned int v120; // edi
  int v121; // edi
  int v122; // r11d
  __int128 v123; // [rsp-30h] [rbp-1C0h]
  unsigned __int64 v124; // [rsp+0h] [rbp-190h]
  const __m128i *v125; // [rsp+10h] [rbp-180h]
  unsigned __int64 v126; // [rsp+10h] [rbp-180h]
  int v127; // [rsp+10h] [rbp-180h]
  _QWORD *v128; // [rsp+10h] [rbp-180h]
  unsigned __int64 v129; // [rsp+18h] [rbp-178h]
  unsigned __int64 v130; // [rsp+18h] [rbp-178h]
  __int64 v131; // [rsp+20h] [rbp-170h]
  int v132; // [rsp+28h] [rbp-168h]
  __int64 v133; // [rsp+28h] [rbp-168h]
  _QWORD *v134; // [rsp+30h] [rbp-160h]
  int v135; // [rsp+30h] [rbp-160h]
  unsigned __int64 v136; // [rsp+30h] [rbp-160h]
  __int64 *v137; // [rsp+30h] [rbp-160h]
  __int64 v138; // [rsp+38h] [rbp-158h]
  __int64 v139; // [rsp+38h] [rbp-158h]
  __int64 v140; // [rsp+40h] [rbp-150h]
  __int64 v141; // [rsp+40h] [rbp-150h]
  unsigned __int64 v142; // [rsp+40h] [rbp-150h]
  __int64 *v143; // [rsp+40h] [rbp-150h]
  _QWORD *v144; // [rsp+40h] [rbp-150h]
  __int64 v145; // [rsp+40h] [rbp-150h]
  __int64 v146; // [rsp+48h] [rbp-148h]
  __int64 v147; // [rsp+48h] [rbp-148h]
  unsigned __int64 v148; // [rsp+50h] [rbp-140h]
  __m128i *v149; // [rsp+50h] [rbp-140h]
  __int64 v150; // [rsp+50h] [rbp-140h]
  unsigned __int64 v151; // [rsp+50h] [rbp-140h]
  __int64 v152; // [rsp+50h] [rbp-140h]
  __int64 v153; // [rsp+50h] [rbp-140h]
  __m128i v154; // [rsp+60h] [rbp-130h] BYREF
  _QWORD *v155; // [rsp+70h] [rbp-120h]
  __int64 v156; // [rsp+78h] [rbp-118h]
  __int64 v157; // [rsp+80h] [rbp-110h]
  __int64 v158; // [rsp+88h] [rbp-108h]
  unsigned __int64 v159; // [rsp+98h] [rbp-F8h]
  unsigned __int64 v160; // [rsp+A0h] [rbp-F0h]
  unsigned __int64 v161; // [rsp+A8h] [rbp-E8h]
  unsigned __int64 v162; // [rsp+B0h] [rbp-E0h]
  __int64 v163; // [rsp+B8h] [rbp-D8h]
  unsigned __int64 v164; // [rsp+C0h] [rbp-D0h]
  unsigned __int64 v165; // [rsp+C8h] [rbp-C8h]
  __m128i *v166; // [rsp+D0h] [rbp-C0h]
  __int64 v167; // [rsp+D8h] [rbp-B8h]
  _QWORD *v168; // [rsp+E0h] [rbp-B0h]
  __int64 v169; // [rsp+E8h] [rbp-A8h]
  __m128i v170; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v171; // [rsp+100h] [rbp-90h]
  __m128i v172; // [rsp+110h] [rbp-80h]
  __int64 v173; // [rsp+120h] [rbp-70h]
  _QWORD *v174; // [rsp+130h] [rbp-60h] BYREF
  __int64 v175; // [rsp+138h] [rbp-58h]
  _QWORD *v176; // [rsp+140h] [rbp-50h]
  __m128i v177; // [rsp+148h] [rbp-48h] BYREF

  v7 = a2;
  LODWORD(v157) = a3;
  LODWORD(v156) = a2;
  if ( (unsigned __int8)sub_3433840(a1, a2, a3) )
  {
    v16 = *(_DWORD *)(a1 + 24);
    if ( v16 == 15 || v16 == 39 )
    {
      v21 = *(_QWORD **)(a6 + 864);
      v22 = (__int64 *)v21[5];
      v157 = v21[2];
      v23 = sub_2E79000(v22);
      v24 = *(_DWORD *)(v23 + 4);
      v25 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v157 + 32LL);
      if ( v25 == sub_2D42F30 )
      {
        v26 = sub_AE2980(v23, v24)[1];
        v27 = 2;
        if ( v26 != 1 )
        {
          v27 = 3;
          if ( v26 != 2 )
          {
            v27 = 4;
            if ( v26 != 4 )
            {
              v27 = 5;
              if ( v26 != 8 )
              {
                v27 = 6;
                if ( v26 != 16 )
                {
                  v27 = 7;
                  if ( v26 != 32 )
                  {
                    v27 = 8;
                    if ( v26 != 64 )
                      v27 = 9 * (v26 == 128);
                  }
                }
              }
            }
          }
        }
      }
      else
      {
        v27 = v25(v157, v23, v24);
      }
      v29 = sub_33EDBD0(v21, *(_DWORD *)(a1 + 96), v27, 0, 1);
      v30 = *(unsigned int *)(a4 + 8);
      v31 = v28;
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        v157 = (__int64)v29;
        v158 = v28;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v30 + 1, 0x10u, (__int64)v29, v28);
        v30 = *(unsigned int *)(a4 + 8);
        v29 = (_QWORD *)v157;
        v31 = v158;
      }
      v32 = (_QWORD *)(*(_QWORD *)a4 + 16 * v30);
      *v32 = v29;
      v32[1] = v31;
      ++*(_DWORD *)(a4 + 8);
      v35 = sub_34335B0(*(_QWORD **)(*(_QWORD *)(a6 + 864) + 40LL), a1);
      v36 = *(unsigned int *)(a5 + 8);
      if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        sub_C8D5F0(a5, (const void *)(a5 + 16), v36 + 1, 8u, v33, v34);
        v36 = *(unsigned int *)(a5 + 8);
      }
      *(_QWORD *)(*(_QWORD *)a5 + 8 * v36) = v35;
      ++*(_DWORD *)(a5 + 8);
      return;
    }
    v17 = 4278124286LL;
    if ( v16 != 51 )
    {
      if ( v16 != 35 && v16 != 11 )
      {
        if ( v16 == 12 || v16 == 36 )
        {
          v18 = sub_C33340();
          v19 = (__int64 *)(*(_QWORD *)(a1 + 96) + 24LL);
          if ( (void *)*v19 == v18 )
            sub_C3E660((__int64)&v174, (__int64)v19);
          else
            sub_C3A850((__int64)&v174, v19);
          v20 = v174;
          if ( (unsigned int)v175 > 0x40 )
            v20 = (_QWORD *)*v174;
          sub_3433B90(a4, (__int64 *)a6, (__int64)v20, a7);
          if ( (unsigned int)v175 > 0x40 )
          {
            if ( v174 )
              j_j___libc_free_0_0((unsigned __int64)v174);
          }
          return;
        }
        goto LABEL_136;
      }
      v58 = *(_QWORD *)(a1 + 96);
      v59 = *(__int64 **)(v58 + 24);
      v60 = *(_DWORD *)(v58 + 32);
      if ( v60 > 0x40 )
      {
        v57 = *v59;
      }
      else
      {
        v57 = 0;
        if ( v60 )
          v57 = (__int64)((_QWORD)v59 << (64 - (unsigned __int8)v60)) >> (64 - (unsigned __int8)v60);
      }
      v17 = v57;
    }
    sub_3433B90(a4, (__int64 *)a6, v17, a7);
    return;
  }
  if ( !(_BYTE)v157 )
  {
    v37 = *(unsigned int *)(a4 + 8);
    if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      sub_C8D5F0(a4, (const void *)(a4 + 16), v37 + 1, 0x10u, v14, v15);
      v37 = *(unsigned int *)(a4 + 8);
    }
    v38 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v37);
    *v38 = a1;
    v38[1] = a2;
    ++*(_DWORD *)(a4 + 8);
    return;
  }
  v39 = sub_33738B0(a6, a2, v12, v13, v14, v15);
  v40 = *(unsigned int *)(a6 + 296);
  v41 = *(_QWORD *)(a6 + 280);
  v148 = v39;
  v42 = v39;
  v157 = (__int64)v43;
  v155 = v43;
  if ( !(_DWORD)v40 )
    goto LABEL_59;
  v44 = 1;
  for ( i = (v40 - 1) & (v7 + ((a1 >> 9) ^ (a1 >> 4))); ; i = (v40 - 1) & v47 )
  {
    v46 = v41 + 32LL * i;
    if ( a1 == *(_QWORD *)v46 && (_DWORD)v156 == *(_DWORD *)(v46 + 8) )
      break;
    if ( !*(_QWORD *)v46 && *(_DWORD *)(v46 + 8) == -1 )
      goto LABEL_59;
    v47 = v44 + i;
    ++v44;
  }
  if ( v46 == v41 + 32 * v40 )
  {
LABEL_59:
    v49 = 0;
  }
  else
  {
    v48 = *(unsigned int *)(v46 + 24);
    v154.m128i_i64[0] = *(_QWORD *)(v46 + 16);
    v49 = v48;
    if ( v154.m128i_i64[0] )
    {
      v50 = 0;
      goto LABEL_43;
    }
  }
  v61 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)v7);
  v138 = v49;
  v131 = a6 + 272;
  v62 = sub_3434150(a6 + 272, *v61, *((_QWORD *)v61 + 1), a6);
  v63 = *(_QWORD *)(a6 + 864);
  LODWORD(v62) = *((_DWORD *)v62 + 24);
  v64 = *(_QWORD *)(v63 + 16);
  v65 = *(__int64 **)(v63 + 40);
  v154.m128i_i64[0] = v63;
  LODWORD(v155) = (_DWORD)v62;
  v140 = v64;
  v66 = sub_2E79000(v65);
  v67 = *(_DWORD *)(v66 + 4);
  v68 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v140 + 32LL);
  if ( v68 == sub_2D42F30 )
  {
    v69 = sub_AE2980(v66, v67);
    v70 = (_QWORD *)v154.m128i_i64[0];
    v71 = v69[1];
    v72 = v138;
    v73 = 2;
    if ( v71 != 1 )
    {
      v73 = 3;
      if ( v71 != 2 )
      {
        v73 = 4;
        if ( v71 != 4 )
        {
          v73 = 5;
          if ( v71 != 8 )
          {
            v73 = 6;
            if ( v71 != 16 )
            {
              v73 = 7;
              if ( v71 != 32 )
              {
                v73 = 8;
                if ( v71 != 64 )
                  v73 = 9 * (v71 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v73 = v68(v140, v66, v67);
    v72 = v138;
    v70 = (_QWORD *)v154.m128i_i64[0];
  }
  v146 = v72;
  v168 = sub_33EDBD0(v70, (int)v155, v73, 0, 1);
  v154.m128i_i64[0] = (__int64)v168;
  v169 = v74;
  v129 = (unsigned int)v74 | v146 & 0xFFFFFFFF00000000LL;
  v134 = *(_QWORD **)(*(_QWORD *)(a6 + 864) + 40LL);
  v141 = v134[6];
  sub_2EAC300((__int64)&v170, (__int64)v134, (unsigned int)v155, 0);
  v174 = 0;
  v175 = 0;
  v176 = 0;
  v75 = _mm_load_si128(&v170);
  v177.m128i_i64[0] = 0;
  v76 = 5LL * (unsigned int)(*(_DWORD *)(v141 + 32) + (_DWORD)v155);
  v77 = *(_QWORD *)(v141 + 8);
  v172 = v75;
  v78 = v77 + 8 * v76;
  v79 = *(_QWORD *)(v78 + 8);
  v80 = *(unsigned __int8 *)(v78 + 16);
  v173 = v171;
  *((_QWORD *)&v123 + 1) = v75.m128i_i64[1];
  if ( v79 > 0x3FFFFFFFFFFFFFFBLL )
    LODWORD(v79) = -2;
  *(_QWORD *)&v123 = v172.m128i_i64[0];
  v81 = sub_2E7BD70(v134, 2u, v79, v80, (int)&v174, 0, v123, v171, 1u, 0, 0);
  v82 = *(_QWORD **)(a6 + 864);
  v83 = *(_DWORD *)(a6 + 848);
  v84 = (const __m128i *)v81;
  v85 = *(_QWORD *)a6;
  v174 = 0;
  v155 = v82;
  v86 = (unsigned __int64)v168;
  LODWORD(v175) = v83;
  v87 = v129;
  if ( v85 )
  {
    if ( &v174 != (_QWORD **)(v85 + 48) )
    {
      v88 = *(_QWORD **)(v85 + 48);
      v174 = v88;
      if ( v88 )
      {
        v125 = v84;
        sub_B96E90((__int64)&v174, (__int64)v88, 1);
        v86 = (unsigned __int64)v168;
        v87 = v129;
        v84 = v125;
      }
    }
  }
  v126 = v86;
  v130 = v87;
  v89 = sub_33F3F90(v155, v148, v157, (__int64)&v174, a1, v7, v86, v87, v84);
  v90 = v126;
  v166 = v89;
  v91 = v89;
  v167 = v92;
  v93 = v130;
  v155 = (_QWORD *)((unsigned int)v92 | v157 & 0xFFFFFFFF00000000LL);
  if ( v174 )
  {
    v149 = v89;
    sub_B91220((__int64)&v174, (__int64)v174);
    v90 = v126;
    v93 = v130;
    v91 = v149;
  }
  v142 = v90;
  v147 = v93;
  v150 = (__int64)v91;
  v50 = (_QWORD *)sub_34335B0(v134, (__int64)v168);
  v94 = *(_DWORD *)(a6 + 296);
  v42 = v150;
  v95 = v142;
  v49 = v147;
  if ( !v94 )
  {
    ++*(_QWORD *)(a6 + 272);
    goto LABEL_91;
  }
  v135 = 1;
  v96 = *(_QWORD *)(a6 + 280);
  v143 = 0;
  v151 = v95;
  v97 = a6;
  v98 = a5;
  v132 = ((a1 >> 9) ^ (a1 >> 4)) + v7;
  v99 = v7;
  v100 = (v94 - 1) & v132;
  v101 = v94;
  for ( j = v94 - 1; ; v100 = j & v104 )
  {
    v103 = (__int64 *)(v96 + 32LL * v100);
    if ( a1 != *v103 )
      break;
    if ( (_DWORD)v156 == *((_DWORD *)v103 + 2) )
    {
      a5 = v98;
      v95 = v151;
      a6 = v97;
      v105 = v103 + 2;
      v49 = v147;
      goto LABEL_89;
    }
LABEL_81:
    v104 = v135 + v100;
    ++v135;
  }
  if ( *v103 )
    goto LABEL_81;
  v111 = *((_DWORD *)v103 + 2);
  if ( v111 != -1 )
  {
    if ( v111 == -2 )
    {
      if ( v143 )
        v103 = v143;
      v143 = v103;
    }
    goto LABEL_81;
  }
  v137 = (__int64 *)*v103;
  v94 = v101;
  v7 = v99;
  a5 = v98;
  a6 = v97;
  v95 = v151;
  v49 = v147;
  if ( v143 )
    v103 = v143;
  v114 = *(_DWORD *)(a6 + 288);
  ++*(_QWORD *)(a6 + 272);
  v112 = v114 + 1;
  if ( 4 * (v114 + 1) < 3 * v94 )
  {
    if ( v94 - *(_DWORD *)(a6 + 292) - v112 > v94 >> 3 )
      goto LABEL_103;
    v124 = v151;
    v128 = v50;
    v153 = v42;
    sub_3434670(v131, v94);
    v115 = *(_DWORD *)(a6 + 296);
    if ( v115 )
    {
      v116 = v115 - 1;
      v117 = 1;
      v42 = v153;
      v118 = v116 & v132;
      v119 = v137;
      v145 = *(_QWORD *)(a6 + 280);
      v50 = v128;
      while ( 1 )
      {
        v103 = (__int64 *)(v145 + 32LL * v118);
        if ( a1 == *v103 )
        {
          if ( (_DWORD)v156 == *((_DWORD *)v103 + 2) )
          {
            v95 = v124;
            v49 = v147;
            goto LABEL_102;
          }
        }
        else if ( !*v103 )
        {
          v122 = *((_DWORD *)v103 + 2);
          if ( v122 == -1 )
          {
            v95 = v124;
            v49 = v147;
            if ( v119 )
              v103 = v119;
            v112 = *(_DWORD *)(a6 + 288) + 1;
            goto LABEL_103;
          }
          if ( v122 == -2 && !v119 )
            v119 = (__int64 *)(v145 + 32LL * v118);
        }
        v120 = v117 + v118;
        ++v117;
        v118 = v116 & v120;
      }
    }
LABEL_135:
    ++*(_DWORD *)(a6 + 288);
LABEL_136:
    BUG();
  }
LABEL_91:
  v136 = v95;
  v139 = v49;
  v144 = v50;
  v152 = v42;
  sub_3434670(v131, 2 * v94);
  v106 = *(_DWORD *)(a6 + 296);
  if ( !v106 )
    goto LABEL_135;
  v107 = 0;
  v42 = v152;
  v95 = v136;
  v108 = 1;
  v49 = v139;
  v133 = *(_QWORD *)(a6 + 280);
  v127 = v106 - 1;
  v109 = (v106 - 1) & (v7 + ((a1 >> 9) ^ (a1 >> 4)));
  v50 = v144;
  while ( 2 )
  {
    v103 = (__int64 *)(v133 + 32LL * v109);
    if ( a1 == *v103 )
    {
      if ( (_DWORD)v156 == *((_DWORD *)v103 + 2) )
      {
LABEL_102:
        v112 = *(_DWORD *)(a6 + 288) + 1;
        goto LABEL_103;
      }
      goto LABEL_95;
    }
    if ( *v103 )
    {
LABEL_95:
      v110 = v108 + v109;
      ++v108;
      v109 = v127 & v110;
      continue;
    }
    break;
  }
  v121 = *((_DWORD *)v103 + 2);
  if ( v121 != -1 )
  {
    if ( !v107 && v121 == -2 )
      v107 = (__int64 *)(v133 + 32LL * v109);
    goto LABEL_95;
  }
  if ( v107 )
    v103 = v107;
  v112 = *(_DWORD *)(a6 + 288) + 1;
LABEL_103:
  *(_DWORD *)(a6 + 288) = v112;
  if ( *v103 || *((_DWORD *)v103 + 2) != -1 )
    --*(_DWORD *)(a6 + 292);
  v165 = v7;
  v105 = v103 + 2;
  v164 = a1;
  *(v105 - 2) = a1;
  v113 = v165;
  *v105 = 0;
  *((_DWORD *)v105 - 2) = v113;
  *((_DWORD *)v105 + 2) = 0;
LABEL_89:
  v163 = v49;
  v162 = v95;
  *v105 = v95;
  *((_DWORD *)v105 + 2) = v163;
LABEL_43:
  v51 = *(unsigned int *)(a4 + 8);
  v177.m128i_i64[1] = v49;
  v175 = v42;
  v176 = v155;
  v52 = *(unsigned int *)(a4 + 12);
  v177.m128i_i64[0] = v154.m128i_i64[0];
  v53 = _mm_loadu_si128(&v177);
  if ( v51 + 1 > v52 )
  {
    v155 = v50;
    v156 = v42;
    v154 = v53;
    sub_C8D5F0(a4, (const void *)(a4 + 16), v51 + 1, 0x10u, v42, v51 + 1);
    v51 = *(unsigned int *)(a4 + 8);
    v53 = _mm_load_si128(&v154);
    v50 = v155;
    v42 = v156;
  }
  *(__m128i *)(*(_QWORD *)a4 + 16 * v51) = v53;
  ++*(_DWORD *)(a4 + 8);
  if ( v50 )
  {
    v54 = *(unsigned int *)(a5 + 8);
    if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      v155 = v50;
      v156 = v42;
      sub_C8D5F0(a5, (const void *)(a5 + 16), v54 + 1, 8u, v42, v54 + 1);
      v54 = *(unsigned int *)(a5 + 8);
      v50 = v155;
      v42 = v156;
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v54) = v50;
    ++*(_DWORD *)(a5 + 8);
  }
  v175 = v42;
  v55 = *(_QWORD *)(a6 + 864);
  v56 = v157 & 0xFFFFFFFF00000000LL | (unsigned int)v176;
  if ( v42 )
  {
    v157 = v42;
    nullsub_1875();
    v161 = v56;
    v160 = v157;
    *(_QWORD *)(v55 + 384) = v157;
    *(_DWORD *)(v55 + 392) = v161;
    sub_33E2B60();
  }
  else
  {
    v159 = v157 & 0xFFFFFFFF00000000LL | (unsigned int)v176;
    *(_QWORD *)(v55 + 384) = 0;
    *(_DWORD *)(v55 + 392) = v159;
  }
}
