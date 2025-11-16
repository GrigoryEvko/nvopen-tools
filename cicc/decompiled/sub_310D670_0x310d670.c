// Function: sub_310D670
// Address: 0x310d670
//
__int64 __fastcall sub_310D670(__int64 a1, int *a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // r14
  __int64 *v6; // r12
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r9
  unsigned __int64 v9; // rdx
  int v10; // r14d
  unsigned __int64 v11; // rcx
  int v12; // eax
  unsigned __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rdx
  int v17; // r14d
  unsigned __int64 v18; // rcx
  int v19; // eax
  __int64 *v20; // r14
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // r9
  unsigned __int64 v23; // rdx
  int v24; // r13d
  unsigned __int64 v25; // rcx
  int v26; // eax
  __int64 v28; // r14
  __int64 v29; // r14
  unsigned __int64 *v30; // r12
  unsigned __int64 v31; // r14
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rsi
  unsigned __int64 v34; // rcx
  int v35; // eax
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rdx
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // rcx
  int v41; // eax
  unsigned __int64 *v42; // r14
  unsigned __int64 v43; // r13
  unsigned __int64 v44; // rdx
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rcx
  int v47; // eax
  __int64 v48; // r15
  __int64 v49; // r15
  float *v50; // r12
  __int64 v51; // r15
  unsigned __int64 v52; // rbx
  float *v53; // r15
  __int64 v54; // r14
  __int64 v55; // r14
  unsigned __int64 v56; // r14
  int *v57; // r12
  unsigned int v58; // r13d
  __int16 v59; // ax
  int v60; // r15d
  unsigned int v61; // r14d
  __int16 v62; // ax
  int v63; // r8d
  unsigned __int16 *v64; // r14
  unsigned int v65; // ecx
  __int16 v66; // bx
  unsigned int v67; // r13d
  int v68; // r15d
  unsigned __int64 v69; // rsi
  __int64 v70; // r14
  unsigned __int64 v71; // r14
  int *v72; // r12
  unsigned int v73; // r13d
  char v74; // al
  int v75; // r15d
  unsigned int v76; // r8d
  char v77; // al
  int v78; // r14d
  char *v79; // r14
  unsigned int v80; // ecx
  unsigned __int8 v81; // bl
  unsigned int v82; // r13d
  int v83; // r15d
  __int64 v84; // r14
  __int64 v85; // r14
  int *v86; // r12
  unsigned int v87; // r13d
  unsigned int v88; // r9d
  unsigned __int64 v89; // rdx
  int v90; // r15d
  unsigned __int64 v91; // rcx
  int v92; // edi
  unsigned __int64 v93; // rax
  unsigned int v94; // ecx
  unsigned int v95; // r15d
  unsigned __int64 v96; // rdx
  int v97; // r14d
  unsigned __int64 v98; // rsi
  int v99; // r8d
  int *v100; // r14
  unsigned int v101; // ebx
  unsigned int v102; // r9d
  unsigned __int64 v103; // rdx
  int v104; // r13d
  unsigned __int64 v105; // rsi
  int v106; // edi
  __int64 v107; // r15
  __int64 v108; // r15
  double *v109; // r12
  __int64 v110; // r15
  unsigned __int64 v111; // rbx
  double *v112; // r15
  __int64 v113; // r14
  __int64 v114; // r14
  unsigned __int64 v115; // r14
  int *v116; // r12
  unsigned int v117; // r13d
  unsigned __int64 v118; // rsi
  int v119; // r15d
  unsigned int v120; // r14d
  unsigned __int64 v121; // rsi
  int v122; // r8d
  unsigned __int16 *v123; // r14
  unsigned int v124; // r13d
  unsigned __int64 v125; // rsi
  int v126; // r15d
  __int64 v127; // r14
  unsigned __int64 v128; // r14
  int *v129; // r12
  unsigned int v130; // r13d
  unsigned __int64 v131; // rsi
  int v132; // r15d
  unsigned int v133; // r8d
  unsigned __int64 v134; // rsi
  int v135; // r14d
  unsigned __int8 *v136; // r14
  unsigned int v137; // r13d
  unsigned __int64 v138; // rsi
  int v139; // r15d
  __int64 v140; // r14
  __int64 v141; // r14
  unsigned int *v142; // r12
  unsigned int v143; // r14d
  unsigned __int64 v144; // rdx
  unsigned __int64 v145; // rsi
  unsigned __int64 v146; // rdi
  int v147; // r9d
  unsigned __int64 v148; // r8
  unsigned int v149; // r14d
  unsigned __int64 v150; // rdx
  unsigned int v151; // ecx
  unsigned __int64 v152; // rsi
  unsigned int v153; // r8d
  unsigned __int64 v154; // rsi
  unsigned int *v155; // r14
  unsigned int v156; // r13d
  unsigned __int64 v157; // rdx
  unsigned __int64 v158; // rsi
  unsigned __int64 v159; // rcx
  int v160; // edi
  unsigned __int64 v163; // [rsp+10h] [rbp-70h]
  unsigned __int64 v164; // [rsp+10h] [rbp-70h]
  unsigned __int64 v165; // [rsp+10h] [rbp-70h]
  int v166; // [rsp+10h] [rbp-70h]
  unsigned int v167; // [rsp+10h] [rbp-70h]
  unsigned int v168; // [rsp+10h] [rbp-70h]
  unsigned __int64 v169; // [rsp+10h] [rbp-70h]
  unsigned __int64 v170; // [rsp+10h] [rbp-70h]
  __int64 v171; // [rsp+18h] [rbp-68h]
  __int64 v172; // [rsp+18h] [rbp-68h]
  unsigned __int64 v173; // [rsp+18h] [rbp-68h]
  __int64 v174; // [rsp+18h] [rbp-68h]
  bool v175; // [rsp+18h] [rbp-68h]
  bool v176; // [rsp+18h] [rbp-68h]
  unsigned __int8 v177; // [rsp+18h] [rbp-68h]
  unsigned __int8 v178; // [rsp+18h] [rbp-68h]
  __int64 v179; // [rsp+18h] [rbp-68h]
  unsigned int v180; // [rsp+18h] [rbp-68h]
  unsigned int v181; // [rsp+18h] [rbp-68h]
  int v182; // [rsp+18h] [rbp-68h]
  unsigned int v183; // [rsp+18h] [rbp-68h]
  __int64 v184; // [rsp+18h] [rbp-68h]
  __int64 *v185; // [rsp+20h] [rbp-60h]
  unsigned __int64 *v186; // [rsp+20h] [rbp-60h]
  float *v187; // [rsp+20h] [rbp-60h]
  __int16 *v188; // [rsp+20h] [rbp-60h]
  int *v189; // [rsp+20h] [rbp-60h]
  int *v190; // [rsp+20h] [rbp-60h]
  double *v191; // [rsp+20h] [rbp-60h]
  unsigned __int16 *v192; // [rsp+20h] [rbp-60h]
  unsigned __int8 *v193; // [rsp+20h] [rbp-60h]
  unsigned int *v194; // [rsp+20h] [rbp-60h]
  char *v196; // [rsp+30h] [rbp-50h] BYREF
  size_t v197; // [rsp+38h] [rbp-48h]
  _QWORD v198[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a1;
  switch ( *(_DWORD *)(a3 + 36) )
  {
    case 0:
    case 0xB:
      BUG();
    case 1:
      v48 = *(_QWORD *)(a3 + 64);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      v49 = v48;
      *(_BYTE *)(a1 + 16) = 0;
      v187 = (float *)&a2[v49];
      if ( &a2[v49] == a2 )
        return v3;
      v50 = (float *)a2;
      v51 = ((v49 * 4) >> 2) - 1;
      do
      {
        sub_11F4620(
          (__int64 *)&v196,
          (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
          58,
          (__int64)"%f",
          *v50);
        v52 = v51 + v197;
        v51 += v197;
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        ++v50;
      }
      while ( v187 != v50 );
      v3 = a1;
      sub_2240E30(a1, v52);
      sub_11F4620(
        (__int64 *)&v196,
        (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
        58,
        (__int64)"%f",
        *(float *)a2);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v53 = (float *)a2;
LABEL_90:
      if ( v187 == ++v53 )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        sub_11F4620(
          (__int64 *)&v196,
          (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
          58,
          (__int64)"%f",
          *v53);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 == (char *)v198 )
          goto LABEL_90;
        ++v53;
        j_j___libc_free_0((unsigned __int64)v196);
        if ( v187 == v53 )
          return v3;
      }
      goto LABEL_382;
    case 2:
      v107 = *(_QWORD *)(a3 + 64);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      v108 = 2 * v107;
      *(_BYTE *)(a1 + 16) = 0;
      v191 = (double *)&a2[v108];
      if ( &a2[v108] == a2 )
        return v3;
      v109 = (double *)a2;
      v110 = ((v108 * 4) >> 3) - 1;
      do
      {
        sub_11F4620(
          (__int64 *)&v196,
          (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
          328,
          (__int64)"%f",
          *v109);
        v111 = v110 + v197;
        v110 += v197;
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        ++v109;
      }
      while ( v191 != v109 );
      v3 = a1;
      sub_2240E30(a1, v111);
      sub_11F4620(
        (__int64 *)&v196,
        (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
        328,
        (__int64)"%f",
        *(double *)a2);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v112 = (double *)a2;
LABEL_199:
      if ( v191 == ++v112 )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        sub_11F4620(
          (__int64 *)&v196,
          (__int64 (__fastcall *)(_BYTE *, __int64, __int64, __va_list_tag *))&vsnprintf,
          328,
          (__int64)"%f",
          *v112);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 == (char *)v198 )
          goto LABEL_199;
        ++v112;
        j_j___libc_free_0((unsigned __int64)v196);
        if ( v191 == v112 )
          return v3;
      }
      goto LABEL_382;
    case 3:
      v70 = *(_QWORD *)(a3 + 64);
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = 0;
      v189 = (int *)((char *)a2 + v70);
      *(_QWORD *)a1 = a1 + 16;
      if ( (int *)((char *)a2 + v70) == a2 )
        return v3;
      v71 = v70 - 1;
      v72 = a2;
      do
      {
        v73 = *(char *)v72;
        v74 = *(_BYTE *)v72 >> 7;
        if ( *(char *)v72 < 0 )
          v73 = -*(char *)v72;
        if ( v73 <= 9 )
          v75 = 1;
        else
          v75 = (v73 > 0x63) + 2;
        v177 = *(_BYTE *)v72 >> 7;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, (v74 + (_BYTE)v75) & 0xF, 45);
        sub_2554A60(&v196[v177], v75, v73);
        v71 += v197;
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        v72 = (int *)((char *)v72 + 1);
      }
      while ( v189 != v72 );
      v3 = a1;
      sub_2240E30(a1, v71);
      v76 = *(char *)a2;
      v77 = *(_BYTE *)a2 >> 7;
      if ( *(char *)a2 < 0 )
        v76 = -*(char *)a2;
      if ( v76 <= 9 )
        v78 = 1;
      else
        v78 = (v76 > 0x63) + 2;
      v178 = *(_BYTE *)a2 >> 7;
      v167 = v76;
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, (v77 + (_BYTE)v78) & 0xF, 45);
      sub_2554A60(&v196[v178], v78, v167);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v79 = (char *)a2 + 1;
      if ( v189 == (int *)((char *)a2 + 1) )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v80 = -*v79;
        v81 = (unsigned __int8)*v79 >> 7;
        if ( *v79 >= 0 )
          v80 = *v79;
        v82 = v80;
        if ( v80 <= 9 )
          v83 = 1;
        else
          v83 = (v80 > 0x63) + 2;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, (v81 + (_BYTE)v83) & 0xF, 45);
        sub_2554A60(&v196[v81], v83, v82);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v189 == (int *)++v79 )
          return v3;
      }
      goto LABEL_382;
    case 4:
      v127 = *(_QWORD *)(a3 + 64);
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = 0;
      v193 = (unsigned __int8 *)a2 + v127;
      *(_QWORD *)a1 = a1 + 16;
      if ( (int *)((char *)a2 + v127) == a2 )
        return v3;
      v128 = v127 - 1;
      v129 = a2;
      do
      {
        v130 = *(unsigned __int8 *)v129;
        if ( v130 <= 9 )
        {
          v131 = 1;
          v132 = 1;
        }
        else
        {
          v131 = 3LL - (v130 < 0x64);
          v132 = 3 - (v130 < 0x64);
        }
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v131, 45);
        sub_2554A60(v196, v132, v130);
        v128 += v197;
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        v129 = (int *)((char *)v129 + 1);
      }
      while ( v193 != (unsigned __int8 *)v129 );
      v3 = a1;
      sub_2240E30(a1, v128);
      v133 = *(unsigned __int8 *)a2;
      if ( v133 <= 9 )
      {
        v134 = 1;
        v135 = 1;
      }
      else
      {
        v134 = 3LL - (v133 < 0x64);
        v135 = 3 - (v133 < 0x64);
      }
      v183 = *(unsigned __int8 *)a2;
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, v134, 45);
      sub_2554A60(v196, v135, v183);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v136 = (unsigned __int8 *)a2 + 1;
      if ( v193 == (unsigned __int8 *)((char *)a2 + 1) )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v137 = *v136;
        if ( v137 <= 9 )
        {
          v138 = 1;
          v139 = 1;
        }
        else
        {
          v138 = 3LL - (v137 < 0x64);
          v139 = 3 - (v137 < 0x64);
        }
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v138, 45);
        sub_2554A60(v196, v139, v137);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v193 == ++v136 )
          return v3;
      }
      goto LABEL_382;
    case 5:
      v54 = *(_QWORD *)(a3 + 64);
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = 0;
      v55 = 2 * v54;
      v188 = (__int16 *)((char *)a2 + v55);
      *(_QWORD *)a1 = a1 + 16;
      if ( (int *)((char *)a2 + v55) == a2 )
        return v3;
      v56 = (v55 >> 1) - 1;
      v57 = a2;
      do
      {
        v58 = *(__int16 *)v57;
        v59 = *(_WORD *)v57 >> 15;
        if ( *(__int16 *)v57 < 0 )
          v58 = -*(__int16 *)v57;
        if ( v58 <= 9 )
        {
          v60 = 1;
        }
        else if ( v58 <= 0x63 )
        {
          v60 = 2;
        }
        else if ( v58 <= 0x3E7 )
        {
          v60 = 3;
        }
        else
        {
          v60 = (v58 > 0x270F) + 4;
        }
        v175 = *(__int16 *)v57 < 0;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, ((_BYTE)v59 + (_BYTE)v60) & 0x1F, 45);
        sub_2554A60(&v196[v175], v60, v58);
        v56 += v197;
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        v57 = (int *)((char *)v57 + 2);
      }
      while ( v188 != (__int16 *)v57 );
      v3 = a1;
      sub_2240E30(a1, v56);
      v61 = *(__int16 *)a2;
      v62 = *(_WORD *)a2 >> 15;
      if ( *(__int16 *)a2 < 0 )
        v61 = -*(__int16 *)a2;
      if ( v61 <= 9 )
      {
        v63 = 1;
      }
      else if ( v61 <= 0x63 )
      {
        v63 = 2;
      }
      else if ( v61 <= 0x3E7 )
      {
        v63 = 3;
      }
      else
      {
        v63 = (v61 > 0x270F) + 4;
      }
      v166 = v63;
      v176 = *(__int16 *)a2 < 0;
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, ((_BYTE)v62 + (_BYTE)v63) & 0x1F, 45);
      sub_2554A60(&v196[v176], v166, v61);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v64 = (unsigned __int16 *)a2 + 1;
      if ( v188 == (__int16 *)((char *)a2 + 2) )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v65 = -(__int16)*v64;
        v66 = *v64 >> 15;
        if ( (*v64 & 0x8000u) == 0 )
          v65 = (__int16)*v64;
        v67 = v65;
        if ( v65 <= 9 )
        {
          v68 = 1;
        }
        else if ( v65 <= 0x63 )
        {
          v68 = 2;
        }
        else if ( v65 <= 0x3E7 )
        {
          v68 = 3;
        }
        else
        {
          v68 = (v65 > 0x270F) + 4;
        }
        v69 = (((*v64 & 0x8000u) != 0) + (_BYTE)v68) & 0x1F;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v69, 45);
        sub_2554A60(&v196[(unsigned __int8)v66], v68, v67);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v188 == (__int16 *)++v64 )
          return v3;
      }
      goto LABEL_382;
    case 6:
      v113 = *(_QWORD *)(a3 + 64);
      *(_BYTE *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = 0;
      v114 = 2 * v113;
      v192 = (unsigned __int16 *)((char *)a2 + v114);
      *(_QWORD *)a1 = a1 + 16;
      if ( (int *)((char *)a2 + v114) == a2 )
        return v3;
      v115 = (v114 >> 1) - 1;
      v116 = a2;
      do
      {
        v117 = *(unsigned __int16 *)v116;
        if ( v117 <= 9 )
        {
          v118 = 1;
          v119 = 1;
        }
        else if ( v117 <= 0x63 )
        {
          v118 = 2;
          v119 = 2;
        }
        else if ( v117 <= 0x3E7 )
        {
          v118 = 3;
          v119 = 3;
        }
        else
        {
          v118 = 5LL - (v117 < 0x2710);
          v119 = 5 - (v117 < 0x2710);
        }
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v118, 45);
        sub_2554A60(v196, v119, v117);
        v115 += v197;
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        v116 = (int *)((char *)v116 + 2);
      }
      while ( v192 != (unsigned __int16 *)v116 );
      v3 = a1;
      sub_2240E30(a1, v115);
      v120 = *(unsigned __int16 *)a2;
      if ( v120 <= 9 )
      {
        v121 = 1;
        v122 = 1;
      }
      else if ( v120 <= 0x63 )
      {
        v121 = 2;
        v122 = 2;
      }
      else if ( v120 <= 0x3E7 )
      {
        v121 = 3;
        v122 = 3;
      }
      else
      {
        v121 = 5LL - (v120 < 0x2710);
        v122 = 5 - (v120 < 0x2710);
      }
      v182 = v122;
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, v121, 45);
      sub_2554A60(v196, v182, v120);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v123 = (unsigned __int16 *)a2 + 1;
      if ( v192 == (unsigned __int16 *)((char *)a2 + 2) )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v124 = *v123;
        if ( v124 <= 9 )
        {
          v125 = 1;
          v126 = 1;
        }
        else if ( v124 <= 0x63 )
        {
          v125 = 2;
          v126 = 2;
        }
        else if ( v124 <= 0x3E7 )
        {
          v125 = 3;
          v126 = 3;
        }
        else
        {
          v125 = 5LL - (v124 < 0x2710);
          v126 = 5 - (v124 < 0x2710);
        }
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v125, 45);
        sub_2554A60(v196, v126, v124);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v192 == ++v123 )
          return v3;
      }
      goto LABEL_382;
    case 7:
      v84 = *(_QWORD *)(a3 + 64);
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
      v85 = v84;
      v190 = &a2[v85];
      *(_QWORD *)a1 = a1 + 16;
      if ( &a2[v85] == a2 )
        return v3;
      v86 = a2;
      v179 = ((v85 * 4) >> 2) - 1;
      do
      {
        v87 = (unsigned int)*v86 >> 31;
        v88 = abs32(*v86);
        if ( v88 <= 9 )
        {
          v90 = 1;
        }
        else if ( v88 <= 0x63 )
        {
          v90 = 2;
        }
        else if ( v88 <= 0x3E7 )
        {
          v90 = 3;
        }
        else
        {
          v89 = v88;
          if ( v88 <= 0x270F )
          {
            v90 = 4;
          }
          else
          {
            v90 = 1;
            while ( 1 )
            {
              v91 = v89;
              v92 = v90;
              v90 += 4;
              v89 /= 0x2710u;
              if ( v91 <= 0x1869F )
                break;
              if ( (unsigned int)v89 <= 0x63 )
              {
                v90 = v92 + 5;
                break;
              }
              if ( (unsigned int)v89 <= 0x3E7 )
              {
                v90 = v92 + 6;
                break;
              }
              if ( (unsigned int)v89 <= 0x270F )
              {
                v90 = v92 + 7;
                break;
              }
            }
          }
        }
        v168 = v88;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v87 + v90, 45);
        sub_2554A60(&v196[(unsigned __int8)v87], v90, v168);
        v93 = v197 + v179;
        v179 += v197;
        if ( v196 != (char *)v198 )
        {
          v169 = v93;
          j_j___libc_free_0((unsigned __int64)v196);
          v93 = v169;
        }
        ++v86;
      }
      while ( v190 != v86 );
      v3 = a1;
      sub_2240E30(a1, v93);
      v94 = (unsigned int)*a2 >> 31;
      v95 = abs32(*a2);
      if ( v95 <= 9 )
      {
        v97 = 1;
      }
      else if ( v95 <= 0x63 )
      {
        v97 = 2;
      }
      else if ( v95 <= 0x3E7 )
      {
        v97 = 3;
      }
      else
      {
        v96 = v95;
        if ( v95 <= 0x270F )
        {
          v97 = 4;
        }
        else
        {
          v97 = 1;
          while ( 1 )
          {
            v98 = v96;
            v99 = v97;
            v97 += 4;
            v96 /= 0x2710u;
            if ( v98 <= 0x1869F )
              break;
            if ( (unsigned int)v96 <= 0x63 )
            {
              v97 = v99 + 5;
              break;
            }
            if ( (unsigned int)v96 <= 0x3E7 )
            {
              v97 = v99 + 6;
              break;
            }
            if ( (unsigned int)v96 <= 0x270F )
            {
              v97 = v99 + 7;
              break;
            }
          }
        }
      }
      v180 = (unsigned int)*a2 >> 31;
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, v94 + v97, 45);
      sub_2554A60(&v196[(unsigned __int8)v180], v97, v95);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v100 = a2 + 1;
      if ( v190 == a2 + 1 )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v101 = (unsigned int)*v100 >> 31;
        v102 = abs32(*v100);
        if ( v102 <= 9 )
        {
          v104 = 1;
        }
        else if ( v102 <= 0x63 )
        {
          v104 = 2;
        }
        else if ( v102 <= 0x3E7 )
        {
          v104 = 3;
        }
        else
        {
          v103 = v102;
          if ( v102 <= 0x270F )
          {
            v104 = 4;
          }
          else
          {
            v104 = 1;
            while ( 1 )
            {
              v105 = v103;
              v106 = v104;
              v104 += 4;
              v103 /= 0x2710u;
              if ( v105 <= 0x1869F )
                break;
              if ( (unsigned int)v103 <= 0x63 )
              {
                v104 = v106 + 5;
                break;
              }
              if ( (unsigned int)v103 <= 0x3E7 )
              {
                v104 = v106 + 6;
                break;
              }
              if ( (unsigned int)v103 <= 0x270F )
              {
                v104 = v106 + 7;
                break;
              }
            }
          }
        }
        v181 = v102;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v101 + v104, 45);
        sub_2554A60(&v196[(unsigned __int8)v101], v104, v181);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v190 == ++v100 )
          return v3;
      }
      goto LABEL_382;
    case 8:
      v140 = *(_QWORD *)(a3 + 64);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      v141 = v140;
      *(_BYTE *)(a1 + 16) = 0;
      v194 = (unsigned int *)&a2[v141];
      if ( &a2[v141] == a2 )
        return v3;
      v142 = (unsigned int *)a2;
      v184 = ((v141 * 4) >> 2) - 1;
      do
      {
        v143 = *v142;
        if ( *v142 <= 9 )
        {
          v145 = 1;
        }
        else if ( v143 <= 0x63 )
        {
          v145 = 2;
        }
        else if ( v143 <= 0x3E7 )
        {
          v145 = 3;
        }
        else
        {
          v144 = v143;
          if ( v143 <= 0x270F )
          {
            v145 = 4;
          }
          else
          {
            LODWORD(v145) = 1;
            while ( 1 )
            {
              v146 = v144;
              v147 = v145;
              v145 = (unsigned int)(v145 + 4);
              v144 /= 0x2710u;
              if ( v146 <= 0x1869F )
                break;
              if ( (unsigned int)v144 <= 0x63 )
              {
                v145 = (unsigned int)(v147 + 5);
                break;
              }
              if ( (unsigned int)v144 <= 0x3E7 )
              {
                v145 = (unsigned int)(v147 + 6);
                break;
              }
              if ( (unsigned int)v144 <= 0x270F )
              {
                v145 = (unsigned int)(v147 + 7);
                break;
              }
            }
          }
        }
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v145, 0);
        sub_2554A60(v196, v197, v143);
        v148 = v197 + v184;
        v184 += v197;
        if ( v196 != (char *)v198 )
        {
          v170 = v148;
          j_j___libc_free_0((unsigned __int64)v196);
          v148 = v170;
        }
        ++v142;
      }
      while ( v194 != v142 );
      v3 = a1;
      sub_2240E30(a1, v148);
      v149 = *a2;
      if ( (unsigned int)*a2 <= 9 )
      {
        v154 = 1;
      }
      else if ( v149 <= 0x63 )
      {
        v154 = 2;
      }
      else if ( v149 <= 0x3E7 )
      {
        v154 = 3;
      }
      else
      {
        v150 = v149;
        if ( v149 <= 0x270F )
        {
          v154 = 4;
        }
        else
        {
          v151 = 1;
          do
          {
            v152 = v150;
            v153 = v151;
            v151 += 4;
            v150 /= 0x2710u;
            if ( v152 <= 0x1869F )
            {
              v154 = v151;
              goto LABEL_272;
            }
            if ( (unsigned int)v150 <= 0x63 )
            {
              v154 = v153 + 5;
              goto LABEL_272;
            }
            if ( (unsigned int)v150 <= 0x3E7 )
            {
              v154 = v153 + 6;
              goto LABEL_272;
            }
          }
          while ( (unsigned int)v150 > 0x270F );
          v154 = v153 + 7;
        }
      }
LABEL_272:
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, v154, 0);
      sub_2554A60(v196, v197, v149);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v155 = (unsigned int *)(a2 + 1);
      if ( v194 == (unsigned int *)(a2 + 1) )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v156 = *v155;
        if ( *v155 <= 9 )
        {
          v158 = 1;
        }
        else if ( v156 <= 0x63 )
        {
          v158 = 2;
        }
        else if ( v156 <= 0x3E7 )
        {
          v158 = 3;
        }
        else
        {
          v157 = v156;
          if ( v156 <= 0x270F )
          {
            v158 = 4;
          }
          else
          {
            LODWORD(v158) = 1;
            while ( 1 )
            {
              v159 = v157;
              v160 = v158;
              v158 = (unsigned int)(v158 + 4);
              v157 /= 0x2710u;
              if ( v159 <= 0x1869F )
                break;
              if ( (unsigned int)v157 <= 0x63 )
              {
                v158 = (unsigned int)(v160 + 5);
                break;
              }
              if ( (unsigned int)v157 <= 0x3E7 )
              {
                v158 = (unsigned int)(v160 + 6);
                break;
              }
              if ( (unsigned int)v157 <= 0x270F )
              {
                v158 = (unsigned int)(v160 + 7);
                break;
              }
            }
          }
        }
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, v158, 0);
        sub_2554A60(v196, v197, v156);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v194 == ++v155 )
          return v3;
      }
      goto LABEL_382;
    case 9:
      v4 = *(_QWORD *)(a3 + 64);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      v5 = 2 * v4;
      *(_BYTE *)(a1 + 16) = 0;
      v185 = (__int64 *)&a2[v5];
      if ( &a2[v5] == a2 )
        return v3;
      v6 = (__int64 *)a2;
      v171 = ((v5 * 4) >> 3) - 1;
      do
      {
        v7 = (unsigned __int64)*v6 >> 63;
        v8 = abs64(*v6);
        if ( v8 <= 9 )
        {
          v10 = 1;
        }
        else if ( v8 <= 0x63 )
        {
          v10 = 2;
        }
        else if ( v8 <= 0x3E7 )
        {
          v10 = 3;
        }
        else if ( v8 <= 0x270F )
        {
          v10 = 4;
        }
        else
        {
          v9 = v8;
          v10 = 1;
          while ( 1 )
          {
            v11 = v9;
            v12 = v10;
            v10 += 4;
            v9 /= 0x2710u;
            if ( v11 <= 0x1869F )
              break;
            if ( v11 <= 0xF423F )
            {
              v10 = v12 + 5;
              break;
            }
            if ( v11 <= (unsigned __int64)&loc_98967F )
            {
              v10 = v12 + 6;
              break;
            }
            if ( v11 <= 0x5F5E0FF )
            {
              v10 = v12 + 7;
              break;
            }
          }
        }
        v163 = v8;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, (unsigned int)(v10 + v7), 45);
        sub_1249540(&v196[v7], v10, v163);
        v13 = v197 + v171;
        v171 += v197;
        if ( v196 != (char *)v198 )
        {
          v164 = v13;
          j_j___libc_free_0((unsigned __int64)v196);
          v13 = v164;
        }
        ++v6;
      }
      while ( v185 != v6 );
      v3 = a1;
      sub_2240E30(a1, v13);
      v14 = *(_QWORD *)a2 >> 63;
      v15 = abs64(*(_QWORD *)a2);
      if ( v15 <= 9 )
      {
        v17 = 1;
      }
      else if ( v15 <= 0x63 )
      {
        v17 = 2;
      }
      else if ( v15 <= 0x3E7 )
      {
        v17 = 3;
      }
      else if ( v15 <= 0x270F )
      {
        v17 = 4;
      }
      else
      {
        v16 = v15;
        v17 = 1;
        while ( 1 )
        {
          v18 = v16;
          v19 = v17;
          v17 += 4;
          v16 /= 0x2710u;
          if ( v18 <= 0x1869F )
            break;
          if ( v18 <= 0xF423F )
          {
            v17 = v19 + 5;
            break;
          }
          if ( v18 <= (unsigned __int64)&loc_98967F )
          {
            v17 = v19 + 6;
            break;
          }
          if ( v18 <= 0x5F5E0FF )
          {
            v17 = v19 + 7;
            break;
          }
        }
      }
      v172 = *(_QWORD *)a2 >> 63;
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, (unsigned int)(v17 + v14), 45);
      sub_1249540(&v196[v172], v17, v15);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      v20 = (__int64 *)(a2 + 2);
      if ( v185 == (__int64 *)(a2 + 2) )
        return v3;
      while ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490((unsigned __int64 *)a1, ",", 1u);
        v21 = (unsigned __int64)*v20 >> 63;
        v22 = abs64(*v20);
        if ( v22 <= 9 )
        {
          v24 = 1;
        }
        else if ( v22 <= 0x63 )
        {
          v24 = 2;
        }
        else if ( v22 <= 0x3E7 )
        {
          v24 = 3;
        }
        else if ( v22 <= 0x270F )
        {
          v24 = 4;
        }
        else
        {
          v23 = v22;
          v24 = 1;
          while ( 1 )
          {
            v25 = v23;
            v26 = v24;
            v24 += 4;
            v23 /= 0x2710u;
            if ( v25 <= 0x1869F )
              break;
            if ( v25 <= 0xF423F )
            {
              v24 = v26 + 5;
              break;
            }
            if ( v25 <= (unsigned __int64)&loc_98967F )
            {
              v24 = v26 + 6;
              break;
            }
            if ( v25 <= 0x5F5E0FF )
            {
              v24 = v26 + 7;
              break;
            }
          }
        }
        v173 = v22;
        v196 = (char *)v198;
        sub_2240A50((__int64 *)&v196, (unsigned int)(v24 + v21), 45);
        sub_1249540(&v196[v21], v24, v173);
        sub_2241490((unsigned __int64 *)a1, v196, v197);
        if ( v196 != (char *)v198 )
          j_j___libc_free_0((unsigned __int64)v196);
        if ( v185 == ++v20 )
          return v3;
      }
      goto LABEL_382;
    case 0xA:
      v28 = *(_QWORD *)(a3 + 64);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      v29 = 2 * v28;
      *(_BYTE *)(a1 + 16) = 0;
      v186 = (unsigned __int64 *)&a2[v29];
      if ( &a2[v29] == a2 )
        return v3;
      v30 = (unsigned __int64 *)a2;
      v174 = ((v29 * 4) >> 3) - 1;
      break;
    default:
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = a1 + 16;
      *(_BYTE *)(a1 + 16) = 0;
      return v3;
  }
  do
  {
    v31 = *v30;
    if ( *v30 <= 9 )
    {
      v33 = 1;
    }
    else if ( v31 <= 0x63 )
    {
      v33 = 2;
    }
    else if ( v31 <= 0x3E7 )
    {
      v33 = 3;
    }
    else if ( v31 <= 0x270F )
    {
      v33 = 4;
    }
    else
    {
      v32 = *v30;
      LODWORD(v33) = 1;
      while ( 1 )
      {
        v34 = v32;
        v35 = v33;
        v33 = (unsigned int)(v33 + 4);
        v32 /= 0x2710u;
        if ( v34 <= 0x1869F )
          break;
        if ( v34 <= 0xF423F )
        {
          v33 = (unsigned int)(v35 + 5);
          break;
        }
        if ( v34 <= (unsigned __int64)&loc_98967F )
        {
          v33 = (unsigned int)(v35 + 6);
          break;
        }
        if ( v34 <= 0x5F5E0FF )
        {
          v33 = (unsigned int)(v35 + 7);
          break;
        }
      }
    }
    v196 = (char *)v198;
    sub_2240A50((__int64 *)&v196, v33, 0);
    sub_1249540(v196, v197, v31);
    v36 = v197 + v174;
    v174 += v197;
    if ( v196 != (char *)v198 )
    {
      v165 = v36;
      j_j___libc_free_0((unsigned __int64)v196);
      v36 = v165;
    }
    ++v30;
  }
  while ( v186 != v30 );
  v3 = a1;
  sub_2240E30(a1, v36);
  v37 = *(_QWORD *)a2;
  if ( *(_QWORD *)a2 <= 9u )
  {
    v39 = 1;
  }
  else if ( v37 <= 0x63 )
  {
    v39 = 2;
  }
  else if ( v37 <= 0x3E7 )
  {
    v39 = 3;
  }
  else if ( v37 <= 0x270F )
  {
    v39 = 4;
  }
  else
  {
    v38 = *(_QWORD *)a2;
    LODWORD(v39) = 1;
    while ( 1 )
    {
      v40 = v38;
      v41 = v39;
      v39 = (unsigned int)(v39 + 4);
      v38 /= 0x2710u;
      if ( v40 <= 0x1869F )
        break;
      if ( v40 <= 0xF423F )
      {
        v39 = (unsigned int)(v41 + 5);
        break;
      }
      if ( v40 <= (unsigned __int64)&loc_98967F )
      {
        v39 = (unsigned int)(v41 + 6);
        break;
      }
      if ( v40 <= 0x5F5E0FF )
      {
        v39 = (unsigned int)(v41 + 7);
        break;
      }
    }
  }
  v196 = (char *)v198;
  sub_2240A50((__int64 *)&v196, v39, 0);
  sub_1249540(v196, v197, v37);
  sub_2241490((unsigned __int64 *)a1, v196, v197);
  if ( v196 != (char *)v198 )
    j_j___libc_free_0((unsigned __int64)v196);
  v42 = (unsigned __int64 *)(a2 + 2);
  if ( v186 != (unsigned __int64 *)(a2 + 2) )
  {
    do
    {
      if ( *(_QWORD *)(a1 + 8) == 0x3FFFFFFFFFFFFFFFLL )
LABEL_382:
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490((unsigned __int64 *)a1, ",", 1u);
      v43 = *v42;
      if ( *v42 <= 9 )
      {
        v45 = 1;
      }
      else if ( v43 <= 0x63 )
      {
        v45 = 2;
      }
      else if ( v43 <= 0x3E7 )
      {
        v45 = 3;
      }
      else if ( v43 <= 0x270F )
      {
        v45 = 4;
      }
      else
      {
        v44 = *v42;
        LODWORD(v45) = 1;
        while ( 1 )
        {
          v46 = v44;
          v47 = v45;
          v45 = (unsigned int)(v45 + 4);
          v44 /= 0x2710u;
          if ( v46 <= 0x1869F )
            break;
          if ( v46 <= 0xF423F )
          {
            v45 = (unsigned int)(v47 + 5);
            break;
          }
          if ( v46 <= (unsigned __int64)&loc_98967F )
          {
            v45 = (unsigned int)(v47 + 6);
            break;
          }
          if ( v46 <= 0x5F5E0FF )
          {
            v45 = (unsigned int)(v47 + 7);
            break;
          }
        }
      }
      v196 = (char *)v198;
      sub_2240A50((__int64 *)&v196, v45, 0);
      sub_1249540(v196, v197, v43);
      sub_2241490((unsigned __int64 *)a1, v196, v197);
      if ( v196 != (char *)v198 )
        j_j___libc_free_0((unsigned __int64)v196);
      ++v42;
    }
    while ( v186 != v42 );
  }
  return v3;
}
