// Function: sub_1011B90
// Address: 0x1011b90
//
__int64 __fastcall sub_1011B90(unsigned __int64 a1, _BYTE *a2, _BYTE *a3, int a4, __m128i *a5, unsigned int a6)
{
  __int64 v6; // r15
  __int64 v8; // r13
  unsigned __int64 v9; // rbx
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rdx
  _QWORD *v13; // rdi
  int v14; // ecx
  int v15; // eax
  __int64 *v16; // rax
  __int64 **v17; // r9
  char v18; // al
  char v19; // al
  unsigned __int8 v20; // al
  __m128i v21; // xmm0
  __int64 v22; // rdi
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __int64 v26; // rax
  __int64 v27; // rax
  int v28; // edx
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  char v32; // al
  char v33; // al
  bool v34; // al
  __int64 v35; // rsi
  __int16 v36; // ax
  bool v37; // zf
  void *v38; // rax
  void *v39; // rdx
  void **v40; // rax
  char v41; // al
  char v42; // al
  void *v43; // rax
  __int64 v44; // rax
  bool v45; // al
  int v46; // edx
  __int64 v47; // rdi
  int v48; // eax
  int v49; // eax
  int v50; // eax
  __int64 v51; // rdx
  _BYTE *v52; // rax
  __m128i v53; // xmm4
  __m128i v54; // xmm5
  __m128i v55; // xmm6
  __m128i v56; // xmm7
  __int64 v57; // rdx
  unsigned __int8 v58; // al
  char v59; // al
  char v60; // al
  bool v61; // al
  __int64 v62; // rdx
  _BYTE *v63; // rax
  void *v64; // rax
  _BYTE *v65; // rdx
  char v66; // al
  __int64 v67; // rax
  char v68; // al
  int v69; // eax
  int v70; // eax
  __int64 *v71; // r10
  char v72; // r8
  __int64 v73; // rdx
  unsigned int v74; // esi
  unsigned __int64 v75; // rdx
  __int64 v76; // rax
  __int16 v77; // dx
  int v78; // eax
  unsigned int v79; // ebx
  void **v80; // rax
  void **v81; // rdx
  char v82; // al
  void **v83; // rdx
  __int64 *v84; // r10
  char v85; // r8
  __int64 v86; // rdx
  unsigned int v87; // esi
  unsigned __int64 v88; // rdx
  char v89; // al
  char v90; // al
  __int64 *v91; // r10
  char v92; // r8
  __int64 v93; // rdx
  unsigned int v94; // esi
  unsigned __int64 v95; // rdx
  char v96; // al
  char v97; // dl
  char v98; // al
  char v99; // al
  int v100; // eax
  unsigned __int8 v101; // al
  char v102; // [rsp+4h] [rbp-15Ch]
  __int64 v103; // [rsp+8h] [rbp-158h]
  __int64 *v104; // [rsp+8h] [rbp-158h]
  __int64 **v105; // [rsp+10h] [rbp-150h]
  __int64 v106; // [rsp+18h] [rbp-148h]
  __int64 **v107; // [rsp+18h] [rbp-148h]
  void **v108; // [rsp+18h] [rbp-148h]
  __int64 *v109; // [rsp+18h] [rbp-148h]
  int v110; // [rsp+20h] [rbp-140h]
  void *v111; // [rsp+20h] [rbp-140h]
  void *v112; // [rsp+20h] [rbp-140h]
  void *v113; // [rsp+20h] [rbp-140h]
  __int64 **v114; // [rsp+20h] [rbp-140h]
  void *v115; // [rsp+20h] [rbp-140h]
  char v116; // [rsp+20h] [rbp-140h]
  __int64 **v117; // [rsp+28h] [rbp-138h]
  __int64 **v118; // [rsp+28h] [rbp-138h]
  __int16 v119; // [rsp+28h] [rbp-138h]
  __int64 v120; // [rsp+28h] [rbp-138h]
  __int64 **v121; // [rsp+28h] [rbp-138h]
  _BYTE *v122; // [rsp+28h] [rbp-138h]
  __int64 **v123; // [rsp+28h] [rbp-138h]
  __int64 **v124; // [rsp+28h] [rbp-138h]
  unsigned __int64 v125; // [rsp+28h] [rbp-138h]
  int v126; // [rsp+28h] [rbp-138h]
  __int64 **v127; // [rsp+28h] [rbp-138h]
  __int64 **v128; // [rsp+28h] [rbp-138h]
  __int64 **v129; // [rsp+28h] [rbp-138h]
  __int64 *v130; // [rsp+28h] [rbp-138h]
  __int64 **v131; // [rsp+28h] [rbp-138h]
  __int64 **v132; // [rsp+30h] [rbp-130h]
  void **v133; // [rsp+30h] [rbp-130h]
  __int64 **v134; // [rsp+30h] [rbp-130h]
  __int64 **v135; // [rsp+30h] [rbp-130h]
  __int64 **v136; // [rsp+30h] [rbp-130h]
  __int64 v137; // [rsp+30h] [rbp-130h]
  __int64 **v138; // [rsp+30h] [rbp-130h]
  unsigned __int64 v139; // [rsp+30h] [rbp-130h]
  __int64 **v140; // [rsp+30h] [rbp-130h]
  __int64 **v141; // [rsp+30h] [rbp-130h]
  __int64 **v142; // [rsp+30h] [rbp-130h]
  char v143; // [rsp+30h] [rbp-130h]
  __int64 **v144; // [rsp+30h] [rbp-130h]
  __int64 **v145; // [rsp+30h] [rbp-130h]
  __int64 **v146; // [rsp+30h] [rbp-130h]
  __int64 **v147; // [rsp+30h] [rbp-130h]
  __int64 **v148; // [rsp+30h] [rbp-130h]
  char v149; // [rsp+38h] [rbp-128h]
  char v150; // [rsp+38h] [rbp-128h]
  char v151; // [rsp+38h] [rbp-128h]
  int v152; // [rsp+38h] [rbp-128h]
  __int64 **v155; // [rsp+40h] [rbp-120h]
  int v156; // [rsp+40h] [rbp-120h]
  __int64 v157; // [rsp+40h] [rbp-120h]
  int v158; // [rsp+40h] [rbp-120h]
  __int64 **v159; // [rsp+40h] [rbp-120h]
  void **v160; // [rsp+50h] [rbp-110h] BYREF
  __int64 v161; // [rsp+5Ch] [rbp-104h]
  unsigned __int64 v162; // [rsp+64h] [rbp-FCh] BYREF
  int v163; // [rsp+6Ch] [rbp-F4h]
  __int64 v164; // [rsp+70h] [rbp-F0h]
  __int64 v165; // [rsp+80h] [rbp-E0h]
  void ***v166; // [rsp+88h] [rbp-D8h] BYREF
  char v167; // [rsp+90h] [rbp-D0h]
  unsigned __int64 v168; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v169; // [rsp+A8h] [rbp-B8h]
  __int64 v170; // [rsp+B0h] [rbp-B0h]
  void ***v171; // [rsp+B8h] [rbp-A8h] BYREF
  char v172; // [rsp+C0h] [rbp-A0h]
  unsigned __int64 *v173; // [rsp+D0h] [rbp-90h] BYREF
  __int64 *v174; // [rsp+D8h] [rbp-88h]
  int v175; // [rsp+E0h] [rbp-80h]
  __m128i v176; // [rsp+E8h] [rbp-78h] BYREF
  __m128i v177; // [rsp+F8h] [rbp-68h]
  __m128i v178; // [rsp+108h] [rbp-58h]
  __m128i v179; // [rsp+118h] [rbp-48h]
  __int64 v180; // [rsp+128h] [rbp-38h]

  v6 = (__int64)a3;
  v8 = (__int64)a2;
  v9 = a1;
  if ( *a2 > 0x15u )
  {
    v11 = (unsigned int)a1;
  }
  else
  {
    if ( *a3 <= 0x15u )
      return sub_9719A0(a1, a2, (__int64)a3, a5->m128i_i64[0], a5->m128i_i64[1], a5[2].m128i_i64[1]);
    v11 = (unsigned int)sub_B52F50(a1);
    v9 = a1 & 0xFFFFFF00FFFFFFFFLL;
    v8 = v6;
    v6 = (__int64)a2;
  }
  v12 = *(_QWORD *)(v8 + 8);
  v13 = *(_QWORD **)v12;
  v14 = *(unsigned __int8 *)(v12 + 8);
  if ( (unsigned int)(v14 - 17) > 1 )
  {
    v17 = (__int64 **)sub_BCB2A0(v13);
    if ( !(_DWORD)v11 )
      return sub_AD6450((__int64)v17);
  }
  else
  {
    v15 = *(_DWORD *)(v12 + 32);
    BYTE4(v161) = (_BYTE)v14 == 18;
    LODWORD(v161) = v15;
    v16 = (__int64 *)sub_BCB2A0(v13);
    v17 = (__int64 **)sub_BCE1B0(v16, v161);
    if ( !(_DWORD)v11 )
      return sub_AD6450((__int64)v17);
  }
  if ( (_DWORD)v11 == 15 )
    return sub_AD6400((__int64)v17);
  if ( *(_BYTE *)v8 == 13 || *(_BYTE *)v6 == 13 )
    return sub_ACADE0(v17);
  v132 = v17;
  v18 = sub_1003090((__int64)a5, (unsigned __int8 *)v8);
  v17 = v132;
  if ( v18 || (v19 = sub_1003090((__int64)a5, (unsigned __int8 *)v6), v17 = v132, (v149 = v19) != 0) )
  {
    v155 = v17;
    v34 = sub_B535C0(v11);
    v17 = v155;
    v35 = v34;
    return sub_AD64C0((__int64)v17, v35, 0);
  }
  if ( v8 == v6 )
  {
    v59 = sub_B535D0(v11);
    v17 = v132;
    if ( !v59 )
    {
      v60 = sub_B53600(v11);
      v17 = v132;
      if ( !v60 )
        goto LABEL_17;
      return sub_AD6450((__int64)v17);
    }
    return sub_AD6400((__int64)v17);
  }
LABEL_17:
  if ( (unsigned int)(v11 - 7) > 1 )
    goto LABEL_18;
  v134 = v17;
  v119 = sub_9B4030((__int64 *)v6, 1023, 0, a5);
  v36 = sub_9B4030((__int64 *)v8, 1023, 0, a5);
  v17 = v134;
  if ( (a4 & 2) != 0 || (((unsigned __int8)v119 | (unsigned __int8)v36) & 3) == 0 )
  {
    v37 = (_DWORD)v11 == 7;
    goto LABEL_41;
  }
  if ( (v119 & 0x3FC) == 0 || (v36 & 0x3FC) == 0 )
  {
    v37 = (_DWORD)v11 == 8;
LABEL_41:
    v35 = v37;
    return sub_AD64C0((__int64)v17, v35, 0);
  }
LABEL_18:
  v20 = *(_BYTE *)v6;
  if ( *(_BYTE *)v6 == 18 )
  {
    v133 = (void **)(v6 + 24);
    goto LABEL_20;
  }
  v51 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
  if ( (unsigned int)v51 <= 1 && v20 <= 0x15u )
  {
    v136 = v17;
    v52 = sub_AD7630(v6, 1, v51);
    v17 = v136;
    if ( v52 )
    {
      v133 = (void **)(v52 + 24);
      if ( *v52 == 18 )
      {
LABEL_20:
        v21 = _mm_loadu_si128(a5);
        v22 = a5[2].m128i_i64[1];
        v162 = 0;
        v173 = &v162;
        v23 = _mm_loadu_si128(a5 + 1);
        v24 = _mm_loadu_si128(a5 + 2);
        v163 = 0;
        v175 = a4;
        v25 = _mm_loadu_si128(a5 + 3);
        v26 = a5[4].m128i_i64[0];
        v174 = (__int64 *)v8;
        v176 = v21;
        v180 = v26;
        v177 = v23;
        v178 = v24;
        v179 = v25;
        if ( !v22 )
          goto LABEL_42;
        v117 = v17;
        v106 = sub_B43CB0(v22);
        v27 = sub_989FD0(v11, v106, v8, (__int64)v133, 1);
        v17 = v117;
        v110 = v28;
        if ( !v27 )
          goto LABEL_42;
        if ( *((_BYTE *)v173 + 8) )
        {
          v29 = *v173;
          v30 = *(unsigned int *)v173;
          goto LABEL_24;
        }
        v71 = v174;
        v72 = v175;
        v73 = v174[1];
        if ( *(_BYTE *)(v73 + 8) == 17 )
        {
          v74 = *(_DWORD *)(v73 + 32);
          LODWORD(v169) = v74;
          if ( v74 > 0x40 )
          {
            v102 = v175;
            v104 = v174;
            sub_C43690((__int64)&v168, -1, 1);
            v17 = v117;
            v71 = v104;
            v72 = v102;
          }
          else
          {
            v75 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v74;
            if ( !v74 )
              v75 = 0;
            v168 = v75;
          }
        }
        else
        {
          LODWORD(v169) = 1;
          v168 = 1;
        }
        if ( (v72 & 2) != 0 )
        {
          v124 = v17;
          if ( (v72 & 4) != 0 )
          {
            v76 = sub_9B3E70(v71, (__int64 *)&v168, 504, 0, &v176);
            v17 = v124;
            v77 = v76 & 0x3FC;
LABEL_135:
            v30 = v77 & 0x1FB;
            goto LABEL_136;
          }
          v76 = sub_9B3E70(v71, (__int64 *)&v168, 1020, 0, &v176);
          v17 = v124;
          v30 = v76 & 0x3FC;
        }
        else
        {
          if ( (v72 & 4) != 0 )
          {
            v127 = v17;
            v76 = sub_9B3E70(v71, (__int64 *)&v168, 507, 0, &v176);
            v17 = v127;
            v77 = v76;
            goto LABEL_135;
          }
          v129 = v17;
          v76 = sub_9B3E70(v71, (__int64 *)&v168, 1023, 0, &v176);
          v17 = v129;
          v30 = (unsigned int)v76;
        }
LABEL_136:
        v125 = v30 | v76 & 0xFFFFFFFF00000000LL;
        if ( (unsigned int)v169 > 0x40 && v168 )
        {
          v103 = v30;
          v105 = v17;
          j_j___libc_free_0_0(v168);
          v30 = v103;
          v17 = v105;
        }
        v29 = v125;
LABEL_24:
        v31 = v30 | v29 & 0xFFFFFFFF00000000LL;
        if ( (_BYTE)v163 )
        {
          LODWORD(v162) = v31;
          WORD2(v162) = WORD2(v31);
        }
        else
        {
          v162 = v31;
          LOBYTE(v163) = 1;
        }
        v168 = (unsigned __int64)a5;
        v169 = v106;
        if ( ((unsigned int)v162 & v110) != 0 )
        {
          if ( ((unsigned __int16)~(_WORD)v110 & (unsigned __int16)v162 & 0x3FF) != 0 )
            goto LABEL_42;
          if ( (v110 & 0x204) == 0 )
            return sub_AD6400((__int64)v17);
        }
        else
        {
          if ( (v110 & 0x204) == 0 )
            return sub_AD6450((__int64)v17);
          v107 = v17;
          v32 = sub_FFEDC0(&v168);
          v17 = v107;
          if ( !v32 )
            return sub_AD6450((__int64)v17);
          if ( ((unsigned __int16)v162 & (unsigned __int16)~(_WORD)v110 & 0x3FF) != 0 )
          {
LABEL_42:
            v120 = (__int64)v17;
            v38 = sub_C33340();
            v17 = (__int64 **)v120;
            v39 = v38;
            if ( *v133 == v38 )
              v40 = (void **)v133[1];
            else
              v40 = v133;
            v41 = *((_BYTE *)v40 + 20);
            if ( (v41 & 7) == 1 )
            {
              v61 = sub_B535C0(v11);
              return sub_AD64C0(v120, v61, 0);
            }
            if ( (v41 & 8) != 0 && (v41 & 7) != 3 )
            {
              if ( (unsigned int)v11 > 0xB )
              {
                if ( (_DWORD)v11 != 14 )
                  goto LABEL_52;
              }
              else if ( (unsigned int)v11 <= 9 )
              {
                if ( (_DWORD)v11 != 1 && (unsigned int)(v11 - 4) > 1 )
                  goto LABEL_52;
                v111 = v39;
                v42 = sub_FFF450((__int64)&v173, 28);
                v17 = (__int64 **)v120;
                v39 = v111;
                if ( (v42 & 0x1C) != 0 )
                  goto LABEL_52;
                return sub_AD6450((__int64)v17);
              }
              v112 = v39;
              v66 = sub_FFF450((__int64)&v173, 28);
              v17 = (__int64 **)v120;
              v39 = v112;
              if ( (v66 & 0x1C) == 0 )
                return sub_AD6400((__int64)v17);
            }
LABEL_52:
            v37 = *(_BYTE *)v8 == 85;
            v164 = 248;
            v165 = 1;
            v166 = &v160;
            v167 = 0;
            if ( !v37 )
            {
LABEL_53:
              v20 = *(_BYTE *)v6;
              goto LABEL_54;
            }
            v67 = *(_QWORD *)(v8 - 32);
            if ( v67 && !*(_BYTE *)v67 && *(_QWORD *)(v67 + 24) == *(_QWORD *)(v8 + 80) && *(_DWORD *)(v67 + 36) == 248 )
            {
              v115 = v39;
              v128 = v17;
              v99 = sub_9940E0((__int64)&v166, *(_QWORD *)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF))));
              v17 = v128;
              v39 = v115;
              if ( v99 )
              {
                if ( v115 == *v160 )
                {
                  v100 = sub_C3E510((__int64)v160, (__int64)v133);
                  v39 = v115;
                  v17 = v128;
                }
                else
                {
                  v100 = sub_C37950((__int64)v160, (__int64)v133);
                  v17 = v128;
                  v39 = v115;
                }
                if ( !v100 )
                  goto LABEL_124;
              }
              v101 = *(_BYTE *)v8;
              v171 = &v160;
              v168 = 237;
              v170 = 1;
              v172 = 0;
              if ( v101 != 85 )
                goto LABEL_53;
              v67 = *(_QWORD *)(v8 - 32);
            }
            else
            {
              v168 = 237;
              v170 = 1;
              v171 = &v160;
              v172 = 0;
            }
            if ( !v67 )
              goto LABEL_53;
            if ( *(_BYTE *)v67 )
              goto LABEL_53;
            if ( *(_QWORD *)(v67 + 24) != *(_QWORD *)(v8 + 80) )
              goto LABEL_53;
            v113 = v39;
            if ( *(_DWORD *)(v67 + 36) != 237 )
              goto LABEL_53;
            v123 = v17;
            v68 = sub_9940E0((__int64)&v171, *(_QWORD *)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF))));
            v17 = v123;
            if ( !v68 )
              goto LABEL_53;
            v69 = v113 == *v160 ? sub_C3E510((__int64)v160, (__int64)v133) : sub_C37950((__int64)v160, (__int64)v133);
            v17 = v123;
            if ( v69 != 2 )
              goto LABEL_53;
LABEL_124:
            v157 = (__int64)v17;
            v70 = sub_987FE0(v8);
            v17 = (__int64 **)v157;
            switch ( (int)v11 )
            {
              case 1:
              case 9:
                return sub_AD6450((__int64)v17);
              case 2:
              case 3:
              case 10:
              case 11:
                return sub_AD64C0(v157, v70 == 237, 0);
              case 4:
              case 5:
              case 12:
              case 13:
                return sub_AD64C0(v157, v70 != 237, 0);
              case 6:
              case 14:
                return sub_AD6400((__int64)v17);
              default:
                BUG();
            }
          }
        }
        v118 = v17;
        v33 = sub_FFEDC0(&v168);
        v17 = v118;
        if ( !v33 )
          return sub_AD6400((__int64)v17);
        goto LABEL_42;
      }
    }
    v20 = *(_BYTE *)v6;
  }
  v53 = _mm_loadu_si128(a5);
  v162 = 0;
  v173 = &v162;
  v54 = _mm_loadu_si128(a5 + 1);
  v55 = _mm_loadu_si128(a5 + 2);
  v56 = _mm_loadu_si128(a5 + 3);
  v163 = 0;
  v57 = a5[4].m128i_i64[0];
  v174 = (__int64 *)v8;
  v175 = a4;
  v180 = v57;
  v176 = v53;
  v177 = v54;
  v178 = v55;
  v179 = v56;
LABEL_54:
  if ( v20 == 18 )
  {
    v135 = v17;
    v43 = sub_C33340();
    v17 = v135;
    if ( *(void **)(v6 + 24) == v43 )
      v44 = *(_QWORD *)(v6 + 32);
    else
      v44 = v6 + 24;
    v45 = (*(_BYTE *)(v44 + 20) & 7) == 3;
    goto LABEL_58;
  }
  v121 = v17;
  v137 = *(_QWORD *)(v6 + 8);
  v62 = (unsigned int)*(unsigned __int8 *)(v137 + 8) - 17;
  if ( (unsigned int)v62 > 1 || v20 > 0x15u )
    goto LABEL_83;
  v63 = sub_AD7630(v6, 0, v62);
  v17 = v121;
  if ( v63 )
  {
    v122 = v63;
    if ( *v63 == 18 )
    {
      v138 = v17;
      v64 = sub_C33340();
      v17 = v138;
      if ( *((void **)v122 + 3) == v64 )
        v65 = (_BYTE *)*((_QWORD *)v122 + 4);
      else
        v65 = v122 + 24;
      v45 = (v65[20] & 7) == 3;
LABEL_58:
      if ( !v45 )
        goto LABEL_83;
      goto LABEL_59;
    }
  }
  if ( *(_BYTE *)(v137 + 8) != 17 )
    goto LABEL_83;
  v78 = *(_DWORD *)(v137 + 32);
  v114 = v17;
  v139 = v9;
  v79 = 0;
  v126 = v78;
  while ( v126 != v79 )
  {
    v80 = (void **)sub_AD69F0((unsigned __int8 *)v6, v79);
    v81 = v80;
    if ( !v80 )
      goto LABEL_82;
    v82 = *(_BYTE *)v80;
    v108 = v81;
    if ( v82 != 13 )
    {
      if ( v82 != 18
        || (v81[3] == sub_C33340() ? (v83 = (void **)v108[4]) : (v83 = v108 + 3), (*((_BYTE *)v83 + 20) & 7) != 3) )
      {
LABEL_82:
        v9 = v139;
        goto LABEL_83;
      }
      v149 = 1;
    }
    ++v79;
  }
  v17 = v114;
  v9 = v139;
  if ( !v149 )
    goto LABEL_83;
LABEL_59:
  if ( (_DWORD)v11 == 11 )
  {
LABEL_63:
    if ( *((_BYTE *)v173 + 8) )
    {
      v46 = *(_DWORD *)v173;
    }
    else
    {
      v91 = v174;
      v92 = v175;
      v93 = v174[1];
      if ( *(_BYTE *)(v93 + 8) == 17 )
      {
        v94 = *(_DWORD *)(v93 + 32);
        LODWORD(v169) = v94;
        if ( v94 > 0x40 )
        {
          v130 = v174;
          v151 = v175;
          v146 = v17;
          sub_C43690((__int64)&v168, -1, 1);
          v17 = v146;
          v92 = v151;
          v91 = v130;
        }
        else
        {
          v95 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v94;
          if ( !v94 )
            v95 = 0;
          v168 = v95;
        }
      }
      else
      {
        LODWORD(v169) = 1;
        v168 = 1;
      }
      v158 = v92 & 2;
      if ( (v92 & 4) != 0 )
      {
        v142 = v17;
        v96 = sub_9B3E70(v91, (__int64 *)&v168, 24, 0, &v176);
        v17 = v142;
        v97 = v96;
        if ( v158 )
          v97 = v96 & 0xFC;
        LOBYTE(v46) = v97 & 0xFB;
      }
      else
      {
        v144 = v17;
        v98 = sub_9B3E70(v91, (__int64 *)&v168, 28, 0, &v176);
        v17 = v144;
        LOBYTE(v46) = v98;
        if ( v158 )
          LOBYTE(v46) = v98 & 0xFC;
      }
      if ( (unsigned int)v169 > 0x40 && v168 )
      {
        v143 = v46;
        v159 = v17;
        j_j___libc_free_0_0(v168);
        LOBYTE(v46) = v143;
        v17 = v159;
      }
    }
    if ( (v46 & 0x1C) == 0 )
    {
      v47 = (__int64)v17;
      if ( (_DWORD)v11 != 11 )
        return sub_AD6450(v47);
      return sub_AD6400(v47);
    }
    goto LABEL_83;
  }
  if ( (unsigned int)v11 > 0xB )
  {
    if ( (_DWORD)v11 != 12 )
      goto LABEL_83;
  }
  else if ( (_DWORD)v11 != 3 )
  {
    if ( (_DWORD)v11 != 4 )
      goto LABEL_83;
    goto LABEL_63;
  }
  v48 = a4 & 2;
  v156 = v48;
  v49 = v48 == 0 ? 31 : 28;
  if ( *((_BYTE *)v173 + 8) )
  {
    v50 = *(_DWORD *)v173;
    goto LABEL_71;
  }
  v84 = v174;
  v85 = v175;
  v86 = v174[1];
  if ( *(_BYTE *)(v86 + 8) == 17 )
  {
    v87 = *(_DWORD *)(v86 + 32);
    LODWORD(v169) = v87;
    if ( v87 > 0x40 )
    {
      v109 = v174;
      v116 = v175;
      v131 = v17;
      v152 = v49;
      sub_C43690((__int64)&v168, -1, 1);
      v49 = v152;
      v17 = v131;
      v85 = v116;
      v84 = v109;
    }
    else
    {
      v88 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v87;
      if ( !v87 )
        v88 = 0;
      v168 = v88;
    }
  }
  else
  {
    LODWORD(v169) = 1;
    v168 = 1;
  }
  if ( (v85 & 2) != 0 )
  {
    if ( (v85 & 4) == 0 )
    {
      v147 = v17;
      LOBYTE(v50) = sub_9B3E70(v84, (__int64 *)&v168, 28, 0, &v176);
      v17 = v147;
      LOBYTE(v50) = v50 & 0xFC;
      goto LABEL_164;
    }
    v140 = v17;
    v89 = sub_9B3E70(v84, (__int64 *)&v168, 24, 0, &v176);
    v17 = v140;
    v90 = v89 & 0xFC;
  }
  else
  {
    if ( (v85 & 4) == 0 )
    {
      v148 = v17;
      LOBYTE(v50) = sub_9B3E70(v84, (__int64 *)&v168, v49, 0, &v176);
      v17 = v148;
      goto LABEL_164;
    }
    v145 = v17;
    v90 = sub_9B3E70(v84, (__int64 *)&v168, v49 & 0x1FB, 0, &v176);
    v17 = v145;
  }
  LOBYTE(v50) = v90 & 0xFB;
LABEL_164:
  if ( (unsigned int)v169 > 0x40 && v168 )
  {
    v150 = v50;
    v141 = v17;
    j_j___libc_free_0_0(v168);
    LOBYTE(v50) = v150;
    v17 = v141;
  }
LABEL_71:
  if ( (v156 || (v50 & 3) == 0) && (v50 & 0x1C) == 0 )
  {
    v47 = (__int64)v17;
    if ( (_DWORD)v11 != 3 )
      return sub_AD6450(v47);
    return sub_AD6400(v47);
  }
LABEL_83:
  v58 = *(_BYTE *)v8;
  if ( *(_BYTE *)v8 == 86 || *(_BYTE *)v6 == 86 )
  {
    v9 = v9 & 0xFFFFFFFF00000000LL | (unsigned int)v11;
    result = sub_10115A0(v9, (_BYTE *)v8, v6, a5, a6);
    if ( result )
      return result;
    v58 = *(_BYTE *)v8;
  }
  if ( v58 == 84 )
    return sub_1012D30(v11 | v9 & 0xFFFFFFFF00000000LL, v8, v6, a5, a6, v17);
  result = 0;
  if ( *(_BYTE *)v6 == 84 )
    return sub_1012D30(v11 | v9 & 0xFFFFFFFF00000000LL, v8, v6, a5, a6, v17);
  return result;
}
