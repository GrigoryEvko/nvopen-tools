// Function: sub_1016D00
// Address: 0x1016d00
//
unsigned __int8 *__fastcall sub_1016D00(unsigned int a1, __int64 **a2, __int64 a3, __int64 a4, __m128i *a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int8 *v7; // r14
  __int64 v8; // r13
  __int64 **v10; // rbx
  __int64 v12; // r9
  unsigned int v14; // ebx
  __int64 v15; // r12
  bool v16; // al
  bool v17; // al
  char v18; // al
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v22; // r9
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // r8
  bool v27; // r13
  unsigned __int8 v28; // al
  void *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned __int8 *v32; // rax
  bool v33; // al
  __int64 v34; // r12
  unsigned int v35; // ebx
  unsigned __int64 v36; // rdi
  unsigned __int8 v37; // dl
  __int64 v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rdx
  char v43; // al
  _BYTE *v44; // r12
  _QWORD *v45; // rax
  unsigned int v46; // eax
  unsigned int v47; // eax
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdi
  _QWORD *v52; // rax
  __int64 v53; // rax
  int v54; // eax
  unsigned int v55; // eax
  __int64 v56; // r15
  _BYTE *v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rcx
  _BYTE *v60; // rdi
  __int64 v61; // r8
  int v62; // eax
  unsigned int v63; // r15d
  char *v64; // rax
  char v65; // al
  void *v66; // rax
  __int64 v67; // r15
  _QWORD *v68; // rax
  __int64 v69; // r8
  _QWORD *v70; // rdx
  __int64 v71; // rax
  bool v72; // al
  __int64 v73; // rax
  __int64 v74; // rax
  unsigned __int8 v75; // al
  __int64 v76; // rsi
  __int64 **v77; // rdi
  __int64 v78; // rdx
  unsigned __int8 *v79; // rax
  unsigned int v80; // eax
  bool v81; // bl
  bool v82; // al
  int v83; // edx
  __int64 v84; // rdx
  __int64 v85; // rsi
  __int64 v86; // rbx
  unsigned int v87; // edi
  int v88; // eax
  unsigned int v89; // edi
  unsigned int v90; // eax
  __m128i v91; // xmm1
  __m128i v92; // xmm2
  __int64 v93; // r12
  __m128i v94; // xmm3
  __m128i v95; // xmm4
  __m128i *v96; // rdi
  __m128i *v97; // rsi
  unsigned __int64 v98; // r12
  __int64 i; // rcx
  bool v100; // al
  __int64 v101; // r15
  _BYTE *v102; // rax
  void *v103; // rax
  __int64 v104; // rbx
  __int64 v105; // rdx
  _BYTE *v106; // rax
  unsigned int v107; // ebx
  __int64 v109; // rdx
  _BYTE *v110; // rax
  int v111; // eax
  unsigned __int8 *v112; // rdx
  __int64 v113; // rdi
  bool v114; // al
  char v115; // al
  _QWORD *v116; // rdx
  int v117; // r14d
  bool v118; // bl
  unsigned int v119; // r15d
  __int64 v120; // rax
  unsigned int v121; // ebx
  bool v122; // al
  bool v123; // al
  int v124; // [rsp-D8h] [rbp-D8h]
  _QWORD *v125; // [rsp-D8h] [rbp-D8h]
  __int64 v126; // [rsp-D0h] [rbp-D0h]
  __int64 v127; // [rsp-D0h] [rbp-D0h]
  bool v128; // [rsp-D0h] [rbp-D0h]
  __int64 v129; // [rsp-D0h] [rbp-D0h]
  __int64 v130; // [rsp-D0h] [rbp-D0h]
  __int64 v131; // [rsp-D0h] [rbp-D0h]
  __int64 v132; // [rsp-C8h] [rbp-C8h]
  __int64 v133; // [rsp-C8h] [rbp-C8h]
  __int64 v134; // [rsp-C8h] [rbp-C8h]
  __int64 v135; // [rsp-C8h] [rbp-C8h]
  __int64 v136; // [rsp-C8h] [rbp-C8h]
  __int64 v137; // [rsp-C8h] [rbp-C8h]
  __int64 v138; // [rsp-C8h] [rbp-C8h]
  unsigned int v139; // [rsp-C0h] [rbp-C0h]
  __int64 v140; // [rsp-C0h] [rbp-C0h]
  __int64 v141; // [rsp-C0h] [rbp-C0h]
  __int64 v142; // [rsp-C0h] [rbp-C0h]
  __int64 v143; // [rsp-C0h] [rbp-C0h]
  __int64 v144; // [rsp-C0h] [rbp-C0h]
  __int64 v145; // [rsp-C0h] [rbp-C0h]
  __int64 v146; // [rsp-C0h] [rbp-C0h]
  unsigned __int8 v147; // [rsp-C0h] [rbp-C0h]
  __int64 v148; // [rsp-C0h] [rbp-C0h]
  _QWORD *v149; // [rsp-C0h] [rbp-C0h]
  __int64 v150; // [rsp-C0h] [rbp-C0h]
  __int64 v151; // [rsp-C0h] [rbp-C0h]
  char v152; // [rsp-C0h] [rbp-C0h]
  _BYTE *v153; // [rsp-C0h] [rbp-C0h]
  __int64 v154; // [rsp-C0h] [rbp-C0h]
  __int64 v155; // [rsp-C0h] [rbp-C0h]
  _QWORD *v156; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v157; // [rsp-B0h] [rbp-B0h] BYREF
  __int64 v158; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v159; // [rsp-A0h] [rbp-A0h]
  __int64 *v160; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v161; // [rsp-90h] [rbp-90h]
  __m128i v162; // [rsp-88h] [rbp-88h] BYREF
  _OWORD v163[3]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v164; // [rsp-48h] [rbp-48h]
  __int64 v165; // [rsp-10h] [rbp-10h]

  if ( a1 > 0x17D )
    return 0;
  v165 = v6;
  v7 = (unsigned __int8 *)a4;
  v8 = a3;
  v10 = a2;
  if ( a1 <= 0xCE )
  {
    if ( a1 == 65 )
    {
      v162.m128i_i64[0] = 0;
      v162.m128i_i64[1] = (__int64)&v160;
      v75 = *(_BYTE *)a3;
      if ( *(_BYTE *)a3 == 55 )
      {
        if ( (unsigned __int8)sub_1006860((__int64 **)&v162, *(_QWORD *)(a3 - 64)) )
        {
          v78 = *(_QWORD *)(v8 - 32);
          if ( v78 )
          {
            *(_QWORD *)v162.m128i_i64[1] = v78;
            return (unsigned __int8 *)v160;
          }
        }
        v75 = *(_BYTE *)v8;
      }
      v162.m128i_i64[0] = 0;
      if ( v75 == 56 )
      {
        v76 = *(_QWORD *)(v8 - 64);
        if ( (unsigned __int8)sub_1006860((__int64 **)&v162, v76) )
          return (unsigned __int8 *)sub_AD6530((__int64)v10, v76);
      }
      return 0;
    }
    if ( a1 <= 0x41 )
    {
      if ( a1 == 1 )
      {
        if ( *(_BYTE *)a3 == 85 )
        {
          v74 = *(_QWORD *)(a3 - 32);
          if ( v74 )
          {
            if ( !*(_BYTE *)v74 && *(_QWORD *)(v74 + 24) == *(_QWORD *)(a3 + 80) && *(_DWORD *)(v74 + 36) == 1 )
              return (unsigned __int8 *)v8;
          }
        }
      }
      else if ( a1 == 26 )
      {
        if ( a4 == a3 )
          return (unsigned __int8 *)v8;
        v160 = (__int64 *)a4;
        if ( sub_1009310(&v160, (unsigned __int8 *)a3) )
          return v7;
        v162.m128i_i64[0] = v8;
        if ( sub_1009310(&v162, v7) )
          return v7;
      }
      return 0;
    }
    if ( a1 != 67 || *(_BYTE *)a3 != 54 )
      return 0;
    v34 = *(_QWORD *)(a3 - 64);
    if ( *(_BYTE *)v34 == 17 )
    {
      v35 = *(_DWORD *)(v34 + 32);
      if ( v35 > 0x40 )
      {
        if ( (unsigned int)sub_C444A0(v34 + 24) == v35 - 1 )
          goto LABEL_62;
        return 0;
      }
      if ( *(_QWORD *)(v34 + 24) != 1 )
        return 0;
    }
    else
    {
      v104 = *(_QWORD *)(v34 + 8);
      v105 = (unsigned int)*(unsigned __int8 *)(v104 + 8) - 17;
      if ( (unsigned int)v105 > 1 || *(_BYTE *)v34 > 0x15u )
        return 0;
      v106 = sub_AD7630(v34, 0, v105);
      if ( !v106 || *v106 != 17 )
      {
        if ( *(_BYTE *)(v104 + 8) == 17 )
        {
          v117 = *(_DWORD *)(v104 + 32);
          if ( v117 )
          {
            v118 = 0;
            v119 = 0;
            while ( 1 )
            {
              v120 = sub_AD69F0((unsigned __int8 *)v34, v119);
              if ( !v120 )
                break;
              if ( *(_BYTE *)v120 != 13 )
              {
                if ( *(_BYTE *)v120 != 17 )
                  break;
                v121 = *(_DWORD *)(v120 + 32);
                v118 = v121 <= 0x40 ? *(_QWORD *)(v120 + 24) == 1 : v121 - 1 == (unsigned int)sub_C444A0(v120 + 24);
                if ( !v118 )
                  break;
              }
              if ( v117 == ++v119 )
              {
                if ( v118 )
                  goto LABEL_62;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v107 = *((_DWORD *)v106 + 8);
      if ( !(v107 <= 0x40 ? *((_QWORD *)v106 + 3) == 1 : v107 - 1 == (unsigned int)sub_C444A0((__int64)(v106 + 24))) )
        return 0;
    }
LABEL_62:
    v12 = *(_QWORD *)(v8 - 32);
    if ( v12 )
      return (unsigned __int8 *)v12;
    return 0;
  }
  switch ( a1 )
  {
    case 0xCFu:
      if ( *(_BYTE *)a3 == 13 )
      {
        v77 = a2;
        return (unsigned __int8 *)sub_ACADE0(v77);
      }
      v52 = *(_QWORD **)(a4 + 24);
      if ( *(_DWORD *)(a4 + 32) > 0x40u )
        v52 = (_QWORD *)*v52;
      v53 = (unsigned __int16)v52 & 0x3FF;
      if ( v53 == 1023 )
        goto LABEL_178;
      if ( !v53 )
      {
        v37 = 0;
        v38 = 0;
        return (unsigned __int8 *)sub_AD64C0((__int64)v10, v38, v37);
      }
      if ( (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)a3) )
        return (unsigned __int8 *)sub_ACA8A0(a2);
      return 0;
    case 0xD1u:
      return sub_1003530(a3, a4, (__int64)a5, 0);
    case 0xD6u:
      if ( *(_BYTE *)a3 > 0x15u || *(_BYTE *)a4 > 0x15u )
        return 0;
      v44 = (_BYTE *)a5->m128i_i64[0];
      v159 = 1;
      v158 = 0;
      if ( (unsigned __int8)sub_96E080(a3, &v156, (__int64)&v158, (__int64)v44, 0)
        && (v45 = (_QWORD *)sub_BD5C60(v8), v143 = sub_BCB2D0(v45), *v7 == 17)
        && *((_DWORD *)v7 + 8) <= 0x40u )
      {
        v46 = sub_AE43F0((__int64)v44, *(_QWORD *)(v8 + 8));
        sub_C44B10((__int64)&v160, (char **)v7 + 3, v46);
        if ( sub_C462E0((__int64)&v160, 4) )
          goto LABEL_235;
        v47 = v161;
        v161 = 0;
        v162.m128i_i32[2] = v47;
        v162.m128i_i64[0] = (__int64)v160;
        v48 = sub_971820(v8, v143, (__int64)&v162, v44);
        if ( v162.m128i_i32[2] > 0x40u && v162.m128i_i64[0] )
          j_j___libc_free_0_0(v162.m128i_i64[0]);
        if ( v48
          && *(_BYTE *)v48 == 5
          && (*(_WORD *)(v48 + 2) != 38
           || (v48 = *(_QWORD *)(v48 - 32LL * (*(_DWORD *)(v48 + 4) & 0x7FFFFFF)), *(_BYTE *)v48 == 5))
          && *(_WORD *)(v48 + 2) == 15
          && (v49 = *(_DWORD *)(v48 + 4) & 0x7FFFFFF, v50 = *(_QWORD *)(v48 - 32 * v49), *(_BYTE *)v50 == 5)
          && *(_WORD *)(v50 + 2) == 47 )
        {
          v144 = *(_QWORD *)(v50 - 32LL * (*(_DWORD *)(v50 + 4) & 0x7FFFFFF));
          v51 = *(_QWORD *)(v48 + 32 * (1 - v49));
          v162.m128i_i32[2] = 1;
          v162.m128i_i64[0] = 0;
          if ( (unsigned __int8)sub_96E080(v51, &v157, (__int64)&v162, (__int64)v44, 0) && v156 == v157 )
          {
            v12 = v144;
            if ( v159 <= 0x40 )
            {
              if ( v158 != v162.m128i_i64[0] )
                v12 = 0;
            }
            else
            {
              v123 = sub_C43C50((__int64)&v158, (const void **)&v162);
              v12 = v144;
              if ( !v123 )
                v12 = 0;
            }
          }
          else
          {
            v12 = 0;
          }
          if ( v162.m128i_i32[2] > 0x40u && v162.m128i_i64[0] )
          {
            v145 = v12;
            j_j___libc_free_0_0(v162.m128i_i64[0]);
            v12 = v145;
          }
        }
        else
        {
LABEL_235:
          v12 = 0;
        }
        if ( v161 > 0x40 && v160 )
        {
          v146 = v12;
          j_j___libc_free_0_0(v160);
          v12 = v146;
        }
      }
      else
      {
        v12 = 0;
      }
      if ( v159 <= 0x40 )
        return (unsigned __int8 *)v12;
      v20 = v158;
      if ( !v158 )
        return (unsigned __int8 *)v12;
      goto LABEL_47;
    case 0xEBu:
    case 0xEDu:
    case 0xF6u:
    case 0xF8u:
      if ( a4 == a3 )
        return v7;
      v22 = a3;
      if ( *(_BYTE *)a3 <= 0x15u )
      {
        v22 = a4;
        v7 = (unsigned __int8 *)a3;
      }
      v23 = (__int64)v7;
      v141 = v22;
      v132 = a6;
      v24 = sub_1003090((__int64)a5, v7);
      v12 = v141;
      v25 = v24;
      if ( (_BYTE)v24 )
        return (unsigned __int8 *)v12;
      v26 = v132;
      v27 = a1 == 235 || a1 == 246;
      v28 = *v7;
      if ( *v7 == 18 )
      {
        v29 = sub_C33340();
        v12 = v141;
        v26 = v132;
        if ( *((void **)v7 + 3) == v29 )
          v32 = (unsigned __int8 *)*((_QWORD *)v7 + 4);
        else
          v32 = v7 + 24;
        v33 = (v32[20] & 7) == 1;
        goto LABEL_56;
      }
      v101 = *((_QWORD *)v7 + 1);
      v152 = v25;
      if ( (unsigned int)*(unsigned __int8 *)(v101 + 8) - 17 > 1 || v28 > 0x15u )
        goto LABEL_242;
      v23 = 0;
      v129 = v132;
      v135 = v12;
      v102 = sub_AD7630((__int64)v7, 0, v25);
      v12 = v135;
      v26 = v129;
      LOBYTE(v30) = v152;
      if ( v102 )
      {
        v153 = v102;
        if ( *v102 == 18 )
        {
          v103 = sub_C33340();
          v12 = v135;
          v26 = v129;
          if ( *((void **)v153 + 3) == v103 )
            v31 = *((_QWORD *)v153 + 4);
          else
            v31 = (__int64)(v153 + 24);
          v33 = (*(_BYTE *)(v31 + 20) & 7) == 1;
LABEL_56:
          if ( v33 )
            goto LABEL_57;
          goto LABEL_149;
        }
      }
      if ( *(_BYTE *)(v101 + 8) == 17 )
      {
        v62 = *(_DWORD *)(v101 + 32);
        v63 = 0;
        v124 = v62;
        if ( v62 )
        {
          do
          {
            v23 = v63;
            v126 = v26;
            v133 = v12;
            v147 = v30;
            v64 = (char *)sub_AD69F0(v7, v63);
            v30 = v147;
            v12 = v133;
            v26 = v126;
            v31 = (__int64)v64;
            if ( !v64 )
              goto LABEL_149;
            v65 = *v64;
            if ( v65 != 13 )
            {
              if ( v65 != 18 )
                goto LABEL_149;
              v148 = v31;
              v66 = sub_C33340();
              v12 = v133;
              v26 = v126;
              v31 = *(void **)(v148 + 24) == v66 ? *(_QWORD *)(v148 + 32) : v148 + 24;
              if ( (*(_BYTE *)(v31 + 20) & 7) != 1 )
                goto LABEL_149;
              v30 = 1;
            }
            ++v63;
          }
          while ( v124 != v63 );
          if ( (_BYTE)v30 )
          {
LABEL_57:
            if ( !v27 )
              return (unsigned __int8 *)v12;
            return sub_10024E0((__int64)v7, v23, v30, v31, v26, v12);
          }
        }
      }
LABEL_149:
      v28 = *v7;
      v67 = (__int64)(v7 + 24);
      if ( *v7 == 18 )
        goto LABEL_150;
      v101 = *((_QWORD *)v7 + 1);
LABEL_242:
      v154 = v26;
      v109 = (unsigned int)*(unsigned __int8 *)(v101 + 8) - 17;
      if ( (unsigned int)v109 > 1 )
        goto LABEL_157;
      if ( v28 > 0x15u )
        goto LABEL_157;
      v136 = v12;
      v110 = sub_AD7630((__int64)v7, 0, v109);
      v12 = v136;
      if ( !v110 || *v110 != 18 )
        goto LABEL_157;
      v26 = v154;
      v67 = (__int64)(v110 + 24);
LABEL_150:
      v127 = v26;
      v134 = v12;
      v149 = *(_QWORD **)v67;
      v68 = sub_C33340();
      v12 = v134;
      v69 = v127;
      v70 = v68;
      if ( v149 == v68 )
      {
        v71 = *(_QWORD *)(v67 + 8);
        if ( (*(_BYTE *)(v71 + 20) & 7) == 0 )
          goto LABEL_153;
      }
      else if ( (*(_BYTE *)(v67 + 20) & 7) == 0 )
      {
LABEL_152:
        v71 = v67;
        goto LABEL_153;
      }
      v125 = v70;
      if ( !v127 )
        goto LABEL_157;
      v113 = v127;
      v130 = v134;
      v137 = v69;
      v114 = sub_B451D0(v113);
      v12 = v130;
      if ( !v114 )
        goto LABEL_157;
      v131 = v137;
      v138 = v12;
      if ( v149 == v125 )
      {
        v115 = sub_C40510((_QWORD **)v67);
        v69 = v131;
        v12 = v138;
        v116 = v125;
      }
      else
      {
        v115 = sub_C33B00(v67);
        v116 = v125;
        v12 = v138;
        v69 = v131;
      }
      if ( !v115 )
        goto LABEL_157;
      if ( v116 != *(_QWORD **)v67 )
        goto LABEL_152;
      v71 = *(_QWORD *)(v67 + 8);
LABEL_153:
      if ( ((*(_BYTE *)(v71 + 20) & 8) != 0) == (((a1 - 246) & 0xFFFFFFFD) == 0) )
      {
        if ( !v27 )
          return (unsigned __int8 *)sub_AD8F10((__int64)v10, (__int64 *)v67);
        if ( v69 )
        {
          v155 = v12;
          v122 = sub_B451C0(v69);
          v12 = v155;
          if ( v122 )
            return (unsigned __int8 *)sub_AD8F10((__int64)v10, (__int64 *)v67);
        }
      }
      else
      {
        if ( v27 )
          return (unsigned __int8 *)v12;
        if ( v69 )
        {
          v150 = v12;
          v72 = sub_B451C0(v69);
          v12 = v150;
          if ( v72 )
            return (unsigned __int8 *)v12;
        }
      }
LABEL_157:
      v151 = v12;
      v73 = sub_FFF8A0(a1, v12, (__int64)v7);
      if ( v73 )
        return (unsigned __int8 *)v73;
      v12 = sub_FFF8A0(a1, (__int64)v7, v151);
      if ( !v12 )
        return 0;
      return (unsigned __int8 *)v12;
    case 0x11Du:
      if ( *(_BYTE *)a4 != 17 )
        return 0;
      v14 = *(_DWORD *)(a4 + 32);
      v15 = a4 + 24;
      if ( v14 <= 0x40 )
        v16 = *(_QWORD *)(a4 + 24) == 0;
      else
        v16 = v14 == (unsigned int)sub_C444A0(a4 + 24);
      if ( v16 )
        return sub_AD8DD0(*(_QWORD *)(v8 + 8), 1.0);
      if ( v14 <= 0x40 )
        v17 = *((_QWORD *)v7 + 3) == 1;
      else
        v17 = v14 - 1 == (unsigned int)sub_C444A0(v15);
      if ( v17 )
        return (unsigned __int8 *)v8;
      return 0;
    case 0x12Bu:
      if ( *(_BYTE *)a3 == 13 || *(_BYTE *)a4 == 13 )
      {
        v77 = *(__int64 ***)(a3 + 8);
        return (unsigned __int8 *)sub_ACADE0(v77);
      }
      a2 = (__int64 **)a3;
      if ( (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)a3) || (unsigned __int8)sub_FFFE90(v8) )
      {
        v19 = *(_QWORD *)(v8 + 8);
        return (unsigned __int8 *)sub_AD6530(v19, (__int64)a2);
      }
      v54 = *v7;
      if ( (unsigned __int8)v54 > 0x1Cu )
      {
        v111 = v54 - 29;
      }
      else
      {
        if ( (_BYTE)v54 != 5 )
          goto LABEL_130;
        v111 = *((unsigned __int16 *)v7 + 1);
      }
      if ( v111 == 47 )
      {
        v112 = (v7[7] & 0x40) != 0
             ? (unsigned __int8 *)*((_QWORD *)v7 - 1)
             : &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
        if ( v8 == *(_QWORD *)v112 )
          return (unsigned __int8 *)v8;
      }
LABEL_130:
      v162.m128i_i64[0] = 0;
      if ( !(unsigned __int8)sub_995B10(&v162, (__int64)v7) && !(unsigned __int8)sub_1003090((__int64)a5, v7) )
      {
        if ( *v7 > 0x15u || *v7 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v7) )
          return 0;
        sub_9AC330((__int64)&v162, v8, 0, a5);
        v55 = sub_BCB060(*((_QWORD *)v7 + 1));
        sub_C44AB0((__int64)&v160, (__int64)&v162, v55);
        v56 = a5->m128i_i64[0];
        v57 = (_BYTE *)sub_AD8D80(*((_QWORD *)v7 + 1), (__int64)&v160);
        v60 = (_BYTE *)sub_96E6C0(0x1Du, (__int64)v7, v57, v56);
        if ( !v60 || !sub_AD7930(v60, (__int64)v7, v58, v59, v61) )
        {
          sub_969240((__int64 *)&v160);
          sub_969240((__int64 *)v163);
          sub_969240(v162.m128i_i64);
          return 0;
        }
        sub_969240((__int64 *)&v160);
        sub_969240((__int64 *)v163);
        sub_969240(v162.m128i_i64);
      }
      return (unsigned __int8 *)v8;
    case 0x137u:
      goto LABEL_26;
    case 0x138u:
    case 0x168u:
      v39 = a3;
      if ( !(unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)a3) )
      {
        v39 = (__int64)v7;
        if ( !(unsigned __int8)sub_1003090((__int64)a5, v7) )
          return 0;
      }
      v162.m128i_i64[0] = sub_AD62B0(*v10[2]);
      v162.m128i_i64[1] = sub_AD6530(v10[2][1], v39);
      return (unsigned __int8 *)sub_AD24A0(v10, v162.m128i_i64, 2);
    case 0x139u:
    case 0x16Au:
      a2 = (__int64 **)a3;
      if ( sub_1016CD0(0x20u, (_BYTE *)a3, (_BYTE *)a4, a5->m128i_i64, 3u) )
        goto LABEL_40;
      if ( a1 != 313 )
      {
        v36 = 36;
        if ( !sub_1016CD0(0x22u, (_BYTE *)v8, v7, a5->m128i_i64, 3u) )
          goto LABEL_72;
LABEL_178:
        v37 = 0;
        v38 = 1;
        return (unsigned __int8 *)sub_AD64C0((__int64)v10, v38, v37);
      }
      v36 = 40;
      if ( sub_1016CD0(0x26u, (_BYTE *)v8, v7, a5->m128i_i64, 3u) )
        goto LABEL_178;
LABEL_72:
      if ( !sub_1016CD0(v36, (_BYTE *)v8, v7, a5->m128i_i64, 3u) )
        return 0;
      v37 = 1;
      v38 = -1;
      return (unsigned __int8 *)sub_AD64C0((__int64)v10, v38, v37);
    case 0x149u:
    case 0x14Au:
    case 0x16Du:
    case 0x16Eu:
      if ( a4 == a3 )
        return v7;
      v139 = sub_BCB060((__int64)a2);
      if ( *(_BYTE *)v8 <= 0x15u && *(_BYTE *)v8 != 5 && !(unsigned __int8)sub_AD6CA0(v8) )
      {
        v79 = (unsigned __int8 *)v8;
        v8 = (__int64)v7;
        v7 = v79;
      }
      if ( (unsigned __int8)sub_1003090((__int64)a5, v7) )
      {
        sub_1002860((__int64)&v162, a1, v139);
        v12 = sub_AD8D80((__int64)a2, (__int64)&v162);
        if ( v162.m128i_i32[2] > 0x40u )
        {
          v20 = v162.m128i_i64[0];
          if ( v162.m128i_i64[0] )
          {
LABEL_47:
            v140 = v12;
            j_j___libc_free_0_0(v20);
            return (unsigned __int8 *)v140;
          }
        }
        return (unsigned __int8 *)v12;
      }
      v162.m128i_i8[8] = 1;
      v162.m128i_i64[0] = (__int64)&v157;
      if ( !(unsigned __int8)sub_991580((__int64)&v162, (__int64)v7) )
        goto LABEL_202;
      sub_1002860((__int64)&v162, a1, v139);
      if ( *((_DWORD *)v157 + 2) <= 0x40u )
        v128 = *v157 == v162.m128i_i64[0];
      else
        v128 = sub_C43C50((__int64)v157, (const void **)&v162);
      sub_969240(v162.m128i_i64);
      if ( v128 )
        return (unsigned __int8 *)sub_AD8D80((__int64)a2, (__int64)v157);
      v80 = sub_9905C0(a1);
      sub_1002860((__int64)&v162, v80, v139);
      if ( *((_DWORD *)v157 + 2) <= 0x40u )
        v81 = *v157 == v162.m128i_i64[0];
      else
        v81 = sub_C43C50((__int64)v157, (const void **)&v162);
      sub_969240(v162.m128i_i64);
      if ( v81 )
        return (unsigned __int8 *)v8;
      v82 = sub_988010(v8);
      if ( !v8 )
        goto LABEL_202;
      if ( !v82 )
        goto LABEL_202;
      if ( (unsigned int)sub_987FE0(v8) != a1 )
        goto LABEL_202;
      v83 = *(_DWORD *)(v8 + 4);
      LOBYTE(v161) = 0;
      v84 = v83 & 0x7FFFFFF;
      v85 = *(_QWORD *)(v8 - 32 * v84);
      v160 = &v158;
      v86 = *(_QWORD *)(v8 + 32 * (1 - v84));
      if ( !(unsigned __int8)sub_991580((__int64)&v160, v85) )
      {
        v162.m128i_i64[0] = (__int64)&v158;
        v162.m128i_i8[8] = 0;
        if ( !(unsigned __int8)sub_991580((__int64)&v162, v86) )
          goto LABEL_202;
      }
      if ( a1 == 365 )
      {
        v87 = 34;
      }
      else if ( a1 > 0x16D )
      {
        v87 = 36;
        if ( a1 != 366 )
          goto LABEL_304;
      }
      else if ( a1 == 329 )
      {
        v87 = 38;
      }
      else
      {
        v87 = 40;
        if ( a1 != 330 )
          goto LABEL_304;
      }
      v88 = sub_B531B0(v87);
      if ( !sub_B532C0(v158, v157, v88) )
      {
LABEL_202:
        v12 = (__int64)sub_10079C0(a1, (char *)v8, (char *)v7);
        if ( v12 )
          return (unsigned __int8 *)v12;
        v12 = (__int64)sub_10079C0(a1, (char *)v7, (char *)v8);
        if ( v12 )
          return (unsigned __int8 *)v12;
        switch ( a1 )
        {
          case 0x16Du:
            v89 = 34;
            break;
          case 0x16Eu:
            v89 = 36;
            break;
          case 0x149u:
            v89 = 38;
            break;
          case 0x14Au:
            v89 = 40;
            break;
          default:
LABEL_304:
            BUG();
        }
        v90 = sub_B531B0(v89);
        v91 = _mm_loadu_si128(a5);
        v92 = _mm_loadu_si128(a5 + 1);
        v93 = v90;
        v94 = _mm_loadu_si128(a5 + 2);
        v95 = _mm_loadu_si128(a5 + 3);
        v164 = a5[4].m128i_i64[0];
        BYTE1(v164) = 0;
        v162 = v91;
        v163[0] = v92;
        v163[1] = v94;
        v163[2] = v95;
        if ( !sub_1016CD0(v90, (_BYTE *)v8, v7, v162.m128i_i64, 3u) )
        {
          v96 = &v162;
          v97 = a5;
          v98 = v93 & 0xFFFFFF00FFFFFFFFLL;
          for ( i = 18; i; --i )
          {
            v96->m128i_i32[0] = v97->m128i_i32[0];
            v97 = (__m128i *)((char *)v97 + 4);
            v96 = (__m128i *)((char *)v96 + 4);
          }
          BYTE1(v164) = 0;
          v100 = sub_1016CD0(v98, v7, (_BYTE *)v8, v162.m128i_i64, 3u);
          v12 = 0;
          if ( v100 )
            return v7;
          return (unsigned __int8 *)v12;
        }
        return (unsigned __int8 *)v8;
      }
      break;
    case 0x14Du:
    case 0x171u:
      if ( !(unsigned __int8)sub_FFFE90(a3) && !(unsigned __int8)sub_FFFE90((__int64)v7) )
        goto LABEL_66;
      goto LABEL_40;
    case 0x152u:
      goto LABEL_34;
    case 0x153u:
    case 0x174u:
      if ( a4 == a3 )
        goto LABEL_40;
LABEL_66:
      a2 = (__int64 **)v8;
      if ( !(unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)v8) )
      {
        a2 = (__int64 **)v7;
        if ( !(unsigned __int8)sub_1003090((__int64)a5, v7) )
          return 0;
      }
      goto LABEL_40;
    case 0x167u:
      v160 = 0;
      if ( (unsigned __int8)sub_995B10(&v160, a3) )
        return (unsigned __int8 *)sub_AD62B0((__int64)a2);
      v162.m128i_i64[0] = 0;
      if ( (unsigned __int8)sub_995B10(&v162, (__int64)v7) )
        return (unsigned __int8 *)sub_AD62B0((__int64)a2);
LABEL_26:
      if ( (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)v8)
        || (unsigned __int8)sub_1003090((__int64)a5, v7) )
      {
        return (unsigned __int8 *)sub_AD62B0((__int64)a2);
      }
      if ( (unsigned __int8)sub_FFFE90((__int64)v7) )
        return (unsigned __int8 *)v8;
      v18 = sub_FFFE90(v8);
      v12 = (__int64)v7;
      if ( !v18 )
        return 0;
      return (unsigned __int8 *)v12;
    case 0x173u:
      if ( (unsigned __int8)sub_FFFE90(a3) )
        goto LABEL_40;
      a2 = (__int64 **)v7;
      v162.m128i_i64[0] = 0;
      if ( (unsigned __int8)sub_995B10(&v162, (__int64)v7) )
        goto LABEL_40;
LABEL_34:
      if ( v7 == (unsigned __int8 *)v8
        || (a2 = (__int64 **)v8, (unsigned __int8)sub_1003090((__int64)a5, (unsigned __int8 *)v8))
        || (a2 = (__int64 **)v7, (unsigned __int8)sub_1003090((__int64)a5, v7)) )
      {
LABEL_40:
        v19 = (__int64)v10;
        return (unsigned __int8 *)sub_AD6530(v19, (__int64)a2);
      }
      if ( (unsigned __int8)sub_FFFE90((__int64)v7) )
        return (unsigned __int8 *)v8;
      return 0;
    case 0x17Du:
      v40 = *(_QWORD *)(a4 + 24);
      if ( *(_DWORD *)(a4 + 32) > 0x40u )
        v40 = **(_QWORD **)(a4 + 24);
      if ( *(_BYTE *)a3 == 85 )
      {
        v41 = *(_QWORD *)(a3 - 32);
        if ( v41 )
        {
          if ( !*(_BYTE *)v41 && *(_QWORD *)(v41 + 24) == *(_QWORD *)(a3 + 80) && *(_DWORD *)(v41 + 36) == 382 )
          {
            v42 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
            v142 = *(_QWORD *)(v8 + 32 * (1 - v42));
            if ( v142 )
            {
              v43 = sub_FFFE90(*(_QWORD *)(v8 + 32 * (2 - v42)));
              if ( !(_DWORD)v40 )
              {
                if ( v43 )
                {
                  v12 = v142;
                  if ( a2 == *(__int64 ***)(v142 + 8) )
                    return (unsigned __int8 *)v12;
                }
              }
            }
          }
        }
      }
      return 0;
    default:
      return 0;
  }
  return (unsigned __int8 *)v8;
}
