// Function: sub_1CA9E90
// Address: 0x1ca9e90
//
__int64 __fastcall sub_1CA9E90(
        _QWORD *a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r12
  const char **v12; // rax
  const char **v13; // r13
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  __int64 *v16; // rsi
  __int64 *v17; // rdi
  __int64 *v18; // r13
  __int64 v19; // r15
  char v20; // al
  _BYTE *v21; // r12
  unsigned __int64 v22; // r14
  __int64 v23; // rax
  unsigned __int8 v24; // al
  bool v25; // al
  __int64 v26; // rax
  __m128i si128; // xmm0
  __m128i v28; // xmm0
  __int64 *v29; // r13
  _BYTE *v30; // rax
  __int64 v31; // rcx
  unsigned __int64 v32; // rdx
  __int64 v33; // rsi
  _BYTE *v34; // rax
  __int64 v35; // rcx
  unsigned __int64 v36; // rdx
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // r15
  __int64 v40; // r15
  char v41; // al
  int v42; // eax
  __int64 v43; // r15
  _BYTE *v44; // rax
  _QWORD *v45; // rdx
  __int64 v46; // rsi
  unsigned __int64 v47; // rcx
  __int64 v48; // rcx
  __int64 *i; // r15
  __int64 v50; // r14
  unsigned int v51; // eax
  unsigned __int64 v52; // r9
  __int64 v53; // rax
  _BYTE *v54; // rax
  _QWORD *v55; // rdx
  __int64 v56; // rsi
  unsigned __int64 v57; // rcx
  __int64 v58; // rcx
  __int64 v59; // rax
  int v60; // esi
  __int64 *v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  bool v66; // zf
  __int64 v67; // rdx
  __int64 v68; // rcx
  unsigned __int64 v69; // rax
  __int64 v70; // rcx
  __int64 *v71; // r13
  __int64 v72; // r15
  __int64 v73; // r14
  int v74; // eax
  __int64 v75; // rcx
  double v76; // xmm4_8
  double v77; // xmm5_8
  __int64 *v78; // rsi
  int v79; // eax
  __int64 v80; // r15
  _BYTE *v81; // rdx
  _QWORD *v82; // rax
  __int64 v83; // rsi
  unsigned __int64 v84; // rcx
  __int64 v85; // rcx
  __int64 v86; // rcx
  _QWORD *v87; // rax
  __int64 v88; // rax
  unsigned __int8 v89; // dl
  __int64 v90; // rax
  __int64 v92; // rdx
  bool v93; // al
  __int64 v94; // rsi
  __int64 *v95; // r9
  __int64 v96; // rdx
  __int64 v97; // rax
  _QWORD *v98; // rax
  __int64 v99; // rsi
  __int64 v100; // rcx
  _BYTE *v101; // r15
  int v102; // r9d
  __int64 v103; // rax
  _BYTE *v104; // r15
  int v105; // r9d
  const char *v106; // rsi
  __int64 v107; // rax
  __int64 v108; // rax
  unsigned int v109; // eax
  __int64 *v110; // rax
  __int64 v111; // rax
  unsigned int v112; // eax
  _BYTE *v113; // rax
  unsigned __int64 v114; // rax
  __int64 v115; // rax
  _BYTE *v116; // rax
  _QWORD *v117; // r15
  __int64 v118; // rcx
  unsigned __int64 v119; // rdx
  __int64 v120; // rdx
  __int64 *v121; // rax
  _BYTE *v122; // rax
  _QWORD *v123; // r15
  __int64 v124; // rcx
  unsigned __int64 v125; // rdx
  __int64 v126; // rcx
  __int64 v127; // rax
  _BYTE *v128; // [rsp+8h] [rbp-138h]
  int v129; // [rsp+10h] [rbp-130h]
  __int64 *v130; // [rsp+10h] [rbp-130h]
  unsigned __int8 v131; // [rsp+10h] [rbp-130h]
  int v132; // [rsp+10h] [rbp-130h]
  const char **v133; // [rsp+18h] [rbp-128h]
  __int64 *v134; // [rsp+18h] [rbp-128h]
  _BYTE *v135; // [rsp+18h] [rbp-128h]
  __int64 *v136; // [rsp+18h] [rbp-128h]
  char v137; // [rsp+18h] [rbp-128h]
  _BYTE *v138; // [rsp+18h] [rbp-128h]
  unsigned int v139; // [rsp+18h] [rbp-128h]
  __int64 v140; // [rsp+18h] [rbp-128h]
  __int64 v141; // [rsp+18h] [rbp-128h]
  _BYTE *v142; // [rsp+20h] [rbp-120h]
  const char *v143; // [rsp+28h] [rbp-118h]
  __int64 v144; // [rsp+28h] [rbp-118h]
  __int64 *v145; // [rsp+28h] [rbp-118h]
  _BYTE *v146; // [rsp+28h] [rbp-118h]
  __int64 *v147; // [rsp+28h] [rbp-118h]
  unsigned __int8 v148; // [rsp+3Bh] [rbp-105h] BYREF
  unsigned int v149; // [rsp+3Ch] [rbp-104h] BYREF
  __int64 v150; // [rsp+40h] [rbp-100h] BYREF
  __int64 v151; // [rsp+48h] [rbp-F8h] BYREF
  __int64 *v152; // [rsp+50h] [rbp-F0h] BYREF
  __int64 *v153; // [rsp+58h] [rbp-E8h]
  __int64 *v154; // [rsp+60h] [rbp-E0h]
  __int64 v155[4]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v156; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v157; // [rsp+98h] [rbp-A8h]
  __int64 v158; // [rsp+A0h] [rbp-A0h]
  int v159; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v160; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v161; // [rsp+B8h] [rbp-88h]
  __int64 v162[4]; // [rsp+C0h] [rbp-80h] BYREF
  const char *v163; // [rsp+E0h] [rbp-60h] BYREF
  unsigned __int64 v164; // [rsp+E8h] [rbp-58h] BYREF
  __int64 v165[10]; // [rsp+F0h] [rbp-50h] BYREF

  v10 = a2;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  sub_1CA49F0((__int64)a1, a2);
  sub_1CA9D60(a1, a2, (__int64)&v156);
  v12 = (const char **)a1[10];
  v13 = (const char **)a1[9];
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v133 = v12;
  if ( v13 != v12 )
  {
    while ( 1 )
    {
      v143 = *v13;
      v142 = (_BYTE *)*((_QWORD *)*v13 - 3);
      v14 = (unsigned int)sub_1CA8350((__int64)a1, *((_BYTE **)*v13 - 6), (__int64)&v156, v10);
      if ( (_DWORD)v14 != (unsigned int)sub_1CA8350((__int64)a1, v142, (__int64)&v156, v10) )
        goto LABEL_3;
      if ( (unsigned int)v14 > 0x10 )
        goto LABEL_3;
      v15 = 65814;
      if ( !_bittest64(&v15, v14) )
        goto LABEL_3;
      v163 = v143;
      v16 = v153;
      if ( v153 == v154 )
      {
        sub_17C2330((__int64)&v152, v153, &v163);
LABEL_3:
        if ( v133 == ++v13 )
          break;
      }
      else
      {
        if ( v153 )
        {
          *v153 = (__int64)v143;
          v16 = v153;
        }
        ++v13;
        v153 = v16 + 1;
        if ( v133 == v13 )
          break;
      }
    }
  }
  v17 = (__int64 *)a1[3];
  v134 = (__int64 *)a1[4];
  if ( v17 != v134 )
  {
    v144 = v10;
    v18 = v17;
    while ( 1 )
    {
      v19 = *v18;
      LODWORD(v150) = 0;
      v155[0] = v19;
      v20 = *(_BYTE *)(v19 + 16);
      switch ( v20 )
      {
        case '6':
          v21 = *(_BYTE **)(v19 - 24);
          v19 = 0;
          break;
        case '7':
          v21 = *(_BYTE **)(v19 - 24);
          break;
        case 'N':
          v103 = *(_QWORD *)(v19 - 24);
          if ( *(_BYTE *)(v103 + 16)
            || (*(_BYTE *)(v103 + 33) & 0x20) == 0
            || !(unsigned __int8)sub_1C98880((__int64)a1, *(_DWORD *)(v103 + 36), &v150) )
          {
            goto LABEL_17;
          }
          v21 = *(_BYTE **)(v19 + 24 * ((unsigned int)v150 - (unsigned __int64)(*(_DWORD *)(v19 + 20) & 0xFFFFFFF)));
          v19 = 0;
          break;
        case ':':
          v21 = *(_BYTE **)(v19 - 72);
          v19 = 0;
          break;
        case ';':
          v21 = *(_BYTE **)(v19 - 48);
          v19 = 0;
          break;
        default:
          goto LABEL_17;
      }
      v22 = (unsigned int)sub_1CA8350((__int64)a1, v21, (__int64)&v156, v144);
      LODWORD(v151) = 0;
      if ( (unsigned __int8)sub_1C98370(a1, v155[0], (__int64)v21, (unsigned int *)&v151) )
        v22 = (unsigned int)v151;
      if ( (unsigned int)v22 > 0x10 )
        goto LABEL_13;
      v23 = 65814;
      if ( !_bittest64(&v23, v22) )
      {
        if ( (_DWORD)v22 != 15 )
        {
LABEL_13:
          if ( v21[16] > 0x17u )
            sub_1CCC6D0(v21, (unsigned int)v22);
        }
        if ( unk_4FBE1ED && *(_BYTE *)a1 )
        {
          v161 = 0;
          v160 = (unsigned __int64)v162;
          LOBYTE(v162[0]) = 0;
          v130 = (__int64 *)(v155[0] + 48);
          sub_15E0530(v144);
          sub_1C315E0((__int64)&v163, v130);
          sub_2241490(&v160, v163, v164);
          if ( v163 != (const char *)v165 )
            j_j___libc_free_0(v163, v165[0] + 1);
          if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v161) <= 0x4A )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(&v160, ": Warning: Cannot tell what pointer points to, assuming global memory space", 75);
          sub_1C3F040((__int64)&v160);
          if ( byte_4FBE000 && byte_4FBDF20 )
            sub_1CCC400(v21);
          if ( (__int64 *)v160 != v162 )
            j_j___libc_free_0(v160, v162[0] + 1);
        }
        goto LABEL_17;
      }
      v24 = *(_BYTE *)(v155[0] + 16);
      if ( v24 <= 0x17u )
        break;
      if ( v24 == 78 )
      {
        v88 = *(_QWORD *)(v155[0] - 24);
        if ( *(_BYTE *)(v88 + 16) || (*(_BYTE *)(v88 + 33) & 0x20) == 0 )
          break;
        v25 = sub_1C98920((__int64)a1, *(_DWORD *)(v88 + 36));
      }
      else
      {
        if ( v24 != 58 && v24 != 59 )
          break;
        v25 = 1;
      }
      if ( v19 )
        goto LABEL_26;
LABEL_27:
      if ( (_DWORD)v22 != 4 || !v25 )
      {
LABEL_87:
        v78 = v153;
        if ( v153 == v154 )
        {
          sub_170B610((__int64)&v152, v153, v155);
        }
        else
        {
          if ( v153 )
          {
            *v153 = v155[0];
            v78 = v153;
          }
          v153 = v78 + 1;
        }
        goto LABEL_17;
      }
      v163 = (const char *)v165;
      v160 = 71;
      v26 = sub_22409D0(&v163, &v160, 0);
      v163 = (const char *)v26;
      v165[0] = v160;
      *(__m128i *)v26 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_42DFCD0);
      *(_DWORD *)(v26 + 64) = 1886593145;
      *(__m128i *)(v26 + 16) = si128;
      v28 = _mm_load_si128((const __m128i *)&xmmword_42DFCE0);
      *(_WORD *)(v26 + 68) = 25441;
      *(__m128i *)(v26 + 32) = v28;
      a3 = (__m128)_mm_load_si128((const __m128i *)&xmmword_42DFCF0);
      *(_BYTE *)(v26 + 70) = 101;
      *(__m128 *)(v26 + 48) = a3;
      v164 = v160;
      v163[v160] = 0;
      sub_1C979E0(v155[0], (__int64)&v163);
      if ( v163 == (const char *)v165 )
      {
LABEL_17:
        if ( v134 == ++v18 )
          goto LABEL_31;
      }
      else
      {
        ++v18;
        j_j___libc_free_0(v163, v165[0] + 1);
        if ( v134 == v18 )
        {
LABEL_31:
          v10 = v144;
          goto LABEL_32;
        }
      }
    }
    if ( !v19 )
      goto LABEL_87;
LABEL_26:
    v25 = 1;
    goto LABEL_27;
  }
LABEL_32:
  v29 = v152;
  LODWORD(v164) = 0;
  v165[1] = (__int64)&v164;
  v165[2] = (__int64)&v164;
  v165[0] = 0;
  v165[3] = 0;
  v145 = v153;
  if ( v152 == v153 )
  {
    v131 = 0;
    goto LABEL_61;
  }
  do
  {
    while ( 1 )
    {
      v40 = *v29;
      v155[1] = (__int64)a1;
      v155[2] = (__int64)&v156;
      v150 = v40;
      v155[0] = (__int64)&v150;
      v41 = *(_BYTE *)(v40 + 16);
      v148 = 0;
      if ( v41 == 75 )
      {
        v129 = sub_1CA7680((__int64)v155, *(_QWORD *)(v40 - 48), &v148);
        v30 = sub_1CA28F0(a1, v10, *(_BYTE **)(v40 - 48), v150, &v163, v129, v148);
        if ( *(_QWORD *)(v40 - 48) )
        {
          v31 = *(_QWORD *)(v40 - 40);
          v32 = *(_QWORD *)(v40 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v32 = v31;
          if ( v31 )
            *(_QWORD *)(v31 + 16) = *(_QWORD *)(v31 + 16) & 3LL | v32;
        }
        *(_QWORD *)(v40 - 48) = v30;
        if ( v30 )
        {
          v33 = *((_QWORD *)v30 + 1);
          *(_QWORD *)(v40 - 40) = v33;
          if ( v33 )
            *(_QWORD *)(v33 + 16) = (v40 - 40) | *(_QWORD *)(v33 + 16) & 3LL;
          *(_QWORD *)(v40 - 32) = (unsigned __int64)(v30 + 8) | *(_QWORD *)(v40 - 32) & 3LL;
          *((_QWORD *)v30 + 1) = v40 - 48;
        }
        v34 = sub_1CA28F0(a1, v10, *(_BYTE **)(v40 - 24), v150, &v163, v129, v148);
        if ( *(_QWORD *)(v40 - 24) )
        {
          v35 = *(_QWORD *)(v40 - 16);
          v36 = *(_QWORD *)(v40 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v36 = v35;
          if ( v35 )
            *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | v36;
        }
        *(_QWORD *)(v40 - 24) = v34;
        if ( v34 )
        {
          v37 = *((_QWORD *)v34 + 1);
          *(_QWORD *)(v40 - 16) = v37;
          if ( v37 )
            *(_QWORD *)(v37 + 16) = (v40 - 16) | *(_QWORD *)(v37 + 16) & 3LL;
          v38 = *(_QWORD *)(v40 - 8);
          v39 = v40 - 24;
          *(_QWORD *)(v39 + 16) = (unsigned __int64)(v34 + 8) | v38 & 3;
          *((_QWORD *)v34 + 1) = v39;
        }
        goto LABEL_48;
      }
      if ( v41 == 54 )
        break;
      switch ( v41 )
      {
        case '7':
          v138 = *(_BYTE **)(v40 - 24);
          v79 = sub_1CA7680((__int64)v155, (__int64)v138, &v148);
          v80 = v150;
          v81 = sub_1CA28F0(a1, v10, v138, v150, &v163, v79, v148);
          if ( (*(_BYTE *)(v80 + 23) & 0x40) != 0 )
            v82 = *(_QWORD **)(v80 - 8);
          else
            v82 = (_QWORD *)(v80 - 24LL * (*(_DWORD *)(v80 + 20) & 0xFFFFFFF));
          if ( v82[3] )
          {
            v83 = v82[4];
            v84 = v82[5] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v84 = v83;
            if ( v83 )
              *(_QWORD *)(v83 + 16) = *(_QWORD *)(v83 + 16) & 3LL | v84;
          }
          v82[3] = v81;
          if ( v81 )
          {
            v85 = *((_QWORD *)v81 + 1);
            v82[4] = v85;
            if ( v85 )
              *(_QWORD *)(v85 + 16) = (unsigned __int64)(v82 + 4) | *(_QWORD *)(v85 + 16) & 3LL;
            v86 = v82[5];
            v87 = v82 + 3;
            v87[2] = (unsigned __int64)(v81 + 8) | v86 & 3;
            *((_QWORD *)v81 + 1) = v87;
          }
          goto LABEL_48;
        case ':':
          v101 = *(_BYTE **)(v40 - 72);
          v102 = sub_1CA7680((__int64)v155, (__int64)v101, &v148);
          if ( ((v102 - 4) & 0xFFFFFFFB) != 0 )
          {
            v140 = v150;
            v116 = sub_1CA28F0(a1, v10, v101, v150, &v163, v102, v148);
            if ( (*(_BYTE *)(v140 + 23) & 0x40) != 0 )
              v117 = *(_QWORD **)(v140 - 8);
            else
              v117 = (_QWORD *)(v140 - 24LL * (*(_DWORD *)(v140 + 20) & 0xFFFFFFF));
            if ( *v117 )
            {
              v118 = v117[1];
              v119 = v117[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v119 = v118;
              if ( v118 )
                *(_QWORD *)(v118 + 16) = *(_QWORD *)(v118 + 16) & 3LL | v119;
            }
            *v117 = v116;
            if ( v116 )
            {
              v120 = *((_QWORD *)v116 + 1);
              v117[1] = v120;
              if ( v120 )
                *(_QWORD *)(v120 + 16) = (unsigned __int64)(v117 + 1) | *(_QWORD *)(v120 + 16) & 3LL;
              v117[2] = (unsigned __int64)(v116 + 8) | v117[2] & 3LL;
              *((_QWORD *)v116 + 1) = v117;
            }
            goto LABEL_48;
          }
          if ( v102 != 8 )
          {
            sub_1C95A60((__int64 *)&v160, ": Warning: Cannot do atomic on constant memory");
            sub_1C979E0(v150, (__int64)&v160);
            if ( (__int64 *)v160 != v162 )
              j_j___libc_free_0(v160, v162[0] + 1);
            goto LABEL_48;
          }
LABEL_218:
          v106 = ": Warning: Cannot do atomic on local memory";
          goto LABEL_180;
        case ';':
          v104 = *(_BYTE **)(v40 - 48);
          v105 = sub_1CA7680((__int64)v155, (__int64)v104, &v148);
          if ( ((v105 - 4) & 0xFFFFFFFB) != 0 )
          {
            v141 = v150;
            v122 = sub_1CA28F0(a1, v10, v104, v150, &v163, v105, v148);
            if ( (*(_BYTE *)(v141 + 23) & 0x40) != 0 )
              v123 = *(_QWORD **)(v141 - 8);
            else
              v123 = (_QWORD *)(v141 - 24LL * (*(_DWORD *)(v141 + 20) & 0xFFFFFFF));
            if ( *v123 )
            {
              v124 = v123[1];
              v125 = v123[2] & 0xFFFFFFFFFFFFFFFCLL;
              *(_QWORD *)v125 = v124;
              if ( v124 )
                *(_QWORD *)(v124 + 16) = *(_QWORD *)(v124 + 16) & 3LL | v125;
            }
            *v123 = v122;
            if ( v122 )
            {
              v126 = *((_QWORD *)v122 + 1);
              v123[1] = v126;
              if ( v126 )
                *(_QWORD *)(v126 + 16) = (unsigned __int64)(v123 + 1) | *(_QWORD *)(v126 + 16) & 3LL;
              v123[2] = v123[2] & 3LL | (unsigned __int64)(v122 + 8);
              *((_QWORD *)v122 + 1) = v123;
            }
            goto LABEL_48;
          }
          if ( v105 == 8 )
            goto LABEL_218;
LABEL_179:
          v106 = ": Warning: Cannot do atomic on constant memory";
          goto LABEL_180;
      }
      v111 = *(_QWORD *)(v40 - 24);
      if ( *(_BYTE *)(v111 + 16) )
LABEL_244:
        BUG();
      v112 = *(_DWORD *)(v111 + 36);
      v149 = 0;
      v139 = v112;
      sub_1C98880((__int64)a1, v112, &v149);
      v128 = *(_BYTE **)(v40 + 24 * (v149 - (unsigned __int64)(*(_DWORD *)(v40 + 20) & 0xFFFFFFF)));
      v132 = sub_1CA7680((__int64)v155, (__int64)v128, &v148);
      if ( !sub_1C302D0(v139) )
      {
        if ( !(unsigned __int8)sub_1C30260(v139) )
          goto LABEL_231;
        if ( ((v132 - 4) & 0xFFFFFFFB) != 0 )
        {
LABEL_197:
          sub_1C98810((__int64)a1, v139);
LABEL_198:
          v113 = sub_1CA28F0(a1, v10, v128, v150, &v163, v132, v148);
          sub_1593B40(
            (_QWORD *)(v40 + 24 * (v149 - (unsigned __int64)(*(_DWORD *)(v40 + 20) & 0xFFFFFFF))),
            (__int64)v113);
          v160 = (unsigned __int64)v162;
          v161 = 0x300000000LL;
          if ( (unsigned __int8)sub_1C30260(v139) )
          {
            v114 = 3 * (v149 - (unsigned __int64)(*(_DWORD *)(v40 + 20) & 0xFFFFFFF));
            goto LABEL_200;
          }
          v127 = *(_DWORD *)(v40 + 20) & 0xFFFFFFF;
          if ( v139 == 137 )
          {
            v151 = **(_QWORD **)(v40 - 24 * v127);
LABEL_239:
            sub_12AA070((__int64)&v160, &v151);
            v151 = **(_QWORD **)(v40 + 24 * (2LL - (*(_DWORD *)(v40 + 20) & 0xFFFFFFF)));
            sub_12AA070((__int64)&v160, &v151);
          }
          else
          {
            if ( (v139 & 0xFFFFFFFD) == 0x85 )
            {
              v151 = **(_QWORD **)(v40 - 24 * v127);
              sub_12AA070((__int64)&v160, &v151);
              v151 = **(_QWORD **)(v40 + 24 * (1LL - (*(_DWORD *)(v40 + 20) & 0xFFFFFFF)));
              goto LABEL_239;
            }
            v114 = 3 * (v149 - v127);
LABEL_200:
            v151 = **(_QWORD **)(v40 + 8 * v114);
            sub_12AA070((__int64)&v160, &v151);
          }
          v115 = sub_15E26F0(*(__int64 **)(v10 + 40), v139, (__int64 *)v160, (unsigned int)v161);
          *(_QWORD *)(v40 + 64) = *(_QWORD *)(*(_QWORD *)v115 + 24LL);
          sub_1593B40((_QWORD *)(v40 - 24), v115);
          if ( (__int64 *)v160 != v162 )
            _libc_free(v160);
          goto LABEL_48;
        }
        if ( v132 == 8 )
          goto LABEL_218;
        goto LABEL_179;
      }
      if ( ((v132 - 4) & 0xFFFFFFFB) != 0 )
      {
        if ( v132 != 2 )
        {
          if ( (unsigned __int8)sub_1C30260(v139) )
            goto LABEL_197;
LABEL_231:
          if ( !sub_1C98810((__int64)a1, v139) || ((v132 - 4) & 0xFFFFFFFB) != 0 )
            goto LABEL_198;
          v106 = ": Warning: cannot perform wmma load or store on local memory";
          if ( v132 != 8 )
            v106 = ": Warning: cannot perform wmma load or store on constant memory";
          goto LABEL_180;
        }
        v106 = ": Warning: Cannot do vector atomic on shared memory";
      }
      else
      {
        v106 = ": Warning: Cannot do vector atomic on local memory";
        if ( v132 != 8 )
          v106 = ": Warning: Cannot do vector atomic on constant memory";
      }
LABEL_180:
      sub_1C95A60((__int64 *)&v160, v106);
      sub_1C979E0(v150, (__int64)&v160);
      sub_2240A30(&v160);
LABEL_48:
      if ( v145 == ++v29 )
        goto LABEL_60;
    }
    v135 = *(_BYTE **)(v40 - 24);
    v42 = sub_1CA7680((__int64)v155, (__int64)v135, &v148);
    v43 = v150;
    v44 = sub_1CA28F0(a1, v10, v135, v150, &v163, v42, v148);
    if ( (*(_BYTE *)(v43 + 23) & 0x40) != 0 )
      v45 = *(_QWORD **)(v43 - 8);
    else
      v45 = (_QWORD *)(v43 - 24LL * (*(_DWORD *)(v43 + 20) & 0xFFFFFFF));
    if ( *v45 )
    {
      v46 = v45[1];
      v47 = v45[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v47 = v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = *(_QWORD *)(v46 + 16) & 3LL | v47;
    }
    *v45 = v44;
    if ( !v44 )
      goto LABEL_48;
    v48 = *((_QWORD *)v44 + 1);
    v45[1] = v48;
    if ( v48 )
      *(_QWORD *)(v48 + 16) = (unsigned __int64)(v45 + 1) | *(_QWORD *)(v48 + 16) & 3LL;
    ++v29;
    v45[2] = (unsigned __int64)(v44 + 8) | v45[2] & 3LL;
    *((_QWORD *)v44 + 1) = v45;
  }
  while ( v145 != v29 );
LABEL_60:
  v131 = 1;
LABEL_61:
  v136 = (__int64 *)a1[7];
  if ( v136 != (__int64 *)a1[6] )
  {
    for ( i = (__int64 *)a1[6]; v136 != i; ++i )
    {
      v50 = *i;
      v146 = *(_BYTE **)(*i + 24 * (1LL - (*(_DWORD *)(*i + 20) & 0xFFFFFFF)));
      v51 = sub_1CA8350((__int64)a1, v146, (__int64)&v156, v10);
      v52 = v51;
      if ( v51 <= 0x10 )
      {
        v53 = 65814;
        if ( _bittest64(&v53, v52) )
        {
          v54 = sub_1CA28F0(a1, v10, v146, v50, &v163, v52, 0);
          v55 = (_QWORD *)(v50 + 24 * (1LL - (*(_DWORD *)(v50 + 20) & 0xFFFFFFF)));
          if ( *v55 )
          {
            v56 = v55[1];
            v57 = v55[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v57 = v56;
            if ( v56 )
              *(_QWORD *)(v56 + 16) = *(_QWORD *)(v56 + 16) & 3LL | v57;
          }
          *v55 = v54;
          if ( v54 )
          {
            v58 = *((_QWORD *)v54 + 1);
            v55[1] = v58;
            if ( v58 )
              *(_QWORD *)(v58 + 16) = (unsigned __int64)(v55 + 1) | *(_QWORD *)(v58 + 16) & 3LL;
            v55[2] = (unsigned __int64)(v54 + 8) | v55[2] & 3LL;
            *((_QWORD *)v54 + 1) = v55;
          }
          v160 = (unsigned __int64)v162;
          v161 = 0x300000000LL;
          v59 = *(_QWORD *)(v50 - 24);
          if ( *(_BYTE *)(v59 + 16) )
            goto LABEL_244;
          v60 = *(_DWORD *)(v59 + 36);
          v61 = *(__int64 **)(v10 + 40);
          v62 = **(_QWORD **)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF));
          LODWORD(v161) = 1;
          v162[0] = v62;
          v63 = **(_QWORD **)(v50 + 24 * (1LL - (*(_DWORD *)(v50 + 20) & 0xFFFFFFF)));
          LODWORD(v161) = 2;
          v162[1] = v63;
          v64 = **(_QWORD **)(v50 + 24 * (2LL - (*(_DWORD *)(v50 + 20) & 0xFFFFFFF)));
          LODWORD(v161) = 3;
          v162[2] = v64;
          v65 = sub_15E26F0(v61, v60, v162, 3);
          v66 = *(_QWORD *)(v50 - 24) == 0;
          v67 = v65;
          *(_QWORD *)(v50 + 64) = *(_QWORD *)(*(_QWORD *)v65 + 24LL);
          if ( !v66 )
          {
            v68 = *(_QWORD *)(v50 - 16);
            v69 = *(_QWORD *)(v50 - 8) & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v69 = v68;
            if ( v68 )
              *(_QWORD *)(v68 + 16) = *(_QWORD *)(v68 + 16) & 3LL | v69;
          }
          *(_QWORD *)(v50 - 24) = v67;
          v70 = *(_QWORD *)(v67 + 8);
          *(_QWORD *)(v50 - 16) = v70;
          if ( v70 )
            *(_QWORD *)(v70 + 16) = (v50 - 16) | *(_QWORD *)(v70 + 16) & 3LL;
          *(_QWORD *)(v50 - 8) = (v67 + 8) | *(_QWORD *)(v50 - 8) & 3LL;
          *(_QWORD *)(v67 + 8) = v50 - 24;
          if ( (__int64 *)v160 != v162 )
            _libc_free(v160);
          v131 = 1;
        }
      }
    }
  }
  v71 = (__int64 *)a1[12];
  v147 = (__int64 *)a1[13];
  if ( v147 != v71 )
  {
    v137 = 0;
    while ( 2 )
    {
      v72 = *v71;
      v73 = *(_QWORD *)(*v71 - 24LL * (*(_DWORD *)(*v71 + 20) & 0xFFFFFFF));
      v74 = sub_1CA7E20((__int64)a1, (_BYTE *)v73, (__int64)&v156, v10);
      switch ( v74 )
      {
        case 1:
        case 4:
          goto LABEL_127;
        case 2:
          v74 = 3;
          goto LABEL_127;
        case 8:
          v74 = 5;
          goto LABEL_127;
        case 16:
          v89 = *(_BYTE *)(v73 + 16);
          v74 = 101;
          if ( v89 <= 0x17u )
            goto LABEL_134;
          goto LABEL_117;
        default:
          LODWORD(v160) = 0;
          if ( (unsigned __int8)sub_1C98370(a1, v72, v73, (unsigned int *)&v160) )
          {
            v74 = v160;
            if ( (_DWORD)v160 && (_DWORD)v160 != 101 )
            {
LABEL_127:
              v92 = *(_QWORD *)(v72 - 24);
              if ( *(_BYTE *)(v92 + 16) )
                goto LABEL_244;
              switch ( *(_DWORD *)(v92 + 36) )
              {
                case 0xFD0:
                  v93 = v74 == 4;
                  goto LABEL_130;
                case 0xFD1:
                  v93 = v74 == 1;
                  goto LABEL_130;
                case 0xFD2:
                  goto LABEL_131;
                case 0xFD3:
                  v93 = v74 == 5;
                  goto LABEL_130;
                case 0xFD4:
                case 0xFD5:
                  v93 = v74 == 3;
LABEL_130:
                  if ( v93 )
                    v94 = sub_15A0600(*(_QWORD *)v72);
                  else
LABEL_131:
                    v94 = sub_15A0640(*(_QWORD *)v72);
                  break;
              }
LABEL_132:
              sub_164D160(v72, v94, a3, a4, a5, a6, v76, v77, a9, a10);
              sub_15F20C0((_QWORD *)v72);
              v137 = 1;
              goto LABEL_120;
            }
          }
          else
          {
            v74 = 0;
          }
          v89 = *(_BYTE *)(v73 + 16);
          if ( v89 <= 0x17u )
          {
LABEL_134:
            if ( v89 == 5 && *(_WORD *)(v73 + 18) == 32 )
            {
LABEL_136:
              if ( (*(_BYTE *)(v73 + 23) & 0x40) != 0 )
                v95 = *(__int64 **)(v73 - 8);
              else
                v95 = (__int64 *)(v73 - 24LL * (*(_DWORD *)(v73 + 20) & 0xFFFFFFF));
              v96 = *v95;
              v97 = *(_DWORD *)(v72 + 20) & 0xFFFFFFF;
              v75 = 4 * v97;
              v98 = (_QWORD *)(v72 - 24 * v97);
              if ( *v98 )
              {
                v99 = v98[1];
                v75 = v98[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v75 = v99;
                if ( v99 )
                {
                  v75 |= *(_QWORD *)(v99 + 16) & 3LL;
                  *(_QWORD *)(v99 + 16) = v75;
                }
              }
              *v98 = v96;
              if ( v96 )
              {
                v100 = *(_QWORD *)(v96 + 8);
                v98[1] = v100;
                if ( v100 )
                  *(_QWORD *)(v100 + 16) = (unsigned __int64)(v98 + 1) | *(_QWORD *)(v100 + 16) & 3LL;
                v75 = (v96 + 8) | v98[2] & 3LL;
                v98[2] = v75;
                *(_QWORD *)(v96 + 8) = v98;
              }
              goto LABEL_120;
            }
          }
          else
          {
LABEL_117:
            if ( v89 == 56 )
              goto LABEL_136;
          }
          if ( !v74 )
          {
            v90 = *(_QWORD *)(v72 - 24LL * (*(_DWORD *)(v72 + 20) & 0xFFFFFFF));
            if ( *(_BYTE *)(v90 + 16) == 78 )
            {
              v107 = *(_QWORD *)(v90 - 24);
              if ( !*(_BYTE *)(v107 + 16) && (*(_BYTE *)(v107 + 33) & 0x20) != 0 && *(_DWORD *)(v107 + 36) == 3770 )
              {
                v108 = *(_QWORD *)(v72 - 24);
                if ( *(_BYTE *)(v108 + 16) )
                  goto LABEL_244;
                v109 = *(_DWORD *)(v108 + 36);
                if ( v109 > 0xFD3 )
                {
                  if ( v109 != 4053 )
                    goto LABEL_120;
                  v121 = (__int64 *)sub_16498A0(v72);
                  v94 = sub_159C4F0(v121);
                }
                else
                {
                  if ( v109 <= 0xFCF )
                    goto LABEL_120;
                  v110 = (__int64 *)sub_16498A0(v72);
                  v94 = sub_159C540(v110);
                }
                if ( v94 )
                  goto LABEL_132;
              }
            }
          }
LABEL_120:
          if ( v147 != ++v71 )
            continue;
          if ( v137 )
          {
            sub_1AF0CE0(v10, 0, 0, (__int64 *)v75, a3, a4, a5, a6, v76, v77, a9, a10);
            v131 = v137;
          }
          break;
      }
      break;
    }
  }
  sub_1C96910(v165[0]);
  j___libc_free_0(0);
  if ( v152 )
    j_j___libc_free_0(v152, (char *)v154 - (char *)v152);
  j___libc_free_0(v157);
  return v131;
}
