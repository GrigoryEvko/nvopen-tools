// Function: sub_27E8360
// Address: 0x27e8360
//
__int64 __fastcall sub_27E8360(__int64 a1, __int64 a2, _QWORD *a3, unsigned int a4, unsigned __int8 *a5)
{
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r15
  unsigned __int8 **v8; // r14
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 *v11; // r15
  __int64 v12; // rbx
  char *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r15
  __int64 *v22; // r8
  __int64 v23; // rsi
  __int64 *v24; // rbx
  __int64 **m; // r14
  unsigned int v26; // r13d
  __int64 *v27; // rdi
  unsigned __int64 v29; // rsi
  char v30; // al
  int v31; // eax
  bool v32; // al
  __int64 v33; // rax
  __int64 v34; // r10
  __int64 v35; // rsi
  unsigned int v36; // ecx
  __int64 v37; // rdi
  __int64 v38; // r9
  __int64 v39; // rdi
  __int64 v40; // rdi
  __int64 v41; // rdi
  int v42; // ecx
  __int64 v43; // rax
  __int64 *v44; // rax
  unsigned __int64 v45; // rax
  __int64 v46; // r13
  int v47; // eax
  int v48; // ebx
  unsigned int n; // r14d
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rax
  unsigned __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 *v55; // r13
  __int64 v56; // rax
  __int64 *v57; // r14
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 *v60; // rbx
  __int64 v61; // r12
  _QWORD *v62; // rsi
  _QWORD *v63; // rax
  _QWORD *v64; // rdx
  __int64 *v65; // rbx
  __int64 v66; // rsi
  _QWORD *v67; // rax
  _QWORD *v68; // rdx
  __int64 v69; // rsi
  __int64 v70; // rdx
  __int64 *v71; // rax
  __int64 v72; // rcx
  __int64 *v73; // rax
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rdx
  unsigned __int64 v78; // rax
  __int64 v79; // r13
  int v80; // ebx
  unsigned int i; // r15d
  unsigned int v82; // esi
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 *v87; // r12
  __int64 *j; // rbx
  _DWORD *v89; // rax
  _BYTE *v90; // rcx
  _BYTE *v91; // rax
  unsigned __int64 k; // rdx
  _QWORD *v93; // rdx
  _QWORD *v94; // rdx
  __int64 *v95; // rdx
  __int64 *v96; // rax
  __int64 *v97; // rax
  __int64 *v98; // rax
  unsigned __int64 v99; // r12
  unsigned int v100; // eax
  __int64 v101; // rcx
  __int64 v102; // rcx
  unsigned __int64 v103; // rax
  int v104; // eax
  unsigned __int64 v105; // rax
  unsigned int v106; // r13d
  __int64 v107; // rbx
  __int64 v108; // rax
  _QWORD *v109; // r14
  __m128i *v110; // rax
  __m128i *v111; // rbx
  __int64 v112; // rsi
  __m128i *v113; // r15
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 v117; // rdi
  __int64 v118; // rcx
  __int64 v119; // rsi
  unsigned __int8 *v120; // rsi
  _QWORD *v121; // rax
  __int64 v122; // rdx
  _QWORD *v123; // rdi
  _QWORD *v124; // rsi
  _QWORD *v125; // rdi
  _QWORD *v126; // rdx
  __int64 v127; // rsi
  __int64 v128; // rsi
  unsigned __int8 v129; // [rsp+8h] [rbp-360h]
  int v130; // [rsp+10h] [rbp-358h]
  __int64 **v131; // [rsp+10h] [rbp-358h]
  __int64 v132; // [rsp+10h] [rbp-358h]
  __int64 v133; // [rsp+10h] [rbp-358h]
  unsigned __int64 v134; // [rsp+18h] [rbp-350h]
  int v135; // [rsp+18h] [rbp-350h]
  __int64 *v137; // [rsp+28h] [rbp-340h]
  char v139; // [rsp+40h] [rbp-328h]
  __int64 *v140; // [rsp+40h] [rbp-328h]
  unsigned __int8 v141; // [rsp+40h] [rbp-328h]
  unsigned __int8 *v142; // [rsp+48h] [rbp-320h]
  __int64 v143; // [rsp+48h] [rbp-320h]
  __int64 v144; // [rsp+48h] [rbp-320h]
  __int64 v145; // [rsp+48h] [rbp-320h]
  __m128i v147; // [rsp+58h] [rbp-310h] BYREF
  __int64 *v148; // [rsp+68h] [rbp-300h] BYREF
  __int64 v149; // [rsp+70h] [rbp-2F8h]
  _BYTE v150[128]; // [rsp+78h] [rbp-2F0h] BYREF
  __int64 **v151; // [rsp+F8h] [rbp-270h] BYREF
  __int64 v152; // [rsp+100h] [rbp-268h]
  __int64 v153; // [rsp+108h] [rbp-260h] BYREF
  unsigned int v154; // [rsp+110h] [rbp-258h]
  _BYTE *v155; // [rsp+118h] [rbp-250h]
  __int64 v156; // [rsp+120h] [rbp-248h]
  _BYTE v157[96]; // [rsp+128h] [rbp-240h] BYREF
  __int64 v158; // [rsp+188h] [rbp-1E0h] BYREF
  char *v159; // [rsp+190h] [rbp-1D8h]
  __int64 v160; // [rsp+198h] [rbp-1D0h]
  int v161; // [rsp+1A0h] [rbp-1C8h]
  char v162; // [rsp+1A4h] [rbp-1C4h]
  char v163; // [rsp+1A8h] [rbp-1C0h] BYREF
  __int64 *v164; // [rsp+228h] [rbp-140h] BYREF
  unsigned __int64 v165; // [rsp+230h] [rbp-138h]
  __int64 v166; // [rsp+238h] [rbp-130h] BYREF
  int v167; // [rsp+240h] [rbp-128h]
  char v168; // [rsp+244h] [rbp-124h]
  char v169; // [rsp+248h] [rbp-120h] BYREF

  v148 = (__int64 *)v150;
  v149 = 0x800000000LL;
  v164 = 0;
  v165 = (unsigned __int64)&v169;
  v166 = 4;
  v167 = 0;
  v168 = 1;
  v139 = sub_27DEC50(a1, (unsigned __int8 *)a2, (__int64)a3, (__int64)&v148, a4, (__int64)&v164, a5);
  if ( !v168 )
    _libc_free(v165);
  if ( !v139 )
  {
    v26 = sub_27E6780(a1, (__int64)a3, a2);
    goto LABEL_43;
  }
  v158 = 0;
  v159 = &v163;
  v7 = 2LL * (unsigned int)v149;
  v164 = &v166;
  v8 = (unsigned __int8 **)&v148[v7];
  v160 = 16;
  v161 = 0;
  v162 = 1;
  v165 = 0x1000000000LL;
  if ( v148 == &v148[v7] )
  {
    v26 = 0;
    goto LABEL_41;
  }
  v9 = 0;
  v10 = 0;
  v11 = v148;
  v12 = v148[1];
LABEL_6:
  v13 = v159;
  v14 = HIDWORD(v160);
  v15 = (__int64)&v159[8 * HIDWORD(v160)];
  if ( v159 == (char *)v15 )
  {
LABEL_46:
    if ( HIDWORD(v160) >= (unsigned int)v160 )
      goto LABEL_12;
    ++HIDWORD(v160);
    *(_QWORD *)v15 = v12;
    ++v158;
LABEL_13:
    v16 = *v11;
    v6 = 0;
    if ( (unsigned int)*(unsigned __int8 *)*v11 - 12 <= 1 )
    {
LABEL_14:
      v14 = (unsigned int)v165;
      if ( (_DWORD)v165 )
      {
        if ( v10 != v6 )
          v10 = -1;
        if ( v16 != v9 )
          v9 = -1;
      }
      else
      {
        v9 = v16;
        v10 = v6;
      }
      v17 = *(_QWORD *)(v12 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v17 != v12 + 48 )
      {
        if ( v17 )
        {
          v15 = (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30;
          if ( (unsigned int)v15 <= 0xA )
          {
            if ( *(_BYTE *)(v17 - 24) == 33 )
              goto LABEL_10;
            v15 = HIDWORD(v165);
            v18 = (unsigned int)v165;
            if ( (unsigned int)v165 >= (unsigned __int64)HIDWORD(v165) )
            {
              if ( HIDWORD(v165) < (unsigned __int64)(unsigned int)v165 + 1 )
              {
                v145 = v6;
                sub_C8D5F0((__int64)&v164, &v166, (unsigned int)v165 + 1LL, 0x10u, v5, v6);
                v18 = (unsigned int)v165;
                v6 = v145;
              }
              v44 = &v164[2 * v18];
              *v44 = v12;
              v44[1] = v6;
              LODWORD(v165) = v165 + 1;
              goto LABEL_10;
            }
            v19 = &v164[2 * (unsigned int)v165];
            if ( v19 )
            {
              *v19 = v12;
              v19[1] = v6;
              LODWORD(v14) = v165;
            }
            v14 = (unsigned int)(v14 + 1);
            v11 += 2;
            LODWORD(v165) = v14;
            if ( v8 != (unsigned __int8 **)v11 )
              goto LABEL_11;
            goto LABEL_27;
          }
        }
      }
LABEL_254:
      BUG();
    }
    v143 = a3[6];
    v29 = v143 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)(v143 & 0xFFFFFFFFFFFFFFF8LL) == a3 + 6
      || !v29
      || (unsigned int)*(unsigned __int8 *)(v29 - 24) - 30 > 0xA )
    {
      goto LABEL_254;
    }
    v30 = *(_BYTE *)(v29 - 24);
    if ( v30 == 31 )
    {
      if ( *(_DWORD *)(v16 + 32) <= 0x40u )
      {
        v32 = *(_QWORD *)(v16 + 24) == 0;
      }
      else
      {
        v130 = *(_DWORD *)(v16 + 32);
        v134 = v143 & 0xFFFFFFFFFFFFFFF8LL;
        v144 = *v11;
        v31 = sub_C444A0(v16 + 24);
        v29 = v134;
        v16 = v144;
        v32 = v130 == v31;
      }
      v6 = *(_QWORD *)(v29 - 32LL * v32 - 56);
      goto LABEL_14;
    }
    if ( v30 != 32 )
    {
      v6 = *(_QWORD *)(v16 - 32);
      goto LABEL_14;
    }
    v5 = ((*(_DWORD *)(v29 - 20) & 0x7FFFFFFu) >> 1) - 1;
    v33 = v5 >> 2;
    if ( v5 >> 2 )
    {
      v34 = 4 * v33;
      v35 = *(_QWORD *)(v29 - 32);
      v36 = 2;
      v33 = 0;
      while ( 1 )
      {
        v38 = v33 + 1;
        v41 = *(_QWORD *)(v35 + 32LL * v36);
        if ( v41 )
        {
          if ( v16 == v41 )
            goto LABEL_68;
        }
        v37 = *(_QWORD *)(v35 + 32LL * (v36 + 2));
        if ( v37 && v16 == v37 )
          goto LABEL_69;
        v38 = v33 + 3;
        v39 = *(_QWORD *)(v35 + 32LL * (v36 + 4));
        if ( v39 && v16 == v39 )
        {
          v38 = v33 + 2;
          goto LABEL_69;
        }
        v33 += 4;
        v40 = *(_QWORD *)(v35 + 32LL * (unsigned int)(2 * v33));
        if ( v40 && v16 == v40 )
          goto LABEL_69;
        v36 += 8;
        if ( v33 == v34 )
        {
          v54 = v5 - v33;
          goto LABEL_94;
        }
      }
    }
    v35 = *(_QWORD *)(v29 - 32);
    v54 = v5;
LABEL_94:
    switch ( v54 )
    {
      case 2LL:
        v38 = v33;
        break;
      case 3LL:
        v38 = v33 + 1;
        v118 = *(_QWORD *)(v35 + 32LL * (unsigned int)(2 * (v33 + 1)));
        if ( v118 && v16 == v118 )
        {
LABEL_68:
          v38 = v33;
          goto LABEL_69;
        }
        break;
      case 1LL:
        goto LABEL_180;
      default:
        goto LABEL_97;
    }
    v33 = v38 + 1;
    v101 = *(_QWORD *)(v35 + 32LL * (unsigned int)(2 * (v38 + 1)));
    if ( v101 && v16 == v101 )
    {
LABEL_69:
      v42 = v38;
      if ( v5 != v38 )
      {
LABEL_70:
        v43 = 32LL * (unsigned int)(2 * v42 + 3);
LABEL_71:
        v6 = *(_QWORD *)(v35 + v43);
        goto LABEL_14;
      }
LABEL_97:
      v43 = 32;
      goto LABEL_71;
    }
LABEL_180:
    v102 = *(_QWORD *)(v35 + 32LL * (unsigned int)(2 * v33 + 2));
    if ( v102 )
    {
      if ( v16 == v102 && v5 != v33 )
      {
        v42 = v33;
        if ( v33 != 4294967294LL )
          goto LABEL_70;
      }
    }
    goto LABEL_97;
  }
  while ( v12 != *(_QWORD *)v13 )
  {
    v13 += 8;
    if ( (char *)v15 == v13 )
      goto LABEL_46;
  }
LABEL_10:
  while ( 1 )
  {
    v11 += 2;
    if ( v8 == (unsigned __int8 **)v11 )
      break;
LABEL_11:
    v12 = v11[1];
    if ( v162 )
      goto LABEL_6;
LABEL_12:
    sub_C8CC70((__int64)&v158, v12, v15, v14, v5, v6);
    if ( (_BYTE)v15 )
      goto LABEL_13;
  }
LABEL_27:
  v20 = (unsigned int)v165;
  v142 = (unsigned __int8 *)v9;
  v21 = v10;
  if ( !(_DWORD)v165 )
  {
    v27 = v164;
    v26 = 0;
    goto LABEL_39;
  }
  if ( (unsigned __int64)(v10 - 1) <= 0xFFFFFFFFFFFFFFFDLL )
  {
    v26 = sub_AA5590((__int64)a3, v165);
    if ( !(_BYTE)v26 )
    {
      v22 = v164;
      v151 = (__int64 **)&v153;
      v24 = &v164[2 * (unsigned int)v165];
      v152 = 0x1000000000LL;
      if ( v24 != v164 )
        goto LABEL_31;
      goto LABEL_36;
    }
    v151 = 0;
    v152 = 0;
    v153 = 0;
    v103 = sub_986580((__int64)a3);
    v104 = sub_B46E30(v103);
    sub_F58D10((const __m128i **)&v151, (unsigned int)(v104 - 1));
    v105 = a3[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v105 != a3 + 6 )
    {
      if ( !v105 )
        goto LABEL_176;
      v132 = v105 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v105 - 24) - 30 <= 0xA )
      {
        v135 = sub_B46E30(v132);
        if ( v135 )
        {
          v129 = v26;
          v106 = 0;
          v141 = 0;
          v107 = v132;
          do
          {
            v108 = sub_B46EC0(v107, v106);
            if ( ((v141 ^ 1) & (v108 == v10)) != 0 )
            {
              v141 = (v141 ^ 1) & (v108 == v10);
            }
            else
            {
              v133 = v108;
              sub_AA5980(v108, (__int64)a3, 1u);
              v147.m128i_i64[0] = (__int64)a3;
              v147.m128i_i64[1] = v133 | 4;
              sub_27E05D0((__int64)&v151, &v147);
            }
            ++v106;
          }
          while ( v135 != v106 );
          v26 = v129;
        }
      }
    }
    v109 = (_QWORD *)sub_986580((__int64)a3);
    v110 = (__m128i *)sub_BD2C40(72, 1u);
    v111 = v110;
    if ( v110 )
      sub_B4C8F0((__int64)v110, v10, 1u, (__int64)(v109 + 3), 0);
    v112 = v109[6];
    v113 = v111 + 3;
    v147.m128i_i64[0] = v112;
    if ( v112 )
    {
      sub_B96E90((__int64)&v147, v112, 1);
      if ( v113 == &v147 )
      {
        if ( v147.m128i_i64[0] )
          sub_B91220((__int64)&v147, v147.m128i_i64[0]);
        goto LABEL_201;
      }
      v119 = v111[3].m128i_i64[0];
      if ( !v119 )
      {
LABEL_215:
        v120 = (unsigned __int8 *)v147.m128i_i64[0];
        v111[3].m128i_i64[0] = v147.m128i_i64[0];
        if ( v120 )
          sub_B976B0((__int64)&v147, v120, (__int64)v111[3].m128i_i64);
        goto LABEL_201;
      }
    }
    else if ( v113 == &v147 || (v119 = v111[3].m128i_i64[0]) == 0 )
    {
LABEL_201:
      sub_B43D60(v109);
      sub_FFDB80(*(_QWORD *)(a1 + 48), (unsigned __int64 *)v151, (v152 - (__int64)v151) >> 4, v114, v115, v116);
      v117 = sub_27DD130((__int64 *)a1);
      if ( v117 )
        sub_FF0C10(v117, (__int64)a3);
      if ( *(_BYTE *)a2 > 0x1Cu )
      {
        if ( *(_QWORD *)(a2 + 16) || (unsigned __int8)sub_B46970((unsigned __int8 *)a2) )
        {
          if ( (unsigned __int64)(v142 - 1) <= 0xFFFFFFFFFFFFFFFDLL )
            sub_27DB9F0(a2, v142, (__int64)a3);
        }
        else
        {
          sub_B43D60((_QWORD *)a2);
        }
      }
      if ( v151 )
        j_j___libc_free_0((unsigned __int64)v151);
      goto LABEL_38;
    }
    sub_B91220((__int64)v111[3].m128i_i64, v119);
    goto LABEL_215;
  }
  if ( v10 != -1 )
    goto LABEL_30;
  v55 = v164;
  v56 = 16LL * (unsigned int)v165;
  v57 = &v164[(unsigned __int64)v56 / 8];
  v58 = v56 >> 4;
  v59 = v56 >> 6;
  if ( !v59 )
    goto LABEL_165;
  v60 = &v164[8 * v59];
  v61 = a1 + 96;
  do
  {
    v5 = v55[1];
    if ( *(_BYTE *)(a1 + 124) )
    {
      v62 = *(_QWORD **)(a1 + 104);
      v63 = &v62[*(unsigned int *)(a1 + 116)];
      v14 = (__int64)v62;
      if ( v62 != v63 )
      {
        v64 = *(_QWORD **)(a1 + 104);
        while ( v5 != *v64 )
        {
          if ( v63 == ++v64 )
            goto LABEL_141;
        }
        goto LABEL_106;
      }
LABEL_141:
      v5 = v55[3];
      v6 = (__int64)(v55 + 2);
      goto LABEL_142;
    }
    if ( sub_C8CA60(v61, v55[1]) )
      goto LABEL_106;
    v5 = v55[3];
    v6 = (__int64)(v55 + 2);
    if ( *(_BYTE *)(a1 + 124) )
    {
      v62 = *(_QWORD **)(a1 + 104);
      v14 = (__int64)v62;
      v63 = &v62[*(unsigned int *)(a1 + 116)];
LABEL_142:
      if ( v63 != v62 )
      {
        v93 = v62;
        while ( *v93 != v5 )
        {
          if ( ++v93 == v63 )
            goto LABEL_147;
        }
        goto LABEL_146;
      }
LABEL_147:
      v5 = v55[5];
      v6 = (__int64)(v55 + 4);
      goto LABEL_148;
    }
    v97 = sub_C8CA60(v61, v55[3]);
    v6 = (__int64)(v55 + 2);
    if ( v97 )
      goto LABEL_146;
    v5 = v55[5];
    v6 = (__int64)(v55 + 4);
    if ( *(_BYTE *)(a1 + 124) )
    {
      v62 = *(_QWORD **)(a1 + 104);
      v14 = (__int64)v62;
      v63 = &v62[*(unsigned int *)(a1 + 116)];
LABEL_148:
      if ( v62 != v63 )
      {
        v94 = v62;
        while ( v5 != *v94 )
        {
          if ( ++v94 == v63 )
            goto LABEL_156;
        }
LABEL_146:
        v55 = (__int64 *)v6;
        goto LABEL_106;
      }
LABEL_156:
      v5 = v55[7];
      v95 = v55 + 6;
      goto LABEL_157;
    }
    v98 = sub_C8CA60(v61, v55[5]);
    v6 = (__int64)(v55 + 4);
    if ( v98 )
      goto LABEL_146;
    v5 = v55[7];
    v95 = v55 + 6;
    if ( !*(_BYTE *)(a1 + 124) )
    {
      v96 = sub_C8CA60(v61, v55[7]);
      v95 = v55 + 6;
      if ( v96 )
        goto LABEL_161;
      goto LABEL_163;
    }
    v62 = *(_QWORD **)(a1 + 104);
    v14 = (__int64)v62;
    v63 = &v62[*(unsigned int *)(a1 + 116)];
LABEL_157:
    if ( v63 != v62 )
    {
      while ( v5 != *(_QWORD *)v14 )
      {
        v14 += 8;
        if ( v63 == (_QWORD *)v14 )
          goto LABEL_163;
      }
LABEL_161:
      v55 = v95;
      goto LABEL_106;
    }
LABEL_163:
    v55 += 8;
  }
  while ( v60 != v55 );
  v58 = ((char *)v57 - (char *)v55) >> 4;
LABEL_165:
  if ( v58 == 2 )
  {
    LOBYTE(v14) = *(_BYTE *)(a1 + 124);
    v61 = a1 + 96;
    goto LABEL_224;
  }
  if ( v58 == 3 )
  {
    v5 = v55[1];
    v14 = *(unsigned __int8 *)(a1 + 124);
    v61 = a1 + 96;
    if ( (_BYTE)v14 )
    {
      v121 = *(_QWORD **)(a1 + 104);
      v122 = *(unsigned int *)(a1 + 116);
      v123 = &v121[v122];
      v124 = v121;
      if ( v121 != v123 )
      {
        while ( v5 != *v124 )
        {
          if ( v123 == ++v124 )
            goto LABEL_251;
        }
        goto LABEL_106;
      }
      v128 = v55[3];
      v55 += 2;
LABEL_226:
      v125 = &v121[v122];
      v14 = (__int64)v121;
      if ( v121 == v125 )
      {
        v127 = v55[3];
        v55 += 2;
        goto LABEL_239;
      }
      while ( *(_QWORD *)v14 != v128 )
      {
        v14 += 8;
        if ( v125 == (_QWORD *)v14 )
          goto LABEL_248;
      }
      goto LABEL_106;
    }
    if ( sub_C8CA60(a1 + 96, v55[1]) )
      goto LABEL_106;
    LOBYTE(v14) = *(_BYTE *)(a1 + 124);
LABEL_251:
    v55 += 2;
LABEL_224:
    v128 = v55[1];
    if ( !(_BYTE)v14 )
    {
      if ( !sub_C8CA60(v61, v128) )
      {
        v139 = *(_BYTE *)(a1 + 124);
LABEL_248:
        v55 += 2;
        goto LABEL_237;
      }
      goto LABEL_106;
    }
    v121 = *(_QWORD **)(a1 + 104);
    v122 = *(unsigned int *)(a1 + 116);
    goto LABEL_226;
  }
  if ( v58 != 1 )
    goto LABEL_168;
  v61 = a1 + 96;
  v139 = *(_BYTE *)(a1 + 124);
LABEL_237:
  v127 = v55[1];
  if ( v139 )
  {
    v121 = *(_QWORD **)(a1 + 104);
    v122 = *(unsigned int *)(a1 + 116);
LABEL_239:
    v126 = &v121[v122];
    if ( v121 == v126 )
    {
LABEL_168:
      v55 = v57;
      goto LABEL_114;
    }
    while ( *v121 != v127 )
    {
      if ( v126 == ++v121 )
        goto LABEL_168;
    }
  }
  else if ( !sub_C8CA60(v61, v127) )
  {
    goto LABEL_168;
  }
LABEL_106:
  if ( v57 != v55 )
  {
    v65 = v55 + 2;
    if ( v57 != v55 + 2 )
    {
      while ( 2 )
      {
        v66 = v65[1];
        if ( *(_BYTE *)(a1 + 124) )
        {
          v67 = *(_QWORD **)(a1 + 104);
          v68 = &v67[*(unsigned int *)(a1 + 116)];
          if ( v67 == v68 )
          {
LABEL_140:
            v55 += 2;
            *(v55 - 2) = *v65;
            *(v55 - 1) = v65[1];
          }
          else
          {
            while ( v66 != *v67 )
            {
              if ( v68 == ++v67 )
                goto LABEL_140;
            }
          }
        }
        else if ( !sub_C8CA60(v61, v66) )
        {
          goto LABEL_140;
        }
        v65 += 2;
        if ( v57 == v65 )
          break;
        continue;
      }
    }
  }
LABEL_114:
  v27 = v164;
  v69 = (char *)&v164[2 * (unsigned int)v165] - (char *)v57;
  v70 = v69 >> 4;
  if ( v69 > 0 )
  {
    v71 = v55;
    do
    {
      v72 = *v57;
      v71 += 2;
      v57 += 2;
      *(v71 - 2) = v72;
      v14 = *(v57 - 1);
      *(v71 - 1) = v14;
      --v70;
    }
    while ( v70 );
    v27 = v164;
    v55 = (__int64 *)((char *)v55 + v69);
  }
  v73 = v55;
  v26 = 0;
  LODWORD(v165) = ((char *)v73 - (char *)v27) >> 4;
  if ( !(_DWORD)v165 )
    goto LABEL_39;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = v157;
  v156 = 0;
  v147.m128i_i64[0] = 0;
  *(_DWORD *)sub_27E80B0((__int64)&v151, v147.m128i_i64, v70, v14, v5, v6) = 0;
  v77 = (__int64)(a3 + 6);
  v78 = a3[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v78 == a3 + 6 )
    goto LABEL_125;
  if ( !v78 )
LABEL_176:
    BUG();
  v79 = v78 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v78 - 24) - 30 <= 0xA )
  {
    v80 = sub_B46E30(v79);
    if ( v80 )
    {
      for ( i = 0; i != v80; ++i )
      {
        v82 = i;
        v147.m128i_i64[0] = sub_B46EC0(v79, v82);
        *(_DWORD *)sub_27E80B0((__int64)&v151, v147.m128i_i64, v83, v84, v85, v86) = 0;
      }
    }
  }
LABEL_125:
  v87 = &v164[2 * (unsigned int)v165];
  if ( v164 != v87 )
  {
    for ( j = v164 + 1; ; j += 2 )
    {
      if ( *j )
      {
        v89 = (_DWORD *)sub_27E80B0((__int64)&v151, j, v77, v74, v75, v76);
        ++*v89;
      }
      if ( v87 == j + 1 )
        break;
    }
  }
  v90 = &v155[16 * (unsigned int)v156];
  if ( v155 == v90 )
  {
    k = (unsigned __int64)v155;
  }
  else
  {
    v91 = v155 + 16;
    for ( k = (unsigned __int64)v155; v90 != v91; v91 += 16 )
    {
      if ( *(_DWORD *)(k + 8) < *((_DWORD *)v91 + 2) )
        k = (unsigned __int64)v91;
    }
  }
  v21 = *(_QWORD *)k;
  if ( v155 != v157 )
    _libc_free((unsigned __int64)v155);
  sub_C7D6A0(v152, 16LL * v154, 8);
  v20 = (unsigned int)v165;
LABEL_30:
  v22 = v164;
  v23 = 2 * v20;
  v151 = (__int64 **)&v153;
  v24 = &v164[v23];
  v152 = 0x1000000000LL;
  if ( v164 != &v164[v23] )
  {
LABEL_31:
    for ( m = (__int64 **)v22; v24 != (__int64 *)m; m += 2 )
    {
      if ( m[1] == (__int64 *)v21 )
      {
        v140 = *m;
        v45 = (*m)[6] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (__int64 *)v45 != *m + 6 )
        {
          if ( !v45 )
            goto LABEL_176;
          v46 = v45 - 24;
          if ( (unsigned int)*(unsigned __int8 *)(v45 - 24) - 30 <= 0xA )
          {
            v47 = sub_B46E30(v46);
            if ( v47 )
            {
              v137 = v24;
              v48 = v47;
              v131 = m;
              for ( n = 0; n != v48; ++n )
              {
                while ( a3 != (_QWORD *)sub_B46EC0(v46, n) )
                {
                  if ( ++n == v48 )
                    goto LABEL_88;
                }
                v52 = (unsigned int)v152;
                v53 = (unsigned int)v152 + 1LL;
                if ( v53 > HIDWORD(v152) )
                {
                  sub_C8D5F0((__int64)&v151, &v153, v53, 8u, v50, v51);
                  v52 = (unsigned int)v152;
                }
                v151[v52] = v140;
                LODWORD(v152) = v152 + 1;
              }
LABEL_88:
              v24 = v137;
              m = v131;
            }
          }
        }
      }
    }
  }
  if ( !v21 )
  {
    v99 = sub_986580((__int64)a3);
    v100 = sub_27DC010((__int64)a3);
    v21 = sub_B46EC0(v99, v100);
  }
LABEL_36:
  v26 = sub_27E5F30(a1, a3, &v151, v21);
  if ( v151 != (__int64 **)&v153 )
    _libc_free((unsigned __int64)v151);
LABEL_38:
  v27 = v164;
LABEL_39:
  if ( v27 != &v166 )
    _libc_free((unsigned __int64)v27);
LABEL_41:
  if ( !v162 )
    _libc_free((unsigned __int64)v159);
LABEL_43:
  if ( v148 != (__int64 *)v150 )
    _libc_free((unsigned __int64)v148);
  return v26;
}
