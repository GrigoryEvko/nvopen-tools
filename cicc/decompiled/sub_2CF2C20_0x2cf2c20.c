// Function: sub_2CF2C20
// Address: 0x2cf2c20
//
__int64 __fastcall sub_2CF2C20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // r12
  char **v8; // rax
  char **v9; // r14
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 *v12; // rsi
  __int64 *v13; // rbx
  __int64 v14; // r14
  char v15; // al
  _BYTE *v16; // r12
  unsigned __int64 v17; // r15
  __int64 v18; // rax
  char v19; // al
  bool v20; // al
  __int64 v21; // rax
  __m128i si128; // xmm0
  __m128i v23; // xmm0
  __m128i v24; // xmm0
  __int64 *v25; // r14
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rbx
  char v33; // al
  int v34; // eax
  __int64 v35; // rbx
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rcx
  __int64 *v40; // r14
  __int64 v41; // r15
  unsigned int v42; // eax
  unsigned __int64 v43; // r9
  __int64 v44; // rax
  unsigned __int64 v45; // rax
  __int64 v46; // rbx
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned int v50; // esi
  __int64 *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rax
  bool v56; // zf
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 *v59; // rbx
  __int64 v60; // r14
  unsigned __int64 v61; // r15
  int v62; // eax
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 *v66; // rsi
  int v67; // eax
  __int64 v68; // rbx
  unsigned __int64 v69; // rdx
  __int64 v70; // rax
  __int64 v71; // rcx
  __int64 v72; // rcx
  __int64 v73; // rax
  char v74; // dl
  __int64 v75; // rax
  __int64 v77; // rdx
  __int64 v78; // rdx
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  bool v82; // al
  __int64 v83; // rsi
  int v84; // eax
  unsigned __int8 *v85; // r11
  int v86; // r9d
  __int64 v87; // rax
  __m128i v88; // xmm0
  unsigned __int64 v89; // rdi
  __int64 v90; // rax
  __int64 v91; // rax
  int v92; // eax
  const char *v93; // rsi
  __int64 v94; // rax
  unsigned int v95; // eax
  char v96; // al
  int v97; // r9d
  int v98; // r9d
  unsigned __int64 v99; // rax
  unsigned __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // rbx
  unsigned __int64 v103; // rax
  __int64 v104; // rbx
  __int64 v105; // rdx
  __int64 v106; // rdx
  __int64 v107; // rdx
  __int64 v108; // rax
  unsigned int v109; // eax
  __int64 *v110; // rax
  const char *v111; // rsi
  char v112; // al
  __int64 *v113; // rax
  bool v114; // al
  __int64 v115; // rax
  __int64 v116; // rsi
  int v117; // [rsp+0h] [rbp-140h]
  int v118; // [rsp+0h] [rbp-140h]
  int v119; // [rsp+0h] [rbp-140h]
  char **v120; // [rsp+10h] [rbp-130h]
  int v121; // [rsp+10h] [rbp-130h]
  unsigned __int8 v122; // [rsp+10h] [rbp-130h]
  __int64 *v123; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v124; // [rsp+10h] [rbp-130h]
  unsigned __int64 v125; // [rsp+18h] [rbp-128h]
  __int64 *v126; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v127; // [rsp+18h] [rbp-128h]
  __int64 *v128; // [rsp+18h] [rbp-128h]
  char v129; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v130; // [rsp+18h] [rbp-128h]
  __int64 v131; // [rsp+18h] [rbp-128h]
  __int64 v132; // [rsp+18h] [rbp-128h]
  unsigned int v133; // [rsp+18h] [rbp-128h]
  char *v134; // [rsp+20h] [rbp-120h]
  __int64 v135; // [rsp+20h] [rbp-120h]
  __int64 *v136; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v137; // [rsp+20h] [rbp-120h]
  __int64 *v138; // [rsp+20h] [rbp-120h]
  unsigned __int8 v139; // [rsp+3Fh] [rbp-101h] BYREF
  __int64 v140; // [rsp+40h] [rbp-100h] BYREF
  unsigned __int64 v141; // [rsp+48h] [rbp-F8h] BYREF
  __int64 *v142; // [rsp+50h] [rbp-F0h] BYREF
  __int64 *v143; // [rsp+58h] [rbp-E8h]
  __int64 *v144; // [rsp+60h] [rbp-E0h]
  __int64 v145[4]; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v146; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v147; // [rsp+98h] [rbp-A8h]
  __int64 v148; // [rsp+A0h] [rbp-A0h]
  unsigned int v149; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v150; // [rsp+B0h] [rbp-90h] BYREF
  unsigned __int64 v151; // [rsp+B8h] [rbp-88h]
  _QWORD v152[4]; // [rsp+C0h] [rbp-80h] BYREF
  char *v153; // [rsp+E0h] [rbp-60h] BYREF
  size_t v154; // [rsp+E8h] [rbp-58h] BYREF
  unsigned __int64 v155[10]; // [rsp+F0h] [rbp-50h] BYREF

  v7 = a2;
  v146 = 0;
  v147 = 0;
  v148 = 0;
  v149 = 0;
  sub_2CEF6B0((__int64)a1, a2, a3, a4, a5, a6);
  sub_2CF2AC0(a1, a2, (__int64)&v146);
  v8 = (char **)a1[10];
  v9 = (char **)a1[9];
  v142 = 0;
  v143 = 0;
  v144 = 0;
  v120 = v8;
  if ( v9 != v8 )
  {
    while ( 1 )
    {
      v134 = *v9;
      v125 = *((_QWORD *)*v9 - 4);
      v10 = (unsigned int)sub_2CED090((__int64)a1, *((_QWORD *)*v9 - 8), (__int64)&v146, v7);
      if ( (_DWORD)v10 != (unsigned int)sub_2CED090((__int64)a1, v125, (__int64)&v146, v7) )
        goto LABEL_3;
      if ( (unsigned int)v10 > 0x10 )
        goto LABEL_3;
      v11 = 65814;
      if ( !_bittest64(&v11, v10) )
        goto LABEL_3;
      v153 = v134;
      v12 = v143;
      if ( v143 == v144 )
      {
        sub_249A840((__int64)&v142, v143, &v153);
LABEL_3:
        if ( v120 == ++v9 )
          break;
      }
      else
      {
        if ( v143 )
        {
          *v143 = (__int64)v134;
          v12 = v143;
        }
        ++v9;
        v143 = v12 + 1;
        if ( v120 == v9 )
          break;
      }
    }
  }
  v13 = (__int64 *)a1[3];
  v126 = (__int64 *)a1[4];
  if ( v13 == v126 )
    goto LABEL_32;
  v135 = v7;
  do
  {
    while ( 1 )
    {
      v14 = *v13;
      LODWORD(v140) = 0;
      v145[0] = v14;
      v15 = *(_BYTE *)v14;
      if ( *(_BYTE *)v14 == 61 )
      {
        v16 = *(_BYTE **)(v14 - 32);
        v14 = 0;
      }
      else
      {
        switch ( v15 )
        {
          case '>':
            v16 = *(_BYTE **)(v14 - 32);
            break;
          case 'U':
            v91 = *(_QWORD *)(v14 - 32);
            if ( !v91
              || *(_BYTE *)v91
              || *(_QWORD *)(v91 + 24) != *(_QWORD *)(v14 + 80)
              || (*(_BYTE *)(v91 + 33) & 0x20) == 0
              || !(unsigned __int8)sub_2CE0320((__int64)a1, *(_DWORD *)(v91 + 36), &v140) )
            {
              goto LABEL_17;
            }
            v16 = *(_BYTE **)(v14 + 32 * ((unsigned int)v140 - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)));
            v14 = 0;
            break;
          case 'A':
            v16 = *(_BYTE **)(v14 - 96);
            v14 = 0;
            break;
          case 'B':
            v16 = *(_BYTE **)(v14 - 64);
            v14 = 0;
            break;
          default:
            goto LABEL_17;
        }
      }
      v17 = (unsigned int)sub_2CED090((__int64)a1, (unsigned __int64)v16, (__int64)&v146, v135);
      LODWORD(v141) = 0;
      if ( (unsigned __int8)sub_2CE0930(a1, v145[0], (__int64)v16, (unsigned int *)&v141) )
        v17 = (unsigned int)v141;
      if ( (unsigned int)v17 > 0x10 )
        goto LABEL_13;
      v18 = 65814;
      if ( !_bittest64(&v18, v17) )
      {
        if ( (_DWORD)v17 != 15 )
        {
LABEL_13:
          if ( *v16 > 0x1Cu )
            sub_CF0BD0((__int64)v16, v17);
        }
        if ( unk_50142AD && *(_BYTE *)a1 )
        {
          v151 = 0;
          v150 = (unsigned __int64)v152;
          LOBYTE(v152[0]) = 0;
          v123 = (__int64 *)(v145[0] + 48);
          sub_B2BE50(v135);
          sub_2C75F20((__int64)&v153, v123);
          sub_2241490(&v150, v153, v154);
          if ( v153 != (char *)v155 )
            j_j___libc_free_0((unsigned __int64)v153);
          if ( 0x3FFFFFFFFFFFFFFFLL - v151 <= 0x4A )
            sub_4262D8((__int64)"basic_string::append");
          sub_2241490(&v150, ": Warning: Cannot tell what pointer points to, assuming global memory space", 0x4Bu);
          sub_CEB650(&v150);
          if ( (_BYTE)qword_50140A8 && (_BYTE)qword_5013FC8 )
            sub_CF0910((unsigned __int64)v16);
          if ( (_QWORD *)v150 != v152 )
            j_j___libc_free_0(v150);
        }
        goto LABEL_17;
      }
      v19 = *(_BYTE *)v145[0];
      if ( *(_BYTE *)v145[0] > 0x1Cu )
      {
        if ( v19 == 85 )
        {
          v73 = *(_QWORD *)(v145[0] - 32);
          if ( v73
            && !*(_BYTE *)v73
            && *(_QWORD *)(v73 + 24) == *(_QWORD *)(v145[0] + 80)
            && (*(_BYTE *)(v73 + 33) & 0x20) != 0 )
          {
            v20 = sub_2CE03D0((__int64)a1, *(_DWORD *)(v73 + 36));
LABEL_96:
            if ( !v14 )
              goto LABEL_27;
            goto LABEL_26;
          }
        }
        else if ( v19 == 65 || v19 == 66 )
        {
          v20 = 1;
          goto LABEL_96;
        }
      }
      if ( !v14 )
        goto LABEL_89;
LABEL_26:
      v20 = 1;
LABEL_27:
      if ( (_DWORD)v17 != 4 || !v20 )
      {
LABEL_89:
        v66 = v143;
        if ( v143 == v144 )
        {
          sub_24454E0((__int64)&v142, v143, v145);
        }
        else
        {
          if ( v143 )
          {
            *v143 = v145[0];
            v66 = v143;
          }
          v143 = v66 + 1;
        }
        goto LABEL_17;
      }
      v153 = (char *)v155;
      v150 = 71;
      v21 = sub_22409D0((__int64)&v153, &v150, 0);
      v153 = (char *)v21;
      v155[0] = v150;
      *(__m128i *)v21 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_42DFCD0);
      *(_DWORD *)(v21 + 64) = 1886593145;
      *(__m128i *)(v21 + 16) = si128;
      v23 = _mm_load_si128((const __m128i *)&xmmword_42DFCE0);
      *(_WORD *)(v21 + 68) = 25441;
      *(__m128i *)(v21 + 32) = v23;
      v24 = _mm_load_si128((const __m128i *)&xmmword_42DFCF0);
      *(_BYTE *)(v21 + 70) = 101;
      *(__m128i *)(v21 + 48) = v24;
      v154 = v150;
      v153[v150] = 0;
      sub_2CDF8F0(v145[0], (__int64)&v153);
      if ( v153 != (char *)v155 )
        break;
LABEL_17:
      if ( v126 == ++v13 )
        goto LABEL_31;
    }
    ++v13;
    j_j___libc_free_0((unsigned __int64)v153);
  }
  while ( v126 != v13 );
LABEL_31:
  v7 = v135;
LABEL_32:
  LODWORD(v154) = 0;
  v155[1] = (unsigned __int64)&v154;
  v155[2] = (unsigned __int64)&v154;
  v155[0] = 0;
  v155[3] = 0;
  v136 = v143;
  if ( v143 != v142 )
  {
    v25 = v142;
    while ( 1 )
    {
      while ( 1 )
      {
        v32 = *v25;
        v139 = 0;
        v145[0] = (__int64)&v140;
        v140 = v32;
        v145[1] = (__int64)a1;
        v145[2] = (__int64)&v146;
        v33 = *(_BYTE *)v32;
        if ( *(_BYTE *)v32 == 82 )
        {
          v121 = sub_2CEC540((__int64)v145, *(_QWORD *)(v32 - 64), &v139);
          v26 = sub_2CEABE0(a1, v7, *(unsigned __int8 **)(v32 - 64), v140, &v153, v121, v139);
          if ( *(_QWORD *)(v32 - 64) )
          {
            v27 = *(_QWORD *)(v32 - 56);
            **(_QWORD **)(v32 - 48) = v27;
            if ( v27 )
              *(_QWORD *)(v27 + 16) = *(_QWORD *)(v32 - 48);
          }
          *(_QWORD *)(v32 - 64) = v26;
          if ( v26 )
          {
            v28 = *(_QWORD *)(v26 + 16);
            *(_QWORD *)(v32 - 56) = v28;
            if ( v28 )
              *(_QWORD *)(v28 + 16) = v32 - 56;
            *(_QWORD *)(v32 - 48) = v26 + 16;
            *(_QWORD *)(v26 + 16) = v32 - 64;
          }
          v29 = sub_2CEABE0(a1, v7, *(unsigned __int8 **)(v32 - 32), v140, &v153, v121, v139);
          if ( *(_QWORD *)(v32 - 32) )
          {
            v30 = *(_QWORD *)(v32 - 24);
            **(_QWORD **)(v32 - 16) = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = *(_QWORD *)(v32 - 16);
          }
          *(_QWORD *)(v32 - 32) = v29;
          if ( v29 )
          {
            v31 = *(_QWORD *)(v29 + 16);
            *(_QWORD *)(v32 - 24) = v31;
            if ( v31 )
              *(_QWORD *)(v31 + 16) = v32 - 24;
            *(_QWORD *)(v32 - 16) = v29 + 16;
            *(_QWORD *)(v29 + 16) = v32 - 32;
          }
          goto LABEL_48;
        }
        if ( v33 != 61 )
          break;
        v127 = *(unsigned __int8 **)(v32 - 32);
        v34 = sub_2CEC540((__int64)v145, (__int64)v127, &v139);
        v35 = v140;
        v36 = sub_2CEABE0(a1, v7, v127, v140, &v153, v34, v139);
        if ( (*(_BYTE *)(v35 + 7) & 0x40) != 0 )
          v37 = *(_QWORD *)(v35 - 8);
        else
          v37 = v35 - 32LL * (*(_DWORD *)(v35 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)v37 )
        {
          v38 = *(_QWORD *)(v37 + 8);
          **(_QWORD **)(v37 + 16) = v38;
          if ( v38 )
            *(_QWORD *)(v38 + 16) = *(_QWORD *)(v37 + 16);
        }
        *(_QWORD *)v37 = v36;
        if ( !v36 )
          goto LABEL_48;
        v39 = *(_QWORD *)(v36 + 16);
        *(_QWORD *)(v37 + 8) = v39;
        if ( v39 )
          *(_QWORD *)(v39 + 16) = v37 + 8;
        *(_QWORD *)(v37 + 16) = v36 + 16;
        ++v25;
        *(_QWORD *)(v36 + 16) = v37;
        if ( v136 == v25 )
        {
LABEL_60:
          v122 = 1;
          goto LABEL_61;
        }
      }
      if ( v33 == 62 )
      {
        v130 = *(unsigned __int8 **)(v32 - 32);
        v67 = sub_2CEC540((__int64)v145, (__int64)v130, &v139);
        v68 = v140;
        v69 = sub_2CEABE0(a1, v7, v130, v140, &v153, v67, v139);
        if ( (*(_BYTE *)(v68 + 7) & 0x40) != 0 )
          v70 = *(_QWORD *)(v68 - 8);
        else
          v70 = v68 - 32LL * (*(_DWORD *)(v68 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)(v70 + 32) )
        {
          v71 = *(_QWORD *)(v70 + 40);
          **(_QWORD **)(v70 + 48) = v71;
          if ( v71 )
            *(_QWORD *)(v71 + 16) = *(_QWORD *)(v70 + 48);
        }
        *(_QWORD *)(v70 + 32) = v69;
        if ( v69 )
        {
          v72 = *(_QWORD *)(v69 + 16);
          *(_QWORD *)(v70 + 40) = v72;
          if ( v72 )
            *(_QWORD *)(v72 + 16) = v70 + 40;
          *(_QWORD *)(v70 + 48) = v69 + 16;
          *(_QWORD *)(v69 + 16) = v70 + 32;
        }
        goto LABEL_48;
      }
      if ( v33 != 65 )
        break;
      v131 = *(_QWORD *)(v32 - 96);
      v84 = sub_2CEC540((__int64)v145, v131, &v139);
      v85 = (unsigned __int8 *)v131;
      v86 = v84;
      if ( ((v84 - 4) & 0xFFFFFFFB) != 0 )
        goto LABEL_208;
      v150 = (unsigned __int64)v152;
      if ( v84 == 8 )
      {
        v141 = 43;
        v87 = sub_22409D0((__int64)&v150, &v141, 0);
        v150 = v87;
        v152[0] = v141;
        *(__m128i *)v87 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
        v88 = _mm_load_si128((const __m128i *)&xmmword_444AF70);
        qmemcpy((void *)(v87 + 32), "ocal memory", 11);
      }
      else
      {
        v141 = 46;
        v87 = sub_22409D0((__int64)&v150, &v141, 0);
        v150 = v87;
        v152[0] = v141;
        *(__m128i *)v87 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
        v88 = _mm_load_si128((const __m128i *)&xmmword_444AF60);
        qmemcpy((void *)(v87 + 32), "onstant memory", 14);
      }
      *(__m128i *)(v87 + 16) = v88;
      v151 = v141;
      *(_BYTE *)(v150 + v141) = 0;
      sub_2CDF8F0(v140, (__int64)&v150);
      v89 = v150;
      if ( (_QWORD *)v150 != v152 )
        goto LABEL_166;
LABEL_48:
      if ( v136 == ++v25 )
        goto LABEL_60;
    }
    if ( v33 == 66 )
    {
      v132 = *(_QWORD *)(v32 - 64);
      v92 = sub_2CEC540((__int64)v145, v132, &v139);
      v85 = (unsigned __int8 *)v132;
      v86 = v92;
      if ( ((v92 - 4) & 0xFFFFFFFB) != 0 )
      {
LABEL_208:
        v102 = v140;
        v103 = sub_2CEABE0(a1, v7, v85, v140, &v153, v86, v139);
        if ( (*(_BYTE *)(v102 + 7) & 0x40) != 0 )
          v104 = *(_QWORD *)(v102 - 8);
        else
          v104 = v102 - 32LL * (*(_DWORD *)(v102 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)v104 )
        {
          v105 = *(_QWORD *)(v104 + 8);
          **(_QWORD **)(v104 + 16) = v105;
          if ( v105 )
            *(_QWORD *)(v105 + 16) = *(_QWORD *)(v104 + 16);
        }
        *(_QWORD *)v104 = v103;
        if ( v103 )
        {
          v106 = *(_QWORD *)(v103 + 16);
          *(_QWORD *)(v104 + 8) = v106;
          if ( v106 )
            *(_QWORD *)(v106 + 16) = v104 + 8;
          *(_QWORD *)(v104 + 16) = v103 + 16;
          *(_QWORD *)(v103 + 16) = v104;
        }
        goto LABEL_48;
      }
      v93 = ": Warning: Cannot do atomic on local memory";
      if ( v92 != 8 )
        v93 = ": Warning: Cannot do atomic on constant memory";
    }
    else
    {
      v94 = *(_QWORD *)(v32 - 32);
      if ( !v94 || *(_BYTE *)v94 || *(_QWORD *)(v94 + 24) != *(_QWORD *)(v32 + 80) )
        goto LABEL_256;
      v95 = *(_DWORD *)(v94 + 36);
      LODWORD(v141) = 0;
      v133 = v95;
      sub_2CE0320((__int64)a1, v95, &v141);
      v124 = *(unsigned __int8 **)(v32 + 32
                                       * ((unsigned int)v141 - (unsigned __int64)(*(_DWORD *)(v32 + 4) & 0x7FFFFFF)));
      v117 = sub_2CEC540((__int64)v145, (__int64)v124, &v139);
      if ( !sub_CEA260(v133) )
      {
        v112 = sub_CEA1F0(v133);
        v97 = v117;
        if ( v112 )
        {
          if ( ((v117 - 4) & 0xFFFFFFFB) != 0 )
          {
LABEL_202:
            v118 = v97;
            sub_2CE02B0((__int64)a1, v133);
            v98 = v118;
LABEL_203:
            v99 = sub_2CEABE0(a1, v7, v124, v140, &v153, v98, v139);
            sub_AC2B30(v32 + 32 * ((unsigned int)v141 - (unsigned __int64)(*(_DWORD *)(v32 + 4) & 0x7FFFFFF)), v99);
            v150 = (unsigned __int64)v152;
            v151 = 0x300000000LL;
            if ( (unsigned __int8)sub_CEA1F0(v133) )
            {
              v100 = (unsigned int)v141 - (unsigned __int64)(*(_DWORD *)(v32 + 4) & 0x7FFFFFF);
              goto LABEL_205;
            }
            v115 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
            if ( v133 == 243 )
            {
              v116 = *(_QWORD *)(*(_QWORD *)(v32 - 32 * v115) + 8LL);
            }
            else
            {
              if ( v133 != 238 && v133 != 241 )
              {
                v100 = (unsigned int)v141 - v115;
LABEL_205:
                sub_94F8E0((__int64)&v150, *(_QWORD *)(*(_QWORD *)(v32 + 32 * v100) + 8LL));
LABEL_206:
                v101 = sub_B6E160(*(__int64 **)(v7 + 40), v133, v150, (unsigned int)v151);
                *(_QWORD *)(v32 + 80) = *(_QWORD *)(v101 + 24);
                sub_AC2B30(v32 - 32, v101);
                if ( (_QWORD *)v150 != v152 )
                  _libc_free(v150);
                goto LABEL_48;
              }
              sub_94F8E0((__int64)&v150, *(_QWORD *)(*(_QWORD *)(v32 - 32 * v115) + 8LL));
              v116 = *(_QWORD *)(*(_QWORD *)(v32 + 32 * (1LL - (*(_DWORD *)(v32 + 4) & 0x7FFFFFF))) + 8LL);
            }
            sub_94F8E0((__int64)&v150, v116);
            sub_94F8E0(
              (__int64)&v150,
              *(_QWORD *)(*(_QWORD *)(v32 + 32 * (2LL - (*(_DWORD *)(v32 + 4) & 0x7FFFFFF))) + 8LL));
            goto LABEL_206;
          }
          v111 = ": Warning: Cannot do atomic on local memory";
          if ( v117 != 8 )
            v111 = ": Warning: Cannot do atomic on constant memory";
          goto LABEL_236;
        }
LABEL_243:
        v119 = v97;
        v114 = sub_2CE02B0((__int64)a1, v133);
        v98 = v119;
        if ( !v114 || ((v119 - 4) & 0xFFFFFFFB) != 0 )
          goto LABEL_203;
        v111 = ": Warning: cannot perform wmma load or store on local memory";
        if ( v119 != 8 )
          v111 = ": Warning: cannot perform wmma load or store on constant memory";
LABEL_236:
        sub_2CDD970((__int64 *)&v150, v111);
        sub_2CDF8F0(v140, (__int64)&v150);
        sub_2240A30(&v150);
        goto LABEL_48;
      }
      if ( ((v117 - 4) & 0xFFFFFFFB) == 0 )
      {
        v111 = ": Warning: Cannot do vector atomic on local memory";
        if ( v117 != 8 )
          v111 = ": Warning: Cannot do vector atomic on constant memory";
        goto LABEL_236;
      }
      if ( v117 != 2 )
      {
        v96 = sub_CEA1F0(v133);
        v97 = v117;
        if ( v96 )
          goto LABEL_202;
        goto LABEL_243;
      }
      v93 = ": Warning: Cannot do vector atomic on shared memory";
    }
    sub_2CDD970((__int64 *)&v150, v93);
    sub_2CDF8F0(v140, (__int64)&v150);
    v89 = v150;
    if ( (_QWORD *)v150 == v152 )
      goto LABEL_48;
LABEL_166:
    j_j___libc_free_0(v89);
    goto LABEL_48;
  }
  v122 = 0;
LABEL_61:
  v128 = (__int64 *)a1[7];
  if ( v128 != (__int64 *)a1[6] )
  {
    v40 = (__int64 *)a1[6];
    do
    {
      v41 = *v40;
      v137 = *(unsigned __int8 **)(*v40 + 32 * (1LL - (*(_DWORD *)(*v40 + 4) & 0x7FFFFFF)));
      v42 = sub_2CED090((__int64)a1, (unsigned __int64)v137, (__int64)&v146, v7);
      v43 = v42;
      if ( v42 <= 0x10 )
      {
        v44 = 65814;
        if ( _bittest64(&v44, v43) )
        {
          v45 = sub_2CEABE0(a1, v7, v137, v41, &v153, v43, 0);
          v46 = v41 + 32 * (1LL - (*(_DWORD *)(v41 + 4) & 0x7FFFFFF));
          if ( *(_QWORD *)v46 )
          {
            v47 = *(_QWORD *)(v46 + 8);
            **(_QWORD **)(v46 + 16) = v47;
            if ( v47 )
              *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 + 16);
          }
          *(_QWORD *)v46 = v45;
          if ( v45 )
          {
            v48 = *(_QWORD *)(v45 + 16);
            *(_QWORD *)(v46 + 8) = v48;
            if ( v48 )
              *(_QWORD *)(v48 + 16) = v46 + 8;
            *(_QWORD *)(v46 + 16) = v45 + 16;
            *(_QWORD *)(v45 + 16) = v46;
          }
          v150 = (unsigned __int64)v152;
          v151 = 0x300000000LL;
          v49 = *(_QWORD *)(v41 - 32);
          if ( !v49 || *(_BYTE *)v49 || *(_QWORD *)(v49 + 24) != *(_QWORD *)(v41 + 80) )
LABEL_256:
            BUG();
          v50 = *(_DWORD *)(v49 + 36);
          v51 = *(__int64 **)(v7 + 40);
          v52 = *(_QWORD *)(*(_QWORD *)(v41 - 32LL * (*(_DWORD *)(v41 + 4) & 0x7FFFFFF)) + 8LL);
          LODWORD(v151) = 1;
          v152[0] = v52;
          v53 = *(_QWORD *)(*(_QWORD *)(v41 + 32 * (1LL - (*(_DWORD *)(v41 + 4) & 0x7FFFFFF))) + 8LL);
          LODWORD(v151) = 2;
          v152[1] = v53;
          v54 = *(_QWORD *)(*(_QWORD *)(v41 + 32 * (2LL - (*(_DWORD *)(v41 + 4) & 0x7FFFFFF))) + 8LL);
          LODWORD(v151) = 3;
          v152[2] = v54;
          v55 = sub_B6E160(v51, v50, (__int64)v152, 3);
          v56 = *(_QWORD *)(v41 - 32) == 0;
          *(_QWORD *)(v41 + 80) = *(_QWORD *)(v55 + 24);
          if ( !v56 )
          {
            v57 = *(_QWORD *)(v41 - 24);
            **(_QWORD **)(v41 - 16) = v57;
            if ( v57 )
              *(_QWORD *)(v57 + 16) = *(_QWORD *)(v41 - 16);
          }
          *(_QWORD *)(v41 - 32) = v55;
          v58 = *(_QWORD *)(v55 + 16);
          *(_QWORD *)(v41 - 24) = v58;
          if ( v58 )
            *(_QWORD *)(v58 + 16) = v41 - 24;
          *(_QWORD *)(v41 - 16) = v55 + 16;
          *(_QWORD *)(v55 + 16) = v41 - 32;
          if ( (_QWORD *)v150 != v152 )
            _libc_free(v150);
          v122 = 1;
        }
      }
      ++v40;
    }
    while ( v128 != v40 );
  }
  v59 = (__int64 *)a1[12];
  v138 = (__int64 *)a1[13];
  if ( v138 != v59 )
  {
    v129 = 0;
    while ( 2 )
    {
      v60 = *v59;
      v61 = *(_QWORD *)(*v59 - 32LL * (*(_DWORD *)(*v59 + 4) & 0x7FFFFFF));
      v62 = sub_2CECAD0((__int64)a1, v61, (__int64)&v146, v7);
      switch ( v62 )
      {
        case 1:
        case 4:
          goto LABEL_133;
        case 2:
          v62 = 3;
          goto LABEL_133;
        case 8:
          v62 = 5;
          goto LABEL_133;
        case 16:
          v74 = *(_BYTE *)v61;
          v62 = 101;
          if ( *(_BYTE *)v61 <= 0x1Cu )
            goto LABEL_138;
          goto LABEL_123;
        default:
          LODWORD(v150) = 0;
          if ( (unsigned __int8)sub_2CE0930(a1, v60, v61, (unsigned int *)&v150) )
          {
            v62 = v150;
            if ( (_DWORD)v150 && (_DWORD)v150 != 101 )
            {
LABEL_133:
              v77 = *(_QWORD *)(v60 - 32);
              if ( v77 && !*(_BYTE *)v77 && *(_QWORD *)(v77 + 24) == *(_QWORD *)(v60 + 80) )
              {
                switch ( *(_DWORD *)(v77 + 36) )
                {
                  case 0x22DF:
                    v82 = v62 == 4;
                    goto LABEL_149;
                  case 0x22E0:
                    v82 = v62 == 1;
                    goto LABEL_149;
                  case 0x22E1:
                    goto LABEL_150;
                  case 0x22E2:
                    v82 = v62 == 5;
                    goto LABEL_149;
                  case 0x22E3:
                  case 0x22E4:
                    v82 = v62 == 3;
LABEL_149:
                    if ( v82 )
                      v83 = sub_AD6400(*(_QWORD *)(v60 + 8));
                    else
LABEL_150:
                      v83 = sub_AD6450(*(_QWORD *)(v60 + 8));
                    goto LABEL_151;
                  default:
                    goto LABEL_256;
                }
              }
              goto LABEL_256;
            }
          }
          else
          {
            v62 = 0;
          }
          v74 = *(_BYTE *)v61;
          if ( *(_BYTE *)v61 <= 0x1Cu )
          {
LABEL_138:
            if ( v74 == 5 && *(_WORD *)(v61 + 2) == 34 )
            {
LABEL_140:
              v78 = *(_QWORD *)(v61 - 32LL * (*(_DWORD *)(v61 + 4) & 0x7FFFFFF));
              v64 = v60 - 32LL * (*(_DWORD *)(v60 + 4) & 0x7FFFFFF);
              v79 = *(_QWORD *)v64;
              if ( v78 )
              {
                if ( v79 )
                {
                  v80 = *(_QWORD *)(v64 + 8);
                  **(_QWORD **)(v64 + 16) = v80;
                  if ( v80 )
                    *(_QWORD *)(v80 + 16) = *(_QWORD *)(v64 + 16);
                }
                *(_QWORD *)v64 = v78;
                v81 = *(_QWORD *)(v78 + 16);
                v63 = v78 + 16;
                *(_QWORD *)(v64 + 8) = v81;
                if ( v81 )
                  *(_QWORD *)(v81 + 16) = v64 + 8;
                *(_QWORD *)(v64 + 16) = v63;
                *(_QWORD *)(v78 + 16) = v64;
              }
              else if ( v79 )
              {
                v90 = *(_QWORD *)(v64 + 8);
                **(_QWORD **)(v64 + 16) = v90;
                if ( v90 )
                  *(_QWORD *)(v90 + 16) = *(_QWORD *)(v64 + 16);
                *(_QWORD *)v64 = 0;
              }
              goto LABEL_126;
            }
          }
          else
          {
LABEL_123:
            if ( v74 == 63 )
              goto LABEL_140;
          }
          if ( !v62 )
          {
            v75 = *(_QWORD *)(v60 - 32LL * (*(_DWORD *)(v60 + 4) & 0x7FFFFFF));
            if ( *(_BYTE *)v75 == 85 )
            {
              v107 = *(_QWORD *)(v75 - 32);
              if ( v107 )
              {
                if ( !*(_BYTE *)v107
                  && *(_QWORD *)(v107 + 24) == *(_QWORD *)(v75 + 80)
                  && (*(_BYTE *)(v107 + 33) & 0x20) != 0
                  && *(_DWORD *)(v107 + 36) == 9005 )
                {
                  v108 = *(_QWORD *)(v60 - 32);
                  if ( !v108 || *(_BYTE *)v108 || *(_QWORD *)(v108 + 24) != *(_QWORD *)(v60 + 80) )
                    goto LABEL_256;
                  v109 = *(_DWORD *)(v108 + 36);
                  if ( v109 > 0x22E2 )
                  {
                    if ( v109 != 8932 )
                      goto LABEL_126;
                    v113 = (__int64 *)sub_BD5C60(v60);
                    v83 = sub_ACD6D0(v113);
                  }
                  else
                  {
                    if ( v109 <= 0x22DE )
                      goto LABEL_126;
                    v110 = (__int64 *)sub_BD5C60(v60);
                    v83 = sub_ACD720(v110);
                  }
                  if ( v83 )
                  {
LABEL_151:
                    sub_BD84D0(v60, v83);
                    sub_B43D60((_QWORD *)v60);
                    v129 = 1;
                  }
                }
              }
            }
          }
LABEL_126:
          if ( v138 != ++v59 )
            continue;
          if ( v129 )
          {
            sub_F62E00(v7, 0, 0, v63, v64, v65);
            v122 = v129;
          }
          break;
      }
      break;
    }
  }
  sub_2CDE470(v155[0]);
  sub_C7D6A0(0, 0, 8);
  if ( v142 )
    j_j___libc_free_0((unsigned __int64)v142);
  sub_C7D6A0(v147, 16LL * v149, 8);
  return v122;
}
