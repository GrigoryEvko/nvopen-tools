// Function: sub_2CEAC10
// Address: 0x2ceac10
//
__int64 __fastcall sub_2CEAC10(_QWORD *a1, __int64 a2)
{
  char **v3; // rbx
  char **v4; // rax
  char *v5; // r12
  int v6; // r9d
  __int64 *v7; // rax
  __int64 *v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rdx
  char v14; // di
  __int64 *v15; // r12
  __int64 *v16; // rbx
  __int64 v17; // r14
  char v18; // al
  __int64 v19; // r15
  char v20; // al
  char **v21; // rsi
  char **v22; // r14
  __int64 v23; // rdx
  char *v24; // rsi
  int v25; // r9d
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  int v29; // r9d
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  char *v33; // r12
  char v34; // al
  __int64 v35; // r11
  char *v36; // rsi
  char v37; // al
  unsigned __int8 *v38; // r11
  int v39; // r9d
  unsigned __int64 v40; // rax
  char *v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 *v44; // r12
  __int64 v45; // r14
  __int64 v46; // r15
  __int64 v47; // rdx
  unsigned int v48; // eax
  unsigned __int64 v49; // rax
  __int64 v50; // rbx
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned int v54; // esi
  __int64 *v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  bool v60; // zf
  __int64 v61; // rdx
  __int64 v62; // rdx
  char *v63; // rdi
  __int64 *v64; // rax
  __int64 *v65; // r9
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 *v68; // rax
  __int64 *v69; // rsi
  __int64 v70; // rcx
  __int64 v71; // rdx
  __int64 v72; // rax
  __m128i si128; // xmm0
  __m128i v74; // xmm0
  __m128i v75; // xmm0
  __int64 v76; // r11
  char *v77; // rsi
  char v78; // al
  unsigned __int8 *v79; // r11
  int v80; // r9d
  unsigned __int64 v81; // rax
  char *v82; // r12
  __int64 v83; // rdx
  __int64 v84; // rdx
  __int64 *v86; // rax
  __int64 *v87; // rsi
  __int64 v88; // rcx
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 *v91; // rax
  __int64 *v92; // rsi
  __int64 v93; // rcx
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // r11
  char *v97; // rsi
  char v98; // al
  __int64 v99; // rax
  __m128i v100; // xmm0
  unsigned __int64 *v101; // rdi
  __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // r11
  char *v105; // rsi
  char v106; // al
  const char *v107; // rsi
  __int64 *v108; // rax
  __int64 *v109; // r8
  __int64 v110; // rcx
  __int64 v111; // rdx
  __int64 v112; // rax
  __int64 v113; // rax
  unsigned int v114; // eax
  __int64 v115; // rax
  unsigned __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rax
  char **v119; // rsi
  __int64 *v120; // rax
  __int64 *v121; // r8
  __int64 v122; // rax
  __int64 *v123; // rax
  __int64 *v124; // rsi
  __int64 v125; // rax
  const char *v126; // rsi
  __int64 v127; // rsi
  int v128; // [rsp+Ch] [rbp-174h]
  __int64 v129; // [rsp+10h] [rbp-170h]
  unsigned __int8 *v130; // [rsp+10h] [rbp-170h]
  __int64 *v131; // [rsp+10h] [rbp-170h]
  unsigned __int64 v132; // [rsp+18h] [rbp-168h]
  unsigned __int8 v133; // [rsp+18h] [rbp-168h]
  int v134; // [rsp+20h] [rbp-160h]
  char **v135; // [rsp+28h] [rbp-158h]
  __int64 *v136; // [rsp+30h] [rbp-150h]
  unsigned __int8 *v138; // [rsp+38h] [rbp-148h]
  unsigned __int8 *v139; // [rsp+40h] [rbp-140h]
  __int64 *v140; // [rsp+40h] [rbp-140h]
  unsigned __int8 *v141; // [rsp+40h] [rbp-140h]
  unsigned __int8 *v142; // [rsp+40h] [rbp-140h]
  unsigned __int8 *v143; // [rsp+40h] [rbp-140h]
  unsigned int v144; // [rsp+40h] [rbp-140h]
  int v145; // [rsp+48h] [rbp-138h]
  char **v146; // [rsp+48h] [rbp-138h]
  unsigned int v147; // [rsp+5Ch] [rbp-124h] BYREF
  unsigned int v148; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v149; // [rsp+64h] [rbp-11Ch] BYREF
  char *v150; // [rsp+68h] [rbp-118h] BYREF
  char **v151; // [rsp+70h] [rbp-110h] BYREF
  char **v152; // [rsp+78h] [rbp-108h]
  char **v153; // [rsp+80h] [rbp-100h]
  unsigned __int64 v154; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v155; // [rsp+98h] [rbp-E8h]
  _QWORD v156[4]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v157; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v158; // [rsp+C8h] [rbp-B8h] BYREF
  __int64 *v159; // [rsp+D0h] [rbp-B0h]
  __int64 *v160; // [rsp+D8h] [rbp-A8h]
  __int64 *v161; // [rsp+E0h] [rbp-A0h]
  __int64 v162; // [rsp+E8h] [rbp-98h]
  char *v163; // [rsp+F0h] [rbp-90h] BYREF
  size_t v164; // [rsp+F8h] [rbp-88h] BYREF
  unsigned __int64 v165[4]; // [rsp+100h] [rbp-80h] BYREF
  unsigned __int64 *v166; // [rsp+120h] [rbp-60h] BYREF
  unsigned __int64 v167; // [rsp+128h] [rbp-58h] BYREF
  unsigned __int64 v168; // [rsp+130h] [rbp-50h] BYREF
  unsigned __int64 *v169; // [rsp+138h] [rbp-48h]
  unsigned __int64 *v170; // [rsp+140h] [rbp-40h]
  __int64 v171; // [rsp+148h] [rbp-38h]

  sub_2CE1400(a1, a2);
  v3 = (char **)a1[9];
  LODWORD(v158) = 0;
  v160 = &v158;
  v161 = &v158;
  v4 = (char **)a1[10];
  v159 = 0;
  v162 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v135 = v4;
  while ( v135 != v3 )
  {
    v5 = *v3;
    LODWORD(v167) = 0;
    v168 = 0;
    v169 = &v167;
    v170 = &v167;
    v171 = 0;
    LODWORD(v150) = *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v5 - 8) + 8LL) + 8LL) >> 8;
    LODWORD(v154) = (_DWORD)v150;
    if ( (unsigned int)sub_2CE8530((__int64)a1, a2, *((unsigned __int8 **)v5 - 8), &v157, &v166, (int *)&v150) == 1
      && (unsigned int)sub_2CE8530((__int64)a1, a2, *((unsigned __int8 **)v5 - 4), &v157, &v166, (int *)&v154) == 1 )
    {
      v6 = (int)v150;
      if ( (_DWORD)v150 == (_DWORD)v154 )
      {
        v7 = v159;
        v8 = &v158;
        if ( !v159 )
          goto LABEL_14;
        do
        {
          while ( 1 )
          {
            v9 = v7[2];
            v10 = v7[3];
            if ( v7[4] >= (unsigned __int64)v5 )
              break;
            v7 = (__int64 *)v7[3];
            if ( !v10 )
              goto LABEL_12;
          }
          v8 = v7;
          v7 = (__int64 *)v7[2];
        }
        while ( v9 );
LABEL_12:
        if ( v8 == &v158 || v8[4] > (unsigned __int64)v5 )
        {
LABEL_14:
          v128 = (int)v150;
          v129 = (__int64)v8;
          v11 = sub_22077B0(0x30u);
          *(_QWORD *)(v11 + 32) = v5;
          *(_DWORD *)(v11 + 40) = 0;
          v132 = v11;
          v12 = sub_2CBB8B0(&v157, v129, (unsigned __int64 *)(v11 + 32));
          if ( v13 )
          {
            v14 = v12 || &v158 == v13 || (unsigned __int64)v5 < v13[4];
            sub_220F040(v14, v132, v13, &v158);
            ++v162;
            v8 = (__int64 *)v132;
            v6 = v128;
          }
          else
          {
            v131 = v12;
            j_j___libc_free_0(v132);
            v6 = v128;
            v8 = v131;
          }
        }
        *((_DWORD *)v8 + 10) = v6;
        v119 = v152;
        v163 = v5;
        if ( v152 == v153 )
        {
          sub_249A840((__int64)&v151, v152, &v163);
        }
        else
        {
          if ( v152 )
          {
            *v152 = v5;
            v119 = v152;
          }
          v152 = v119 + 1;
        }
      }
    }
    ++v3;
    sub_2CDF380(v168);
  }
  v15 = (__int64 *)a1[4];
  v16 = (__int64 *)a1[3];
  if ( v16 != v15 )
  {
    while ( 1 )
    {
      v17 = *v16;
      v147 = 0;
      v150 = (char *)v17;
      v18 = *(_BYTE *)v17;
      if ( *(_BYTE *)v17 == 61 )
      {
        v19 = *(_QWORD *)(v17 - 32);
        v17 = 0;
      }
      else
      {
        switch ( v18 )
        {
          case '>':
            v19 = *(_QWORD *)(v17 - 32);
            break;
          case 'U':
            v103 = *(_QWORD *)(v17 - 32);
            if ( !v103
              || *(_BYTE *)v103
              || *(_QWORD *)(v103 + 24) != *(_QWORD *)(v17 + 80)
              || (*(_BYTE *)(v103 + 33) & 0x20) == 0
              || !(unsigned __int8)sub_2CE0320((__int64)a1, *(_DWORD *)(v103 + 36), &v147) )
            {
              goto LABEL_33;
            }
            v19 = *(_QWORD *)(v17 + 32 * (v147 - (unsigned __int64)(*(_DWORD *)(v17 + 4) & 0x7FFFFFF)));
            v17 = 0;
            break;
          case 'A':
            v19 = *(_QWORD *)(v17 - 96);
            v17 = 0;
            break;
          case 'B':
            v19 = *(_QWORD *)(v17 - 64);
            v17 = 0;
            break;
          default:
            goto LABEL_33;
        }
      }
      v148 = *(_DWORD *)(*(_QWORD *)(v19 + 8) + 8LL) >> 8;
      if ( !v148 )
        break;
LABEL_33:
      if ( v15 == ++v16 )
        goto LABEL_34;
    }
    LODWORD(v167) = 0;
    v168 = 0;
    v169 = &v167;
    v170 = &v167;
    v171 = 0;
    v145 = sub_2CE8530((__int64)a1, a2, (unsigned __int8 *)v19, &v157, &v166, (int *)&v148);
    v149 = 0;
    if ( (unsigned __int8)sub_2CE0930(a1, (__int64)v150, v19, &v149) )
    {
      v148 = v149;
    }
    else if ( v145 != 1 )
    {
      if ( !unk_50142AD || !*(_BYTE *)a1 )
        goto LABEL_32;
      v155 = 0;
      v154 = (unsigned __int64)v156;
      LOBYTE(v156[0]) = 0;
      v140 = (__int64 *)(v150 + 48);
      sub_B2BE50(a2);
      sub_2C75F20((__int64)&v163, v140);
      sub_2241490(&v154, v163, v164);
      if ( v163 != (char *)v165 )
        j_j___libc_free_0((unsigned __int64)v163);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v155) <= 0x4A )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v154, ": Warning: Cannot tell what pointer points to, assuming global memory space", 0x4Bu);
      sub_CEB650(&v154);
      if ( (_BYTE)qword_50140A8 && (_BYTE)qword_5013FC8 )
        sub_CF0A70((_BYTE *)v19);
      v63 = (char *)v154;
      if ( (_QWORD *)v154 == v156 )
        goto LABEL_32;
LABEL_103:
      j_j___libc_free_0((unsigned __int64)v63);
LABEL_32:
      sub_2CDF380(v168);
      goto LABEL_33;
    }
    v20 = *v150;
    if ( (unsigned __int8)*v150 > 0x1Cu )
    {
      if ( v20 == 85 )
      {
        v102 = *((_QWORD *)v150 - 4);
        if ( v102
          && !*(_BYTE *)v102
          && *(_QWORD *)(v102 + 24) == *((_QWORD *)v150 + 10)
          && (*(_BYTE *)(v102 + 33) & 0x20) != 0 )
        {
          if ( !sub_2CE03D0((__int64)a1, *(_DWORD *)(v102 + 36)) && !v17 )
          {
            v21 = v152;
            if ( v152 != v153 )
              goto LABEL_29;
LABEL_178:
            sub_24454E0((__int64)&v151, v21, &v150);
            goto LABEL_32;
          }
LABEL_27:
          if ( v148 != 4 )
            goto LABEL_28;
LABEL_128:
          v154 = 71;
          v163 = (char *)v165;
          v72 = sub_22409D0((__int64)&v163, &v154, 0);
          v163 = (char *)v72;
          v165[0] = v154;
          *(__m128i *)v72 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
          si128 = _mm_load_si128((const __m128i *)&xmmword_42DFCD0);
          *(_DWORD *)(v72 + 64) = 1886593145;
          *(__m128i *)(v72 + 16) = si128;
          v74 = _mm_load_si128((const __m128i *)&xmmword_42DFCE0);
          *(_WORD *)(v72 + 68) = 25441;
          *(__m128i *)(v72 + 32) = v74;
          v75 = _mm_load_si128((const __m128i *)&xmmword_42DFCF0);
          *(_BYTE *)(v72 + 70) = 101;
          *(__m128i *)(v72 + 48) = v75;
          v164 = v154;
          v163[v154] = 0;
          sub_2CDF8F0((__int64)v150, (__int64)&v163);
          v63 = v163;
          if ( v163 == (char *)v165 )
            goto LABEL_32;
          goto LABEL_103;
        }
      }
      else if ( v20 == 65 )
      {
        if ( v148 == 4 )
          goto LABEL_128;
        goto LABEL_28;
      }
    }
    if ( v20 == 66 || v17 )
      goto LABEL_27;
LABEL_28:
    v21 = v152;
    if ( v152 != v153 )
    {
LABEL_29:
      if ( v21 )
      {
        *v21 = v150;
        v21 = v152;
      }
      v152 = v21 + 1;
      goto LABEL_32;
    }
    goto LABEL_178;
  }
LABEL_34:
  v22 = v151;
  LODWORD(v164) = 0;
  v165[1] = (unsigned __int64)&v164;
  v165[2] = (unsigned __int64)&v164;
  v165[0] = 0;
  v165[3] = 0;
  v146 = v152;
  if ( v152 != v151 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v33 = *v22;
        v34 = **v22;
        switch ( v34 )
        {
          case 'R':
            v23 = *((_QWORD *)v33 - 8);
            v24 = *v22;
            LODWORD(v150) = 0;
            v154 = v23;
            if ( (unsigned __int8)sub_2CE0930(a1, (__int64)v24, v23, (unsigned int *)&v150) )
            {
              v25 = (int)v150;
LABEL_38:
              v26 = sub_2CE9A90(a1, a2, *((unsigned __int8 **)v33 - 8), (__int64)v33, &v163, v25, 0);
              if ( *((_QWORD *)v33 - 8) )
              {
                v27 = *((_QWORD *)v33 - 7);
                **((_QWORD **)v33 - 6) = v27;
                if ( v27 )
                  *(_QWORD *)(v27 + 16) = *((_QWORD *)v33 - 6);
              }
              *((_QWORD *)v33 - 8) = v26;
              if ( v26 )
              {
                v28 = *(_QWORD *)(v26 + 16);
                *((_QWORD *)v33 - 7) = v28;
                if ( v28 )
                  *(_QWORD *)(v28 + 16) = v33 - 56;
                *((_QWORD *)v33 - 6) = v26 + 16;
                *(_QWORD *)(v26 + 16) = v33 - 64;
              }
              v154 = *((_QWORD *)v33 - 4);
              LODWORD(v150) = 0;
              if ( (unsigned __int8)sub_2CE0930(a1, (__int64)v33, v154, (unsigned int *)&v150) )
              {
                v29 = (int)v150;
                goto LABEL_47;
              }
              v64 = v159;
              if ( v159 )
              {
                v65 = &v158;
                do
                {
                  while ( 1 )
                  {
                    v66 = v64[2];
                    v67 = v64[3];
                    if ( v64[4] >= v154 )
                      break;
                    v64 = (__int64 *)v64[3];
                    if ( !v67 )
                      goto LABEL_112;
                  }
                  v65 = v64;
                  v64 = (__int64 *)v64[2];
                }
                while ( v66 );
LABEL_112:
                if ( v65 == &v158 || v65[4] > v154 )
                {
LABEL_114:
                  v166 = &v154;
                  v65 = (__int64 *)sub_2CE6580(&v157, (__int64)v65, &v166);
                }
                v29 = *((_DWORD *)v65 + 10);
LABEL_47:
                v30 = sub_2CE9A90(a1, a2, *((unsigned __int8 **)v33 - 4), (__int64)v33, &v163, v29, 0);
                if ( *((_QWORD *)v33 - 4) )
                {
                  v31 = *((_QWORD *)v33 - 3);
                  **((_QWORD **)v33 - 2) = v31;
                  if ( v31 )
                    *(_QWORD *)(v31 + 16) = *((_QWORD *)v33 - 2);
                }
                *((_QWORD *)v33 - 4) = v30;
                if ( v30 )
                {
                  v32 = *(_QWORD *)(v30 + 16);
                  *((_QWORD *)v33 - 3) = v32;
                  if ( v32 )
                    *(_QWORD *)(v32 + 16) = v33 - 24;
                  *((_QWORD *)v33 - 2) = v30 + 16;
                  *(_QWORD *)(v30 + 16) = v33 - 32;
                }
                goto LABEL_54;
              }
              v65 = &v158;
              goto LABEL_114;
            }
            v68 = v159;
            if ( v159 )
            {
              v69 = &v158;
              do
              {
                while ( 1 )
                {
                  v70 = v68[2];
                  v71 = v68[3];
                  if ( v68[4] >= v154 )
                    break;
                  v68 = (__int64 *)v68[3];
                  if ( !v71 )
                    goto LABEL_121;
                }
                v69 = v68;
                v68 = (__int64 *)v68[2];
              }
              while ( v70 );
LABEL_121:
              if ( v69 == &v158 || v69[4] > v154 )
              {
LABEL_123:
                v166 = &v154;
                v69 = (__int64 *)sub_2CE6580(&v157, (__int64)v69, &v166);
              }
              v25 = *((_DWORD *)v69 + 10);
              goto LABEL_38;
            }
            v69 = &v158;
            goto LABEL_123;
          case '=':
            v35 = *((_QWORD *)v33 - 4);
            v36 = *v22;
            LODWORD(v150) = 0;
            v154 = v35;
            v139 = (unsigned __int8 *)v35;
            v37 = sub_2CE0930(a1, (__int64)v36, v35, (unsigned int *)&v150);
            v38 = v139;
            if ( v37 )
            {
              v39 = (int)v150;
              goto LABEL_59;
            }
            v86 = v159;
            if ( v159 )
            {
              v87 = &v158;
              do
              {
                while ( 1 )
                {
                  v88 = v86[2];
                  v89 = v86[3];
                  if ( v86[4] >= v154 )
                    break;
                  v86 = (__int64 *)v86[3];
                  if ( !v89 )
                    goto LABEL_151;
                }
                v87 = v86;
                v86 = (__int64 *)v86[2];
              }
              while ( v88 );
LABEL_151:
              if ( v87 == &v158 || v87[4] > v154 )
              {
LABEL_153:
                v166 = &v154;
                v90 = sub_2CE6580(&v157, (__int64)v87, &v166);
                v38 = v139;
                v87 = (__int64 *)v90;
              }
              v39 = *((_DWORD *)v87 + 10);
              goto LABEL_59;
            }
            v87 = &v158;
            goto LABEL_153;
          case '>':
            v76 = *((_QWORD *)v33 - 4);
            v77 = *v22;
            LODWORD(v150) = 0;
            v154 = v76;
            v141 = (unsigned __int8 *)v76;
            v78 = sub_2CE0930(a1, (__int64)v77, v76, (unsigned int *)&v150);
            v79 = v141;
            if ( v78 )
            {
              v80 = (int)v150;
LABEL_133:
              v81 = sub_2CE9A90(a1, a2, v79, (__int64)v33, &v163, v80, 0);
              if ( (v33[7] & 0x40) != 0 )
                v82 = (char *)*((_QWORD *)v33 - 1);
              else
                v82 = &v33[-32 * (*((_DWORD *)v33 + 1) & 0x7FFFFFF)];
              if ( *((_QWORD *)v82 + 4) )
              {
                v83 = *((_QWORD *)v82 + 5);
                **((_QWORD **)v82 + 6) = v83;
                if ( v83 )
                  *(_QWORD *)(v83 + 16) = *((_QWORD *)v82 + 6);
              }
              *((_QWORD *)v82 + 4) = v81;
              if ( v81 )
              {
                v84 = *(_QWORD *)(v81 + 16);
                *((_QWORD *)v82 + 5) = v84;
                if ( v84 )
                  *(_QWORD *)(v84 + 16) = v82 + 40;
                *((_QWORD *)v82 + 6) = v81 + 16;
                *(_QWORD *)(v81 + 16) = v82 + 32;
              }
              goto LABEL_54;
            }
            v91 = v159;
            if ( v159 )
            {
              v92 = &v158;
              do
              {
                while ( 1 )
                {
                  v93 = v91[2];
                  v94 = v91[3];
                  if ( v91[4] >= v154 )
                    break;
                  v91 = (__int64 *)v91[3];
                  if ( !v94 )
                    goto LABEL_160;
                }
                v92 = v91;
                v91 = (__int64 *)v91[2];
              }
              while ( v93 );
LABEL_160:
              if ( v92 == &v158 || v92[4] > v154 )
              {
LABEL_162:
                v166 = &v154;
                v95 = sub_2CE6580(&v157, (__int64)v92, &v166);
                v79 = v141;
                v92 = (__int64 *)v95;
              }
              v80 = *((_DWORD *)v92 + 10);
              goto LABEL_133;
            }
            v92 = &v158;
            goto LABEL_162;
        }
        if ( v34 != 65 )
          break;
        v96 = *((_QWORD *)v33 - 12);
        v97 = *v22;
        LODWORD(v150) = 0;
        v154 = v96;
        v142 = (unsigned __int8 *)v96;
        v98 = sub_2CE0930(a1, (__int64)v97, v96, (unsigned int *)&v150);
        v38 = v142;
        if ( v98 )
        {
          v39 = (int)v150;
          goto LABEL_167;
        }
        v108 = v159;
        if ( v159 )
        {
          v109 = &v158;
          do
          {
            while ( 1 )
            {
              v110 = v108[2];
              v111 = v108[3];
              if ( v108[4] >= v154 )
                break;
              v108 = (__int64 *)v108[3];
              if ( !v111 )
                goto LABEL_199;
            }
            v109 = v108;
            v108 = (__int64 *)v108[2];
          }
          while ( v110 );
LABEL_199:
          if ( v109 != &v158 && v109[4] <= v154 )
            goto LABEL_202;
        }
        else
        {
          v109 = &v158;
        }
        v166 = &v154;
        v112 = sub_2CE6580(&v157, (__int64)v109, &v166);
        v38 = v142;
        v109 = (__int64 *)v112;
LABEL_202:
        v39 = *((_DWORD *)v109 + 10);
LABEL_167:
        if ( (unsigned int)(v39 - 4) > 1 )
          goto LABEL_59;
        if ( v39 == 5 )
          goto LABEL_220;
        v166 = &v168;
        v154 = 46;
        v99 = sub_22409D0((__int64)&v166, &v154, 0);
        v166 = (unsigned __int64 *)v99;
        v168 = v154;
        *(__m128i *)v99 = _mm_load_si128((const __m128i *)&xmmword_42DFCC0);
        v100 = _mm_load_si128((const __m128i *)&xmmword_444AF60);
        qmemcpy((void *)(v99 + 32), "onstant memory", 14);
        *(__m128i *)(v99 + 16) = v100;
        v167 = v154;
        *((_BYTE *)v166 + v154) = 0;
        sub_2CDF8F0((__int64)v33, (__int64)&v166);
        v101 = v166;
        if ( v166 != &v168 )
          goto LABEL_170;
LABEL_54:
        if ( v146 == ++v22 )
          goto LABEL_68;
      }
      if ( v34 != 66 )
      {
        v113 = *((_QWORD *)v33 - 4);
        if ( !v113 || *(_BYTE *)v113 || *(_QWORD *)(v113 + 24) != *((_QWORD *)v33 + 10) )
LABEL_274:
          BUG();
        v114 = *(_DWORD *)(v113 + 36);
        v149 = 0;
        v144 = v114;
        sub_2CE0320((__int64)a1, v114, &v149);
        v115 = *(_QWORD *)&v33[32 * (v149 - (unsigned __int64)(*((_DWORD *)v33 + 1) & 0x7FFFFFF))];
        LODWORD(v150) = 0;
        v130 = (unsigned __int8 *)v115;
        v154 = v115;
        if ( (unsigned __int8)sub_2CE0930(a1, (__int64)v33, v115, (unsigned int *)&v150) )
        {
          v134 = (int)v150;
          goto LABEL_210;
        }
        v123 = v159;
        if ( v159 )
        {
          v124 = &v158;
          do
          {
            if ( v123[4] < v154 )
            {
              v123 = (__int64 *)v123[3];
            }
            else
            {
              v124 = v123;
              v123 = (__int64 *)v123[2];
            }
          }
          while ( v123 );
          if ( v124 == &v158 || v124[4] > v154 )
          {
LABEL_248:
            v166 = &v154;
            v124 = (__int64 *)sub_2CE6580(&v157, (__int64)v124, &v166);
          }
          v134 = *((_DWORD *)v124 + 10);
LABEL_210:
          if ( sub_CEA260(v144) && (unsigned int)(v134 - 3) <= 2 )
          {
            v126 = ": Warning: Cannot do vector atomic on local memory";
            if ( v134 != 5 )
            {
              v126 = ": Warning: Cannot do vector atomic on constant memory";
              if ( v134 != 4 )
                v126 = ": Warning: Cannot to vector atomic on shared memory";
            }
          }
          else if ( (unsigned __int8)sub_CEA1F0(v144) )
          {
            if ( (unsigned int)(v134 - 4) > 1 )
            {
              sub_2CE02B0((__int64)a1, v144);
LABEL_215:
              v116 = sub_2CE9A90(a1, a2, v130, (__int64)v33, &v163, v134, 0);
              sub_AC2B30((__int64)&v33[32 * (v149 - (unsigned __int64)(*((_DWORD *)v33 + 1) & 0x7FFFFFF))], v116);
              v166 = &v168;
              v167 = 0x300000000LL;
              if ( (unsigned __int8)sub_CEA1F0(v144) )
              {
                v117 = v149 - (unsigned __int64)(*((_DWORD *)v33 + 1) & 0x7FFFFFF);
                goto LABEL_217;
              }
              v125 = *((_DWORD *)v33 + 1) & 0x7FFFFFF;
              if ( v144 == 243 )
              {
                v127 = *(_QWORD *)(*(_QWORD *)&v33[-32 * v125] + 8LL);
              }
              else
              {
                if ( v144 != 238 && v144 != 241 )
                {
                  v117 = v149 - v125;
LABEL_217:
                  sub_94F8E0((__int64)&v166, *(_QWORD *)(*(_QWORD *)&v33[32 * v117] + 8LL));
LABEL_218:
                  v118 = sub_B6E160(*(__int64 **)(a2 + 40), v144, (__int64)v166, (unsigned int)v167);
                  *((_QWORD *)v33 + 10) = *(_QWORD *)(v118 + 24);
                  sub_AC2B30((__int64)(v33 - 32), v118);
                  if ( v166 != &v168 )
                    _libc_free((unsigned __int64)v166);
                  goto LABEL_54;
                }
                sub_94F8E0((__int64)&v166, *(_QWORD *)(*(_QWORD *)&v33[-32 * v125] + 8LL));
                v127 = *(_QWORD *)(*(_QWORD *)&v33[32 * (1LL - (*((_DWORD *)v33 + 1) & 0x7FFFFFF))] + 8LL);
              }
              sub_94F8E0((__int64)&v166, v127);
              sub_94F8E0(
                (__int64)&v166,
                *(_QWORD *)(*(_QWORD *)&v33[32 * (2LL - (*((_DWORD *)v33 + 1) & 0x7FFFFFF))] + 8LL));
              goto LABEL_218;
            }
            v126 = ": Warning: Cannot do atomic on local memory";
            if ( v134 != 5 )
              v126 = ": Warning: Cannot do atomic on constant memory";
          }
          else
          {
            if ( !sub_2CE02B0((__int64)a1, v144) || (unsigned int)(v134 - 4) > 1 )
              goto LABEL_215;
            v126 = ": Warning: cannot perform wmma load or store on local memory";
            if ( v134 != 5 )
              v126 = ": Warning: cannot perform wmma load or store on constant memory";
          }
          sub_2CDD970((__int64 *)&v166, v126);
          sub_2CDF8F0((__int64)v33, (__int64)&v166);
          sub_2240A30((unsigned __int64 *)&v166);
          goto LABEL_54;
        }
        v124 = &v158;
        goto LABEL_248;
      }
      v104 = *((_QWORD *)v33 - 8);
      v105 = *v22;
      LODWORD(v150) = 0;
      v154 = v104;
      v143 = (unsigned __int8 *)v104;
      v106 = sub_2CE0930(a1, (__int64)v105, v104, (unsigned int *)&v150);
      v38 = v143;
      if ( v106 )
      {
        v39 = (int)v150;
        goto LABEL_190;
      }
      v120 = v159;
      if ( v159 )
      {
        v121 = &v158;
        do
        {
          if ( v120[4] < v154 )
          {
            v120 = (__int64 *)v120[3];
          }
          else
          {
            v121 = v120;
            v120 = (__int64 *)v120[2];
          }
        }
        while ( v120 );
        if ( v121 != &v158 && v121[4] <= v154 )
          goto LABEL_239;
      }
      else
      {
        v121 = &v158;
      }
      v166 = &v154;
      v122 = sub_2CE6580(&v157, (__int64)v121, &v166);
      v38 = v143;
      v121 = (__int64 *)v122;
LABEL_239:
      v39 = *((_DWORD *)v121 + 10);
LABEL_190:
      if ( (unsigned int)(v39 - 4) <= 1 )
      {
        v107 = ": Warning: Cannot do atomic on constant memory";
        if ( v39 == 5 )
LABEL_220:
          v107 = ": Warning: Cannot do atomic on local memory";
        sub_2CDD970((__int64 *)&v166, v107);
        sub_2CDF8F0((__int64)v33, (__int64)&v166);
        v101 = v166;
        if ( v166 == &v168 )
          goto LABEL_54;
LABEL_170:
        j_j___libc_free_0((unsigned __int64)v101);
        goto LABEL_54;
      }
LABEL_59:
      v40 = sub_2CE9A90(a1, a2, v38, (__int64)v33, &v163, v39, 0);
      if ( (v33[7] & 0x40) != 0 )
        v41 = (char *)*((_QWORD *)v33 - 1);
      else
        v41 = &v33[-32 * (*((_DWORD *)v33 + 1) & 0x7FFFFFF)];
      if ( *(_QWORD *)v41 )
      {
        v42 = *((_QWORD *)v41 + 1);
        **((_QWORD **)v41 + 2) = v42;
        if ( v42 )
          *(_QWORD *)(v42 + 16) = *((_QWORD *)v41 + 2);
      }
      *(_QWORD *)v41 = v40;
      if ( !v40 )
        goto LABEL_54;
      v43 = *(_QWORD *)(v40 + 16);
      *((_QWORD *)v41 + 1) = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = v41 + 8;
      *((_QWORD *)v41 + 2) = v40 + 16;
      ++v22;
      *(_QWORD *)(v40 + 16) = v41;
      if ( v146 == v22 )
      {
LABEL_68:
        v133 = 1;
        goto LABEL_69;
      }
    }
  }
  v133 = 0;
LABEL_69:
  v44 = (__int64 *)a1[6];
  v136 = (__int64 *)a1[7];
  if ( v136 != v44 )
  {
    v45 = a2;
    do
    {
      v46 = *v44;
      v47 = *(_QWORD *)(*v44 + 32 * (1LL - (*(_DWORD *)(*v44 + 4) & 0x7FFFFFF)));
      v138 = (unsigned __int8 *)v47;
      v48 = *(_DWORD *)(*(_QWORD *)(v47 + 8) + 8LL);
      LODWORD(v167) = 0;
      v168 = 0;
      v171 = 0;
      LODWORD(v150) = v48 >> 8;
      v169 = &v167;
      v170 = &v167;
      if ( (unsigned int)sub_2CE8530((__int64)a1, v45, (unsigned __int8 *)v47, &v157, &v166, (int *)&v150) == 1 )
      {
        v49 = sub_2CE9A90(a1, v45, v138, v46, &v163, (int)v150, 0);
        v50 = v46 + 32 * (1LL - (*(_DWORD *)(v46 + 4) & 0x7FFFFFF));
        if ( *(_QWORD *)v50 )
        {
          v51 = *(_QWORD *)(v50 + 8);
          **(_QWORD **)(v50 + 16) = v51;
          if ( v51 )
            *(_QWORD *)(v51 + 16) = *(_QWORD *)(v50 + 16);
        }
        *(_QWORD *)v50 = v49;
        if ( v49 )
        {
          v52 = *(_QWORD *)(v49 + 16);
          *(_QWORD *)(v50 + 8) = v52;
          if ( v52 )
            *(_QWORD *)(v52 + 16) = v50 + 8;
          *(_QWORD *)(v50 + 16) = v49 + 16;
          *(_QWORD *)(v49 + 16) = v50;
        }
        v154 = (unsigned __int64)v156;
        v155 = 0x300000000LL;
        v53 = *(_QWORD *)(v46 - 32);
        if ( !v53 || *(_BYTE *)v53 || *(_QWORD *)(v53 + 24) != *(_QWORD *)(v46 + 80) )
          goto LABEL_274;
        v54 = *(_DWORD *)(v53 + 36);
        v55 = *(__int64 **)(v45 + 40);
        v56 = *(_QWORD *)(*(_QWORD *)(v46 - 32LL * (*(_DWORD *)(v46 + 4) & 0x7FFFFFF)) + 8LL);
        LODWORD(v155) = 1;
        v156[0] = v56;
        v57 = *(_QWORD *)(*(_QWORD *)(v46 + 32 * (1LL - (*(_DWORD *)(v46 + 4) & 0x7FFFFFF))) + 8LL);
        LODWORD(v155) = 2;
        v156[1] = v57;
        v58 = *(_QWORD *)(*(_QWORD *)(v46 + 32 * (2LL - (*(_DWORD *)(v46 + 4) & 0x7FFFFFF))) + 8LL);
        LODWORD(v155) = 3;
        v156[2] = v58;
        v59 = sub_B6E160(v55, v54, (__int64)v156, 3);
        v60 = *(_QWORD *)(v46 - 32) == 0;
        *(_QWORD *)(v46 + 80) = *(_QWORD *)(v59 + 24);
        if ( !v60 )
        {
          v61 = *(_QWORD *)(v46 - 24);
          **(_QWORD **)(v46 - 16) = v61;
          if ( v61 )
            *(_QWORD *)(v61 + 16) = *(_QWORD *)(v46 - 16);
        }
        *(_QWORD *)(v46 - 32) = v59;
        v62 = *(_QWORD *)(v59 + 16);
        *(_QWORD *)(v46 - 24) = v62;
        if ( v62 )
          *(_QWORD *)(v62 + 16) = v46 - 24;
        *(_QWORD *)(v46 - 16) = v59 + 16;
        *(_QWORD *)(v59 + 16) = v46 - 32;
        if ( (_QWORD *)v154 != v156 )
          _libc_free(v154);
        v133 = 1;
      }
      ++v44;
      sub_2CDF380(v168);
    }
    while ( v136 != v44 );
  }
  sub_2CDE470(v165[0]);
  sub_C7D6A0(0, 0, 8);
  if ( v151 )
    j_j___libc_free_0((unsigned __int64)v151);
  sub_2CDE640((unsigned __int64)v159);
  return v133;
}
