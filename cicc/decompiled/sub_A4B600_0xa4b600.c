// Function: sub_A4B600
// Address: 0xa4b600
//
__int64 __fastcall sub_A4B600(__int64 a1, __int64 a2, int a3, __int64 a4, size_t *a5)
{
  __int64 v5; // r15
  __int64 v7; // rbx
  __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // r10
  unsigned __int8 v13; // dl
  __int64 v14; // r15
  unsigned int v15; // r14d
  __int64 v16; // rdx
  unsigned __int8 v17; // al
  int v18; // eax
  __int64 v19; // r8
  char v20; // dl
  unsigned int v21; // ecx
  unsigned int v22; // r10d
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  unsigned __int64 v26; // rdx
  __int64 v27; // rsi
  unsigned __int64 v28; // rdx
  unsigned int v29; // r10d
  unsigned __int64 v30; // rcx
  unsigned __int64 v31; // rax
  __int64 v32; // rdi
  __int64 v33; // r12
  __int64 v34; // rax
  __m128i v35; // xmm0
  __int64 v36; // rax
  int v38; // edx
  __int64 v39; // r8
  int v40; // ecx
  char v41; // dl
  unsigned __int64 v42; // rdx
  int v43; // r14d
  __int64 v44; // rsi
  __int64 v45; // r15
  __int64 v46; // r13
  int v47; // ebx
  __int64 v48; // rax
  __int64 v49; // r9
  char v50; // al
  __int64 v51; // rdx
  __int64 v52; // r12
  __int64 v53; // rax
  __m128i v54; // xmm0
  __int64 v55; // rax
  __int64 v56; // rax
  char v57; // dl
  __int64 v58; // rax
  __int64 v59; // r11
  size_t v60; // r12
  __int64 v61; // rax
  __int64 v62; // r12
  __int64 v63; // r8
  __int64 v64; // rdx
  char v65; // al
  __int64 v66; // rdi
  __int64 v67; // rsi
  int v68; // eax
  unsigned __int64 v69; // rdx
  unsigned int v70; // r10d
  __int64 v71; // r13
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // r12
  __int64 v76; // rax
  __m128i v77; // xmm0
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // r8
  __int64 v81; // rdx
  __int64 v82; // rcx
  char *v83; // r13
  __int64 v84; // rax
  size_t v85; // rax
  size_t v86; // r10
  _QWORD *v87; // rdx
  __int64 v88; // rax
  char *v89; // rdi
  __int64 v90; // rax
  __int64 v91; // r13
  char v92; // al
  __int64 v93; // rdi
  __int64 v94; // r14
  __int64 v95; // rax
  __m128i si128; // xmm0
  __int64 v97; // rax
  __int64 v98; // rcx
  int v99; // edx
  int v100; // r14d
  __int64 v101; // rax
  unsigned int *v102; // rbx
  __int64 v103; // r13
  __int64 v104; // rax
  __int64 v105; // r9
  char v106; // al
  unsigned int v107; // r14d
  int v108; // r13d
  __int64 v109; // r8
  __int64 v110; // rax
  char v111; // al
  int v112; // r14d
  __int64 v113; // rax
  unsigned int *v114; // rbx
  __int64 v115; // rax
  __int64 v116; // r9
  char v117; // al
  unsigned __int64 v118; // rax
  __int64 v119; // rax
  _QWORD *v120; // rdi
  unsigned __int64 v121; // rdi
  __int64 v122; // r12
  __int64 v123; // rax
  __m128i v124; // xmm0
  char *v125; // rax
  unsigned __int64 v126; // rax
  __int64 v127; // rax
  __int64 v128; // r8
  __int64 v129; // rdx
  __int64 v130; // rcx
  char *v131; // r12
  __int64 v132; // rax
  size_t v133; // rax
  size_t v134; // r8
  __int64 v135; // rax
  char *v136; // rdi
  __int64 v137; // rax
  __int64 v138; // rax
  __m128i v139; // xmm0
  __int64 v140; // rax
  __m128i *v141; // rax
  __m128i v142; // xmm0
  __int64 v143; // rax
  __int64 v144; // rax
  __m128i v145; // xmm0
  _QWORD *v146; // rdi
  __int64 v147; // rax
  unsigned __int64 v148; // [rsp+0h] [rbp-180h]
  int v149; // [rsp+Ch] [rbp-174h]
  __int64 v150; // [rsp+10h] [rbp-170h]
  size_t n; // [rsp+20h] [rbp-160h]
  size_t na; // [rsp+20h] [rbp-160h]
  size_t nd; // [rsp+20h] [rbp-160h]
  size_t nb; // [rsp+20h] [rbp-160h]
  size_t nc; // [rsp+20h] [rbp-160h]
  unsigned int v156; // [rsp+28h] [rbp-158h]
  __int64 v157; // [rsp+28h] [rbp-158h]
  int v158; // [rsp+28h] [rbp-158h]
  unsigned int v159; // [rsp+28h] [rbp-158h]
  unsigned int v160; // [rsp+28h] [rbp-158h]
  __int64 v161; // [rsp+28h] [rbp-158h]
  __int64 v163; // [rsp+30h] [rbp-150h]
  size_t v164; // [rsp+30h] [rbp-150h]
  int v165; // [rsp+38h] [rbp-148h]
  __int64 v167; // [rsp+40h] [rbp-140h]
  __int64 v168; // [rsp+48h] [rbp-138h]
  int v169; // [rsp+48h] [rbp-138h]
  __int64 v170; // [rsp+50h] [rbp-130h] BYREF
  __int64 v171; // [rsp+58h] [rbp-128h] BYREF
  __int64 v172; // [rsp+60h] [rbp-120h] BYREF
  char v173; // [rsp+68h] [rbp-118h]
  __int64 v174; // [rsp+70h] [rbp-110h] BYREF
  char v175; // [rsp+78h] [rbp-108h]
  __int64 v176; // [rsp+80h] [rbp-100h] BYREF
  char v177; // [rsp+88h] [rbp-F8h]
  _QWORD v178[2]; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v179; // [rsp+A0h] [rbp-E0h] BYREF
  char *s; // [rsp+B0h] [rbp-D0h]
  __int64 v181; // [rsp+B8h] [rbp-C8h]
  __m128i v182; // [rsp+C0h] [rbp-C0h] BYREF
  _QWORD v183[2]; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v184; // [rsp+E0h] [rbp-A0h] BYREF
  char *v185; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v186; // [rsp+F8h] [rbp-88h]
  __m128i v187; // [rsp+100h] [rbp-80h] BYREF
  __int64 v188; // [rsp+110h] [rbp-70h] BYREF
  size_t v189; // [rsp+118h] [rbp-68h]
  _QWORD v190[2]; // [rsp+120h] [rbp-60h] BYREF
  __int64 v191; // [rsp+130h] [rbp-50h] BYREF
  char *v192; // [rsp+138h] [rbp-48h]
  _QWORD dest[8]; // [rsp+140h] [rbp-40h] BYREF

  v5 = a1;
  v7 = a2;
  if ( a3 == 3 )
  {
    sub_9CE2D0((__int64)&v172, a2, 6, a4);
    v30 = v173 & 1;
    v27 = (unsigned int)(2 * v30);
    v38 = v27 | v173 & 0xFD;
    v173 = (2 * v30) | v173 & 0xFD;
    if ( (_BYTE)v30 )
    {
      v56 = v172;
      v28 = v38 & 0xFFFFFFFD;
      *(_BYTE *)(a1 + 8) |= 3u;
      v173 = v28;
      v172 = 0;
      *(_QWORD *)a1 = v56 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_44;
    }
    v169 = v172;
    sub_9CE2D0((__int64)&v174, v7, 6, v30);
    v40 = v175 & 1;
    v27 = (unsigned int)(2 * v40);
    v41 = (2 * v40) | v175 & 0xFD;
    v175 = v41;
    if ( !(_BYTE)v40 )
    {
      v42 = 8LL * *(_QWORD *)(v7 + 8);
      v43 = v174;
      if ( (unsigned int)v174 >= v42 )
      {
        v94 = sub_2241E50(&v174, v27, v42, (unsigned int)v174, v39);
        v191 = (__int64)dest;
        v188 = 21;
        v95 = sub_22409D0(&v191, &v188, 0);
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F23130);
        v191 = v95;
        dest[0] = v188;
        *(_DWORD *)(v95 + 16) = 1818388851;
        *(_BYTE *)(v95 + 20) = 101;
        v27 = (__int64)&v191;
        *(__m128i *)v95 = si128;
        v192 = (char *)v188;
        *(_BYTE *)(v191 + v188) = 0;
        sub_C63F00(&v188, &v191, 84, v94);
        if ( (_QWORD *)v191 != dest )
        {
          v27 = dest[0] + 1LL;
          j_j___libc_free_0(v191, dest[0] + 1LL);
        }
        v97 = v188;
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v97 & 0xFFFFFFFFFFFFFFFELL;
      }
      else
      {
        v28 = (unsigned int)v174 + (unsigned __int64)*(unsigned int *)(a4 + 8);
        v30 = *(unsigned int *)(a4 + 12);
        if ( v28 > v30 )
        {
          v27 = a4 + 16;
          sub_C8D5F0(a4, a4 + 16, v28, 8);
        }
        if ( v43 )
        {
          v44 = a4 + 16;
          v45 = a4;
          v46 = v7;
          v157 = v44;
          v47 = 0;
          while ( 1 )
          {
            v27 = v46;
            sub_A4B2C0((__int64)&v191, v46, 6, v30);
            v28 = (unsigned __int8)v192 & 1;
            v30 = (unsigned int)(2 * v28);
            LOBYTE(v192) = (2 * v28) | (unsigned __int8)v192 & 0xFD;
            if ( (_BYTE)v28 )
              break;
            v48 = *(unsigned int *)(v45 + 8);
            v30 = *(unsigned int *)(v45 + 12);
            v49 = v191;
            if ( v48 + 1 > v30 )
            {
              v27 = v157;
              na = v191;
              sub_C8D5F0(v45, v157, v48 + 1, 8);
              v48 = *(unsigned int *)(v45 + 8);
              v49 = na;
            }
            v28 = *(_QWORD *)v45;
            *(_QWORD *)(*(_QWORD *)v45 + 8 * v48) = v49;
            v50 = (char)v192;
            ++*(_DWORD *)(v45 + 8);
            if ( (v50 & 2) != 0 )
LABEL_105:
              sub_9CDF70(&v191);
            if ( (v50 & 1) != 0 && v191 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v191 + 8LL))(v191);
            if ( v43 == ++v47 )
            {
              v5 = a1;
              goto LABEL_153;
            }
          }
          v5 = a1;
          v118 = v191 & 0xFFFFFFFFFFFFFFFELL;
          *(_BYTE *)(a1 + 8) |= 3u;
          *(_QWORD *)a1 = v118;
        }
        else
        {
LABEL_153:
          *(_BYTE *)(v5 + 8) = *(_BYTE *)(v5 + 8) & 0xFC | 2;
          *(_DWORD *)v5 = v169;
        }
      }
      goto LABEL_94;
    }
    v78 = v174;
    v174 = 0;
    v175 = v41 & 0xFD;
    v170 = v78 | 1;
    sub_C64870(v178, &v170);
    v79 = sub_2241130(v178, 0, 0, "Failed to read size: ", 21);
    s = (char *)&v182;
    v81 = v79 + 16;
    if ( *(_QWORD *)v79 == v79 + 16 )
    {
      v182 = _mm_loadu_si128((const __m128i *)(v79 + 16));
    }
    else
    {
      s = *(char **)v79;
      v182.m128i_i64[0] = *(_QWORD *)(v79 + 16);
    }
    v82 = *(_QWORD *)(v79 + 8);
    *(_BYTE *)(v79 + 16) = 0;
    v181 = v82;
    *(_QWORD *)v79 = v81;
    v83 = s;
    *(_QWORD *)(v79 + 8) = 0;
    v84 = sub_2241E50(v178, 0, v81, v82, v80);
    v188 = (__int64)v190;
    v163 = v84;
    if ( !v83 )
      goto LABEL_198;
    v85 = strlen(v83);
    v191 = v85;
    v86 = v85;
    if ( v85 > 0xF )
    {
      nd = v85;
      v119 = sub_22409D0(&v188, &v191, 0);
      v86 = nd;
      v188 = v119;
      v120 = (_QWORD *)v119;
      v190[0] = v191;
    }
    else
    {
      if ( v85 == 1 )
      {
        LOBYTE(v190[0]) = *v83;
        v87 = v190;
LABEL_86:
        v189 = v85;
        *((_BYTE *)v87 + v85) = 0;
        v27 = (__int64)&v188;
        sub_C63F00(&v191, &v188, 84, v163);
        if ( (_QWORD *)v188 != v190 )
        {
          v27 = v190[0] + 1LL;
          j_j___libc_free_0(v188, v190[0] + 1LL);
        }
        v88 = v191;
        *(_BYTE *)(v5 + 8) |= 3u;
        v89 = s;
        *(_QWORD *)v5 = v88 & 0xFFFFFFFFFFFFFFFELL;
        if ( v89 != (char *)&v182 )
        {
          v27 = v182.m128i_i64[0] + 1;
          j_j___libc_free_0(v89, v182.m128i_i64[0] + 1);
        }
        if ( (__int64 *)v178[0] != &v179 )
        {
          v27 = v179 + 1;
          j_j___libc_free_0(v178[0], v179 + 1);
        }
        if ( (v170 & 1) != 0 || (v170 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_C63C30(&v170);
LABEL_94:
        if ( (v175 & 2) != 0 )
          sub_9CE230(&v174);
        if ( (v175 & 1) != 0 && v174 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v174 + 8LL))(v174);
        if ( (v173 & 2) != 0 )
          sub_9CE230(&v172);
        if ( (v173 & 1) == 0 )
          return v5;
LABEL_44:
        v32 = v172;
        if ( !v172 )
          return v5;
        goto LABEL_19;
      }
      if ( !v85 )
      {
        v87 = v190;
        goto LABEL_86;
      }
      v120 = v190;
    }
    memcpy(v120, v83, v86);
    v85 = v191;
    v87 = (_QWORD *)v188;
    goto LABEL_86;
  }
  v8 = *(_QWORD *)(a2 + 40);
  v9 = (unsigned int)(a3 - 4);
  v10 = (*(_QWORD *)(v7 + 48) - v8) >> 4;
  if ( v9 >= v10 )
  {
    v33 = sub_2241E50(a1, v8, v10, v9, a5);
    v191 = (__int64)dest;
    v188 = 21;
    v34 = sub_22409D0(&v191, &v188, 0);
    v35 = _mm_load_si128((const __m128i *)&xmmword_3F23140);
    v191 = v34;
    dest[0] = v188;
    *(_DWORD *)(v34 + 16) = 1700949365;
    *(_BYTE *)(v34 + 20) = 114;
    *(__m128i *)v34 = v35;
    v192 = (char *)v188;
    *(_BYTE *)(v191 + v188) = 0;
    sub_C63F00(&v188, &v191, 84, v33);
    if ( (_QWORD *)v191 != dest )
      j_j___libc_free_0(v191, dest[0] + 1LL);
    v36 = v188;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v36 & 0xFFFFFFFFFFFFFFFELL;
    return v5;
  }
  v11 = 16 * v9;
  v12 = **(_QWORD **)(v8 + v11);
  v168 = *(_QWORD *)(v8 + v11);
  v13 = *(_BYTE *)(v12 + 8);
  if ( (v13 & 1) != 0 )
  {
    v149 = *(_DWORD *)v12;
  }
  else
  {
    v51 = (unsigned __int8)(((v13 >> 1) & 7) - 3) & 0xFD;
    if ( !(_DWORD)v51 )
    {
      v52 = sub_2241E50(a1, v8, v51, v11, a5);
      v191 = (__int64)dest;
      v188 = 43;
      v53 = sub_22409D0(&v191, &v188, 0);
      v191 = v53;
      dest[0] = v188;
      *(__m128i *)v53 = _mm_load_si128((const __m128i *)&xmmword_3F23150);
      v54 = _mm_load_si128((const __m128i *)&xmmword_3F23160);
      qmemcpy((void *)(v53 + 32), "y or a Blob", 11);
      *(__m128i *)(v53 + 16) = v54;
      v192 = (char *)v188;
      *(_BYTE *)(v191 + v188) = 0;
      sub_C63F00(&v188, &v191, 84, v52);
      if ( (_QWORD *)v191 != dest )
        j_j___libc_free_0(v191, dest[0] + 1LL);
      v55 = v188;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v55 & 0xFFFFFFFFFFFFFFFELL;
      return v5;
    }
    sub_A4B540((__int64)&v191, v7, v12, v11);
    if ( ((unsigned __int8)v192 & 1) != 0 )
      goto LABEL_48;
    v149 = v191;
  }
  v165 = *(_DWORD *)(v168 + 8);
  if ( v165 == 1 )
  {
LABEL_59:
    *(_BYTE *)(v5 + 8) = *(_BYTE *)(v5 + 8) & 0xFC | 2;
    *(_DWORD *)v5 = v149;
    return v5;
  }
  v14 = a4;
  v15 = 1;
  v150 = a4 + 16;
  while ( 1 )
  {
    v16 = *(_QWORD *)v168 + 16LL * v15;
    v17 = *(_BYTE *)(v16 + 8);
    if ( (v17 & 1) != 0 )
    {
      v61 = *(unsigned int *)(v14 + 8);
      v11 = *(unsigned int *)(v14 + 12);
      v62 = *(_QWORD *)v16;
      if ( v61 + 1 > v11 )
      {
        sub_C8D5F0(v14, v150, v61 + 1, 8);
        v61 = *(unsigned int *)(v14 + 8);
      }
      ++v15;
      *(_QWORD *)(*(_QWORD *)v14 + 8 * v61) = v62;
      ++*(_DWORD *)(v14 + 8);
      goto LABEL_57;
    }
    v18 = (v17 >> 1) & 7;
    if ( v18 == 3 )
    {
      sub_9CE2D0((__int64)&v176, v7, 6, v11);
      v64 = v177 & 1;
      v65 = (2 * v64) | v177 & 0xFD;
      v177 = v65;
      if ( !(_BYTE)v64 )
      {
        v66 = *(_QWORD *)(v7 + 8);
        v67 = (unsigned int)v176;
        v68 = v176;
        if ( (unsigned int)v176 >= (unsigned __int64)(8 * v66) )
        {
          v5 = a1;
          v137 = sub_2241E50(v66, (unsigned int)v176, v64, 8 * v66, v63);
          v191 = (__int64)dest;
          v75 = v137;
          v188 = 21;
          v138 = sub_22409D0(&v191, &v188, 0);
          v139 = _mm_load_si128((const __m128i *)&xmmword_3F23130);
          v191 = v138;
          dest[0] = v188;
          *(_DWORD *)(v138 + 16) = 1818388851;
          *(_BYTE *)(v138 + 20) = 101;
          *(__m128i *)v138 = v139;
        }
        else
        {
          v11 = *(unsigned int *)(v14 + 12);
          v69 = (unsigned int)v176 + (unsigned __int64)*(unsigned int *)(v14 + 8);
          if ( v69 > v11 )
          {
            v158 = v176;
            v67 = v150;
            v66 = v14;
            sub_C8D5F0(v14, v150, v69, 8);
            v68 = v158;
          }
          v70 = v15 + 2;
          if ( v15 + 2 == v165 )
          {
            v71 = *(_QWORD *)v168 + 16LL * (v15 + 1);
            v72 = *(unsigned __int8 *)(v71 + 8);
            if ( (v72 & 1) == 0 )
            {
              v73 = ((unsigned __int8)v72 >> 1) & 7;
              switch ( (_BYTE)v73 )
              {
                case 2:
                  if ( v68 )
                  {
                    v159 = v15 + 2;
                    v112 = v68;
                    v113 = v7;
                    v114 = (unsigned int *)v71;
                    v103 = v113;
                    do
                    {
                      v27 = v103;
                      sub_A4B2C0((__int64)&v191, v103, *v114, v11);
                      v28 = (unsigned __int8)v192 & 1;
                      v30 = (unsigned int)(2 * v28);
                      LOBYTE(v192) = (2 * v28) | (unsigned __int8)v192 & 0xFD;
                      if ( (_BYTE)v28 )
                        goto LABEL_174;
                      v115 = *(unsigned int *)(v14 + 8);
                      v11 = *(unsigned int *)(v14 + 12);
                      v116 = v191;
                      if ( v115 + 1 > v11 )
                      {
                        nc = v191;
                        sub_C8D5F0(v14, v150, v115 + 1, 8);
                        v115 = *(unsigned int *)(v14 + 8);
                        v116 = nc;
                      }
                      *(_QWORD *)(*(_QWORD *)v14 + 8 * v115) = v116;
                      v117 = (char)v192;
                      ++*(_DWORD *)(v14 + 8);
                      if ( (v117 & 2) != 0 )
                        goto LABEL_105;
                      if ( (v117 & 1) != 0 && v191 )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v191 + 8LL))(v191);
                      --v112;
                    }
                    while ( v112 );
LABEL_130:
                    v70 = v159;
                    v7 = v103;
                  }
                  break;
                case 4:
                  if ( v68 )
                  {
                    v107 = v15 + 2;
                    v108 = v68;
                    do
                    {
                      v27 = v7;
                      sub_9C66D0((__int64)&v191, v7, 6, v11);
                      v28 = (unsigned __int8)v192 & 1;
                      v30 = (unsigned int)(2 * v28);
                      LOBYTE(v192) = (2 * v28) | (unsigned __int8)v192 & 0xFD;
                      if ( (_BYTE)v28 )
                        goto LABEL_174;
                      v11 = *(unsigned int *)(v14 + 12);
                      v109 = aAbcdefghijklmn[(unsigned int)v191];
                      v110 = *(unsigned int *)(v14 + 8);
                      if ( v110 + 1 > v11 )
                      {
                        v161 = aAbcdefghijklmn[(unsigned int)v191];
                        sub_C8D5F0(v14, v150, v110 + 1, 8);
                        v110 = *(unsigned int *)(v14 + 8);
                        v109 = v161;
                      }
                      *(_QWORD *)(*(_QWORD *)v14 + 8 * v110) = v109;
                      v111 = (char)v192;
                      ++*(_DWORD *)(v14 + 8);
                      if ( (v111 & 2) != 0 )
                        goto LABEL_105;
                      if ( (v111 & 1) != 0 && v191 )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v191 + 8LL))(v191);
                      --v108;
                    }
                    while ( v108 );
                    v70 = v107;
                  }
                  break;
                case 1:
                  if ( v68 )
                  {
                    v159 = v15 + 2;
                    v100 = v68;
                    v101 = v7;
                    v102 = (unsigned int *)v71;
                    v103 = v101;
                    while ( 1 )
                    {
                      v27 = v103;
                      sub_9C66D0((__int64)&v191, v103, *v102, v11);
                      v28 = (unsigned __int8)v192 & 1;
                      v30 = (unsigned int)(2 * v28);
                      LOBYTE(v192) = (2 * v28) | (unsigned __int8)v192 & 0xFD;
                      if ( (_BYTE)v28 )
                        break;
                      v104 = *(unsigned int *)(v14 + 8);
                      v11 = *(unsigned int *)(v14 + 12);
                      v105 = v191;
                      if ( v104 + 1 > v11 )
                      {
                        nb = v191;
                        sub_C8D5F0(v14, v150, v104 + 1, 8);
                        v104 = *(unsigned int *)(v14 + 8);
                        v105 = nb;
                      }
                      *(_QWORD *)(*(_QWORD *)v14 + 8 * v104) = v105;
                      v106 = (char)v192;
                      ++*(_DWORD *)(v14 + 8);
                      if ( (v106 & 2) != 0 )
                        goto LABEL_105;
                      if ( (v106 & 1) != 0 && v191 )
                        (*(void (__fastcall **)(__int64))(*(_QWORD *)v191 + 8LL))(v191);
                      if ( !--v100 )
                        goto LABEL_130;
                    }
LABEL_174:
                    v5 = a1;
                    v126 = v191 & 0xFFFFFFFFFFFFFFFELL;
                    *(_BYTE *)(a1 + 8) |= 3u;
                    *(_QWORD *)a1 = v126;
LABEL_76:
                    if ( (v177 & 2) != 0 )
LABEL_161:
                      sub_9CE230(&v176);
                    if ( (v177 & 1) != 0 )
                    {
                      v32 = v176;
                      if ( v176 )
                        goto LABEL_19;
                    }
                    return v5;
                  }
                  break;
                default:
                  v5 = a1;
                  v74 = sub_2241E50(v66, v168, v73, v11, v63);
                  v191 = (__int64)dest;
                  v75 = v74;
                  v188 = 46;
                  v76 = sub_22409D0(&v191, &v188, 0);
                  v191 = v76;
                  dest[0] = v188;
                  *(__m128i *)v76 = _mm_load_si128((const __m128i *)&xmmword_3F23180);
                  v77 = _mm_load_si128((const __m128i *)&xmmword_3F231B0);
                  qmemcpy((void *)(v76 + 32), "rray or a Blob", 14);
                  *(__m128i *)(v76 + 16) = v77;
                  goto LABEL_73;
              }
              if ( (v177 & 2) != 0 )
                goto LABEL_161;
              if ( (v177 & 1) != 0 && v176 )
              {
                v160 = v70;
                (*(void (__fastcall **)(__int64))(*(_QWORD *)v176 + 8LL))(v176);
                v70 = v160;
              }
              v15 = v70;
              goto LABEL_57;
            }
            v5 = a1;
            v143 = sub_2241E50(v66, v168, v72, v11, v63);
            v191 = (__int64)dest;
            v75 = v143;
            v188 = 50;
            v144 = sub_22409D0(&v191, &v188, 0);
            v191 = v144;
            dest[0] = v188;
            *(__m128i *)v144 = _mm_load_si128((const __m128i *)&xmmword_3F23180);
            v145 = _mm_load_si128((const __m128i *)&xmmword_3F23190);
            *(_WORD *)(v144 + 48) = 25968;
            *(__m128i *)(v144 + 16) = v145;
            *(__m128i *)(v144 + 32) = _mm_load_si128((const __m128i *)&xmmword_3F231A0);
          }
          else
          {
            v5 = a1;
            v140 = sub_2241E50(v66, v67, v69, v11, v63);
            v191 = (__int64)dest;
            v75 = v140;
            v188 = 27;
            v141 = (__m128i *)sub_22409D0(&v191, &v188, 0);
            v142 = _mm_load_si128((const __m128i *)&xmmword_3F23170);
            v191 = (__int64)v141;
            dest[0] = v188;
            qmemcpy(&v141[1], "ond to last", 11);
            *v141 = v142;
          }
        }
LABEL_73:
        v27 = (__int64)&v191;
        v192 = (char *)v188;
        *(_BYTE *)(v191 + v188) = 0;
        sub_C63F00(&v188, &v191, 84, v75);
        if ( (_QWORD *)v191 != dest )
        {
          v27 = dest[0] + 1LL;
          j_j___libc_free_0(v191, dest[0] + 1LL);
        }
        *(_BYTE *)(v5 + 8) |= 3u;
        *(_QWORD *)v5 = v188 & 0xFFFFFFFFFFFFFFFELL;
        goto LABEL_76;
      }
      v5 = a1;
      v177 = v65 & 0xFD;
      v171 = v176 | 1;
      v176 = 0;
      sub_C64870(v183, &v171);
      v127 = sub_2241130(v183, 0, 0, "Failed to read size: ", 21);
      v185 = (char *)&v187;
      v129 = v127 + 16;
      if ( *(_QWORD *)v127 == v127 + 16 )
      {
        v187 = _mm_loadu_si128((const __m128i *)(v127 + 16));
      }
      else
      {
        v185 = *(char **)v127;
        v187.m128i_i64[0] = *(_QWORD *)(v127 + 16);
      }
      v186 = *(_QWORD *)(v127 + 8);
      v130 = v186;
      *(_QWORD *)v127 = v129;
      *(_QWORD *)(v127 + 8) = 0;
      *(_BYTE *)(v127 + 16) = 0;
      v131 = v185;
      v132 = sub_2241E50(v183, 0, v129, v130, v128);
      v191 = (__int64)dest;
      v167 = v132;
      if ( v131 )
      {
        v133 = strlen(v131);
        v188 = v133;
        v134 = v133;
        if ( v133 > 0xF )
        {
          v164 = v133;
          v147 = sub_22409D0(&v191, &v188, 0);
          v134 = v164;
          v191 = v147;
          v146 = (_QWORD *)v147;
          dest[0] = v188;
        }
        else
        {
          if ( v133 == 1 )
          {
            LOBYTE(dest[0]) = *v131;
LABEL_181:
            v27 = (__int64)&v191;
            v192 = (char *)v188;
            *(_BYTE *)(v191 + v188) = 0;
            sub_C63F00(&v188, &v191, 84, v167);
            if ( (_QWORD *)v191 != dest )
            {
              v27 = dest[0] + 1LL;
              j_j___libc_free_0(v191, dest[0] + 1LL);
            }
            v135 = v188;
            *(_BYTE *)(v5 + 8) |= 3u;
            v136 = v185;
            *(_QWORD *)v5 = v135 & 0xFFFFFFFFFFFFFFFELL;
            if ( v136 != (char *)&v187 )
            {
              v27 = v187.m128i_i64[0] + 1;
              j_j___libc_free_0(v136, v187.m128i_i64[0] + 1);
            }
            if ( (__int64 *)v183[0] != &v184 )
            {
              v27 = v184 + 1;
              j_j___libc_free_0(v183[0], v184 + 1);
            }
            if ( (v171 & 1) != 0 || (v171 & 0xFFFFFFFFFFFFFFFELL) != 0 )
              sub_C63C30(&v171);
            goto LABEL_76;
          }
          if ( !v133 )
            goto LABEL_181;
          v146 = dest;
        }
        memcpy(v146, v131, v134);
        goto LABEL_181;
      }
LABEL_198:
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    }
    if ( v18 != 5 )
    {
      sub_A4B540((__int64)&v191, v7, v16, v11);
      v57 = (unsigned __int8)v192 & 1;
      LOBYTE(v192) = (2 * ((unsigned __int8)v192 & 1)) | (unsigned __int8)v192 & 0xFD;
      if ( !v57 )
      {
        v90 = *(unsigned int *)(v14 + 8);
        v11 = *(unsigned int *)(v14 + 12);
        v91 = v191;
        if ( v90 + 1 > v11 )
        {
          sub_C8D5F0(v14, v150, v90 + 1, 8);
          v90 = *(unsigned int *)(v14 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v14 + 8 * v90) = v91;
        v92 = (char)v192;
        ++*(_DWORD *)(v14 + 8);
        if ( (v92 & 2) != 0 )
          goto LABEL_105;
        if ( (v92 & 1) == 0 )
          goto LABEL_56;
        v93 = v191;
        if ( !v191 )
          goto LABEL_56;
        goto LABEL_103;
      }
      v5 = a1;
LABEL_48:
      v58 = v191;
      *(_BYTE *)(v5 + 8) |= 3u;
      *(_QWORD *)v5 = v58 & 0xFFFFFFFFFFFFFFFELL;
      return v5;
    }
    sub_9CE2D0((__int64)&v188, v7, 6, v11);
    v20 = v189 & 1;
    LOBYTE(v189) = (2 * (v189 & 1)) | v189 & 0xFD;
    if ( v20 )
    {
      v5 = a1;
      v121 = v188 & 0xFFFFFFFFFFFFFFFELL;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v121;
      return v5;
    }
    v21 = *(_DWORD *)(v7 + 32);
    v22 = v188;
    if ( v21 > 0x1F )
    {
      *(_DWORD *)(v7 + 32) = 32;
      v23 = 32;
      *(_QWORD *)(v7 + 24) >>= (unsigned __int8)v21 - 32;
    }
    else
    {
      *(_DWORD *)(v7 + 32) = 0;
      v23 = 0;
    }
    v24 = *(_QWORD *)(v7 + 16);
    v25 = 8 * v24 - v23;
    v26 = v25 + 32LL * ((v22 != 0) + ((v22 - (v22 != 0)) >> 2));
    if ( v26 >> 3 > *(_QWORD *)(v7 + 8) )
      break;
    v27 = v7;
    n = v25;
    v156 = v22;
    sub_9CDFE0(&v191, v7, v26, v25);
    v29 = v156;
    v30 = n;
    v31 = v191 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v191 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v5 = a1;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v31;
      goto LABEL_16;
    }
    v11 = n >> 3;
    v59 = v156;
    v60 = (n >> 3) + *(_QWORD *)v7;
    if ( a5 )
    {
      *a5 = v60;
      a5[1] = v156;
    }
    else
    {
      v98 = *(unsigned int *)(v14 + 8);
      v99 = *(_DWORD *)(v14 + 8);
      if ( (unsigned __int64)v156 + v98 > *(unsigned int *)(v14 + 12) )
      {
        v148 = v191 & 0xFFFFFFFFFFFFFFFELL;
        sub_C8D5F0(v14, v150, v156 + v98, 8);
        v98 = *(unsigned int *)(v14 + 8);
        v31 = v148;
        v29 = v156;
        v59 = v156;
        v99 = *(_DWORD *)(v14 + 8);
      }
      v11 = *(_QWORD *)v14 + 8 * v98;
      if ( v29 )
      {
        do
        {
          *(_QWORD *)(v11 + 8 * v31) = *(unsigned __int8 *)(v60 + v31);
          ++v31;
        }
        while ( v59 != v31 );
        v99 = *(_DWORD *)(v14 + 8);
      }
      *(_DWORD *)(v14 + 8) = v99 + v29;
    }
    if ( (v189 & 2) != 0 )
      goto LABEL_106;
    if ( (v189 & 1) == 0 )
      goto LABEL_56;
    v93 = v188;
    if ( !v188 )
      goto LABEL_56;
LABEL_103:
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v93 + 8LL))(v93);
LABEL_56:
    ++v15;
LABEL_57:
    if ( v165 == v15 )
    {
      v5 = a1;
      goto LABEL_59;
    }
  }
  v5 = a1;
  v122 = sub_2241E50(&v188, v24, v26, v25, v19);
  v191 = (__int64)dest;
  v185 = (char *)18;
  v123 = sub_22409D0(&v191, &v185, 0);
  v27 = (__int64)&v191;
  v124 = _mm_load_si128((const __m128i *)&xmmword_3F231C0);
  v191 = v123;
  dest[0] = v185;
  *(_WORD *)(v123 + 16) = 28271;
  *(__m128i *)v123 = v124;
  v192 = v185;
  v185[v191] = 0;
  sub_C63F00(&v185, &v191, 84, v122);
  if ( (_QWORD *)v191 != dest )
  {
    v27 = dest[0] + 1LL;
    j_j___libc_free_0(v191, dest[0] + 1LL);
  }
  v125 = v185;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = (unsigned __int64)v125 & 0xFFFFFFFFFFFFFFFELL;
LABEL_16:
  if ( (v189 & 2) != 0 )
LABEL_106:
    sub_9CE230(&v188);
  if ( (v189 & 1) != 0 )
  {
    v32 = v188;
    if ( v188 )
LABEL_19:
      (*(void (__fastcall **)(__int64, __int64, unsigned __int64, unsigned __int64))(*(_QWORD *)v32 + 8LL))(
        v32,
        v27,
        v28,
        v30);
  }
  return v5;
}
