// Function: sub_37D6250
// Address: 0x37d6250
//
void __fastcall sub_37D6250(__int64 a1, __int64 a2, _QWORD ***a3, _QWORD *a4, _QWORD *a5, __int64 a6)
{
  _BYTE *v9; // rsi
  _BYTE *v10; // rax
  _BYTE *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 i; // rax
  int v25; // edi
  __int64 v26; // rcx
  __int64 v27; // r9
  _BYTE *v28; // rax
  _DWORD *v29; // rdi
  _QWORD *v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rdx
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 *v36; // rax
  _BYTE *v37; // rcx
  _BYTE *v38; // rcx
  _BYTE *v39; // rcx
  unsigned int v40; // eax
  char v41; // dl
  __int64 v42; // rdi
  _QWORD *v43; // r8
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // esi
  __int64 v48; // rax
  char v49; // di
  unsigned int *v50; // r15
  unsigned int *v51; // rdx
  unsigned int *v52; // r13
  unsigned int *v53; // r8
  __int64 v54; // rcx
  __int64 v55; // rcx
  unsigned int *v56; // r8
  __int64 v57; // rax
  __int64 v58; // r9
  __int64 v59; // r12
  unsigned int *v60; // rbx
  unsigned __int64 v61; // r10
  __int64 v62; // r11
  __int64 v63; // r8
  _QWORD *v64; // rax
  unsigned int v65; // ecx
  unsigned __int64 v66; // rdx
  _BYTE *v67; // rax
  __int64 v68; // rdi
  __int64 v69; // rax
  __int64 v70; // r8
  char v71; // si
  __int64 v72; // rax
  _QWORD *v73; // rdx
  __int64 v74; // rcx
  bool v75; // di
  __int64 *v76; // r13
  __int64 *v77; // r12
  __int64 v78; // rax
  unsigned int v79; // esi
  __int64 v80; // r8
  int v81; // r11d
  __int64 v82; // rdi
  unsigned int v83; // ecx
  __int64 v84; // rdx
  __int64 v85; // r10
  unsigned int v86; // r15d
  int v87; // r11d
  __int64 v88; // rdx
  unsigned int v89; // edi
  __int64 v90; // rax
  __int64 v91; // rcx
  unsigned int v92; // eax
  _QWORD *v93; // rax
  _QWORD *v94; // rax
  char v95; // dl
  __int64 *v96; // rax
  _BYTE *v97; // rsi
  _BYTE *v98; // rsi
  _BYTE *v99; // rcx
  unsigned int v100; // edi
  __int64 v101; // rsi
  __int64 v102; // r8
  __int64 v103; // rdx
  unsigned int *v104; // rsi
  unsigned int *v105; // r8
  char v106; // dl
  __int64 *v107; // rax
  _BYTE *v108; // rsi
  _BYTE *v109; // rsi
  __int64 v110; // r8
  __int64 v111; // rdx
  __int64 v112; // r13
  __int64 v113; // rcx
  int v114; // ecx
  int v115; // edx
  int v116; // r11d
  int v117; // r11d
  __int64 v118; // r10
  int v119; // eax
  __int64 v120; // rdi
  int v121; // eax
  int v122; // r11d
  int v123; // r11d
  __int64 v124; // rsi
  __int64 v125; // r10
  __int64 v126; // rdi
  int v127; // r15d
  __int64 v128; // r8
  int v129; // r15d
  __int64 v130; // r11
  unsigned int v131; // ecx
  int v132; // r10d
  __int64 v133; // rsi
  int v134; // r15d
  int v135; // r15d
  __int64 v136; // r11
  int v137; // r10d
  unsigned int v138; // ecx
  unsigned int *v139; // rcx
  _QWORD *v140; // [rsp+0h] [rbp-5E0h]
  __int64 v142; // [rsp+20h] [rbp-5C0h]
  __int64 v143; // [rsp+20h] [rbp-5C0h]
  __int64 v144; // [rsp+28h] [rbp-5B8h]
  _QWORD *v145; // [rsp+28h] [rbp-5B8h]
  __int64 v146; // [rsp+28h] [rbp-5B8h]
  __int64 v147; // [rsp+38h] [rbp-5A8h]
  __int64 v148; // [rsp+38h] [rbp-5A8h]
  __int64 v149; // [rsp+38h] [rbp-5A8h]
  char v151; // [rsp+48h] [rbp-598h]
  unsigned int v152; // [rsp+48h] [rbp-598h]
  __int64 v153; // [rsp+58h] [rbp-588h] BYREF
  _BYTE *v154; // [rsp+60h] [rbp-580h] BYREF
  _BYTE *v155; // [rsp+68h] [rbp-578h]
  _BYTE *v156; // [rsp+70h] [rbp-570h]
  _BYTE *v157; // [rsp+80h] [rbp-560h] BYREF
  _BYTE *v158; // [rsp+88h] [rbp-558h]
  _BYTE *v159; // [rsp+90h] [rbp-550h]
  __int64 v160; // [rsp+A0h] [rbp-540h] BYREF
  void *s; // [rsp+A8h] [rbp-538h]
  _BYTE v162[12]; // [rsp+B0h] [rbp-530h]
  char v163; // [rsp+BCh] [rbp-524h]
  _BYTE v164[128]; // [rsp+C0h] [rbp-520h] BYREF
  __int64 v165; // [rsp+140h] [rbp-4A0h] BYREF
  _BYTE *v166; // [rsp+148h] [rbp-498h]
  __int64 v167; // [rsp+150h] [rbp-490h]
  int v168; // [rsp+158h] [rbp-488h]
  char v169; // [rsp+15Ch] [rbp-484h]
  _BYTE v170[128]; // [rsp+160h] [rbp-480h] BYREF
  __int64 v171; // [rsp+1E0h] [rbp-400h] BYREF
  __int64 *v172; // [rsp+1E8h] [rbp-3F8h]
  __int64 v173; // [rsp+1F0h] [rbp-3F0h]
  int v174; // [rsp+1F8h] [rbp-3E8h]
  char v175; // [rsp+1FCh] [rbp-3E4h]
  char v176; // [rsp+200h] [rbp-3E0h] BYREF
  __int64 v177; // [rsp+280h] [rbp-360h] BYREF
  char *v178; // [rsp+288h] [rbp-358h]
  __int64 v179; // [rsp+290h] [rbp-350h]
  int v180; // [rsp+298h] [rbp-348h]
  char v181; // [rsp+29Ch] [rbp-344h]
  char v182; // [rsp+2A0h] [rbp-340h] BYREF
  _BYTE *v183; // [rsp+3A0h] [rbp-240h] BYREF
  __int64 v184; // [rsp+3A8h] [rbp-238h]
  _BYTE v185[560]; // [rsp+3B0h] [rbp-230h] BYREF

  s = v164;
  v166 = v170;
  v178 = &v182;
  v154 = 0;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v160 = 0;
  *(_QWORD *)v162 = 16;
  *(_DWORD *)&v162[8] = 0;
  v163 = 1;
  v165 = 0;
  v167 = 16;
  v168 = 0;
  v169 = 1;
  v177 = 0;
  v179 = 32;
  v180 = 0;
  v181 = 1;
  LODWORD(v171) = 0;
  if ( *(_DWORD *)(a1 + 680) )
  {
    sub_B8BBF0((__int64)&v154, 0, &v171);
LABEL_14:
    v10 = v155;
    while ( 1 )
    {
      v11 = v154;
      v12 = *((unsigned int *)v10 - 1);
      v13 = v10 - v154;
      v14 = (v13 >> 2) - 1;
      v15 = ((v13 >> 2) - 2) / 2;
      if ( v14 > 0 )
      {
        while ( 1 )
        {
          v139 = (unsigned int *)&v11[4 * v14];
          v16 = *(_DWORD *)&v11[4 * v15];
          if ( (unsigned int)v12 >= v16 )
            break;
          *v139 = v16;
          v14 = v15;
          if ( v15 <= 0 )
          {
            v139 = (unsigned int *)&v11[4 * v15];
            break;
          }
          v15 = (v15 - 1) / 2;
        }
      }
      else
      {
        v139 = (unsigned int *)&v154[v13 - 4];
      }
      *v139 = v12;
      sub_37BC2F0(
        (__int64)&v183,
        (__int64)&v165,
        *(__int64 **)(*(_QWORD *)(a1 + 600) + 8LL * (unsigned int)v171),
        (__int64)v139,
        v12,
        a6);
      sub_37BC2F0(
        (__int64)&v183,
        (__int64)&v177,
        *(__int64 **)(*(_QWORD *)(a1 + 600) + 8LL * (unsigned int)v171),
        v17,
        v18,
        v19);
      v20 = v171 + 1;
      LODWORD(v171) = v20;
      if ( v20 >= *(_DWORD *)(a1 + 680) )
        break;
      v9 = v155;
      if ( v155 == v156 )
      {
        sub_B8BBF0((__int64)&v154, v155, &v171);
        goto LABEL_14;
      }
      if ( v155 )
      {
        *(_DWORD *)v155 = v20;
        v9 = v155;
      }
      v10 = v9 + 4;
      v155 = v9 + 4;
    }
  }
  v21 = *(_QWORD *)(a1 + 408);
  v22 = *(_DWORD *)(v21 + 40);
  if ( v22 )
  {
    v23 = v22;
    for ( i = 0; i != v23; ++i )
    {
      v25 = (_DWORD)i << 8;
      v26 = ***a3 + 8 * i;
      *(_QWORD *)v26 &= 0xFFFFFF0000000000LL;
      *(_DWORD *)(v26 + 4) = v25;
    }
    v21 = *(_QWORD *)(a1 + 408);
  }
  *(_DWORD *)(v21 + 304) = 0;
  sub_37D5200(a1, a2, (__int64)&v177, (__int64)a3, (__int64)a5, a6);
  v171 = 0;
  v172 = (__int64 *)&v176;
  v173 = 16;
  v174 = 0;
  v175 = 1;
  v140 = a4;
  while ( 1 )
  {
    v28 = v155;
    v29 = v154;
    if ( v155 == v154 && v158 == v157 )
      break;
    v30 = v140;
    v183 = v185;
    v184 = 0x2000000000LL;
    while ( v28 != (_BYTE *)v29 )
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * (unsigned int)*v29);
        *(_DWORD *)(a1 + 416) = *(_DWORD *)(v31 + 24);
        if ( v28 - (_BYTE *)v29 > 4 )
        {
          v65 = *((_DWORD *)v28 - 1);
          *((_DWORD *)v28 - 1) = *v29;
          sub_37B6A70((__int64)v29, 0, (v28 - 4 - (_BYTE *)v29) >> 2, v65);
          v28 = v155;
        }
        v32 = *(int *)(v31 + 24);
        v155 = v28 - 4;
        v35 = (unsigned int)sub_37BB7D0(a1, v31, (__int64)&v171, v30, (*a3)[v32], v27);
        if ( v175 )
        {
          v36 = v172;
          v34 = HIDWORD(v173);
          v33 = &v172[HIDWORD(v173)];
          if ( v172 != v33 )
          {
            while ( v31 != *v36 )
            {
              if ( v33 == ++v36 )
                goto LABEL_42;
            }
            goto LABEL_31;
          }
LABEL_42:
          if ( HIDWORD(v173) < (unsigned int)v173 )
            break;
        }
        v151 = v35;
        sub_C8CC70((__int64)&v171, v31, (__int64)v33, v34, v35, v27);
        LOBYTE(v35) = v41 | v151;
LABEL_31:
        if ( (_BYTE)v35 )
          goto LABEL_44;
        v28 = v155;
        v29 = v154;
        if ( v155 == v154 )
          goto LABEL_33;
      }
      ++HIDWORD(v173);
      *v33 = v31;
      ++v171;
LABEL_44:
      v42 = *(_QWORD *)(a1 + 408);
      v43 = (*a3)[*(int *)(v31 + 24)];
      v44 = *(unsigned int *)(v42 + 40);
      *(_DWORD *)(v42 + 280) = *(_DWORD *)(a1 + 416);
      v45 = 0;
      if ( (_DWORD)v44 )
      {
        do
        {
          *(_QWORD *)(*(_QWORD *)(v42 + 32) + v45) = *(_QWORD *)(*v43 + v45);
          v45 += 8;
        }
        while ( 8 * v44 != v45 );
      }
      v46 = *(unsigned int *)(a1 + 416);
      LODWORD(v184) = 0;
      v47 = v46;
      v48 = *a5 + 80 * v46;
      if ( *(_DWORD *)(v48 + 8) >> 1 )
      {
        v49 = *(_BYTE *)(v48 + 8) & 1;
        if ( v49 )
        {
          v50 = (unsigned int *)(v48 + 16);
          v51 = (unsigned int *)(v48 + 80);
          goto LABEL_49;
        }
        v54 = *(unsigned int *)(v48 + 24);
        v53 = *(unsigned int **)(v48 + 16);
        v50 = v53;
        v51 = &v53[4 * v54];
        if ( v53 == v51 )
        {
          v52 = *(unsigned int **)(v48 + 16);
LABEL_54:
          v55 = 4 * v54;
          goto LABEL_55;
        }
LABEL_49:
        while ( *v50 > 0xFFFFFFFD )
        {
          v50 += 4;
          if ( v50 == v51 )
          {
            v52 = v50;
            goto LABEL_52;
          }
        }
        v52 = v50;
        v50 = v51;
      }
      else
      {
        v49 = *(_BYTE *)(v48 + 8) & 1;
        if ( v49 )
        {
          v112 = v48 + 16;
          v113 = 64;
        }
        else
        {
          v112 = *(_QWORD *)(v48 + 16);
          v113 = 16LL * *(unsigned int *)(v48 + 24);
        }
        v52 = (unsigned int *)(v113 + v112);
        v50 = v52;
      }
LABEL_52:
      if ( !v49 )
      {
        v53 = *(unsigned int **)(v48 + 16);
        v54 = *(unsigned int *)(v48 + 24);
        goto LABEL_54;
      }
      v53 = (unsigned int *)(v48 + 16);
      v55 = 16;
LABEL_55:
      v56 = &v53[v55];
      v57 = 0;
      if ( v56 != v52 )
      {
        v58 = (__int64)v30;
        v59 = v31;
        v60 = v56;
        while ( 1 )
        {
          v61 = v57 + 1;
          if ( (v52[2] & 0xFFFFF) == v47 && (*((_QWORD *)v52 + 1) & 0xFFFFF00000LL) == 0 )
          {
            v62 = *v52;
            v63 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 408) + 32LL) + 8LL * (v52[3] >> 8));
            if ( v61 > HIDWORD(v184) )
            {
              v142 = v58;
              v144 = *v52;
              v147 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 408) + 32LL) + 8LL * (v52[3] >> 8));
              sub_C8D5F0((__int64)&v183, v185, v57 + 1, 0x10u, v63, v58);
              v57 = (unsigned int)v184;
              v58 = v142;
              v62 = v144;
              v63 = v147;
            }
          }
          else
          {
            v62 = *v52;
            v63 = *((_QWORD *)v52 + 1);
            if ( v61 > HIDWORD(v184) )
            {
              v143 = v58;
              v146 = *((_QWORD *)v52 + 1);
              v149 = *v52;
              sub_C8D5F0((__int64)&v183, v185, v57 + 1, 0x10u, v63, v58);
              v57 = (unsigned int)v184;
              v58 = v143;
              v63 = v146;
              v62 = v149;
            }
          }
          v64 = &v183[16 * v57];
          *v64 = v62;
          v64[1] = v63;
          v57 = (unsigned int)(v184 + 1);
          LODWORD(v184) = v184 + 1;
          do
            v52 += 4;
          while ( v52 != v50 && *v52 > 0xFFFFFFFD );
          if ( v60 == v52 )
            break;
          v47 = *(_DWORD *)(a1 + 416);
        }
        v66 = (unsigned __int64)v183;
        v31 = v59;
        v30 = (_QWORD *)v58;
        v67 = &v183[16 * v57];
        if ( v67 != v183 )
        {
          do
          {
            v68 = *(_QWORD *)(v66 + 8);
            v66 += 16LL;
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 408) + 32LL) + 8LL * *(unsigned int *)(v66 - 16)) = v68;
          }
          while ( v67 != (_BYTE *)v66 );
        }
      }
      v27 = *(_QWORD *)(a1 + 408);
      v69 = *(unsigned int *)(v27 + 40);
      if ( (_DWORD)v69 )
      {
        v70 = 8 * v69;
        v71 = 0;
        v72 = 0;
        do
        {
          v73 = (_QWORD *)(v72 + **(_QWORD **)(*v30 + 8LL * *(int *)(v31 + 24)));
          v74 = *(_QWORD *)(v72 + *(_QWORD *)(v27 + 32));
          v75 = *v73 != v74;
          v72 += 8;
          *v73 = v74;
          v71 |= v75;
        }
        while ( v70 != v72 );
        *(_DWORD *)(*(_QWORD *)(a1 + 408) + 304LL) = 0;
        if ( v71 )
        {
          v76 = *(__int64 **)(v31 + 112);
          if ( v76 != &v76[*(unsigned int *)(v31 + 120)] )
          {
            v145 = v30;
            v77 = &v76[*(unsigned int *)(v31 + 120)];
            v152 = ((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4);
            v148 = a1 + 664;
            while ( 1 )
            {
              v78 = *v76;
              v79 = *(_DWORD *)(a1 + 688);
              v153 = *v76;
              if ( v79 )
              {
                v27 = v79 - 1;
                v80 = *(_QWORD *)(a1 + 672);
                v81 = 1;
                v82 = 0;
                v83 = v27 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
                v84 = v80 + 16LL * v83;
                v85 = *(_QWORD *)v84;
                if ( v78 == *(_QWORD *)v84 )
                {
LABEL_78:
                  v86 = *(_DWORD *)(v84 + 8);
                  goto LABEL_79;
                }
                while ( v85 != -4096 )
                {
                  if ( !v82 && v85 == -8192 )
                    v82 = v84;
                  v83 = v27 & (v81 + v83);
                  v84 = v80 + 16LL * v83;
                  v85 = *(_QWORD *)v84;
                  if ( v78 == *(_QWORD *)v84 )
                    goto LABEL_78;
                  ++v81;
                }
                v114 = *(_DWORD *)(a1 + 680);
                if ( !v82 )
                  v82 = v84;
                ++*(_QWORD *)(a1 + 664);
                v115 = v114 + 1;
                if ( 4 * (v114 + 1) < 3 * v79 )
                {
                  if ( v79 - *(_DWORD *)(a1 + 684) - v115 > v79 >> 3 )
                    goto LABEL_142;
                  sub_2E515B0(v148, v79);
                  v134 = *(_DWORD *)(a1 + 688);
                  if ( !v134 )
                  {
LABEL_222:
                    ++*(_DWORD *)(a1 + 680);
                    BUG();
                  }
                  v128 = v153;
                  v135 = v134 - 1;
                  v136 = *(_QWORD *)(a1 + 672);
                  v137 = 1;
                  v115 = *(_DWORD *)(a1 + 680) + 1;
                  v133 = 0;
                  v138 = v135 & (((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4));
                  v82 = v136 + 16LL * v138;
                  v78 = *(_QWORD *)v82;
                  if ( v153 == *(_QWORD *)v82 )
                    goto LABEL_142;
                  while ( v78 != -4096 )
                  {
                    if ( !v133 && v78 == -8192 )
                      v133 = v82;
                    v138 = v135 & (v137 + v138);
                    v82 = v136 + 16LL * v138;
                    v78 = *(_QWORD *)v82;
                    if ( v153 == *(_QWORD *)v82 )
                      goto LABEL_142;
                    ++v137;
                  }
                  goto LABEL_170;
                }
              }
              else
              {
                ++*(_QWORD *)(a1 + 664);
              }
              sub_2E515B0(v148, 2 * v79);
              v127 = *(_DWORD *)(a1 + 688);
              if ( !v127 )
                goto LABEL_222;
              v128 = v153;
              v129 = v127 - 1;
              v130 = *(_QWORD *)(a1 + 672);
              v115 = *(_DWORD *)(a1 + 680) + 1;
              v131 = v129 & (((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4));
              v82 = v130 + 16LL * v131;
              v78 = *(_QWORD *)v82;
              if ( v153 == *(_QWORD *)v82 )
                goto LABEL_142;
              v132 = 1;
              v133 = 0;
              while ( v78 != -4096 )
              {
                if ( !v133 && v78 == -8192 )
                  v133 = v82;
                v131 = v129 & (v132 + v131);
                v82 = v130 + 16LL * v131;
                v78 = *(_QWORD *)v82;
                if ( v153 == *(_QWORD *)v82 )
                  goto LABEL_142;
                ++v132;
              }
LABEL_170:
              v78 = v128;
              if ( v133 )
                v82 = v133;
LABEL_142:
              *(_DWORD *)(a1 + 680) = v115;
              if ( *(_QWORD *)v82 != -4096 )
                --*(_DWORD *)(a1 + 684);
              *(_QWORD *)v82 = v78;
              *(_DWORD *)(v82 + 8) = 0;
              v79 = *(_DWORD *)(a1 + 688);
              if ( !v79 )
              {
                ++*(_QWORD *)(a1 + 664);
                v86 = 0;
                goto LABEL_146;
              }
              v80 = *(_QWORD *)(a1 + 672);
              v27 = v79 - 1;
              v86 = 0;
LABEL_79:
              v87 = 1;
              v88 = 0;
              v89 = v27 & v152;
              v90 = v80 + 16LL * ((unsigned int)v27 & v152);
              v91 = *(_QWORD *)v90;
              if ( v31 == *(_QWORD *)v90 )
                goto LABEL_80;
              while ( 1 )
              {
                if ( v91 == -4096 )
                {
                  if ( !v88 )
                    v88 = v90;
                  v121 = *(_DWORD *)(a1 + 680);
                  ++*(_QWORD *)(a1 + 664);
                  v119 = v121 + 1;
                  if ( 4 * v119 >= 3 * v79 )
                  {
LABEL_146:
                    sub_2E515B0(v148, 2 * v79);
                    v116 = *(_DWORD *)(a1 + 688);
                    if ( v116 )
                    {
                      v117 = v116 - 1;
                      v118 = *(_QWORD *)(a1 + 672);
                      v91 = v117 & v152;
                      v119 = *(_DWORD *)(a1 + 680) + 1;
                      v88 = v118 + 16 * v91;
                      v120 = *(_QWORD *)v88;
                      if ( v31 != *(_QWORD *)v88 )
                      {
                        v27 = 1;
                        v124 = 0;
                        while ( v120 != -4096 )
                        {
                          if ( !v124 && v120 == -8192 )
                            v124 = v88;
                          v80 = (unsigned int)(v27 + 1);
                          v91 = v117 & (unsigned int)(v27 + v91);
                          v88 = v118 + 16LL * (unsigned int)v91;
                          v120 = *(_QWORD *)v88;
                          if ( v31 == *(_QWORD *)v88 )
                            goto LABEL_148;
                          v27 = (unsigned int)v80;
                        }
LABEL_161:
                        if ( v124 )
                          v88 = v124;
                      }
LABEL_148:
                      *(_DWORD *)(a1 + 680) = v119;
                      if ( *(_QWORD *)v88 != -4096 )
                        --*(_DWORD *)(a1 + 684);
                      *(_QWORD *)v88 = v31;
                      v92 = 0;
                      *(_DWORD *)(v88 + 8) = 0;
                      goto LABEL_81;
                    }
                  }
                  else
                  {
                    v91 = v79 >> 3;
                    if ( v79 - (v119 + *(_DWORD *)(a1 + 684)) > (unsigned int)v91 )
                      goto LABEL_148;
                    sub_2E515B0(v148, v79);
                    v122 = *(_DWORD *)(a1 + 688);
                    if ( v122 )
                    {
                      v123 = v122 - 1;
                      v124 = 0;
                      v125 = *(_QWORD *)(a1 + 672);
                      v27 = 1;
                      v91 = v123 & v152;
                      v119 = *(_DWORD *)(a1 + 680) + 1;
                      v88 = v125 + 16 * v91;
                      v126 = *(_QWORD *)v88;
                      if ( v31 != *(_QWORD *)v88 )
                      {
                        while ( v126 != -4096 )
                        {
                          if ( !v124 && v126 == -8192 )
                            v124 = v88;
                          v80 = (unsigned int)(v27 + 1);
                          v91 = v123 & (unsigned int)(v27 + v91);
                          v88 = v125 + 16LL * (unsigned int)v91;
                          v126 = *(_QWORD *)v88;
                          if ( v31 == *(_QWORD *)v88 )
                            goto LABEL_148;
                          v27 = (unsigned int)v80;
                        }
                        goto LABEL_161;
                      }
                      goto LABEL_148;
                    }
                  }
                  ++*(_DWORD *)(a1 + 680);
                  BUG();
                }
                if ( v88 || v91 != -8192 )
                  v90 = v88;
                v88 = (unsigned int)(v87 + 1);
                v89 = v27 & (v87 + v89);
                v91 = *(_QWORD *)(v80 + 16LL * v89);
                if ( v31 == v91 )
                  break;
                ++v87;
                v88 = v90;
                v90 = v80 + 16LL * v89;
              }
              v90 = v80 + 16LL * v89;
LABEL_80:
              v92 = *(_DWORD *)(v90 + 8);
LABEL_81:
              if ( v86 <= v92 )
              {
                if ( v163 )
                {
                  v94 = s;
                  v91 = *(unsigned int *)&v162[4];
                  v88 = (__int64)s + 8 * *(unsigned int *)&v162[4];
                  if ( s != (void *)v88 )
                  {
                    while ( *v94 != v153 )
                    {
                      if ( (_QWORD *)v88 == ++v94 )
                        goto LABEL_120;
                    }
                    goto LABEL_87;
                  }
LABEL_120:
                  if ( *(_DWORD *)&v162[4] < *(_DWORD *)v162 )
                  {
                    ++*(_DWORD *)&v162[4];
                    *(_QWORD *)v88 = v153;
                    ++v160;
LABEL_110:
                    v107 = sub_2E51790(v148, &v153);
                    v108 = v158;
                    if ( v158 == v159 )
                    {
                      sub_B8BBF0((__int64)&v157, v158, v107);
                      v109 = v158;
                    }
                    else
                    {
                      if ( v158 )
                      {
                        *(_DWORD *)v158 = *(_DWORD *)v107;
                        v108 = v158;
                      }
                      v109 = v108 + 4;
                      v158 = v109;
                    }
                    v99 = v157;
                    v100 = *((_DWORD *)v109 - 1);
                    v101 = v109 - v157;
                    v110 = (v101 >> 2) - 1;
                    v111 = ((v101 >> 2) - 2) / 2;
                    if ( v110 <= 0 )
                      goto LABEL_198;
                    while ( 1 )
                    {
                      v104 = (unsigned int *)&v99[4 * v111];
                      v105 = (unsigned int *)&v99[4 * v110];
                      if ( v100 >= *v104 )
                        break;
                      *v105 = *v104;
                      v110 = v111;
                      if ( v111 <= 0 )
                      {
LABEL_124:
                        *v104 = v100;
                        goto LABEL_107;
                      }
                      v111 = (v111 - 1) / 2;
                    }
LABEL_106:
                    *v105 = v100;
                    goto LABEL_107;
                  }
                }
                sub_C8CC70((__int64)&v160, v153, v88, v91, v80, v27);
                if ( v106 )
                  goto LABEL_110;
LABEL_87:
                if ( v77 == ++v76 )
                  goto LABEL_88;
              }
              else
              {
                if ( !v169 )
                  goto LABEL_96;
                v93 = v166;
                v91 = HIDWORD(v167);
                v88 = (__int64)&v166[8 * HIDWORD(v167)];
                if ( v166 != (_BYTE *)v88 )
                {
                  while ( *v93 != v153 )
                  {
                    if ( (_QWORD *)v88 == ++v93 )
                      goto LABEL_122;
                  }
                  goto LABEL_87;
                }
LABEL_122:
                if ( HIDWORD(v167) >= (unsigned int)v167 )
                {
LABEL_96:
                  sub_C8CC70((__int64)&v165, v153, v88, v91, v80, v27);
                  if ( v95 )
                    goto LABEL_97;
                  goto LABEL_87;
                }
                ++HIDWORD(v167);
                *(_QWORD *)v88 = v153;
                ++v165;
LABEL_97:
                v96 = sub_2E51790(v148, &v153);
                v97 = v155;
                if ( v155 == v156 )
                {
                  sub_B8BBF0((__int64)&v154, v155, v96);
                  v98 = v155;
                }
                else
                {
                  if ( v155 )
                  {
                    *(_DWORD *)v155 = *(_DWORD *)v96;
                    v97 = v155;
                  }
                  v98 = v97 + 4;
                  v155 = v98;
                }
                v99 = v154;
                v100 = *((_DWORD *)v98 - 1);
                v101 = v98 - v154;
                v102 = (v101 >> 2) - 1;
                v103 = ((v101 >> 2) - 2) / 2;
                if ( v102 > 0 )
                {
                  while ( 1 )
                  {
                    v104 = (unsigned int *)&v99[4 * v103];
                    v105 = (unsigned int *)&v99[4 * v102];
                    if ( v100 >= *v104 )
                      goto LABEL_106;
                    *v105 = *v104;
                    v102 = v103;
                    if ( v103 <= 0 )
                      goto LABEL_124;
                    v103 = (v103 - 1) / 2;
                  }
                }
LABEL_198:
                *(_DWORD *)&v99[v101 - 4] = v100;
LABEL_107:
                if ( v77 == ++v76 )
                {
LABEL_88:
                  v30 = v145;
                  break;
                }
              }
            }
          }
        }
      }
      else
      {
        *(_DWORD *)(v27 + 304) = 0;
      }
      v28 = v155;
      v29 = v154;
    }
LABEL_33:
    v37 = v157;
    v157 = v28;
    v154 = v37;
    v38 = v158;
    v158 = v28;
    v155 = v38;
    v39 = v159;
    v159 = v156;
    v156 = v39;
    sub_C8CFE0((__int64)&v160, v164, v170, (__int64)&v165);
    ++v160;
    if ( v163 )
      goto LABEL_38;
    v40 = 4 * (*(_DWORD *)&v162[4] - *(_DWORD *)&v162[8]);
    if ( v40 < 0x20 )
      v40 = 32;
    if ( *(_DWORD *)v162 > v40 )
    {
      sub_C8C990((__int64)&v160, (__int64)v164);
    }
    else
    {
      memset(s, -1, 8LL * *(unsigned int *)v162);
LABEL_38:
      *(_QWORD *)&v162[4] = 0;
    }
    if ( v183 != v185 )
      _libc_free((unsigned __int64)v183);
  }
  if ( !v175 )
    _libc_free((unsigned __int64)v172);
  if ( !v181 )
    _libc_free((unsigned __int64)v178);
  if ( !v169 )
    _libc_free((unsigned __int64)v166);
  if ( !v163 )
    _libc_free((unsigned __int64)s);
  if ( v157 )
    j_j___libc_free_0((unsigned __int64)v157);
  if ( v154 )
    j_j___libc_free_0((unsigned __int64)v154);
}
