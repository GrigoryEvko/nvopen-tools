// Function: sub_19DA750
// Address: 0x19da750
//
__int64 __fastcall sub_19DA750(
        __int64 a1,
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
  unsigned int v10; // eax
  unsigned int v11; // r14d
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 *v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rax
  __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // r8
  __int64 v31; // rdx
  __int64 v32; // rdi
  __int64 v33; // r9
  char v34; // si
  unsigned int v35; // r12d
  __int64 v36; // rax
  int v37; // r12d
  __int64 v38; // r11
  __int64 v39; // rdi
  __int64 v40; // r15
  __int64 v41; // rbx
  unsigned int v42; // esi
  __int64 v43; // rdx
  int v44; // eax
  __int64 v45; // rcx
  __int64 v46; // rdx
  __int64 v47; // r12
  char v48; // di
  unsigned int v49; // esi
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rcx
  unsigned __int64 v54; // r15
  __int64 v55; // rax
  char v56; // cl
  bool v57; // al
  __int64 v58; // rdx
  int v59; // r11d
  int v60; // r12d
  unsigned int v61; // ecx
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 *v64; // r12
  __int64 *i; // rbx
  __int64 v66; // rdi
  __int64 v67; // rdi
  char *v68; // r15
  char *v69; // rbx
  unsigned __int64 v70; // r15
  __int64 v71; // r12
  char v72; // al
  double v73; // xmm4_8
  double v74; // xmm5_8
  char *v75; // r15
  char *v76; // rbx
  char v77; // r12
  unsigned int v78; // ecx
  unsigned int v79; // esi
  __int64 v80; // rax
  __int64 v81; // rdx
  char *v82; // rdx
  __int64 v83; // rsi
  __int64 v84; // r15
  __int64 v85; // rax
  __int64 *v86; // rax
  unsigned __int64 v87; // rcx
  __int64 v88; // rcx
  _QWORD *v89; // rax
  __int64 v90; // rbx
  _QWORD *v91; // rdi
  __int64 v92; // rdi
  int v93; // r12d
  __int64 v94; // rbx
  unsigned __int64 v95; // r15
  unsigned __int64 v96; // rax
  double v97; // xmm4_8
  double v98; // xmm5_8
  unsigned __int64 v99; // rdx
  unsigned __int64 v100; // rax
  void **v101; // rax
  __int64 *v102; // rdx
  unsigned __int64 v103; // rdi
  __int64 v104; // rax
  unsigned __int64 v105; // r15
  __int64 v106; // rdi
  __int64 v107; // rdi
  __int64 v108; // r12
  unsigned int v109; // eax
  unsigned int v110; // eax
  __int64 *v111; // rax
  __int64 v112; // rsi
  char *v113; // r15
  char *v114; // r12
  char *v115; // rbx
  __int64 v116; // rdi
  __int64 v117; // rdi
  __int64 *v118; // r12
  __int64 *v119; // r15
  signed __int64 v120; // rbx
  unsigned __int64 v121; // rax
  __int64 *v122; // rbx
  __int64 *v123; // rdi
  int v124; // edx
  __int64 *v125; // r8
  __int64 v126; // r15
  unsigned int v127; // eax
  __int64 v128; // rdx
  __int64 v129; // r12
  __int64 *v130; // rdi
  __int64 *v131; // rdx
  __int64 v132; // rbx
  __int64 v133; // rcx
  char *v134; // rax
  char *v135; // r11
  __int64 v136; // rsi
  signed __int64 v137; // rsi
  int v138; // esi
  __int64 *v139; // rbx
  __int64 *k; // r12
  __int64 v141; // rdi
  __int64 v142; // rdi
  __int64 v143; // [rsp+10h] [rbp-1D0h]
  __int64 v144; // [rsp+18h] [rbp-1C8h]
  __int64 v145; // [rsp+18h] [rbp-1C8h]
  __int64 v146; // [rsp+20h] [rbp-1C0h]
  int v147; // [rsp+20h] [rbp-1C0h]
  __int64 v148; // [rsp+20h] [rbp-1C0h]
  __int64 v149; // [rsp+20h] [rbp-1C0h]
  __int64 v150; // [rsp+30h] [rbp-1B0h]
  __int64 v151; // [rsp+30h] [rbp-1B0h]
  __int64 v152; // [rsp+30h] [rbp-1B0h]
  char v153; // [rsp+30h] [rbp-1B0h]
  __int64 *v154; // [rsp+30h] [rbp-1B0h]
  __int64 n; // [rsp+38h] [rbp-1A8h]
  __int64 *v156; // [rsp+40h] [rbp-1A0h]
  __int64 v157; // [rsp+48h] [rbp-198h]
  unsigned __int8 v158; // [rsp+57h] [rbp-189h]
  char v159; // [rsp+58h] [rbp-188h]
  __int64 v160; // [rsp+58h] [rbp-188h]
  __int64 *v161; // [rsp+60h] [rbp-180h] BYREF
  __int64 *v162; // [rsp+68h] [rbp-178h]
  __int64 *j; // [rsp+70h] [rbp-170h]
  char v164[8]; // [rsp+80h] [rbp-160h] BYREF
  __int64 *v165; // [rsp+88h] [rbp-158h]
  int v166; // [rsp+90h] [rbp-150h]
  int v167; // [rsp+98h] [rbp-148h]
  __int64 v168; // [rsp+A0h] [rbp-140h] BYREF
  char *v169; // [rsp+A8h] [rbp-138h]
  char *v170; // [rsp+B0h] [rbp-130h]
  __int64 *v171; // [rsp+B8h] [rbp-128h]
  __int64 v172; // [rsp+C0h] [rbp-120h]
  __int64 v173; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v174; // [rsp+D8h] [rbp-108h]
  __int64 v175; // [rsp+E0h] [rbp-100h]
  char v176; // [rsp+E8h] [rbp-F8h]
  __int64 v177; // [rsp+F0h] [rbp-F0h]
  __int64 v178; // [rsp+F8h] [rbp-E8h]
  __int64 v179; // [rsp+100h] [rbp-E0h] BYREF
  unsigned int v180; // [rsp+108h] [rbp-D8h]
  __int64 v181; // [rsp+110h] [rbp-D0h]
  __int64 v182; // [rsp+118h] [rbp-C8h]
  __int64 v183; // [rsp+120h] [rbp-C0h] BYREF
  unsigned int v184; // [rsp+128h] [rbp-B8h]
  int v185; // [rsp+130h] [rbp-B0h]
  __int64 v186[20]; // [rsp+140h] [rbp-A0h] BYREF

  v10 = sub_1636880(a1, a2);
  if ( (_BYTE)v10 )
    return 0;
  v13 = *(__int64 **)(a1 + 8);
  v11 = v10;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_283:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F9B6E8 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_283;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F9B6E8);
  v17 = *(__int64 **)(a1 + 8);
  v18 = v16;
  v157 = v16 + 360;
  v19 = *v17;
  v20 = v17[1];
  if ( v19 == v20 )
LABEL_281:
    BUG();
  while ( *(_UNKNOWN **)v19 != &unk_4F9D3C0 )
  {
    v19 += 16;
    if ( v20 == v19 )
      goto LABEL_281;
  }
  v21 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v19 + 8) + 104LL))(*(_QWORD *)(v19 + 8), &unk_4F9D3C0);
  v22 = (__int64 *)sub_14A4050(v21, a2);
  if ( !sub_14A2F00(v22, 1u) )
  {
    v101 = (void **)&v186[5];
    v186[7] = 0;
    v186[3] = 0x100000002LL;
    v186[8] = (__int64)&v186[12];
    v186[9] = (__int64)&v186[12];
    v102 = &v186[6];
    v186[1] = (__int64)&v186[5];
    v186[2] = (__int64)&v186[5];
    v186[10] = 2;
    LODWORD(v186[11]) = 0;
    LODWORD(v186[4]) = 0;
    v186[5] = (__int64)&unk_4F9EE48;
    v186[0] = 1;
    goto LABEL_134;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v18 + 360) + 72LL) & 0xC) == 0 || *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8LL) == a2 + 72 )
    goto LABEL_126;
  v158 = v11;
  v23 = *(_QWORD *)(*(_QWORD *)(a2 + 80) + 8LL);
  v24 = a2 + 72;
  v159 = 0;
  do
  {
    if ( !v23 )
      BUG();
    v25 = *(_QWORD *)(v23 + 24);
    if ( !v25 )
      BUG();
    if ( *(_BYTE *)(v25 - 8) == 77 )
    {
      v26 = *(_DWORD *)(v25 - 4) & 0xFFFFFFF;
      if ( (unsigned int)v26 > 1 )
      {
        v27 = 3 * v26;
        v28 = 8 * v26;
        v29 = 0;
        v30 = v25 - 24 - 8 * v27;
        v31 = 0;
        do
        {
          while ( 1 )
          {
            v32 = v30;
            if ( (*(_BYTE *)(v25 - 1) & 0x40) != 0 )
              v32 = *(_QWORD *)(v25 - 32);
            v33 = *(_QWORD *)(v32 + 3 * v31);
            v34 = *(_BYTE *)(v33 + 16);
            if ( v34 != 13 )
              break;
            v31 += 8;
            if ( v28 == v31 )
              goto LABEL_28;
          }
          if ( v29 )
            goto LABEL_18;
          if ( v34 != 75 )
            goto LABEL_18;
          v29 = *(_QWORD *)(v31 + v32 + 24LL * *(unsigned int *)(v25 + 32) + 8);
          if ( *(_QWORD *)(v33 + 40) != v29 )
            goto LABEL_18;
          v31 += 8;
        }
        while ( v28 != v31 );
LABEL_28:
        if ( v29 && *(_QWORD *)(v25 + 16) == sub_157F1C0(v29) )
        {
          v150 = v25 - 24;
          v35 = *(_DWORD *)(v25 - 4) & 0xFFFFFFF;
          if ( !v35 )
          {
            MEMORY[0] = v29;
            BUG();
          }
          v36 = 8LL * v35;
          v37 = v35 - 1;
          n = v36;
          v156 = (__int64 *)sub_22077B0(v36);
          memset(v156, 0, n);
          v38 = v25 - 24;
          if ( v37 <= 0 )
            goto LABEL_44;
          v39 = v29;
          v40 = v25;
          v41 = v37;
          while ( !*(_WORD *)(v39 + 18) )
          {
            v156[v41] = v39;
            v39 = sub_157F0B0(v39);
            if ( !v39 )
              break;
            v42 = *(_DWORD *)(v40 - 4) & 0xFFFFFFF;
            if ( !v42 )
              break;
            v43 = 24LL * *(unsigned int *)(v40 + 32) + 8;
            v44 = 0;
            while ( 1 )
            {
              v45 = v150 - 24LL * v42;
              if ( (*(_BYTE *)(v40 - 1) & 0x40) != 0 )
                v45 = *(_QWORD *)(v40 - 32);
              if ( v39 == *(_QWORD *)(v45 + v43) )
                break;
              ++v44;
              v43 += 8;
              if ( v42 == v44 )
                goto LABEL_157;
            }
            if ( v44 < 0 )
              break;
            if ( (int)--v41 <= 0 )
            {
              v25 = v40;
              v38 = v150;
              v29 = v39;
LABEL_44:
              *v156 = v29;
              if ( !n )
                break;
              v168 = v38;
              v169 = 0;
              v170 = 0;
              v171 = 0;
              v161 = 0;
              v162 = 0;
              j = 0;
              if ( !(n >> 3) )
                goto LABEL_93;
              v143 = v38;
              v151 = 0;
LABEL_47:
              v46 = 0x17FFFFFFE8LL;
              v47 = v156[v151];
              v48 = *(_BYTE *)(v25 - 1) & 0x40;
              v49 = *(_DWORD *)(v25 - 4) & 0xFFFFFFF;
              if ( !v49 )
                goto LABEL_54;
              v50 = 24LL * *(unsigned int *)(v25 + 32) + 8;
              v51 = 0;
              do
              {
                v52 = v143 - 24LL * v49;
                if ( v48 )
                  v52 = *(_QWORD *)(v25 - 32);
                if ( v47 == *(_QWORD *)(v52 + v50) )
                {
                  v46 = 24 * v51;
                  goto LABEL_54;
                }
                ++v51;
                v50 += 8;
              }
              while ( v49 != (_DWORD)v51 );
              v46 = 0x17FFFFFFE8LL;
LABEL_54:
              if ( v48 )
                v53 = *(_QWORD *)(v25 - 32);
              else
                v53 = v143 - 24LL * v49;
              if ( v47 + 40 == (*(_QWORD *)(v47 + 40) & 0xFFFFFFFFFFFFFFF8LL)
                || (v144 = v46, v146 = v53, v54 = sub_157EBA0(v156[v151]), *(_BYTE *)(v54 + 16) != 26) )
              {
                v174 = 0;
                v175 = 0;
                v176 = 0;
                v177 = 0;
                v178 = 0;
                v180 = 1;
                v179 = 0;
                v184 = 1;
                v185 = 0;
                v173 = v47;
LABEL_80:
                if ( v180 > 0x40 && v179 )
                  j_j___libc_free_0_0(v179);
                v64 = v162;
                for ( i = v161; v64 != i; i += 13 )
                {
                  if ( *((_DWORD *)i + 22) > 0x40u )
                  {
                    v66 = i[10];
                    if ( v66 )
                      j_j___libc_free_0_0(v66);
                  }
                  if ( *((_DWORD *)i + 14) > 0x40u )
                  {
                    v67 = i[6];
                    if ( v67 )
                      j_j___libc_free_0_0(v67);
                  }
                }
                goto LABEL_91;
              }
              v55 = *(_QWORD *)(v146 + v144);
              v56 = *(_BYTE *)(v55 + 16);
              if ( (*(_DWORD *)(v54 + 20) & 0xFFFFFFF) == 1 )
              {
                if ( v56 != 75 )
                  goto LABEL_64;
                v124 = 32;
              }
              else
              {
                if ( v56 != 13 )
                  BUG();
                if ( *(_DWORD *)(v55 + 32) <= 0x40u )
                {
                  v57 = *(_QWORD *)(v55 + 24) == 0;
                }
                else
                {
                  v147 = *(_DWORD *)(v55 + 32);
                  v57 = v147 == (unsigned int)sub_16A57B0(v55 + 24);
                }
                if ( !v57 || (v55 = *(_QWORD *)(v54 - 72), *(_BYTE *)(v55 + 16) != 75) )
                {
LABEL_64:
                  v174 = 0;
                  v175 = 0;
                  v176 = 0;
                  v177 = 0;
                  v178 = 0;
                  v180 = 1;
                  v179 = 0;
                  v181 = 0;
                  v182 = 0;
                  v184 = 1;
                  v183 = 0;
                  v185 = 0;
                  goto LABEL_65;
                }
                v124 = (*(_QWORD *)(v25 + 16) != *(_QWORD *)(v54 - 48)) + 32;
              }
              v148 = v55;
              sub_19D8AF0((__int64)v186, v55, v124);
              v175 = v54;
              v174 = v148;
              v176 = v186[3];
              v177 = v186[4];
              v178 = v186[5];
              v180 = v186[7];
              v179 = v186[6];
              v181 = v186[8];
              v182 = v186[9];
              v184 = v186[11];
              v183 = v186[10];
              v185 = v186[12];
LABEL_65:
              v173 = v47;
              if ( !v177
                || !*(_QWORD *)(v177 - 24LL * (*(_DWORD *)(v177 + 20) & 0xFFFFFFF))
                || !v181
                || !*(_QWORD *)(v181 - 24LL * (*(_DWORD *)(v181 + 20) & 0xFFFFFFF)) )
              {
                goto LABEL_78;
              }
              v186[1] = v181;
              v186[0] = v177;
              v186[2] = v178;
              v186[3] = v182;
              v186[4] = v174;
              v186[5] = v175;
              sub_19D91E0((__int64)v164, v186, 6);
              v58 = *(_QWORD *)(v173 + 48);
              if ( v58 == v173 + 40 )
              {
LABEL_165:
                j___libc_free_0(v165);
                v108 = (__int64)v162;
                if ( v162 == j )
                  goto LABEL_253;
                goto LABEL_166;
              }
              v59 = v167 - 1;
              while ( 1 )
              {
                v63 = v58 - 24;
                if ( !v58 )
                  v63 = 0;
                if ( !v167 )
                  goto LABEL_76;
                v60 = 1;
                v61 = v59 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
                v62 = v165[v61];
                if ( v63 != v62 )
                  break;
LABEL_72:
                v58 = *(_QWORD *)(v58 + 8);
                if ( v173 + 40 == v58 )
                  goto LABEL_165;
              }
              while ( v62 != -8 )
              {
                v61 = v59 & (v60 + v61);
                v62 = v165[v61];
                if ( v63 == v62 )
                  goto LABEL_72;
                ++v60;
              }
LABEL_76:
              j___libc_free_0(v165);
              if ( v162 != v161 )
              {
LABEL_78:
                if ( v184 > 0x40 && v183 )
                  j_j___libc_free_0_0(v183);
                goto LABEL_80;
              }
              v186[0] = v177;
              v186[1] = v181;
              v186[2] = v178;
              v186[3] = v182;
              v186[4] = v174;
              v186[5] = v175;
              sub_19D91E0((__int64)v164, v186, 6);
              v125 = v165;
              v126 = *(_QWORD *)(v173 + 48);
              v145 = v173 + 40;
              if ( v126 == v173 + 40 )
                goto LABEL_252;
              v149 = v25;
LABEL_207:
              v129 = v126 - 24;
              if ( !v126 )
                v129 = 0;
              if ( v167 )
              {
                v127 = (v167 - 1) & (((unsigned int)v129 >> 9) ^ ((unsigned int)v129 >> 4));
                v128 = v125[v127];
                if ( v129 != v128 )
                {
                  v138 = 1;
                  while ( v128 != -8 )
                  {
                    v127 = (v167 - 1) & (v138 + v127);
                    v128 = v125[v127];
                    if ( v129 == v128 )
                      goto LABEL_206;
                    ++v138;
                  }
                  goto LABEL_210;
                }
              }
              else
              {
LABEL_210:
                if ( (unsigned __int8)sub_15F3040(v129) || sub_15F3330(v129) )
                {
                  v25 = v149;
                  v125 = v165;
                  goto LABEL_212;
                }
                v125 = v165;
                v130 = &v165[v167];
                if ( v166 && v130 != v165 )
                {
                  v131 = v165;
                  while ( *v131 == -8 || *v131 == -16 )
                  {
                    if ( v130 == ++v131 )
                      goto LABEL_206;
                  }
                  if ( v130 != v131 )
                  {
                    v132 = 24LL * (*(_DWORD *)(v129 + 20) & 0xFFFFFFF);
LABEL_222:
                    v133 = *v131;
                    if ( (*(_BYTE *)(v129 + 23) & 0x40) != 0 )
                    {
                      v134 = *(char **)(v129 - 8);
                      v135 = &v134[v132];
                    }
                    else
                    {
                      v135 = (char *)v129;
                      v134 = (char *)(v129 - v132);
                    }
                    if ( (__int64)(*(_DWORD *)(v129 + 20) & 0xFFFFFFF) >> 2 )
                    {
                      v136 = (__int64)(*(_DWORD *)(v129 + 20) & 0xFFFFFFF) >> 2;
                      while ( v133 != *(_QWORD *)v134 )
                      {
                        if ( v133 == *((_QWORD *)v134 + 3) )
                        {
                          v134 += 24;
                          break;
                        }
                        if ( v133 == *((_QWORD *)v134 + 6) )
                        {
                          v134 += 48;
                          break;
                        }
                        if ( v133 == *((_QWORD *)v134 + 9) )
                        {
                          v134 += 72;
                          break;
                        }
                        v134 += 96;
                        if ( !--v136 )
                          goto LABEL_239;
                      }
LABEL_231:
                      if ( v134 == v135 )
                        goto LABEL_232;
                      v25 = v149;
LABEL_212:
                      j___libc_free_0(v125);
                      goto LABEL_173;
                    }
LABEL_239:
                    v137 = v135 - v134;
                    if ( v135 - v134 != 48 )
                    {
                      if ( v137 != 72 )
                      {
                        if ( v137 != 24 )
                          goto LABEL_232;
                        goto LABEL_242;
                      }
                      if ( v133 == *(_QWORD *)v134 )
                        goto LABEL_231;
                      v134 += 24;
                    }
                    if ( v133 == *(_QWORD *)v134 )
                      goto LABEL_231;
                    v134 += 24;
LABEL_242:
                    if ( v133 == *(_QWORD *)v134 )
                      goto LABEL_231;
LABEL_232:
                    while ( v130 != ++v131 )
                    {
                      if ( *v131 != -16 && *v131 != -8 )
                      {
                        if ( v130 != v131 )
                          goto LABEL_222;
                        break;
                      }
                    }
                  }
                }
              }
LABEL_206:
              v126 = *(_QWORD *)(v126 + 8);
              if ( v145 == v126 )
              {
                v25 = v149;
LABEL_252:
                j___libc_free_0(v125);
                v176 = 1;
                v108 = (__int64)v162;
                if ( v162 == j )
                {
LABEL_253:
                  sub_19D6410((__int64 *)&v161, v108, (__int64)&v173);
                  goto LABEL_173;
                }
LABEL_166:
                if ( v108 )
                {
                  *(_QWORD *)v108 = v173;
                  *(_QWORD *)(v108 + 8) = v174;
                  *(_QWORD *)(v108 + 16) = v175;
                  *(_BYTE *)(v108 + 24) = v176;
                  *(_QWORD *)(v108 + 32) = v177;
                  *(_QWORD *)(v108 + 40) = v178;
                  v109 = v180;
                  *(_DWORD *)(v108 + 56) = v180;
                  if ( v109 > 0x40 )
                    sub_16A4FD0(v108 + 48, (const void **)&v179);
                  else
                    *(_QWORD *)(v108 + 48) = v179;
                  *(_QWORD *)(v108 + 64) = v181;
                  *(_QWORD *)(v108 + 72) = v182;
                  v110 = v184;
                  *(_DWORD *)(v108 + 88) = v184;
                  if ( v110 > 0x40 )
                    sub_16A4FD0(v108 + 80, (const void **)&v183);
                  else
                    *(_QWORD *)(v108 + 80) = v183;
                  *(_DWORD *)(v108 + 96) = v185;
                }
                v162 += 13;
LABEL_173:
                if ( v184 > 0x40 && v183 )
                  j_j___libc_free_0_0(v183);
                if ( v180 > 0x40 && v179 )
                  j_j___libc_free_0_0(v179);
                if ( ++v151 == n >> 3 )
                {
                  v111 = v161;
                  if ( v162 != v161 )
                  {
                    v112 = *v161;
                    v113 = v169;
                    v161 = 0;
                    v114 = v170;
                    v169 = (char *)v111;
                    v172 = v112;
                    v115 = v113;
                    v154 = v171;
                    v170 = (char *)v162;
                    v171 = j;
                    v162 = 0;
                    for ( j = 0; v114 != v115; v115 += 104 )
                    {
                      if ( *((_DWORD *)v115 + 22) > 0x40u )
                      {
                        v116 = *((_QWORD *)v115 + 10);
                        if ( v116 )
                          j_j___libc_free_0_0(v116);
                      }
                      if ( *((_DWORD *)v115 + 14) > 0x40u )
                      {
                        v117 = *((_QWORD *)v115 + 6);
                        if ( v117 )
                          j_j___libc_free_0_0(v117);
                      }
                    }
                    if ( v113 )
                      j_j___libc_free_0(v113, (char *)v154 - v113);
                    v118 = (__int64 *)v170;
                    v119 = (__int64 *)v169;
                    if ( v170 != v169 )
                    {
                      v120 = v170 - v169;
                      _BitScanReverse64(&v121, 0x4EC4EC4EC4EC4EC5LL * ((v170 - v169) >> 3));
                      sub_19D7EE0((__int64 *)v169, (__int64 *)v170, 2LL * (int)(63 - (v121 ^ 0x3F)));
                      if ( v120 <= 1664 )
                      {
                        sub_19D71F0((__int64)v119, (__int64)v118);
                      }
                      else
                      {
                        v122 = v119 + 208;
                        sub_19D71F0((__int64)v119, (__int64)(v119 + 208));
                        while ( v118 != v122 )
                        {
                          v123 = v122;
                          v122 += 13;
                          sub_19D6DB0(v123);
                        }
                      }
                    }
                    v139 = v162;
                    for ( k = v161; v139 != k; k += 13 )
                    {
                      if ( *((_DWORD *)k + 22) > 0x40u )
                      {
                        v141 = k[10];
                        if ( v141 )
                          j_j___libc_free_0_0(v141);
                      }
                      if ( *((_DWORD *)k + 14) > 0x40u )
                      {
                        v142 = k[6];
                        if ( v142 )
                          j_j___libc_free_0_0(v142);
                      }
                    }
                  }
LABEL_91:
                  if ( v161 )
                    j_j___libc_free_0(v161, (char *)j - (char *)v161);
LABEL_93:
                  v68 = v170;
                  v69 = v169;
                  if ( (int)(-991146299 * ((v170 - v169) >> 3)) > 1 && (unsigned __int64)(v170 - v169) > 0x68 )
                  {
                    v70 = 1;
                    v71 = 104;
                    while ( 1 )
                    {
                      v72 = sub_19D6B50((__int64)&v69[v71 - 104], (__int64)&v69[v71]);
                      if ( v72 )
                        break;
                      v69 = v169;
                      ++v70;
                      v71 += 104;
                      if ( 0x4EC4EC4EC4EC4EC5LL * ((v170 - v169) >> 3) <= v70 )
                      {
                        v68 = v170;
                        goto LABEL_147;
                      }
                    }
                    v75 = v170;
                    v76 = v169;
                    v77 = v72;
                    while ( v75 != v76 )
                    {
                      v78 = *(_DWORD *)(v168 + 20) & 0xFFFFFFF;
                      if ( v78 )
                      {
                        v79 = 0;
                        v80 = 24LL * *(unsigned int *)(v168 + 56) + 8;
                        while ( 1 )
                        {
                          v81 = v168 - 24LL * v78;
                          if ( (*(_BYTE *)(v168 + 23) & 0x40) != 0 )
                            v81 = *(_QWORD *)(v168 - 8);
                          if ( *(_QWORD *)v76 == *(_QWORD *)(v81 + v80) )
                            break;
                          ++v79;
                          v80 += 8;
                          if ( v78 == v79 )
                            goto LABEL_158;
                        }
                      }
                      else
                      {
LABEL_158:
                        v79 = -1;
                      }
                      v76 += 104;
                      sub_15F5350(v168, v79, 0);
                    }
                    v82 = v169;
                    v83 = *(_QWORD *)v169;
                    v84 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v169 + 56LL) + 80LL);
                    if ( v84 )
                      v84 -= 24;
                    v85 = 0x4EC4EC4EC4EC4EC5LL * ((v170 - v169) >> 3);
                    if ( (unsigned __int64)(v170 - v169) <= 0x68 )
                    {
                      v92 = v172;
                      if ( v83 != v172 )
                        goto LABEL_117;
                    }
                    else
                    {
                      v86 = (__int64 *)(v169 + 104);
                      v87 = 1;
                      while ( *v86 != v84 )
                      {
                        ++v87;
                        v86 += 13;
                        if ( v87 >= 0x4EC4EC4EC4EC4EC5LL * ((v170 - v169) >> 3) )
                        {
                          v92 = v172;
                          if ( v83 != v172 )
                            goto LABEL_117;
                          goto LABEL_119;
                        }
                      }
                      v88 = *(_QWORD *)(v84 + 56);
                      LOWORD(v186[2]) = 257;
                      v152 = v88;
                      v160 = sub_157E9C0(v84);
                      v89 = (_QWORD *)sub_22077B0(64);
                      v90 = (__int64)v89;
                      if ( v89 )
                        sub_157FB60(v89, v160, (__int64)v186, v152, v84);
                      v91 = sub_1648A60(56, 1u);
                      if ( v91 )
                        sub_15F8590((__int64)v91, v84, v90);
                      v82 = v169;
                      v92 = v172;
                      v83 = *(_QWORD *)v169;
                      if ( v172 == *(_QWORD *)v169 )
                        goto LABEL_118;
LABEL_117:
                      sub_164D160(v92, v83, a3, a4, a5, a6, v73, v74, a9, a10);
                      v82 = v169;
                      v172 = *(_QWORD *)v169;
LABEL_118:
                      if ( (unsigned __int64)(v170 - v82) > 0x68 )
                      {
LABEL_119:
                        v153 = v77;
                        v93 = 1;
                        v94 = 104;
                        v95 = 1;
                        do
                        {
                          if ( (unsigned __int8)sub_19D6B50((__int64)&v82[v94 - 104], (__int64)&v82[v94]) )
                          {
                            ++v93;
                          }
                          else
                          {
                            v99 = v93;
                            v100 = v95 - v93;
                            v93 = 1;
                            sub_19D94E0(
                              (__int64)&v168,
                              (__int64 *)&v169[104 * v100],
                              v99,
                              *(_QWORD *)&v169[v94],
                              v168,
                              v157,
                              a3,
                              a4,
                              a5,
                              a6,
                              v97,
                              v98,
                              a9,
                              a10);
                          }
                          v82 = v169;
                          ++v95;
                          v94 += 104;
                          v96 = 0x4EC4EC4EC4EC4EC5LL * ((v170 - v169) >> 3);
                        }
                        while ( v96 > v95 );
                        v105 = v93;
                        v77 = v153;
                        sub_19D94E0(
                          (__int64)&v168,
                          (__int64 *)&v169[104 * (v96 - v105)],
                          v105,
                          0,
                          v168,
                          v157,
                          a3,
                          a4,
                          a5,
                          a6,
                          v97,
                          v98,
                          a9,
                          a10);
                        goto LABEL_146;
                      }
                      v85 = 0x4EC4EC4EC4EC4EC5LL * ((v170 - v82) >> 3);
                    }
                    sub_19D94E0(
                      (__int64)&v168,
                      (__int64 *)&v82[104 * v85 - 104],
                      1u,
                      0,
                      v168,
                      v157,
                      a3,
                      a4,
                      a5,
                      a6,
                      v73,
                      v74,
                      a9,
                      a10);
LABEL_146:
                    v159 = v77;
                    v68 = v170;
                    v69 = v169;
                  }
LABEL_147:
                  while ( v68 != v69 )
                  {
                    if ( *((_DWORD *)v69 + 22) > 0x40u )
                    {
                      v106 = *((_QWORD *)v69 + 10);
                      if ( v106 )
                        j_j___libc_free_0_0(v106);
                    }
                    if ( *((_DWORD *)v69 + 14) > 0x40u )
                    {
                      v107 = *((_QWORD *)v69 + 6);
                      if ( v107 )
                        j_j___libc_free_0_0(v107);
                    }
                    v69 += 104;
                  }
                  if ( v169 )
                    j_j___libc_free_0(v169, (char *)v171 - v169);
                  break;
                }
                goto LABEL_47;
              }
              goto LABEL_207;
            }
          }
LABEL_157:
          j_j___libc_free_0(v156, n);
        }
      }
    }
LABEL_18:
    v23 = *(_QWORD *)(v23 + 8);
  }
  while ( v23 != v24 );
  v11 = v158;
  if ( v159 )
  {
    memset(v186, 0, 0x70u);
    v101 = (void **)&v186[5];
    LODWORD(v186[3]) = 2;
    v186[1] = (__int64)&v186[5];
    v186[2] = (__int64)&v186[5];
    v186[8] = (__int64)&v186[12];
    v186[9] = (__int64)&v186[12];
    LODWORD(v186[10]) = 2;
  }
  else
  {
LABEL_126:
    v101 = (void **)&v186[5];
    v186[7] = 0;
    v186[3] = 0x100000002LL;
    v186[1] = (__int64)&v186[5];
    v186[2] = (__int64)&v186[5];
    v186[8] = (__int64)&v186[12];
    v186[9] = (__int64)&v186[12];
    v186[10] = 2;
    LODWORD(v186[11]) = 0;
    LODWORD(v186[4]) = 0;
    v186[5] = (__int64)&unk_4F9EE48;
    v186[0] = 1;
  }
  v102 = &v186[HIDWORD(v186[3]) + 5];
  if ( v102 != &v186[5] )
  {
LABEL_134:
    while ( *v101 != &unk_4F9EE48 )
    {
      if ( v102 == (__int64 *)++v101 )
        goto LABEL_128;
    }
    if ( v102 != (__int64 *)v101 )
    {
      while ( (unsigned __int64)*v101 >= 0xFFFFFFFFFFFFFFFELL )
      {
        if ( v102 == (__int64 *)++v101 )
          goto LABEL_128;
      }
      if ( v102 != (__int64 *)v101 )
      {
        v103 = v186[9];
        v104 = v186[8];
        goto LABEL_141;
      }
    }
  }
LABEL_128:
  v103 = v186[9];
  v104 = v186[8];
  v11 = 1;
LABEL_141:
  if ( v104 != v103 )
    _libc_free(v103);
  if ( v186[2] != v186[1] )
    _libc_free(v186[2]);
  return v11;
}
