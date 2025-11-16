// Function: sub_D4F720
// Address: 0xd4f720
//
__int64 __fastcall sub_D4F720(__int64 a1, __int64 *a2)
{
  bool v3; // zf
  __int64 *v4; // rax
  __int64 *v5; // rsi
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rax
  int v8; // edx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  _QWORD *i; // rdx
  __int64 *v13; // rax
  __int64 **v14; // r13
  __int64 **v15; // r12
  __int64 *v16; // rbx
  int v17; // esi
  __int64 v18; // rdi
  int v19; // esi
  unsigned int v20; // ecx
  __int64 **v21; // rdx
  __int64 *v22; // r8
  __int64 *v23; // rdx
  __int64 *v24; // rcx
  __int64 *j; // rcx
  int v26; // esi
  int v27; // r14d
  __int64 *v28; // r11
  unsigned int v29; // edi
  __int64 *v30; // rcx
  __int64 v31; // r8
  _QWORD **v32; // r15
  char *v33; // rax
  char *v34; // rdx
  char *v35; // rsi
  __int64 **v36; // rdi
  __int64 **v37; // rdx
  __int64 **v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 *v41; // rcx
  __int64 *v42; // r10
  int v43; // r11d
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rax
  _BYTE *v48; // rsi
  _QWORD *v49; // r12
  int v50; // esi
  int v51; // edi
  __int64 v52; // rax
  _BYTE *v53; // rsi
  __int64 *v54; // rax
  _BYTE *v55; // rsi
  _BYTE *v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rdi
  __int64 v59; // rsi
  __int64 v61; // rsi
  __int64 v62; // rdx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rax
  __int64 *v66; // rbx
  __int64 *v67; // r13
  int v68; // ecx
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // r8
  _QWORD **v72; // r15
  _QWORD **v73; // rax
  _QWORD **v74; // r12
  __int64 v75; // r15
  __int64 v76; // rdx
  __int64 v77; // rdi
  unsigned int v78; // esi
  int v79; // r11d
  __int64 *v80; // r10
  unsigned int v81; // ecx
  __int64 *v82; // rax
  __int64 v83; // r8
  __int64 v84; // rsi
  int v85; // ecx
  __int64 v86; // rdi
  unsigned int v87; // esi
  unsigned int v88; // ecx
  __int64 *v89; // rax
  __int64 v90; // r8
  int v91; // eax
  int v92; // r9d
  __int64 v93; // r12
  __int64 v94; // rcx
  int v95; // eax
  int v96; // esi
  unsigned int v97; // edx
  __int64 *v98; // rax
  __int64 v99; // rdi
  _QWORD **v100; // rbx
  _QWORD **v101; // rdx
  __int64 *v102; // rbx
  __int64 v103; // rdx
  __int64 v104; // rax
  _BYTE *v105; // rsi
  __int64 v106; // rbx
  __int64 v107; // r12
  __int64 v108; // r9
  int v109; // r11d
  __int64 v110; // rdx
  __int64 v111; // rdi
  unsigned int v112; // ecx
  _QWORD *v113; // rax
  __int64 v114; // r8
  _DWORD *v115; // rax
  int v116; // eax
  int v117; // edx
  __int64 v118; // rdx
  int v119; // eax
  int v120; // edx
  int v121; // eax
  int v122; // r9d
  __int64 *v123; // rcx
  int v124; // edi
  __int64 v125; // r8
  __int64 v126; // r9
  int v127; // edi
  unsigned int v128; // edx
  __int64 *v129; // rax
  __int64 v130; // r11
  _BYTE *v131; // rsi
  _BYTE *v132; // rax
  __int64 v133; // rax
  __int64 v134; // r13
  __int64 *v135; // rbx
  __int64 v136; // r14
  unsigned int v137; // r12d
  int v138; // eax
  int v139; // r13d
  __int64 *v140; // rax
  int v141; // eax
  int v142; // r12d
  int v143; // eax
  int v144; // eax
  int v145; // eax
  __int64 v146; // rax
  int v147; // r8d
  int v148; // r9d
  char v150; // [rsp+17h] [rbp-4F9h]
  char v151; // [rsp+18h] [rbp-4F8h]
  __int64 v152; // [rsp+18h] [rbp-4F8h]
  __int64 v153; // [rsp+28h] [rbp-4E8h] BYREF
  __int64 v154[2]; // [rsp+30h] [rbp-4E0h] BYREF
  __int64 *v155; // [rsp+40h] [rbp-4D0h] BYREF
  __int64 v156; // [rsp+48h] [rbp-4C8h]
  _QWORD v157[2]; // [rsp+50h] [rbp-4C0h] BYREF
  _QWORD *v158; // [rsp+60h] [rbp-4B0h]
  __int64 v159; // [rsp+68h] [rbp-4A8h]
  unsigned int v160; // [rsp+70h] [rbp-4A0h]
  __int64 *v161; // [rsp+78h] [rbp-498h] BYREF
  __int64 *v162; // [rsp+80h] [rbp-490h]
  __int64 v163; // [rsp+88h] [rbp-488h]
  __int64 v164; // [rsp+90h] [rbp-480h] BYREF
  __int64 v165; // [rsp+98h] [rbp-478h]
  __int64 v166; // [rsp+A0h] [rbp-470h]
  unsigned int v167; // [rsp+A8h] [rbp-468h]
  char v168; // [rsp+B0h] [rbp-460h]
  __int64 *v169; // [rsp+C0h] [rbp-450h] BYREF
  _BYTE *v170; // [rsp+C8h] [rbp-448h] BYREF
  __int64 v171; // [rsp+D0h] [rbp-440h]
  _BYTE v172[328]; // [rsp+D8h] [rbp-438h] BYREF
  __int64 v173[44]; // [rsp+220h] [rbp-2F0h] BYREF
  __int64 *v174; // [rsp+380h] [rbp-190h] BYREF
  _BYTE *v175; // [rsp+388h] [rbp-188h] BYREF
  __int64 v176; // [rsp+390h] [rbp-180h]
  _BYTE v177[376]; // [rsp+398h] [rbp-178h] BYREF

  v3 = *a2 == 0;
  v4 = (__int64 *)a2[4];
  v5 = (__int64 *)a2[5];
  if ( !v3 )
  {
    v156 = a1;
    v157[1] = 0;
    v6 = (unsigned int)(v5 - v4);
    v155 = a2;
    v157[0] = a2;
    v7 = ((((((((v6 | (v6 >> 1)) >> 2) | v6 | (v6 >> 1)) >> 4) | ((v6 | (v6 >> 1)) >> 2) | v6 | (v6 >> 1)) >> 8)
         | ((((v6 | (v6 >> 1)) >> 2) | v6 | (v6 >> 1)) >> 4)
         | ((v6 | (v6 >> 1)) >> 2)
         | v6
         | (v6 >> 1)) >> 16)
       | ((((((v6 | (v6 >> 1)) >> 2) | v6 | (v6 >> 1)) >> 4) | ((v6 | (v6 >> 1)) >> 2) | v6 | (v6 >> 1)) >> 8)
       | ((((v6 | (v6 >> 1)) >> 2) | v6 | (v6 >> 1)) >> 4)
       | ((v6 | (v6 >> 1)) >> 2)
       | v6
       | (v6 >> 1);
    if ( (_DWORD)v7 == -1 )
    {
      v158 = 0;
      v159 = 0;
      v160 = 0;
    }
    else
    {
      v8 = v7 + 1;
      v9 = (((((((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
             | (4 * v8 / 3u + 1)
             | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 4)
           | (((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
           | (4 * v8 / 3u + 1)
           | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
           | (4 * v8 / 3u + 1)
           | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 4)
         | (((4 * v8 / 3u + 1) | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1)) >> 2)
         | (4 * v8 / 3u + 1)
         | ((unsigned __int64)(4 * v8 / 3u + 1) >> 1);
      v10 = ((v9 >> 16) | v9) + 1;
      v160 = v10;
      v11 = (_QWORD *)sub_C7D670(16 * v10, 8);
      v159 = 0;
      v158 = v11;
      for ( i = &v11[2 * v160]; i != v11; v11 += 2 )
      {
        if ( v11 )
          *v11 = -4096;
      }
      v6 = (unsigned int)((a2[5] - a2[4]) >> 3);
    }
    v161 = 0;
    v162 = 0;
    v163 = 0;
    sub_D4ACD0((__int64)&v161, v6);
    v13 = v155;
    v164 = 0;
    v165 = 0;
    v166 = 0;
    v167 = 0;
    v168 = 0;
    v14 = (__int64 **)v155[5];
    v15 = (__int64 **)v155[4];
    if ( !(unsigned int)(v14 - v15) )
    {
LABEL_10:
      if ( v15 != v14 )
      {
        while ( 1 )
        {
          v16 = *v15;
          v17 = *(_DWORD *)(v156 + 24);
          v18 = *(_QWORD *)(v156 + 8);
          if ( !v17 )
            break;
          v19 = v17 - 1;
          v20 = v19 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v21 = (__int64 **)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v16 != *v21 )
          {
            v120 = 1;
            while ( v22 != (__int64 *)-4096LL )
            {
              v148 = v120 + 1;
              v20 = v19 & (v120 + v20);
              v21 = (__int64 **)(v18 + 16LL * v20);
              v22 = *v21;
              if ( v16 == *v21 )
                goto LABEL_13;
              v120 = v148;
            }
            v173[0] = 0;
            v23 = 0;
            if ( v13 )
              goto LABEL_24;
            v23 = 0;
            goto LABEL_18;
          }
LABEL_13:
          v23 = v21[1];
          v173[0] = (__int64)v23;
          if ( v23 == v13 )
            goto LABEL_18;
          if ( v23 )
          {
            v24 = v23;
            do
            {
              v24 = (__int64 *)*v24;
              if ( v24 == v13 )
                goto LABEL_18;
            }
            while ( v24 );
          }
LABEL_24:
          v32 = (_QWORD **)*v13;
          if ( (__int64 *)*v13 != v23 )
          {
            do
            {
              v174 = v16;
              v33 = (char *)sub_D463D0(v32[4], (__int64)v32[5], (__int64 *)&v174);
              v34 = (char *)v32[5];
              v35 = v33 + 8;
              if ( v34 != v33 + 8 )
              {
                memmove(v33, v35, v34 - v35);
                v35 = (char *)v32[5];
              }
              v3 = *((_BYTE *)v32 + 84) == 0;
              v32[5] = v35 - 8;
              if ( v3 )
              {
                v54 = sub_C8CA60((__int64)(v32 + 7), (__int64)v174);
                if ( v54 )
                {
                  *v54 = -2;
                  ++*((_DWORD *)v32 + 20);
                  v32[7] = (_QWORD *)((char *)v32[7] + 1);
                }
              }
              else
              {
                v36 = (__int64 **)v32[8];
                v37 = &v36[*((unsigned int *)v32 + 19)];
                v38 = v36;
                if ( v36 != v37 )
                {
                  while ( v174 != *v38 )
                  {
                    if ( v37 == ++v38 )
                      goto LABEL_33;
                  }
                  v39 = (unsigned int)(*((_DWORD *)v32 + 19) - 1);
                  *((_DWORD *)v32 + 19) = v39;
                  *v38 = v36[v39];
                  v32[7] = (_QWORD *)((char *)v32[7] + 1);
                }
              }
LABEL_33:
              v32 = (_QWORD **)*v32;
            }
            while ( (_QWORD **)v173[0] != v32 );
            v13 = v155;
          }
          if ( v14 == ++v15 )
            goto LABEL_36;
        }
        v173[0] = 0;
        v23 = 0;
        if ( v13 )
          goto LABEL_24;
        v23 = 0;
LABEL_18:
        for ( j = (__int64 *)*v23; j != v13; j = (__int64 *)*j )
        {
          v173[0] = (__int64)j;
          v23 = j;
        }
        v26 = v167;
        if ( v167 )
        {
          v27 = 1;
          v28 = 0;
          v29 = (v167 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v30 = (__int64 *)(v165 + 16LL * v29);
          v31 = *v30;
          if ( v23 == (__int64 *)*v30 )
          {
LABEL_22:
            v23 = (__int64 *)v30[1];
LABEL_23:
            v173[0] = (__int64)v23;
            goto LABEL_24;
          }
          while ( v31 != -4096 )
          {
            if ( v31 == -8192 && !v28 )
              v28 = v30;
            v29 = (v167 - 1) & (v27 + v29);
            v30 = (__int64 *)(v165 + 16LL * v29);
            v31 = *v30;
            if ( (__int64 *)*v30 == v23 )
              goto LABEL_22;
            ++v27;
          }
          if ( v28 )
            v30 = v28;
          ++v164;
          v145 = v166 + 1;
          v174 = v30;
          if ( 4 * ((int)v166 + 1) < 3 * v167 )
          {
            if ( v167 - HIDWORD(v166) - v145 > v167 >> 3 )
            {
LABEL_205:
              LODWORD(v166) = v145;
              if ( *v30 != -4096 )
                --HIDWORD(v166);
              v146 = v173[0];
              v23 = 0;
              v30[1] = 0;
              *v30 = v146;
              v13 = v155;
              goto LABEL_23;
            }
LABEL_210:
            sub_D4EA50((__int64)&v164, v26);
            sub_D4C320((__int64)&v164, v173, &v174);
            v30 = v174;
            v145 = v166 + 1;
            goto LABEL_205;
          }
        }
        else
        {
          ++v164;
          v174 = 0;
        }
        v26 = 2 * v167;
        goto LABEL_210;
      }
LABEL_36:
      v40 = v13[2];
      if ( v40 == v13[1] )
      {
LABEL_60:
        v55 = *(_BYTE **)(*a2 + 8);
        if ( *(__int64 **)v55 != a2 )
        {
          v56 = v55 + 8;
          do
          {
            v55 = v56;
            v56 += 8;
          }
          while ( *((__int64 **)v56 - 1) != a2 );
        }
        sub_D4C9B0(*a2 + 8, v55);
        v57 = v167;
        v58 = v165;
        *a2 = 0;
        sub_C7D6A0(v58, 16 * v57, 8);
        if ( v161 )
          j_j___libc_free_0(v161, v163 - (_QWORD)v161);
        v59 = 16LL * v160;
        sub_C7D6A0((__int64)v158, v59, 8);
        return sub_D47BB0((__int64)a2, v59);
      }
      while ( 1 )
      {
        v173[0] = *(_QWORD *)(v40 - 8);
        v49 = *(_QWORD **)(v40 - 8);
        sub_D4C9B0((__int64)(v13 + 1), (_BYTE *)(v40 - 8));
        *v49 = 0;
        v50 = v167;
        if ( !v167 )
          break;
        v41 = (__int64 *)v173[0];
        v42 = 0;
        v43 = 1;
        v44 = (v167 - 1) & ((LODWORD(v173[0]) >> 9) ^ (LODWORD(v173[0]) >> 4));
        v45 = (__int64 *)(v165 + 16LL * v44);
        v46 = *v45;
        if ( v173[0] != *v45 )
        {
          while ( v46 != -4096 )
          {
            if ( !v42 && v46 == -8192 )
              v42 = v45;
            v44 = (v167 - 1) & (v43 + v44);
            v45 = (__int64 *)(v165 + 16LL * v44);
            v46 = *v45;
            if ( v173[0] == *v45 )
              goto LABEL_39;
            ++v43;
          }
          if ( v42 )
            v45 = v42;
          ++v164;
          v51 = v166 + 1;
          v174 = v45;
          if ( 4 * ((int)v166 + 1) < 3 * v167 )
          {
            if ( v167 - HIDWORD(v166) - v51 <= v167 >> 3 )
            {
LABEL_48:
              sub_D4EA50((__int64)&v164, v50);
              sub_D4C320((__int64)&v164, v173, &v174);
              v41 = (__int64 *)v173[0];
              v51 = v166 + 1;
              v45 = v174;
            }
            LODWORD(v166) = v51;
            if ( *v45 != -4096 )
              --HIDWORD(v166);
            *v45 = (__int64)v41;
            v45[1] = 0;
            v41 = (__int64 *)v173[0];
            goto LABEL_52;
          }
LABEL_47:
          v50 = 2 * v167;
          goto LABEL_48;
        }
LABEL_39:
        v47 = v45[1];
        if ( v47 )
        {
          v174 = (__int64 *)v173[0];
          *(_QWORD *)v173[0] = v47;
          v48 = *(_BYTE **)(v47 + 16);
          if ( v48 == *(_BYTE **)(v47 + 24) )
          {
            sub_D4C7F0(v47 + 8, v48, &v174);
          }
          else
          {
            if ( v48 )
            {
              *(_QWORD *)v48 = v174;
              v48 = *(_BYTE **)(v47 + 16);
            }
            *(_QWORD *)(v47 + 16) = v48 + 8;
          }
          goto LABEL_44;
        }
LABEL_52:
        v52 = v156;
        v174 = v41;
        v53 = *(_BYTE **)(v156 + 40);
        if ( v53 == *(_BYTE **)(v156 + 48) )
        {
          sub_D4C7F0(v156 + 32, v53, &v174);
        }
        else
        {
          if ( v53 )
          {
            *(_QWORD *)v53 = v41;
            v53 = *(_BYTE **)(v52 + 40);
          }
          *(_QWORD *)(v52 + 40) = v53 + 8;
        }
LABEL_44:
        v13 = v155;
        v40 = v155[2];
        if ( v40 == v155[1] )
          goto LABEL_60;
      }
      ++v164;
      v174 = 0;
      goto LABEL_47;
    }
    v154[0] = (__int64)v157;
    v154[1] = v156;
    v61 = **(_QWORD **)(v157[0] + 32LL);
    sub_D4E110(&v174, v61, v154);
    v170 = v172;
    v169 = v174;
    v171 = 0x800000000LL;
    if ( (_DWORD)v176 )
    {
      v61 = (__int64)&v175;
      sub_D4C550((__int64)&v170, (__int64)&v175, v62, (unsigned int)v176, v63, v64);
    }
    if ( v175 != v177 )
      _libc_free(v175, v61);
    v174 = v154;
    v173[0] = (__int64)v154;
    v175 = v177;
    v176 = 0x800000000LL;
LABEL_72:
    v65 = (unsigned int)v171;
    while ( 1 )
    {
      if ( !v65 )
      {
        if ( v170 != v172 )
          _libc_free(v170, v61);
        v150 = v168;
        if ( !v168 )
        {
LABEL_91:
          v13 = v155;
          v15 = (__int64 **)v155[4];
          v14 = (__int64 **)v155[5];
          goto LABEL_10;
        }
        while ( 2 )
        {
          v66 = v161;
          v67 = v162;
          if ( v161 == v162 )
            goto LABEL_91;
          v151 = 0;
LABEL_88:
          v84 = *v66;
          v85 = *(_DWORD *)(v156 + 24);
          v86 = *(_QWORD *)(v156 + 8);
          if ( v85 )
          {
            v68 = v85 - 1;
            v69 = v68 & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
            v70 = (__int64 *)(v86 + 16LL * v69);
            v71 = *v70;
            if ( v84 != *v70 )
            {
              v91 = 1;
              while ( v71 != -4096 )
              {
                v92 = v91 + 1;
                v69 = v68 & (v91 + v69);
                v70 = (__int64 *)(v86 + 16LL * v69);
                v71 = *v70;
                if ( v84 == *v70 )
                  goto LABEL_80;
                v91 = v92;
              }
              goto LABEL_89;
            }
LABEL_80:
            v72 = (_QWORD **)v70[1];
          }
          else
          {
LABEL_89:
            v72 = 0;
          }
          v73 = sub_D4ED70((__int64)&v155, v84, v72);
          v74 = v73;
          if ( v73 == v72 )
            goto LABEL_87;
          v75 = v156;
          v76 = *v66;
          v173[0] = *v66;
          v77 = *(_QWORD *)(v156 + 8);
          v78 = *(_DWORD *)(v156 + 24);
          if ( !v73 )
          {
            if ( v78 )
            {
              v87 = v78 - 1;
              v88 = v87 & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
              v89 = (__int64 *)(v77 + 16LL * v88);
              v90 = *v89;
              if ( v76 == *v89 )
              {
LABEL_94:
                *v89 = -8192;
                --*(_DWORD *)(v75 + 16);
                ++*(_DWORD *)(v75 + 20);
              }
              else
              {
                v121 = 1;
                while ( v90 != -4096 )
                {
                  v122 = v121 + 1;
                  v88 = v87 & (v121 + v88);
                  v89 = (__int64 *)(v77 + 16LL * v88);
                  v90 = *v89;
                  if ( v76 == *v89 )
                    goto LABEL_94;
                  v121 = v122;
                }
              }
            }
            goto LABEL_86;
          }
          if ( v78 )
          {
            v79 = 1;
            v80 = 0;
            v81 = (v78 - 1) & (((unsigned int)v76 >> 9) ^ ((unsigned int)v76 >> 4));
            v82 = (__int64 *)(v77 + 16LL * v81);
            v83 = *v82;
            if ( v76 == *v82 )
              goto LABEL_85;
            while ( v83 != -4096 )
            {
              if ( v83 == -8192 && !v80 )
                v80 = v82;
              v81 = (v78 - 1) & (v79 + v81);
              v82 = (__int64 *)(v77 + 16LL * v81);
              v83 = *v82;
              if ( v76 == *v82 )
                goto LABEL_85;
              ++v79;
            }
            if ( !v80 )
              v80 = v82;
            v174 = v80;
            v116 = *(_DWORD *)(v156 + 16);
            ++*(_QWORD *)v156;
            v117 = v116 + 1;
            if ( 4 * (v116 + 1) < 3 * v78 )
            {
              if ( v78 - *(_DWORD *)(v75 + 20) - v117 > v78 >> 3 )
              {
LABEL_123:
                *(_DWORD *)(v75 + 16) = v117;
                v82 = v174;
                if ( *v174 != -4096 )
                  --*(_DWORD *)(v75 + 20);
                v118 = v173[0];
                v82[1] = 0;
                *v82 = v118;
LABEL_85:
                v82[1] = (__int64)v74;
LABEL_86:
                v151 = v150;
LABEL_87:
                if ( v67 == ++v66 )
                {
                  if ( !v151 )
                    goto LABEL_91;
                  continue;
                }
                goto LABEL_88;
              }
LABEL_128:
              sub_D4F150(v75, v78);
              sub_D4C730(v75, v173, &v174);
              v117 = *(_DWORD *)(v75 + 16) + 1;
              goto LABEL_123;
            }
          }
          else
          {
            v174 = 0;
            ++*(_QWORD *)v156;
          }
          break;
        }
        v78 *= 2;
        goto LABEL_128;
      }
      v93 = *(_QWORD *)&v170[40 * v65 - 8];
      v94 = *(_QWORD *)(v156 + 8);
      v95 = *(_DWORD *)(v156 + 24);
      if ( v95 )
      {
        v96 = v95 - 1;
        v97 = (v95 - 1) & (((unsigned int)v93 >> 9) ^ ((unsigned int)v93 >> 4));
        v98 = (__int64 *)(v94 + 16LL * v97);
        v99 = *v98;
        if ( v93 == *v98 )
        {
LABEL_101:
          v100 = (_QWORD **)v98[1];
          goto LABEL_102;
        }
        v119 = 1;
        while ( v99 != -4096 )
        {
          v147 = v119 + 1;
          v97 = v96 & (v119 + v97);
          v98 = (__int64 *)(v94 + 16LL * v97);
          v99 = *v98;
          if ( v93 == *v98 )
            goto LABEL_101;
          v119 = v147;
        }
      }
      v100 = 0;
LABEL_102:
      v101 = sub_D4ED70((__int64)&v155, v93, v100);
      if ( v101 != v100 )
        sub_D4F540(v156, v93, (__int64)v101);
      v102 = v169;
      v103 = *(_QWORD *)&v170[40 * (unsigned int)v171 - 8];
      v153 = v103;
      v104 = *v169;
      v105 = *(_BYTE **)(*v169 + 48);
      if ( v105 == *(_BYTE **)(*v169 + 56) )
      {
        sub_9319A0(v104 + 40, v105, &v153);
      }
      else
      {
        if ( v105 )
        {
          *(_QWORD *)v105 = v103;
          v105 = *(_BYTE **)(v104 + 48);
        }
        *(_QWORD *)(v104 + 48) = v105 + 8;
      }
      v106 = *v102;
      v61 = *(unsigned int *)(v106 + 32);
      v107 = (__int64)(*(_QWORD *)(v106 + 48) - *(_QWORD *)(v106 + 40)) >> 3;
      if ( !(_DWORD)v61 )
      {
        v173[0] = 0;
        ++*(_QWORD *)(v106 + 8);
        goto LABEL_193;
      }
      v108 = *(_QWORD *)(v106 + 16);
      v109 = 1;
      v110 = 0;
      v111 = v153;
      v112 = (v61 - 1) & (((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4));
      v113 = (_QWORD *)(v108 + 16LL * v112);
      v114 = *v113;
      if ( v153 != *v113 )
      {
        while ( v114 != -4096 )
        {
          if ( v114 == -8192 && !v110 )
            v110 = (__int64)v113;
          v112 = (v61 - 1) & (v109 + v112);
          v113 = (_QWORD *)(v108 + 16LL * v112);
          v114 = *v113;
          if ( v153 == *v113 )
            goto LABEL_110;
          ++v109;
        }
        if ( !v110 )
          v110 = (__int64)v113;
        v173[0] = v110;
        v143 = *(_DWORD *)(v106 + 24);
        ++*(_QWORD *)(v106 + 8);
        v144 = v143 + 1;
        if ( 4 * v144 >= (unsigned int)(3 * v61) )
        {
LABEL_193:
          LODWORD(v61) = 2 * v61;
        }
        else if ( (int)v61 - *(_DWORD *)(v106 + 28) - v144 > (unsigned int)v61 >> 3 )
        {
LABEL_189:
          *(_DWORD *)(v106 + 24) = v144;
          if ( *(_QWORD *)v110 != -4096 )
            --*(_DWORD *)(v106 + 28);
          *(_QWORD *)v110 = v111;
          v115 = (_DWORD *)(v110 + 8);
          *(_DWORD *)(v110 + 8) = 0;
          goto LABEL_111;
        }
        sub_B23080(v106 + 8, v61);
        v61 = (__int64)&v153;
        sub_B1C700(v106 + 8, &v153, v173);
        v111 = v153;
        v110 = v173[0];
        v144 = *(_DWORD *)(v106 + 24) + 1;
        goto LABEL_189;
      }
LABEL_110:
      v115 = v113 + 1;
LABEL_111:
      *v115 = v107;
      v3 = (_DWORD)v171 == 1;
      v65 = (unsigned int)(v171 - 1);
      LODWORD(v171) = v171 - 1;
      if ( !v3 )
      {
        sub_D4DD40(&v169);
        goto LABEL_72;
      }
    }
  }
  v123 = v4;
  if ( v5 != v4 )
  {
    while ( 1 )
    {
      v124 = *(_DWORD *)(a1 + 24);
      v125 = *v123;
      v126 = *(_QWORD *)(a1 + 8);
      if ( !v124 )
        goto LABEL_154;
      v127 = v124 - 1;
      v128 = v127 & (((unsigned int)v125 >> 9) ^ ((unsigned int)v125 >> 4));
      v129 = (__int64 *)(v126 + 16LL * v128);
      v130 = *v129;
      if ( v125 != *v129 )
      {
        v136 = *v129;
        v137 = v127 & (((unsigned int)v125 >> 9) ^ ((unsigned int)v125 >> 4));
        v138 = 1;
        while ( v136 != -4096 )
        {
          v139 = v138 + 1;
          v137 = v127 & (v138 + v137);
          v140 = (__int64 *)(v126 + 16LL * v137);
          v136 = *v140;
          if ( v125 == *v140 )
          {
            if ( (__int64 *)v140[1] == a2 )
            {
              v141 = 1;
              while ( v130 != -4096 )
              {
                v142 = v141 + 1;
                v128 = v127 & (v141 + v128);
                v129 = (__int64 *)(v126 + 16LL * v128);
                v130 = *v129;
                if ( v125 == *v129 )
                  goto LABEL_158;
                v141 = v142;
              }
            }
            goto LABEL_154;
          }
          v138 = v139;
        }
        goto LABEL_154;
      }
      if ( (__int64 *)v129[1] == a2 )
      {
LABEL_158:
        ++v123;
        *v129 = -8192;
        --*(_DWORD *)(a1 + 16);
        ++*(_DWORD *)(a1 + 20);
        if ( v5 == v123 )
          break;
      }
      else
      {
LABEL_154:
        if ( v5 == ++v123 )
          break;
      }
    }
  }
  v131 = *(_BYTE **)(a1 + 32);
  if ( *(__int64 **)v131 != a2 )
  {
    v132 = v131 + 8;
    do
    {
      v131 = v132;
      v132 += 8;
    }
    while ( *((__int64 **)v132 - 1) != a2 );
  }
  sub_D4C9B0(a1 + 32, v131);
  v133 = a2[2];
  if ( v133 != a2[1] )
  {
    v152 = a1 + 32;
    v134 = a1;
    do
    {
      while ( 1 )
      {
        v135 = *(__int64 **)(v133 - 8);
        sub_D4C9B0((__int64)(a2 + 1), (_BYTE *)(v133 - 8));
        *v135 = 0;
        v131 = *(_BYTE **)(v134 + 40);
        v174 = v135;
        if ( v131 != *(_BYTE **)(v134 + 48) )
          break;
        sub_D4C7F0(v152, v131, &v174);
        v133 = a2[2];
        if ( v133 == a2[1] )
          return sub_D47BB0((__int64)a2, (__int64)v131);
      }
      if ( v131 )
      {
        *(_QWORD *)v131 = v135;
        v131 = *(_BYTE **)(v134 + 40);
      }
      v131 += 8;
      *(_QWORD *)(v134 + 40) = v131;
      v133 = a2[2];
    }
    while ( v133 != a2[1] );
  }
  return sub_D47BB0((__int64)a2, (__int64)v131);
}
