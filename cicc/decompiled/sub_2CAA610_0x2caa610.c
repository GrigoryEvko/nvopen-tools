// Function: sub_2CAA610
// Address: 0x2caa610
//
__int64 __fastcall sub_2CAA610(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned int v7; // edx
  __int64 *v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // r14
  char v14; // al
  _QWORD *v15; // r12
  __int64 *v16; // r13
  __int64 *v17; // r14
  bool v18; // al
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  unsigned int v28; // edi
  __int64 *v29; // rdx
  __int64 v30; // r10
  __int64 *v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r12
  unsigned __int64 v39; // rdi
  int v40; // ecx
  __int64 v41; // rax
  _QWORD *v42; // rdi
  _QWORD *i; // rax
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // r12
  __int64 *v48; // r13
  __int64 *v49; // rax
  __int64 v50; // rax
  int v51; // edi
  int v52; // r10d
  __int64 v53; // rdi
  int v54; // r14d
  unsigned int v55; // eax
  unsigned __int64 v56; // rcx
  unsigned __int64 v57; // rdx
  __int64 v58; // r9
  unsigned int v59; // esi
  __int64 v60; // r14
  __int64 v61; // r11
  __int64 v62; // rdx
  unsigned int v63; // eax
  _QWORD *v64; // rdi
  __int64 v65; // r9
  int v66; // eax
  int v67; // eax
  __int64 *v68; // rax
  int v69; // edx
  _QWORD *v70; // rax
  _BYTE *v71; // rsi
  unsigned __int8 v72; // dl
  _QWORD *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // r12
  unsigned __int64 v76; // rdx
  const char *v77; // r14
  size_t v78; // r13
  __int64 v79; // rax
  _QWORD *v80; // rdx
  unsigned __int8 *v81; // rsi
  size_t v82; // rdx
  __int64 v83; // rdi
  _BYTE *v84; // rax
  _QWORD *v85; // r12
  _QWORD *v86; // rax
  _QWORD *v87; // rbx
  unsigned __int64 *v88; // r13
  unsigned int v89; // edx
  _QWORD *v90; // r9
  __int64 v91; // rsi
  unsigned int v92; // ecx
  unsigned int v93; // edx
  int v94; // edx
  unsigned __int64 v95; // rax
  int v96; // ebx
  __int64 v97; // r12
  _QWORD *v98; // rax
  _QWORD *j; // rdx
  _BYTE *v100; // rdx
  int v101; // r12d
  __int64 v102; // rbx
  int v103; // r14d
  unsigned int v104; // eax
  unsigned __int64 v105; // r8
  unsigned __int64 v106; // rcx
  __int64 v107; // r9
  int v108; // esi
  __int64 v109; // rbx
  __int64 *v110; // r13
  int v111; // edx
  __int64 v112; // rax
  unsigned __int64 *v113; // r13
  unsigned int v114; // eax
  __int64 v115; // rbx
  __int64 v116; // r12
  unsigned __int64 v117; // rdi
  _QWORD *v118; // rax
  _QWORD *v119; // r12
  _QWORD *v120; // rbx
  __int64 v121; // r13
  void *v122; // rax
  unsigned __int64 v123; // rdx
  const char *v124; // r15
  size_t v125; // r14
  __int64 v126; // rax
  _QWORD *v127; // rdx
  unsigned __int8 *v128; // rsi
  size_t v129; // rdx
  __int64 v130; // rax
  _WORD *v131; // rdx
  _QWORD *v132; // rdi
  int v133; // ecx
  unsigned __int8 v134; // dl
  unsigned __int64 *v135; // r13
  int v136; // r9d
  _QWORD *v137; // rdi
  _QWORD *v138; // rsi
  unsigned __int64 v139; // [rsp+0h] [rbp-260h]
  __int64 v141; // [rsp+18h] [rbp-248h]
  int v142; // [rsp+20h] [rbp-240h]
  __int64 v143; // [rsp+20h] [rbp-240h]
  __int64 v144; // [rsp+28h] [rbp-238h]
  unsigned __int8 v146; // [rsp+36h] [rbp-22Ah]
  __int64 v149; // [rsp+40h] [rbp-220h]
  __int64 v150; // [rsp+40h] [rbp-220h]
  _QWORD *v151; // [rsp+40h] [rbp-220h]
  __int64 v152; // [rsp+40h] [rbp-220h]
  __int64 v153; // [rsp+48h] [rbp-218h]
  int v154; // [rsp+5Ch] [rbp-204h] BYREF
  __int64 v155; // [rsp+60h] [rbp-200h] BYREF
  __int64 v156; // [rsp+68h] [rbp-1F8h] BYREF
  _BYTE *v157; // [rsp+70h] [rbp-1F0h] BYREF
  _BYTE *v158; // [rsp+78h] [rbp-1E8h]
  _BYTE *v159; // [rsp+80h] [rbp-1E0h]
  __int64 v160; // [rsp+90h] [rbp-1D0h] BYREF
  _QWORD *v161; // [rsp+98h] [rbp-1C8h]
  __int64 v162; // [rsp+A0h] [rbp-1C0h]
  unsigned int v163; // [rsp+A8h] [rbp-1B8h]
  __int64 v164; // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 v165; // [rsp+B8h] [rbp-1A8h]
  __int64 v166; // [rsp+C0h] [rbp-1A0h]
  unsigned int v167; // [rsp+C8h] [rbp-198h]
  _QWORD *v168; // [rsp+D0h] [rbp-190h] BYREF
  __int64 v169; // [rsp+D8h] [rbp-188h]
  _QWORD v170[8]; // [rsp+E0h] [rbp-180h] BYREF
  _BYTE *v171; // [rsp+120h] [rbp-140h] BYREF
  __int64 v172; // [rsp+128h] [rbp-138h]
  _BYTE v173[304]; // [rsp+130h] [rbp-130h] BYREF

  v171 = v173;
  v172 = 0x2000000000LL;
  v160 = 0;
  v161 = 0;
  v162 = 0;
  v163 = 0;
  v157 = 0;
  v158 = 0;
  v159 = 0;
  v164 = 0;
  v165 = 0;
  v166 = 0;
  v167 = 0;
  v154 = 0;
  if ( !dword_5012828 )
    goto LABEL_2;
  v121 = (__int64)sub_CB72A0();
  v122 = *(void **)(v121 + 32);
  if ( *(_QWORD *)(v121 + 24) - (_QWORD)v122 <= 0xCu )
  {
    v121 = sub_CB6200(v121, "\n\nProcessing ", 0xDu);
  }
  else
  {
    qmemcpy(v122, "\n\nProcessing ", 13);
    *(_QWORD *)(v121 + 32) += 13LL;
  }
  v124 = sub_BD5D20(a2);
  v125 = v123;
  if ( v124 )
  {
    v156 = v123;
    v126 = v123;
    v168 = v170;
    if ( v123 > 0xF )
    {
      v168 = (_QWORD *)sub_22409D0((__int64)&v168, (unsigned __int64 *)&v156, 0);
      v137 = v168;
      v170[0] = v156;
    }
    else
    {
      if ( v123 == 1 )
      {
        LOBYTE(v170[0]) = *v124;
        v127 = v170;
LABEL_194:
        v169 = v126;
        *((_BYTE *)v127 + v126) = 0;
        v128 = (unsigned __int8 *)v168;
        v129 = v169;
        goto LABEL_195;
      }
      if ( !v123 )
      {
        v127 = v170;
        goto LABEL_194;
      }
      v137 = v170;
    }
    memcpy(v137, v124, v125);
    v126 = v156;
    v127 = v168;
    goto LABEL_194;
  }
  LOBYTE(v170[0]) = 0;
  v129 = 0;
  v168 = v170;
  v128 = (unsigned __int8 *)v170;
  v169 = 0;
LABEL_195:
  v130 = sub_CB6200(v121, v128, v129);
  v131 = *(_WORD **)(v130 + 32);
  if ( *(_QWORD *)(v130 + 24) - (_QWORD)v131 <= 1u )
  {
    sub_CB6200(v130, (unsigned __int8 *)"\n\n", 2u);
  }
  else
  {
    *v131 = 2570;
    *(_QWORD *)(v130 + 32) += 2LL;
  }
  if ( v168 != v170 )
    j_j___libc_free_0((unsigned __int64)v168);
LABEL_2:
  v146 = 0;
  v141 = a2 + 72;
  v144 = *(_QWORD *)(a2 + 80);
  if ( v144 == a2 + 72 )
    goto LABEL_49;
  do
  {
    v5 = v144 - 24;
    if ( !v144 )
      v5 = 0;
    v155 = v5;
    if ( !a4 )
    {
      if ( dword_5012828 <= 1 )
        goto LABEL_7;
      v73 = sub_CB72A0();
      v74 = v73[4];
      v75 = (__int64)v73;
      if ( (unsigned __int64)(v73[3] - v74) <= 5 )
      {
        v75 = sub_CB6200((__int64)v73, "Block ", 6u);
      }
      else
      {
        *(_DWORD *)v74 = 1668246594;
        *(_WORD *)(v74 + 4) = 8299;
        v73[4] += 6LL;
      }
      v77 = sub_BD5D20(v155);
      v78 = v76;
      if ( !v77 )
      {
        LOBYTE(v170[0]) = 0;
        v82 = 0;
        v168 = v170;
        v81 = (unsigned __int8 *)v170;
        v169 = 0;
        goto LABEL_108;
      }
      v156 = v76;
      v79 = v76;
      v168 = v170;
      if ( v76 > 0xF )
      {
        v168 = (_QWORD *)sub_22409D0((__int64)&v168, (unsigned __int64 *)&v156, 0);
        v132 = v168;
        v170[0] = v156;
      }
      else
      {
        if ( v76 == 1 )
        {
          LOBYTE(v170[0]) = *v77;
          v80 = v170;
LABEL_106:
          v169 = v79;
          *((_BYTE *)v80 + v79) = 0;
          v81 = (unsigned __int8 *)v168;
          v82 = v169;
LABEL_108:
          v83 = sub_CB6200(v75, v81, v82);
          v84 = *(_BYTE **)(v83 + 32);
          if ( *(_BYTE **)(v83 + 24) == v84 )
          {
            sub_CB6200(v83, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v84 = 10;
            ++*(_QWORD *)(v83 + 32);
          }
          if ( v168 != v170 )
            j_j___libc_free_0((unsigned __int64)v168);
          v5 = v155;
LABEL_7:
          v6 = *(_QWORD *)(v5 + 56);
          v153 = v5 + 48;
          if ( v5 + 48 != v6 )
            goto LABEL_12;
LABEL_32:
          v36 = (unsigned int)v172;
          if ( (unsigned int)v172 > 1 )
          {
            v72 = sub_2CA8B00(a1, (unsigned __int64 *)&v157, (__int64)&v160, *(_QWORD *)(a1 + 200), &v154);
            v36 = (unsigned int)v172;
            if ( v72 )
              v146 = v72;
          }
          if ( (_DWORD)v36 )
          {
            v37 = 8 * v36;
            v38 = 0;
            do
            {
              v39 = *(_QWORD *)&v171[v38];
              if ( v39 )
                j_j___libc_free_0(v39);
              v38 += 8;
            }
            while ( v37 != v38 );
          }
          LODWORD(v172) = 0;
          v40 = v162;
          if ( !(_DWORD)v162 )
          {
            ++v160;
            goto LABEL_40;
          }
          v42 = v161;
          v85 = &v161[2 * v163];
          if ( v161 == v85 )
            goto LABEL_118;
          v86 = v161;
          while ( 1 )
          {
            v87 = v86;
            if ( *v86 != -4096 && *v86 != -8192 )
              break;
            v86 += 2;
            if ( v85 == v86 )
              goto LABEL_118;
          }
          if ( v86 == v85 )
          {
LABEL_118:
            ++v160;
          }
          else
          {
            do
            {
              v88 = (unsigned __int64 *)v87[1];
              if ( v88 )
              {
                if ( (unsigned __int64 *)*v88 != v88 + 2 )
                  _libc_free(*v88);
                j_j___libc_free_0((unsigned __int64)v88);
              }
              v87 += 2;
              if ( v87 == v85 )
                break;
              while ( *v87 == -4096 || *v87 == -8192 )
              {
                v87 += 2;
                if ( v85 == v87 )
                  goto LABEL_128;
              }
            }
            while ( v85 != v87 );
LABEL_128:
            v40 = v162;
            ++v160;
            if ( !(_DWORD)v162 )
            {
LABEL_40:
              if ( HIDWORD(v162) )
              {
                v41 = v163;
                if ( v163 <= 0x40 )
                {
                  v42 = v161;
LABEL_43:
                  for ( i = &v42[2 * v41]; i != v42; v42 += 2 )
                    *v42 = -4096;
                  v162 = 0;
                  goto LABEL_46;
                }
                sub_C7D6A0((__int64)v161, 16LL * v163, 8);
                v161 = 0;
                v162 = 0;
                v163 = 0;
              }
LABEL_46:
              if ( v157 != v158 )
                v158 = v157;
              goto LABEL_48;
            }
            v42 = v161;
          }
          v89 = 4 * v40;
          v41 = v163;
          if ( (unsigned int)(4 * v40) < 0x40 )
            v89 = 64;
          if ( v89 >= v163 )
            goto LABEL_43;
          v90 = v42;
          v91 = 2LL * v163;
          v92 = v40 - 1;
          if ( v92 )
          {
            _BitScanReverse(&v93, v92);
            v94 = 1 << (33 - (v93 ^ 0x1F));
            if ( v94 < 64 )
              v94 = 64;
            if ( v94 == v163 )
            {
              v162 = 0;
              v138 = &v42[v91];
              do
              {
                if ( v90 )
                  *v90 = -4096;
                v90 += 2;
              }
              while ( v138 != v90 );
              goto LABEL_46;
            }
            v95 = (((4 * v94 / 3u + 1)
                  | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)
                  | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)) >> 4)
                | (4 * v94 / 3u + 1)
                | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)
                | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)
                | (((((4 * v94 / 3u + 1)
                    | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)
                    | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)) >> 4)
                  | (4 * v94 / 3u + 1)
                  | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)
                  | (((4 * v94 / 3u + 1) | ((unsigned __int64)(4 * v94 / 3u + 1) >> 1)) >> 2)) >> 8);
            v96 = (v95 | (v95 >> 16)) + 1;
            v97 = 16 * ((v95 | (v95 >> 16)) + 1);
          }
          else
          {
            v97 = 2048;
            v96 = 128;
          }
          sub_C7D6A0((__int64)v42, v91 * 8, 8);
          v163 = v96;
          v98 = (_QWORD *)sub_C7D670(v97, 8);
          v162 = 0;
          v161 = v98;
          for ( j = &v98[2 * v163]; j != v98; v98 += 2 )
          {
            if ( v98 )
              *v98 = -4096;
          }
          goto LABEL_46;
        }
        if ( !v76 )
        {
          v80 = v170;
          goto LABEL_106;
        }
        v132 = v170;
      }
      memcpy(v132, v77, v78);
      v79 = v156;
      v80 = v168;
      goto LABEL_106;
    }
    v6 = *(_QWORD *)(v5 + 56);
    v153 = v5 + 48;
    if ( v6 == v5 + 48 )
      goto LABEL_48;
    do
    {
      while ( 1 )
      {
LABEL_12:
        v10 = *(unsigned int *)(a3 + 24);
        v11 = v6;
        v12 = *(_QWORD *)(a3 + 8);
        v6 = *(_QWORD *)(v6 + 8);
        v13 = v11 - 24;
        if ( (_DWORD)v10 )
        {
          v7 = (v10 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v8 = (__int64 *)(v12 + 8LL * v7);
          v9 = *v8;
          if ( v13 == *v8 )
          {
LABEL_10:
            if ( v8 != (__int64 *)(v12 + 8 * v10) )
              goto LABEL_11;
          }
          else
          {
            v51 = 1;
            while ( v9 != -4096 )
            {
              v52 = v51 + 1;
              v7 = (v10 - 1) & (v51 + v7);
              v8 = (__int64 *)(v12 + 8LL * v7);
              v9 = *v8;
              if ( v13 == *v8 )
                goto LABEL_10;
              v51 = v52;
            }
          }
        }
        v14 = *(_BYTE *)(v11 - 24);
        if ( v14 != 61 )
        {
          if ( v14 != 62 )
            goto LABEL_11;
          v150 = *(_QWORD *)(v11 - 56);
          v15 = (_QWORD *)sub_22077B0(0x20u);
          if ( !v15 )
            goto LABEL_11;
          v15[2] = v13;
          v48 = *(__int64 **)(a1 + 184);
          v15[3] = v150;
          v49 = sub_DD8400((__int64)v48, v150);
          sub_2C95190(v15, (__int64)v49, v48);
          goto LABEL_21;
        }
        v149 = *(_QWORD *)(v11 - 56);
        v15 = (_QWORD *)sub_22077B0(0x20u);
        if ( v15 )
          break;
LABEL_11:
        if ( v6 == v153 )
          goto LABEL_31;
      }
      v15[2] = v13;
      v15[3] = v149;
      v16 = *(__int64 **)(a1 + 184);
      v17 = sub_DD8400((__int64)v16, v149);
      v18 = sub_D968A0((__int64)v17);
      v15[1] = v17;
      if ( v18 )
      {
        v50 = sub_D95540((__int64)v17);
        *v15 = sub_DA2C50((__int64)v16, v50, 0, 0);
      }
      else
      {
        v168 = v170;
        v169 = 0x800000000LL;
        v19 = sub_D95540((__int64)v17);
        *v15 = sub_DA2C50((__int64)v16, v19, 0, 0);
        sub_2C94930((__int64)v17, 0, (__int64)&v168, v16, (__int64)v15, v20);
        if ( (_DWORD)v169 )
        {
          if ( (unsigned int)v169 == 1 )
          {
            v24 = v168;
            v15[1] = *v168;
          }
          else
          {
            v23 = sub_DC7EB0(v16, (__int64)&v168, 0, 0);
            v24 = v168;
            v15[1] = v23;
          }
          if ( v24 != v170 )
LABEL_20:
            _libc_free((unsigned __int64)v24);
        }
        else
        {
          v15[1] = 0;
          v24 = v168;
          if ( v168 != v170 )
            goto LABEL_20;
        }
      }
LABEL_21:
      v25 = (unsigned int)v172;
      v26 = (unsigned int)v172 + 1LL;
      if ( v26 > HIDWORD(v172) )
      {
        sub_C8D5F0((__int64)&v171, v173, v26, 8u, v21, v22);
        v25 = (unsigned int)v172;
      }
      *(_QWORD *)&v171[8 * v25] = v15;
      v27 = v15[1];
      LODWORD(v172) = v172 + 1;
      v156 = v27;
      if ( !v27 )
      {
        j_j___libc_free_0((unsigned __int64)v15);
        goto LABEL_11;
      }
      if ( v163 )
      {
        v28 = (v163 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v29 = &v161[2 * v28];
        v30 = *v29;
        if ( v27 == *v29 )
        {
LABEL_26:
          if ( &v161[2 * v163] != v29 )
            goto LABEL_27;
        }
        else
        {
          v69 = 1;
          while ( v30 != -4096 )
          {
            v136 = v69 + 1;
            v28 = (v163 - 1) & (v69 + v28);
            v29 = &v161[2 * v28];
            v30 = *v29;
            if ( v27 == *v29 )
              goto LABEL_26;
            v69 = v136;
          }
        }
      }
      v70 = (_QWORD *)sub_22077B0(0x50u);
      if ( v70 )
      {
        *v70 = v70 + 2;
        v70[1] = 0x800000000LL;
      }
      *sub_2C931E0((__int64)&v160, &v156) = (__int64)v70;
      v71 = v158;
      if ( v158 == v159 )
      {
        sub_2C95B10((__int64)&v157, v158, &v156);
      }
      else
      {
        if ( v158 )
        {
          *(_QWORD *)v158 = v156;
          v71 = v158;
        }
        v158 = v71 + 8;
      }
LABEL_27:
      if ( !a4 )
        goto LABEL_28;
      if ( !v167 )
      {
        ++v164;
        v168 = 0;
        goto LABEL_232;
      }
      v53 = v156;
      v54 = 1;
      v55 = (v167 - 1) & (((unsigned int)v156 >> 9) ^ ((unsigned int)v156 >> 4));
      v56 = v165 + 40LL * v55;
      v57 = 0;
      v58 = *(_QWORD *)v56;
      if ( v156 != *(_QWORD *)v56 )
      {
        while ( v58 != -4096 )
        {
          if ( v58 == -8192 && !v57 )
            v57 = v56;
          v55 = (v167 - 1) & (v54 + v55);
          v56 = v165 + 40LL * v55;
          v58 = *(_QWORD *)v56;
          if ( v156 == *(_QWORD *)v56 )
            goto LABEL_74;
          ++v54;
        }
        if ( !v57 )
          v57 = v56;
        ++v164;
        v133 = v166 + 1;
        v168 = (_QWORD *)v57;
        if ( 4 * ((int)v166 + 1) >= 3 * v167 )
        {
LABEL_232:
          sub_2C9CAD0((__int64)&v164, 2 * v167);
        }
        else
        {
          if ( v167 - HIDWORD(v166) - v133 > v167 >> 3 )
          {
LABEL_213:
            LODWORD(v166) = v133;
            if ( *(_QWORD *)v57 != -4096 )
              --HIDWORD(v166);
            *(_QWORD *)v57 = v53;
            v61 = v57 + 8;
            *(_QWORD *)(v57 + 8) = 0;
            *(_QWORD *)(v57 + 16) = 0;
            *(_QWORD *)(v57 + 24) = 0;
            *(_DWORD *)(v57 + 32) = 0;
LABEL_201:
            v168 = 0;
            v59 = 0;
            ++*(_QWORD *)v61;
            goto LABEL_202;
          }
          sub_2C9CAD0((__int64)&v164, v167);
        }
        sub_2C96100((__int64)&v164, &v156, &v168);
        v53 = v156;
        v57 = (unsigned __int64)v168;
        v133 = v166 + 1;
        goto LABEL_213;
      }
LABEL_74:
      v59 = *(_DWORD *)(v56 + 32);
      v60 = *(_QWORD *)(v56 + 16);
      v61 = v56 + 8;
      if ( !v59 )
        goto LABEL_201;
      v62 = v155;
      v63 = (v59 - 1) & (((unsigned int)v155 >> 9) ^ ((unsigned int)v155 >> 4));
      v64 = (_QWORD *)(v60 + 8LL * v63);
      v65 = *v64;
      if ( v155 == *v64 )
        goto LABEL_28;
      v142 = 1;
      v151 = 0;
      v139 = v56;
      while ( v65 != -4096 )
      {
        if ( v65 == -8192 )
        {
          if ( v151 )
            v64 = v151;
          v151 = v64;
        }
        v63 = (v59 - 1) & (v142 + v63);
        v64 = (_QWORD *)(v60 + 8LL * v63);
        v65 = *v64;
        if ( v155 == *v64 )
          goto LABEL_28;
        ++v142;
      }
      if ( v151 )
        v64 = v151;
      v168 = v64;
      v66 = *(_DWORD *)(v56 + 24);
      ++*(_QWORD *)(v56 + 8);
      v67 = v66 + 1;
      if ( 4 * v67 < 3 * v59 )
      {
        if ( v59 - *(_DWORD *)(v56 + 28) - v67 <= v59 >> 3 )
        {
          v152 = v56 + 8;
          sub_CF28B0(v61, v59);
          sub_D6B660(v152, &v155, &v168);
          v62 = v155;
          v61 = v152;
          v67 = *(_DWORD *)(v139 + 24) + 1;
        }
        goto LABEL_83;
      }
LABEL_202:
      v143 = v61;
      sub_CF28B0(v61, 2 * v59);
      sub_D6B660(v143, &v155, &v168);
      v61 = v143;
      v62 = v155;
      v67 = *(_DWORD *)(v143 + 16) + 1;
LABEL_83:
      *(_DWORD *)(v61 + 16) = v67;
      v68 = v168;
      if ( *v168 != -4096 )
        --*(_DWORD *)(v61 + 20);
      *v68 = v62;
LABEL_28:
      v31 = sub_2C931E0((__int64)&v160, &v156);
      v34 = *v31;
      v35 = *(unsigned int *)(*v31 + 8);
      if ( v35 + 1 > (unsigned __int64)*(unsigned int *)(v34 + 12) )
      {
        sub_C8D5F0(v34, (const void *)(v34 + 16), v35 + 1, 8u, v32, v33);
        v35 = *(unsigned int *)(v34 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v34 + 8 * v35) = v15;
      ++*(_DWORD *)(v34 + 8);
    }
    while ( v6 != v153 );
LABEL_31:
    if ( !a4 )
      goto LABEL_32;
LABEL_48:
    v144 = *(_QWORD *)(v144 + 8);
  }
  while ( v141 != v144 );
LABEL_49:
  if ( !a4 )
    goto LABEL_50;
  v100 = v157;
  v101 = 0;
  v102 = 0;
  if ( v158 != v157 )
  {
    while ( 1 )
    {
      v108 = v167;
      v109 = 8 * v102;
      v110 = (__int64 *)&v100[v109];
      if ( !v167 )
        break;
      v103 = 1;
      v104 = (v167 - 1) & (((unsigned int)*v110 >> 9) ^ ((unsigned int)*v110 >> 4));
      v105 = v165 + 40LL * v104;
      v106 = 0;
      v107 = *(_QWORD *)v105;
      if ( *v110 != *(_QWORD *)v105 )
      {
        while ( v107 != -4096 )
        {
          if ( v107 == -8192 && !v106 )
            v106 = v105;
          v104 = (v167 - 1) & (v103 + v104);
          v105 = v165 + 40LL * v104;
          v107 = *(_QWORD *)v105;
          if ( *v110 == *(_QWORD *)v105 )
            goto LABEL_150;
          ++v103;
        }
        if ( !v106 )
          v106 = v105;
        ++v164;
        v111 = v166 + 1;
        v168 = (_QWORD *)v106;
        if ( 4 * ((int)v166 + 1) < 3 * v167 )
        {
          if ( v167 - HIDWORD(v166) - v111 > v167 >> 3 )
            goto LABEL_168;
          goto LABEL_155;
        }
LABEL_154:
        v108 = 2 * v167;
LABEL_155:
        sub_2C9CAD0((__int64)&v164, v108);
        sub_2C96100((__int64)&v164, v110, &v168);
        v106 = (unsigned __int64)v168;
        v111 = v166 + 1;
LABEL_168:
        LODWORD(v166) = v111;
        if ( *(_QWORD *)v106 != -4096 )
          --HIDWORD(v166);
        v112 = *v110;
        *(_QWORD *)(v106 + 8) = 0;
        *(_QWORD *)(v106 + 16) = 0;
        *(_QWORD *)(v106 + 24) = 0;
        *(_DWORD *)(v106 + 32) = 0;
        *(_QWORD *)v106 = v112;
        v100 = v157;
LABEL_171:
        v113 = (unsigned __int64 *)*sub_2C931E0((__int64)&v160, (__int64 *)&v100[v109]);
        if ( v113 )
        {
          if ( (unsigned __int64 *)*v113 != v113 + 2 )
            _libc_free(*v113);
          j_j___libc_free_0((unsigned __int64)v113);
        }
        *sub_2C931E0((__int64)&v160, (__int64 *)&v157[v109]) = 0;
        v100 = v157;
        goto LABEL_151;
      }
LABEL_150:
      if ( *(_DWORD *)(v105 + 24) <= 1u )
        goto LABEL_171;
LABEL_151:
      v102 = (unsigned int)++v101;
      if ( v101 == (v158 - v100) >> 3 )
        goto LABEL_176;
    }
    ++v164;
    v168 = 0;
    goto LABEL_154;
  }
LABEL_176:
  v114 = v172;
  if ( (unsigned int)v172 > 1 )
  {
    v134 = sub_2CA8B00(a1, (unsigned __int64 *)&v157, (__int64)&v160, *(_QWORD *)(a1 + 200), &v154);
    v114 = v172;
    if ( v134 )
      v146 = v134;
  }
  v115 = 0;
  v116 = 8LL * v114;
  if ( v114 )
  {
    do
    {
      v117 = *(_QWORD *)&v171[v115];
      if ( v117 )
        j_j___libc_free_0(v117);
      v115 += 8;
    }
    while ( v115 != v116 );
  }
  if ( (_DWORD)v162 )
  {
    v118 = v161;
    v119 = &v161[2 * v163];
    if ( v161 != v119 )
    {
      while ( 1 )
      {
        v120 = v118;
        if ( *v118 != -8192 && *v118 != -4096 )
          break;
        v118 += 2;
        if ( v119 == v118 )
          goto LABEL_50;
      }
      while ( v120 != v119 )
      {
        v135 = (unsigned __int64 *)v120[1];
        if ( v135 )
        {
          if ( (unsigned __int64 *)*v135 != v135 + 2 )
            _libc_free(*v135);
          j_j___libc_free_0((unsigned __int64)v135);
        }
        v120 += 2;
        if ( v120 == v119 )
          break;
        while ( *v120 == -8192 || *v120 == -4096 )
        {
          v120 += 2;
          if ( v119 == v120 )
            goto LABEL_50;
        }
      }
    }
  }
LABEL_50:
  v44 = v167;
  if ( v167 )
  {
    v45 = v165;
    v46 = v165 + 40LL * v167;
    do
    {
      if ( *(_QWORD *)v45 != -4096 && *(_QWORD *)v45 != -8192 )
        sub_C7D6A0(*(_QWORD *)(v45 + 16), 8LL * *(unsigned int *)(v45 + 32), 8);
      v45 += 40;
    }
    while ( v46 != v45 );
    v44 = v167;
  }
  sub_C7D6A0(v165, 40 * v44, 8);
  if ( v157 )
    j_j___libc_free_0((unsigned __int64)v157);
  sub_C7D6A0((__int64)v161, 16LL * v163, 8);
  if ( v171 != v173 )
    _libc_free((unsigned __int64)v171);
  return v146;
}
