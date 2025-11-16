// Function: sub_1A99D30
// Address: 0x1a99d30
//
__int64 __fastcall sub_1A99D30(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, _QWORD *a5, __int64 a6)
{
  __int64 *v8; // r15
  _QWORD **v9; // rcx
  unsigned int v10; // r8d
  unsigned int v11; // edx
  __int64 *v12; // rax
  _QWORD **v13; // rdi
  __int64 v14; // rax
  _QWORD *v15; // r12
  unsigned int v16; // r13d
  __int64 v17; // rdi
  _QWORD *v18; // rax
  int v19; // r9d
  _QWORD *v20; // rbx
  int v21; // esi
  int v22; // edi
  __int64 v23; // rbx
  __int64 v24; // r13
  unsigned int v25; // edx
  _QWORD **v26; // rsi
  _QWORD **v27; // rax
  _QWORD *v28; // r15
  __int64 v29; // rdi
  _QWORD *v30; // rax
  _QWORD *v31; // r12
  int v32; // esi
  _QWORD **v33; // rcx
  int v34; // r9d
  unsigned int v35; // r8d
  unsigned int v36; // edx
  __int64 *v37; // rax
  _QWORD **v38; // rdi
  __int64 v39; // rax
  __int64 v40; // rbx
  int v41; // r8d
  int v42; // r9d
  __int64 *v43; // r13
  __int64 *v44; // rbx
  unsigned int v45; // edi
  __int64 *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // r15
  _QWORD *v49; // rax
  __int64 v50; // r12
  __int64 v51; // r14
  __int64 v52; // r12
  __int64 v53; // rax
  int v54; // ecx
  __int64 *v55; // rdx
  __int64 v56; // r8
  int v58; // r10d
  __int64 *v59; // r8
  __int64 v60; // r15
  int v61; // r9d
  __int64 v62; // rsi
  int v63; // r8d
  int v64; // r11d
  __int64 *v65; // r10
  int v66; // edi
  __int64 v67; // rax
  __int64 v68; // rsi
  __int64 *v69; // r11
  __int64 *v70; // r11
  __int64 *v71; // rax
  __int64 *v72; // r12
  __int64 *v73; // rbx
  __int64 v74; // rsi
  __int64 v75; // r15
  __int64 v76; // rdx
  _WORD *i; // rax
  __int64 *v78; // rax
  __int64 *v79; // rdx
  __int64 v80; // r12
  __int64 *v81; // rbx
  _QWORD *v82; // r14
  unsigned int v83; // eax
  int v84; // r8d
  int v85; // r9d
  __int64 v86; // r15
  __int64 *v87; // r13
  unsigned int v88; // ebx
  __int64 v89; // r15
  _QWORD *v90; // rax
  int v91; // r8d
  int v92; // r9d
  unsigned __int64 *v93; // r13
  __int64 v94; // r15
  unsigned __int64 v95; // rax
  unsigned __int64 *v96; // r15
  unsigned __int64 v97; // rcx
  unsigned __int64 v98; // rdx
  unsigned __int64 *v99; // rax
  unsigned __int64 *v100; // rsi
  __int64 *v101; // rdi
  char *v102; // r13
  char *v103; // rax
  __int64 v104; // rsi
  char *v105; // rdx
  __int64 *v106; // r13
  _QWORD *v107; // rax
  __int64 v108; // r15
  __int64 v109; // rbx
  int v110; // ecx
  __int64 v111; // r15
  unsigned int v112; // eax
  __int64 v113; // rdx
  __int64 v114; // rax
  _QWORD *v115; // rax
  _QWORD *v116; // r13
  __int64 v117; // rdx
  _QWORD *v118; // r8
  __int64 v119; // rcx
  unsigned __int64 v120; // rdx
  __int64 v121; // rdx
  _QWORD *v122; // rax
  __int64 v123; // r13
  unsigned __int8 v124; // al
  __int64 v125; // rax
  _QWORD *v126; // rcx
  unsigned __int64 *v127; // rsi
  int v128; // r10d
  __int64 *v129; // r9
  __int64 *v130; // r12
  _QWORD *v131; // [rsp+20h] [rbp-900h]
  __int64 *v133; // [rsp+28h] [rbp-8F8h]
  _QWORD *v134; // [rsp+30h] [rbp-8F0h]
  __int64 *v135; // [rsp+30h] [rbp-8F0h]
  __int64 v136; // [rsp+38h] [rbp-8E8h]
  _QWORD *v137; // [rsp+40h] [rbp-8E0h]
  __int64 *v138; // [rsp+40h] [rbp-8E0h]
  unsigned __int64 v140; // [rsp+48h] [rbp-8D8h]
  __int64 *v141; // [rsp+50h] [rbp-8D0h]
  unsigned int v142; // [rsp+50h] [rbp-8D0h]
  __int64 v143; // [rsp+58h] [rbp-8C8h]
  __int64 v144; // [rsp+58h] [rbp-8C8h]
  __int64 v145; // [rsp+58h] [rbp-8C8h]
  unsigned __int64 *v146; // [rsp+58h] [rbp-8C8h]
  char *v147; // [rsp+58h] [rbp-8C8h]
  _QWORD *v148; // [rsp+58h] [rbp-8C8h]
  _QWORD **v149; // [rsp+68h] [rbp-8B8h] BYREF
  __int64 v150; // [rsp+70h] [rbp-8B0h] BYREF
  __int64 *v151; // [rsp+78h] [rbp-8A8h]
  __int64 v152; // [rsp+80h] [rbp-8A0h]
  unsigned int v153; // [rsp+88h] [rbp-898h]
  void *src; // [rsp+90h] [rbp-890h] BYREF
  __int64 v155; // [rsp+98h] [rbp-888h]
  _WORD v156[256]; // [rsp+A0h] [rbp-880h] BYREF
  void *v157; // [rsp+2A0h] [rbp-680h] BYREF
  __int64 v158; // [rsp+2A8h] [rbp-678h]
  _BYTE v159[1648]; // [rsp+2B0h] [rbp-670h] BYREF

  v157 = v159;
  v134 = a5;
  v137 = a5;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v158 = 0xC800000000LL;
  if ( a4 > 0xC8 )
    sub_16CD150((__int64)&v157, v159, a4, 8, (int)a5, a6);
  v8 = a3;
  v136 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v141 = &a3[a4];
  if ( v141 != a3 )
  {
    while ( 1 )
    {
      v149 = (_QWORD **)*v8;
      v15 = *v149;
      v16 = *(_DWORD *)(v136 + 4);
      v156[0] = 257;
      v17 = *(_QWORD *)(a1 + 80);
      if ( v17 )
        v17 -= 24;
      v143 = sub_157ED20(v17);
      v18 = sub_1648A60(64, 1u);
      v20 = v18;
      if ( v18 )
        sub_15F8BE0((__int64)v18, v15, v16, (__int64)&src, v143);
      v21 = v153;
      if ( !v153 )
        break;
      v9 = v149;
      v10 = (unsigned int)v151;
      v11 = (v153 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
      v12 = &v151[2 * v11];
      v13 = (_QWORD **)*v12;
      if ( v149 != (_QWORD **)*v12 )
      {
        v19 = 1;
        v69 = 0;
        while ( v13 != (_QWORD **)-8LL )
        {
          if ( v13 != (_QWORD **)-16LL || v69 )
            v12 = v69;
          v11 = (v153 - 1) & (v19 + v11);
          v130 = &v151[2 * v11];
          v13 = (_QWORD **)*v130;
          if ( v149 == (_QWORD **)*v130 )
          {
            v12 = &v151[2 * v11];
            goto LABEL_6;
          }
          ++v19;
          v69 = v12;
          v12 = &v151[2 * v11];
        }
        if ( v69 )
          v12 = v69;
        ++v150;
        v22 = v152 + 1;
        if ( 4 * ((int)v152 + 1) < 3 * v153 )
        {
          v10 = v153 >> 3;
          if ( v153 - HIDWORD(v152) - v22 <= v153 >> 3 )
            goto LABEL_16;
LABEL_98:
          LODWORD(v152) = v22;
          if ( *v12 != -8 )
            --HIDWORD(v152);
          *v12 = (__int64)v9;
          v12[1] = 0;
          goto LABEL_6;
        }
LABEL_15:
        v21 = 2 * v153;
LABEL_16:
        sub_176F940((__int64)&v150, v21);
        sub_176A9A0((__int64)&v150, (__int64 *)&v149, &src);
        v12 = (__int64 *)src;
        v9 = v149;
        v22 = v152 + 1;
        goto LABEL_98;
      }
LABEL_6:
      v12[1] = (__int64)v20;
      v14 = (unsigned int)v158;
      if ( (unsigned int)v158 >= HIDWORD(v158) )
      {
        sub_16CD150((__int64)&v157, v159, 0, 8, v10, v19);
        v14 = (unsigned int)v158;
      }
      ++v8;
      *((_QWORD *)v157 + v14) = v20;
      LODWORD(v158) = v158 + 1;
      if ( v141 == v8 )
        goto LABEL_17;
    }
    ++v150;
    goto LABEL_15;
  }
LABEL_17:
  v131 = &v134[23 * a6];
  if ( v131 != v134 )
  {
    while ( 1 )
    {
      v23 = v134[20];
      v24 = v134[21];
      if ( v23 != v24 )
        break;
LABEL_32:
      v134 += 23;
      if ( v131 == v134 )
      {
        while ( 1 )
        {
          v40 = v137[14];
          v145 = v40;
          sub_1A99820(*(_QWORD *)(v40 + 8), 0, (__int64)&v150);
          if ( *(_BYTE *)(v40 + 16) == 29 )
            sub_1A99820(*(_QWORD *)(v137[15] + 8LL), 0, (__int64)&v150);
          v43 = (__int64 *)v137[21];
          v44 = (__int64 *)v137[20];
          if ( v44 != v43 )
            break;
LABEL_49:
          if ( byte_4FB5EE8 )
          {
            v155 = 0x4000000000LL;
            src = v156;
            if ( !(_DWORD)v152 || (v71 = v151, v72 = &v151[2 * v153], v151 == v72) )
            {
LABEL_86:
              if ( *(_BYTE *)(v145 + 16) != 29 )
              {
LABEL_87:
                v67 = *(_QWORD *)(v145 + 32);
                if ( v67 == *(_QWORD *)(v145 + 40) + 40LL || !v67 )
                  v68 = 0;
                else
                  v68 = v67 - 24;
LABEL_90:
                sub_1A95180((__int64)&src, v68);
                if ( src != v156 )
                  _libc_free((unsigned __int64)src);
                goto LABEL_50;
              }
            }
            else
            {
              while ( 1 )
              {
                v73 = v71;
                if ( *v71 != -16 && *v71 != -8 )
                  break;
                v71 += 2;
                if ( v72 == v71 )
                  goto LABEL_109;
              }
              if ( v71 != v72 )
              {
                v75 = v71[1];
                v76 = 0;
                for ( i = v156; ; i = src )
                {
                  *(_QWORD *)&i[4 * v76] = v75;
                  v73 += 2;
                  v76 = (unsigned int)(v155 + 1);
                  LODWORD(v155) = v155 + 1;
                  if ( v73 == v72 )
                    break;
                  while ( *v73 == -16 || *v73 == -8 )
                  {
                    v73 += 2;
                    if ( v72 == v73 )
                      goto LABEL_86;
                  }
                  if ( v72 == v73 )
                    break;
                  v75 = v73[1];
                  if ( HIDWORD(v155) <= (unsigned int)v76 )
                  {
                    sub_16CD150((__int64)&src, v156, 0, 8, v41, v42);
                    v76 = (unsigned int)v155;
                  }
                }
                goto LABEL_86;
              }
LABEL_109:
              if ( *(_BYTE *)(v145 + 16) != 29 )
                goto LABEL_87;
            }
            v74 = sub_157EE30(*(_QWORD *)(v145 - 48));
            if ( v74 )
              v74 -= 24;
            sub_1A95180((__int64)&src, v74);
            v68 = sub_157EE30(*(_QWORD *)(v145 - 24));
            if ( v68 )
              v68 -= 24;
            goto LABEL_90;
          }
LABEL_50:
          j___libc_free_0(0);
          v137 += 23;
          if ( v131 == v137 )
            goto LABEL_51;
        }
        while ( 1 )
        {
          v51 = *v44;
          v52 = v44[1];
          if ( v153 )
          {
            v45 = (v153 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
            v46 = &v151[2 * v45];
            v47 = *v46;
            if ( v52 == *v46 )
            {
              v48 = v46[1];
              goto LABEL_39;
            }
            v58 = 1;
            v55 = 0;
            while ( v47 != -8 )
            {
              if ( v55 || v47 != -16 )
                v46 = v55;
              v45 = (v153 - 1) & (v58 + v45);
              v70 = &v151[2 * v45];
              v47 = *v70;
              if ( v52 == *v70 )
              {
                v48 = v70[1];
                goto LABEL_39;
              }
              ++v58;
              v55 = v46;
              v46 = &v151[2 * v45];
            }
            if ( !v55 )
              v55 = v46;
            ++v150;
            v54 = v152 + 1;
            if ( 4 * ((int)v152 + 1) < 3 * v153 )
            {
              if ( v153 - HIDWORD(v152) - v54 <= v153 >> 3 )
              {
                sub_176F940((__int64)&v150, v153);
                if ( !v153 )
                {
LABEL_240:
                  LODWORD(v152) = v152 + 1;
                  BUG();
                }
                v59 = 0;
                LODWORD(v60) = (v153 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
                v61 = 1;
                v54 = v152 + 1;
                v55 = &v151[2 * (unsigned int)v60];
                v62 = *v55;
                if ( *v55 != v52 )
                {
                  while ( v62 != -8 )
                  {
                    if ( !v59 && v62 == -16 )
                      v59 = v55;
                    v60 = (v153 - 1) & ((_DWORD)v60 + v61);
                    v55 = &v151[2 * v60];
                    v62 = *v55;
                    if ( v52 == *v55 )
                      goto LABEL_46;
                    ++v61;
                  }
                  if ( v59 )
                    v55 = v59;
                }
              }
              goto LABEL_46;
            }
          }
          else
          {
            ++v150;
          }
          sub_176F940((__int64)&v150, 2 * v153);
          if ( !v153 )
            goto LABEL_240;
          LODWORD(v53) = (v153 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
          v54 = v152 + 1;
          v55 = &v151[2 * (unsigned int)v53];
          v56 = *v55;
          if ( v52 != *v55 )
          {
            v128 = 1;
            v129 = 0;
            while ( v56 != -8 )
            {
              if ( !v129 && v56 == -16 )
                v129 = v55;
              v53 = (v153 - 1) & ((_DWORD)v53 + v128);
              v55 = &v151[2 * v53];
              v56 = *v55;
              if ( v52 == *v55 )
                goto LABEL_46;
              ++v128;
            }
            if ( v129 )
              v55 = v129;
          }
LABEL_46:
          LODWORD(v152) = v54;
          if ( *v55 != -8 )
            --HIDWORD(v152);
          *v55 = v52;
          v48 = 0;
          v55[1] = 0;
LABEL_39:
          v49 = sub_1648A60(64, 2u);
          v50 = (__int64)v49;
          if ( v49 )
            sub_15F9650((__int64)v49, v51, v48, 0, 0);
          v44 += 2;
          sub_15F2180(v50, v51);
          if ( v43 == v44 )
            goto LABEL_49;
        }
      }
    }
    while ( 1 )
    {
      v27 = *(_QWORD ***)(v23 + 8);
      if ( !v153 )
        goto LABEL_23;
      v25 = (v153 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v26 = (_QWORD **)v151[2 * v25];
      if ( v27 == v26 )
      {
LABEL_21:
        v23 += 16;
        if ( v24 == v23 )
          goto LABEL_32;
      }
      else
      {
        v63 = 1;
        while ( v26 != (_QWORD **)-8LL )
        {
          v25 = (v153 - 1) & (v63 + v25);
          v26 = (_QWORD **)v151[2 * v25];
          if ( v27 == v26 )
            goto LABEL_21;
          ++v63;
        }
LABEL_23:
        v149 = *(_QWORD ***)(v23 + 8);
        v28 = *v27;
        v29 = *(_QWORD *)(a1 + 80);
        v142 = *(_DWORD *)(v136 + 4);
        v156[0] = 257;
        if ( v29 )
          v29 -= 24;
        v144 = sub_157ED20(v29);
        v30 = sub_1648A60(64, 1u);
        v31 = v30;
        if ( v30 )
          sub_15F8BE0((__int64)v30, v28, v142, (__int64)&src, v144);
        v32 = v153;
        if ( !v153 )
        {
          ++v150;
          goto LABEL_83;
        }
        v33 = v149;
        v34 = v153 - 1;
        v35 = (unsigned int)v151;
        v36 = (v153 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
        v37 = &v151[2 * v36];
        v38 = (_QWORD **)*v37;
        if ( v149 != (_QWORD **)*v37 )
        {
          v64 = 1;
          v65 = 0;
          while ( v38 != (_QWORD **)-8LL )
          {
            if ( !v65 && v38 == (_QWORD **)-16LL )
              v65 = v37;
            v36 = v34 & (v64 + v36);
            v37 = &v151[2 * v36];
            v38 = (_QWORD **)*v37;
            if ( v149 == (_QWORD **)*v37 )
              goto LABEL_29;
            ++v64;
          }
          if ( v65 )
            v37 = v65;
          ++v150;
          v66 = v152 + 1;
          if ( 4 * ((int)v152 + 1) >= 3 * v153 )
          {
LABEL_83:
            v32 = 2 * v153;
          }
          else
          {
            v35 = v153 >> 3;
            if ( v153 - HIDWORD(v152) - v66 > v153 >> 3 )
            {
LABEL_79:
              LODWORD(v152) = v66;
              if ( *v37 != -8 )
                --HIDWORD(v152);
              *v37 = (__int64)v33;
              v37[1] = 0;
              goto LABEL_29;
            }
          }
          sub_176F940((__int64)&v150, v32);
          sub_176A9A0((__int64)&v150, (__int64 *)&v149, &src);
          v37 = (__int64 *)src;
          v33 = v149;
          v66 = v152 + 1;
          goto LABEL_79;
        }
LABEL_29:
        v37[1] = (__int64)v31;
        v39 = (unsigned int)v158;
        if ( (unsigned int)v158 >= HIDWORD(v158) )
        {
          sub_16CD150((__int64)&v157, v159, 0, 8, v35, v34);
          v39 = (unsigned int)v158;
        }
        v23 += 16;
        *((_QWORD *)v157 + v39) = v31;
        LODWORD(v158) = v158 + 1;
        if ( v24 == v23 )
          goto LABEL_32;
      }
    }
  }
LABEL_51:
  if ( (_DWORD)v152 )
  {
    v78 = v151;
    v79 = &v151[2 * v153];
    v135 = v79;
    if ( v151 != v79 )
    {
      while ( 1 )
      {
        v80 = *v78;
        v81 = v78;
        if ( *v78 != -16 && v80 != -8 )
          break;
        v78 += 2;
        if ( v79 == v78 )
          goto LABEL_52;
      }
      if ( v79 != v78 )
      {
        while ( 1 )
        {
          v82 = (_QWORD *)v81[1];
          src = v156;
          v155 = 0x1400000000LL;
          v83 = sub_1648EF0(v80);
          if ( HIDWORD(v155) < v83 )
            sub_16CD150((__int64)&src, v156, v83, 8, v84, v85);
          v86 = (unsigned int)v155;
          if ( *(_QWORD *)(v80 + 8) )
          {
            v87 = v81;
            v88 = v155;
            v89 = *(_QWORD *)(v80 + 8);
            do
            {
              v90 = sub_1648700(v89);
              if ( *((_BYTE *)v90 + 16) != 5 )
              {
                if ( HIDWORD(v155) <= v88 )
                {
                  v148 = v90;
                  sub_16CD150((__int64)&src, v156, 0, 8, v91, v92);
                  v88 = v155;
                  v90 = v148;
                }
                *((_QWORD *)src + v88) = v90;
                v88 = v155 + 1;
                LODWORD(v155) = v155 + 1;
              }
              v89 = *(_QWORD *)(v89 + 8);
            }
            while ( v89 );
            v86 = v88;
            v81 = v87;
          }
          v93 = (unsigned __int64 *)src;
          v94 = 8 * v86;
          if ( (char *)src + v94 == src )
            goto LABEL_209;
          v146 = (unsigned __int64 *)((char *)src + v94);
          _BitScanReverse64(&v95, v94 >> 3);
          sub_1A95210((char *)src, (unsigned __int64 *)((char *)src + v94), 2LL * (int)(63 - (v95 ^ 0x3F)));
          if ( (unsigned __int64)v94 <= 0x80 )
          {
            sub_1A94FB0(v93, v146);
          }
          else
          {
            v96 = v93 + 16;
            sub_1A94FB0(v93, v93 + 16);
            if ( v146 != v93 + 16 )
            {
              do
              {
                while ( 1 )
                {
                  v97 = *v96;
                  v98 = *(v96 - 1);
                  v99 = v96 - 1;
                  if ( v98 > *v96 )
                    break;
                  v127 = v96++;
                  *v127 = v97;
                  if ( v146 == v96 )
                    goto LABEL_148;
                }
                do
                {
                  v99[1] = v98;
                  v100 = v99;
                  v98 = *--v99;
                }
                while ( v97 < v98 );
                ++v96;
                *v100 = v97;
              }
              while ( v146 != v96 );
            }
          }
LABEL_148:
          v101 = (__int64 *)src;
          v102 = (char *)src + 8 * (unsigned int)v155;
          if ( src == v102 )
          {
LABEL_209:
            LODWORD(v155) = 0;
            goto LABEL_181;
          }
          v103 = (char *)src;
          while ( 1 )
          {
            v105 = v103;
            v103 += 8;
            if ( v102 == v103 )
              break;
            v104 = *((_QWORD *)v103 - 1);
            if ( v104 == *(_QWORD *)v103 )
            {
              if ( v102 == v105 )
              {
                v103 = (char *)src + 8 * (unsigned int)v155;
              }
              else
              {
                v126 = v105 + 16;
                if ( v102 != v105 + 16 )
                {
                  while ( 1 )
                  {
                    if ( v104 != *v126 )
                    {
                      *((_QWORD *)v105 + 1) = *v126;
                      v105 += 8;
                    }
                    if ( v102 == (char *)++v126 )
                      break;
                    v104 = *(_QWORD *)v105;
                  }
                  v103 = v105 + 8;
                }
              }
              break;
            }
          }
          LODWORD(v155) = (v103 - (char *)v101) >> 3;
          v147 = (char *)&v101[(unsigned int)v155];
          if ( v101 != (__int64 *)v147 )
            break;
LABEL_181:
          v122 = sub_1648A60(64, 2u);
          v123 = (__int64)v122;
          if ( v122 )
            sub_15F9650((__int64)v122, v80, (__int64)v82, 0, 0);
          v124 = *(_BYTE *)(v80 + 16);
          if ( v124 <= 0x17u )
          {
            sub_15F2180(v123, (__int64)v82);
          }
          else if ( v124 == 29 )
          {
            v125 = sub_157ED20(*(_QWORD *)(v80 - 48));
            sub_15F2120(v123, v125);
          }
          else
          {
            sub_15F2180(v123, v80);
          }
          if ( src != v156 )
            _libc_free((unsigned __int64)src);
          v81 += 2;
          if ( v81 != v135 )
          {
            while ( 1 )
            {
              v80 = *v81;
              if ( *v81 != -8 && v80 != -16 )
                break;
              v81 += 2;
              if ( v135 == v81 )
                goto LABEL_52;
            }
            if ( v81 != v135 )
              continue;
          }
          goto LABEL_52;
        }
        v133 = v81;
        v106 = v101;
        while ( 1 )
        {
          v109 = *v106;
          if ( *(_BYTE *)(*v106 + 16) != 77 )
            break;
          v110 = *(_DWORD *)(v109 + 20);
          v111 = 0;
          v112 = v110 & 0xFFFFFFF;
          if ( (v110 & 0xFFFFFFF) != 0 )
          {
            v138 = v106;
            do
            {
              while ( 1 )
              {
                v113 = (*(_BYTE *)(v109 + 23) & 0x40) != 0 ? *(_QWORD *)(v109 - 8) : v109 - 24LL * v112;
                v114 = *(_QWORD *)(v113 + 24 * v111);
                if ( v80 == v114 )
                {
                  if ( v114 )
                    break;
                }
                ++v111;
                v112 = v110 & 0xFFFFFFF;
                if ( (v110 & 0xFFFFFFFu) <= (unsigned int)v111 )
                  goto LABEL_179;
              }
              v140 = sub_157EBA0(*(_QWORD *)(v113 + 8 * v111 + 24LL * *(unsigned int *)(v109 + 56) + 8));
              v115 = sub_1648A60(64, 1u);
              v116 = v115;
              if ( v115 )
                sub_15F9100((__int64)v115, v82, byte_3F871B3, v140);
              if ( (*(_BYTE *)(v109 + 23) & 0x40) != 0 )
                v117 = *(_QWORD *)(v109 - 8);
              else
                v117 = v109 - 24LL * (*(_DWORD *)(v109 + 20) & 0xFFFFFFF);
              v118 = (_QWORD *)(v117 + 24 * v111);
              if ( *v118 )
              {
                v119 = v118[1];
                v120 = v118[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v120 = v119;
                if ( v119 )
                  *(_QWORD *)(v119 + 16) = *(_QWORD *)(v119 + 16) & 3LL | v120;
              }
              *v118 = v116;
              if ( v116 )
              {
                v121 = v116[1];
                v118[1] = v121;
                if ( v121 )
                  *(_QWORD *)(v121 + 16) = (unsigned __int64)(v118 + 1) | *(_QWORD *)(v121 + 16) & 3LL;
                v118[2] = (unsigned __int64)(v116 + 1) | v118[2] & 3LL;
                v116[1] = v118;
              }
              v110 = *(_DWORD *)(v109 + 20);
              ++v111;
              v112 = v110 & 0xFFFFFFF;
            }
            while ( (v110 & 0xFFFFFFFu) > (unsigned int)v111 );
LABEL_179:
            v106 = v138 + 1;
            if ( v147 == (char *)(v138 + 1) )
            {
LABEL_180:
              v81 = v133;
              goto LABEL_181;
            }
          }
          else
          {
LABEL_157:
            if ( v147 == (char *)++v106 )
              goto LABEL_180;
          }
        }
        v107 = sub_1648A60(64, 1u);
        v108 = (__int64)v107;
        if ( v107 )
          sub_15F9100((__int64)v107, v82, byte_3F871B3, v109);
        sub_1648780(v109, v80, v108);
        goto LABEL_157;
      }
    }
  }
LABEL_52:
  if ( (_DWORD)v158 )
    sub_1B3B3D0(v157);
  if ( v157 != v159 )
    _libc_free((unsigned __int64)v157);
  return j___libc_free_0(v151);
}
