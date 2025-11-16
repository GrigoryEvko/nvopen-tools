// Function: sub_33FCE10
// Address: 0x33fce10
//
_QWORD *__fastcall sub_33FCE10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8,
        __int64 a9,
        const void *a10,
        __int64 a11)
{
  const void *v13; // r10
  unsigned int v14; // esi
  __int64 v15; // r9
  bool v16; // zf
  __int64 v17; // r15
  __int64 v18; // r8
  __int64 (*v19)(); // rax
  int v20; // edi
  __int64 v21; // r15
  __int64 v22; // rax
  char v23; // si
  __int64 v24; // r13
  char v25; // r9
  int v26; // edx
  int v27; // r11d
  int v28; // r10d
  _BYTE *v29; // r8
  __int64 v30; // rax
  char v31; // di
  char v32; // r14
  __int64 j; // rdx
  __int64 v34; // r15
  unsigned int *v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rax
  unsigned __int16 *v38; // rcx
  __int64 v39; // rsi
  _QWORD *v40; // rax
  _BYTE *v41; // r10
  int v42; // r11d
  _QWORD *v43; // rcx
  bool v44; // al
  __int16 *v45; // rax
  unsigned __int16 v46; // bx
  __int64 v47; // r13
  __int64 *v48; // rax
  int v49; // r9d
  __int64 v50; // r14
  _QWORD *v51; // r12
  __int64 v53; // r15
  __m128i *v54; // r14
  int v55; // edx
  __int64 v56; // r8
  __int64 v57; // rax
  _QWORD **v58; // r9
  __int64 v59; // rbx
  int v60; // r13d
  _QWORD *v61; // rax
  char *v62; // rdi
  _QWORD *v63; // rax
  unsigned int v64; // edx
  unsigned int v65; // r15d
  __int64 v66; // rax
  int v67; // edx
  _QWORD *v68; // rax
  unsigned int v69; // edx
  unsigned int v70; // r14d
  unsigned int v71; // edx
  _QWORD *v72; // r14
  unsigned int v73; // r8d
  int v74; // esi
  int *v75; // rax
  int *v76; // r9
  int v77; // edx
  int v78; // edi
  unsigned int v79; // eax
  __int64 v80; // rax
  int v81; // esi
  int *v82; // rax
  int *v83; // r8
  int v84; // edx
  int v85; // edi
  unsigned int v86; // edi
  __int64 v87; // rdx
  _DWORD *v88; // rcx
  __int64 v89; // r8
  __int64 v90; // rax
  _BYTE *v91; // rdi
  unsigned __int64 i; // rax
  __int64 v93; // rdx
  unsigned __int64 v94; // rdx
  __int64 v95; // rsi
  __int64 v96; // rax
  _BYTE *v97; // rdi
  __int64 v98; // rsi
  unsigned __int64 k; // rax
  __int64 v100; // rdx
  unsigned __int64 v101; // rdx
  int *v102; // rcx
  int v103; // edx
  unsigned __int64 v104; // rdx
  __int64 v105; // r10
  __int64 v106; // rdx
  __int64 v107; // rsi
  char *v108; // rbx
  __int64 *v109; // r13
  int v110; // r10d
  int v111; // r9d
  unsigned __int8 *v112; // rsi
  __int64 v113; // rdi
  __int64 v114; // rax
  unsigned __int64 v115; // rax
  __int64 v116; // rax
  __int16 v117; // dx
  __int64 v118; // rax
  __int64 v119; // [rsp+8h] [rbp-1A8h]
  int v120; // [rsp+10h] [rbp-1A0h]
  unsigned int src; // [rsp+18h] [rbp-198h]
  int srca; // [rsp+18h] [rbp-198h]
  int srcb; // [rsp+18h] [rbp-198h]
  __int64 v124; // [rsp+20h] [rbp-190h]
  int v127; // [rsp+48h] [rbp-168h]
  unsigned int v128; // [rsp+50h] [rbp-160h]
  unsigned int v129; // [rsp+50h] [rbp-160h]
  __int64 v130; // [rsp+58h] [rbp-158h]
  unsigned int v131; // [rsp+60h] [rbp-150h]
  _QWORD **v132; // [rsp+60h] [rbp-150h]
  _QWORD *v133; // [rsp+60h] [rbp-150h]
  _QWORD *v134; // [rsp+60h] [rbp-150h]
  __int64 v135; // [rsp+68h] [rbp-148h]
  int v136; // [rsp+68h] [rbp-148h]
  int v137; // [rsp+68h] [rbp-148h]
  __int64 v138; // [rsp+80h] [rbp-130h] BYREF
  __int64 v139; // [rsp+88h] [rbp-128h]
  __int64 *v140; // [rsp+90h] [rbp-120h] BYREF
  unsigned __int8 *v141; // [rsp+98h] [rbp-118h] BYREF
  __int64 v142; // [rsp+A0h] [rbp-110h] BYREF
  unsigned __int64 v143; // [rsp+A8h] [rbp-108h]
  __int64 v144; // [rsp+B0h] [rbp-100h]
  unsigned __int64 v145; // [rsp+B8h] [rbp-F8h]
  void *v146; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v147; // [rsp+C8h] [rbp-E8h]
  _BYTE v148[32]; // [rsp+D0h] [rbp-E0h] BYREF
  _QWORD *v149; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v150; // [rsp+F8h] [rbp-B8h]
  _BYTE v151[48]; // [rsp+100h] [rbp-B0h] BYREF
  int v152; // [rsp+130h] [rbp-80h]

  v13 = a10;
  v138 = a2;
  v14 = a6;
  v15 = a8;
  v16 = *(_DWORD *)(a5 + 24) == 51;
  v139 = a3;
  v130 = a5;
  v128 = v14;
  v135 = a8;
  v131 = a9;
  if ( !v16 || *(_DWORD *)(a8 + 24) != 51 )
  {
    v17 = 4 * a11;
    v146 = v148;
    v18 = (4 * a11) >> 2;
    v147 = 0x800000000LL;
    if ( (unsigned __int64)(4 * a11) > 0x20 )
    {
      sub_C8D5F0((__int64)&v146, v148, (4 * a11) >> 2, 4u, v18, a8);
      v18 = (4 * a11) >> 2;
      v15 = a8;
      v13 = a10;
      v62 = (char *)v146 + 4 * (unsigned int)v147;
    }
    else
    {
      if ( !v17 )
      {
LABEL_5:
        LODWORD(v147) = v17 + v18;
        if ( v14 == (_DWORD)a9 && a5 == v15 )
        {
          v149 = 0;
          LODWORD(v150) = 0;
          v63 = sub_33F17F0((_QWORD *)a1, 51, (__int64)&v149, v138, v139);
          v65 = v64;
          if ( v149 )
          {
            v133 = v63;
            sub_B91220((__int64)&v149, (__int64)v149);
            v63 = v133;
          }
          v135 = (__int64)v63;
          v131 = v65;
          if ( (_DWORD)a11 )
          {
            v66 = 0;
            do
            {
              v67 = *(_DWORD *)((char *)v146 + v66);
              if ( (int)a11 <= v67 )
                *(_DWORD *)((char *)v146 + v66) = v67 - a11;
              v66 += 4;
            }
            while ( 4LL * (unsigned int)(a11 - 1) + 4 != v66 );
          }
        }
        if ( *(_DWORD *)(a5 + 24) == 51 )
        {
          v81 = v147;
          v82 = (int *)v146;
          if ( (_DWORD)v147 )
          {
            v83 = (int *)((char *)v146 + 4 * (unsigned int)(v147 - 1) + 4);
            do
            {
              v84 = *v82;
              if ( *v82 >= 0 )
              {
                v85 = v84 - v81;
                if ( v84 < v81 )
                  v85 = v81 + v84;
                *v82 = v85;
              }
              ++v82;
            }
            while ( v83 != v82 );
          }
          v86 = v131;
          v131 = v128;
          v128 = v86;
          v130 = v135;
          v135 = a5;
        }
        v19 = *(__int64 (**)())(**(_QWORD **)(a1 + 16) + 1496LL);
        if ( v19 != sub_2FE34F0 && (unsigned __int8)v19() )
        {
          if ( *(_DWORD *)(v130 + 24) == 156 )
          {
            v152 = 0;
            v149 = v151;
            v150 = 0x600000000LL;
            v90 = sub_33D2250(v130, (__int64)&v149, v87, (__int64)v88, v89, (__int64)&v149);
            v91 = v149;
            if ( v90 && (int)a11 > 0 )
            {
              v89 = (unsigned int)(a11 - 1);
              for ( i = 0; ; i = v87 )
              {
                v88 = (char *)v146 + 4 * i;
                v94 = (unsigned int)*v88;
                if ( (v94 & 0x80000000) != 0LL || (int)a11 <= (int)v94 )
                  goto LABEL_107;
                v95 = *(_QWORD *)&v91[8 * ((unsigned int)v94 >> 6)];
                if ( !_bittest64(&v95, v94) )
                  break;
                *v88 = -1;
                v87 = i + 1;
                v91 = v149;
                if ( v89 == i )
                  goto LABEL_113;
LABEL_108:
                ;
              }
              v93 = *(_QWORD *)&v91[8 * ((unsigned int)i >> 6)];
              if ( !_bittest64(&v93, i) )
              {
                *v88 = i;
                v91 = v149;
              }
LABEL_107:
              v87 = i + 1;
              if ( v89 == i )
                goto LABEL_113;
              goto LABEL_108;
            }
LABEL_113:
            if ( v91 != v151 )
              _libc_free((unsigned __int64)v91);
          }
          v20 = *(_DWORD *)(v135 + 24);
          if ( v20 != 156 )
          {
LABEL_10:
            if ( !(_DWORD)a11 )
              goto LABEL_56;
            v21 = (unsigned int)(a11 - 1);
            v22 = 0;
            v23 = 1;
            v24 = 4 * v21 + 4;
            v25 = 1;
            do
            {
              while ( 1 )
              {
                v26 = *(_DWORD *)((char *)v146 + v22);
                if ( (int)a11 > v26 )
                  break;
                if ( v20 == 51 )
                  *(_DWORD *)((char *)v146 + v22) = -1;
                else
                  v25 = 0;
                v22 += 4;
                if ( v24 == v22 )
                  goto LABEL_19;
              }
              if ( v26 >= 0 )
                v23 = 0;
              v22 += 4;
            }
            while ( v24 != v22 );
LABEL_19:
            if ( v25 && v23 )
            {
LABEL_56:
              v149 = 0;
              LODWORD(v150) = 0;
              v51 = sub_33F17F0((_QWORD *)a1, 51, (__int64)&v149, v138, v139);
              if ( v149 )
                sub_B91220((__int64)&v149, (__int64)v149);
              goto LABEL_51;
            }
            if ( v20 != 51 && v25 )
            {
              v149 = 0;
              LODWORD(v150) = 0;
              v68 = sub_33F17F0((_QWORD *)a1, 51, (__int64)&v149, v138, v139);
              v70 = v69;
              if ( v149 )
              {
                v134 = v68;
                sub_B91220((__int64)&v149, (__int64)v149);
                v68 = v134;
              }
              v135 = (__int64)v68;
              v131 = v70;
            }
            else if ( v23 )
            {
              v149 = 0;
              LODWORD(v150) = 0;
              v72 = sub_33F17F0((_QWORD *)a1, 51, (__int64)&v149, v138, v139);
              v73 = v71;
              if ( v149 )
              {
                v129 = v71;
                sub_B91220((__int64)&v149, (__int64)v149);
                v73 = v129;
              }
              v74 = v147;
              v75 = (int *)v146;
              if ( (_DWORD)v147 )
              {
                v76 = (int *)((char *)v146 + 4 * (unsigned int)(v147 - 1) + 4);
                do
                {
                  v77 = *v75;
                  if ( *v75 >= 0 )
                  {
                    v78 = v77 - v74;
                    if ( v77 < v74 )
                      v78 = v74 + v77;
                    *v75 = v78;
                  }
                  ++v75;
                }
                while ( v76 != v75 );
              }
              v79 = v131;
              v131 = v73;
              v128 = v79;
              v80 = v135;
              v135 = (__int64)v72;
              v130 = v80;
            }
            v27 = *(_DWORD *)(v135 + 24);
            v28 = *(_DWORD *)(v130 + 24);
            if ( v28 == 51 && v27 == 51 )
            {
              v114 = sub_3288990(a1, (unsigned int)v138, v139);
              v29 = v146;
              v51 = (_QWORD *)v114;
            }
            else
            {
              v29 = v146;
              v30 = 0;
              v31 = 1;
              v32 = 1;
              for ( j = *(unsigned int *)v146; ; j = *((unsigned int *)v146 + ++v30) )
              {
                if ( (_DWORD)j != (_DWORD)v30 && (int)j >= 0 )
                  v31 = 0;
                if ( *(_DWORD *)v146 != (_DWORD)j )
                  v32 = 0;
                if ( v30 == v21 )
                  break;
              }
              if ( !v31 )
              {
                if ( v27 != 51 )
                  goto LABEL_58;
                if ( v28 == 234 )
                {
                  v34 = v130;
                  do
                  {
                    v35 = *(unsigned int **)(v34 + 40);
                    v34 = *(_QWORD *)v35;
                    v36 = v35[2];
                    v28 = *(_DWORD *)(*(_QWORD *)v35 + 24LL);
                  }
                  while ( v28 == 234 );
                }
                else
                {
                  v36 = v128;
                  v34 = v130;
                }
                if ( v28 != 156 )
                  goto LABEL_58;
                src = v36;
                v149 = v151;
                v150 = 0x600000000LL;
                v152 = 0;
                v37 = sub_33D2250(v34, (__int64)&v149, j, v36, (__int64)v146, (__int64)&v149);
                if ( v37 )
                {
                  if ( *(_DWORD *)(v37 + 24) == 51 )
                  {
                    v118 = sub_3288990(a1, (unsigned int)v138, v139);
                    v41 = v149;
                    v51 = (_QWORD *)v118;
                  }
                  else
                  {
                    v38 = (unsigned __int16 *)(*(_QWORD *)(v34 + 48) + 16LL * src);
                    v119 = v37;
                    v39 = *v38;
                    v143 = *((_QWORD *)v38 + 1);
                    LOWORD(v142) = v39;
                    srca = sub_3281500(&v142, v39);
                    sub_3281500(&v138, v39);
                    v40 = sub_33C7FB0(v149, (__int64)&v149[(unsigned int)v150]);
                    if ( v43 != v40 )
                    {
                      v44 = srca == v42;
LABEL_43:
                      if ( v32 && v44 )
                      {
                        v45 = *(__int16 **)(v34 + 48);
                        v46 = *v45;
                        v47 = *((_QWORD *)v45 + 1);
                        v48 = (__int64 *)(*(_QWORD *)(v34 + 40) + 40LL * *(unsigned int *)v146);
                        v50 = sub_32886A0(a1, v46, v47, a4, *v48, v48[1]);
                        if ( v46 != (_WORD)v138 || v139 != v47 && !v46 )
                          v50 = (__int64)sub_33FAF80(a1, 234, a4, (unsigned int)v138, v139, v49, a7);
                        v41 = v149;
                        v51 = (_QWORD *)v50;
                        goto LABEL_49;
                      }
                      goto LABEL_149;
                    }
                    if ( srca != v42 )
                    {
                      if ( !sub_33CF170(v119) )
                      {
LABEL_149:
                        if ( v149 != (_QWORD *)v151 )
                          _libc_free((unsigned __int64)v149);
LABEL_58:
                        v53 = 0;
                        v54 = sub_33ED250(a1, (unsigned int)v138, v139);
                        v120 = v55;
                        v142 = v130;
                        v149 = v151;
                        v150 = 0x2000000000LL;
                        v144 = v135;
                        v143 = v128 | a6 & 0xFFFFFFFF00000000LL;
                        v145 = v131 | a9 & 0xFFFFFFFF00000000LL;
                        sub_33C9670(
                          (__int64)&v149,
                          165,
                          (unsigned __int64)v54,
                          (unsigned __int64 *)&v142,
                          2,
                          (__int64)&v149);
                        v57 = (unsigned int)v150;
                        v58 = &v149;
                        v59 = v24;
                        do
                        {
                          v60 = *(_DWORD *)((char *)v146 + v53);
                          if ( v57 + 1 > (unsigned __int64)HIDWORD(v150) )
                          {
                            v132 = v58;
                            sub_C8D5F0((__int64)v58, v151, v57 + 1, 4u, v56, (__int64)v58);
                            v57 = (unsigned int)v150;
                            v58 = v132;
                          }
                          v53 += 4;
                          *((_DWORD *)v149 + v57) = v60;
                          v57 = (unsigned int)(v150 + 1);
                          LODWORD(v150) = v150 + 1;
                        }
                        while ( v59 != v53 );
                        v140 = 0;
                        v61 = sub_33CCCF0(a1, (__int64)v58, a4, (__int64 *)&v140);
                        if ( v61 )
                        {
                          v51 = v61;
                          goto LABEL_64;
                        }
                        v106 = *(_QWORD *)(a1 + 544);
                        v107 = 4LL * (int)a11;
                        *(_QWORD *)(a1 + 624) += v107;
                        v108 = (char *)((v106 + 3) & 0xFFFFFFFFFFFFFFFCLL);
                        if ( *(_QWORD *)(a1 + 552) >= (unsigned __int64)&v108[v107] && v106 )
                          *(_QWORD *)(a1 + 544) = &v108[v107];
                        else
                          v108 = (char *)sub_9D1E70(a1 + 544, v107, v107, 2);
                        if ( 4LL * (unsigned int)v147 )
                          memmove(v108, v146, 4LL * (unsigned int)v147);
                        v109 = *(__int64 **)(a1 + 416);
                        v110 = *(_DWORD *)(a4 + 8);
                        if ( v109 )
                        {
                          *(_QWORD *)(a1 + 416) = *v109;
                        }
                        else
                        {
                          v137 = *(_DWORD *)(a4 + 8);
                          v115 = sub_33E48B0((__int64 *)(a1 + 424));
                          v110 = v137;
                          v109 = (__int64 *)v115;
                          if ( !v115 )
                          {
LABEL_142:
                            sub_33E4EC0(a1, (__int64)v109, (__int64)&v142, 2);
                            sub_C657C0((__int64 *)(a1 + 520), v109, v140, (__int64)off_4A367D0);
                            v113 = a1;
                            v51 = v109;
                            sub_33CC420(v113, (__int64)v109);
LABEL_64:
                            if ( v149 != (_QWORD *)v151 )
                              _libc_free((unsigned __int64)v149);
                            goto LABEL_51;
                          }
                        }
                        v111 = v120;
                        v112 = *(unsigned __int8 **)a4;
                        v141 = v112;
                        if ( v112 )
                        {
                          v136 = v110;
                          sub_B96E90((__int64)&v141, (__int64)v112, 1);
                          v111 = v120;
                          v110 = v136;
                        }
                        sub_33CA300((__int64)v109, 165, v110, &v141, (__int64)v54, v111);
                        if ( v141 )
                          sub_B91220((__int64)&v141, (__int64)v141);
                        v109[12] = (__int64)v108;
                        goto LABEL_142;
                      }
                      v41 = v149;
                    }
                    v51 = (_QWORD *)v130;
                  }
LABEL_49:
                  if ( v41 != v151 )
                    _libc_free((unsigned __int64)v41);
LABEL_51:
                  v29 = v146;
                  goto LABEL_52;
                }
                v116 = *(_QWORD *)(v34 + 48) + 16LL * src;
                v117 = *(_WORD *)v116;
                v143 = *(_QWORD *)(v116 + 8);
                LOWORD(v142) = v117;
                srcb = sub_3281500(&v142, (__int64)&v149);
                v44 = srcb == (unsigned int)sub_3281500(&v138, (__int64)&v149);
                goto LABEL_43;
              }
              v51 = (_QWORD *)v130;
            }
LABEL_52:
            if ( v29 != v148 )
              _libc_free((unsigned __int64)v29);
            return v51;
          }
          v152 = 0;
          v149 = v151;
          v150 = 0x600000000LL;
          v96 = sub_33D2250(v135, (__int64)&v149, v87, (__int64)v88, v89, (__int64)&v149);
          v97 = v149;
          if ( v96 && (int)a11 > 0 )
          {
            v98 = (unsigned int)(a11 - 1);
            for ( k = 0; ; k = v101 )
            {
              v102 = (int *)((char *)v146 + 4 * k);
              v103 = *v102;
              if ( (int)a11 > *v102 || v103 >= 2 * (int)a11 )
                goto LABEL_121;
              v104 = (unsigned int)(v103 - a11);
              v105 = *(_QWORD *)&v97[8 * ((unsigned int)v104 >> 6)];
              if ( !_bittest64(&v105, v104) )
                break;
              *v102 = -1;
              v101 = k + 1;
              v97 = v149;
              if ( v98 == k )
                goto LABEL_127;
LABEL_122:
              ;
            }
            v100 = *(_QWORD *)&v97[8 * ((unsigned int)k >> 6)];
            if ( !_bittest64(&v100, k) )
            {
              *v102 = a11 + k;
              v97 = v149;
            }
LABEL_121:
            v101 = k + 1;
            if ( v98 == k )
              goto LABEL_127;
            goto LABEL_122;
          }
LABEL_127:
          if ( v97 != v151 )
            _libc_free((unsigned __int64)v97);
        }
        v20 = *(_DWORD *)(v135 + 24);
        goto LABEL_10;
      }
      v62 = v148;
    }
    v124 = v15;
    v127 = v18;
    memcpy(v62, v13, 4 * a11);
    LODWORD(v17) = v147;
    v15 = v124;
    LODWORD(v18) = v127;
    goto LABEL_5;
  }
  v149 = 0;
  LODWORD(v150) = 0;
  v51 = sub_33F17F0((_QWORD *)a1, 51, (__int64)&v149, v138, a3);
  if ( v149 )
    sub_B91220((__int64)&v149, (__int64)v149);
  return v51;
}
