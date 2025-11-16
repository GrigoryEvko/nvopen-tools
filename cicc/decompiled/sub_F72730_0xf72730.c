// Function: sub_F72730
// Address: 0xf72730
//
__int64 __fastcall sub_F72730(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        int a7,
        int a8,
        __int64 a9)
{
  __int64 *v10; // rsi
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rcx
  char v17; // bl
  _BYTE *v18; // rdi
  __int64 v19; // r13
  char v20; // al
  __int64 *v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rax
  unsigned __int8 *v24; // r15
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r14
  __int64 v31; // r15
  __int64 v32; // rax
  unsigned __int8 *v33; // rbx
  __int64 v34; // r12
  int v35; // ecx
  __int64 v36; // rdi
  int v37; // ecx
  unsigned int v38; // edx
  __int64 **v39; // rax
  __int64 *v40; // r9
  __int64 *v41; // rax
  __int64 **v42; // rax
  __int64 **v43; // rdx
  int v44; // eax
  _BYTE *v45; // rdi
  __int64 v46; // rdx
  bool v47; // al
  bool v48; // zf
  int v49; // eax
  unsigned __int8 *v50; // r15
  __int64 v51; // rbx
  __int64 v52; // r12
  _BYTE *v53; // r8
  __int64 v54; // r10
  __int64 v55; // r12
  __int64 *v56; // rax
  __int64 *v57; // rdi
  __int16 v58; // ax
  __int64 v59; // r9
  char v60; // r12
  __int64 v61; // rcx
  int v62; // eax
  unsigned __int64 v63; // rdx
  int v64; // r8d
  __int64 *v65; // r11
  __int64 v66; // rdx
  unsigned __int64 v67; // rcx
  int v68; // eax
  unsigned __int64 *v69; // rdi
  __int64 v70; // rax
  __int64 *v71; // rcx
  __int64 v72; // r11
  int v73; // eax
  int v74; // r8d
  unsigned int v75; // edx
  __int64 **v76; // rax
  __int64 *v77; // rdi
  __int64 *v78; // rax
  unsigned int v79; // ecx
  __int64 **v80; // rdx
  __int64 *v81; // rdi
  __int64 *v82; // rdx
  __int64 v83; // rdx
  unsigned __int8 *v84; // rcx
  int v85; // eax
  __int64 i; // rbx
  __int64 v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 *v90; // r8
  __int64 *v91; // rax
  __int64 v92; // rdx
  _QWORD *v93; // r8
  unsigned int v94; // edi
  __int64 v95; // rax
  __int64 v96; // r12
  __int64 **v97; // rax
  __int64 **v98; // rdx
  _QWORD *v99; // rdx
  __int64 v100; // rcx
  __int64 v101; // r8
  __int64 v102; // r9
  __int64 v103; // r12
  __int64 v104; // rbx
  _QWORD *v105; // rax
  __int64 v106; // rax
  unsigned __int64 v107; // rdx
  char *v108; // r15
  __int64 v109; // r13
  __int64 v110; // r14
  __int64 v111; // r15
  __int64 v112; // rbx
  unsigned __int8 *v113; // rdi
  unsigned __int64 v114; // r8
  const __m128i *v115; // rbx
  __m128i *v116; // rax
  __m128i v117; // xmm1
  int v118; // edx
  int v119; // r9d
  int v120; // eax
  int v121; // r9d
  __int64 v122; // rax
  char *v123; // rbx
  __int64 v124; // [rsp+8h] [rbp-328h]
  unsigned __int8 *v128; // [rsp+48h] [rbp-2E8h]
  __int64 v129; // [rsp+50h] [rbp-2E0h]
  __int64 v130; // [rsp+58h] [rbp-2D8h]
  char v131; // [rsp+58h] [rbp-2D8h]
  _BYTE *v132; // [rsp+60h] [rbp-2D0h]
  __int64 v133; // [rsp+68h] [rbp-2C8h]
  __int64 v135; // [rsp+78h] [rbp-2B8h]
  _BYTE *v136; // [rsp+80h] [rbp-2B0h]
  __int64 v137; // [rsp+80h] [rbp-2B0h]
  __int64 v138; // [rsp+88h] [rbp-2A8h]
  char v139; // [rsp+88h] [rbp-2A8h]
  __int64 v140; // [rsp+88h] [rbp-2A8h]
  __int64 *v141; // [rsp+90h] [rbp-2A0h]
  int v142; // [rsp+90h] [rbp-2A0h]
  int v143; // [rsp+98h] [rbp-298h]
  unsigned int v144; // [rsp+98h] [rbp-298h]
  __int64 *v145; // [rsp+A8h] [rbp-288h] BYREF
  _BYTE *v146; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v147; // [rsp+B8h] [rbp-278h]
  _BYTE v148[64]; // [rsp+C0h] [rbp-270h] BYREF
  _QWORD *v149; // [rsp+100h] [rbp-230h] BYREF
  __int64 v150; // [rsp+108h] [rbp-228h]
  _QWORD v151[8]; // [rsp+110h] [rbp-220h] BYREF
  __int64 v152; // [rsp+150h] [rbp-1E0h] BYREF
  __int64 v153; // [rsp+158h] [rbp-1D8h]
  __int64 v154; // [rsp+160h] [rbp-1D0h] BYREF
  unsigned __int8 *v155; // [rsp+168h] [rbp-1C8h]
  unsigned __int8 *v156; // [rsp+170h] [rbp-1C0h] BYREF
  __int64 *v157; // [rsp+178h] [rbp-1B8h]
  _BYTE *v158; // [rsp+180h] [rbp-1B0h]
  __int64 v159; // [rsp+188h] [rbp-1A8h]
  _BYTE v160[32]; // [rsp+190h] [rbp-1A0h] BYREF
  __int64 *v161; // [rsp+1B0h] [rbp-180h] BYREF
  __int64 v162; // [rsp+1B8h] [rbp-178h]
  _BYTE v163[368]; // [rsp+1C0h] [rbp-170h] BYREF

  v10 = (__int64 *)&v146;
  v11 = a1;
  v146 = v148;
  v147 = 0x800000000LL;
  sub_D474A0(a1, (__int64)&v146);
  v161 = (__int64 *)v163;
  v162 = 0x800000000LL;
  v136 = v146;
  v132 = &v146[8 * (unsigned int)v147];
  if ( v146 != v132 )
  {
    v135 = a2;
    v12 = a1;
    do
    {
      v13 = *(_QWORD *)(*(_QWORD *)v136 + 56LL);
      if ( !v13 )
        BUG();
      if ( *(_BYTE *)(v13 - 24) == 84 )
      {
        v14 = v13 - 24;
        v15 = v12;
        v138 = *(_QWORD *)(v13 + 8);
        v143 = *(_DWORD *)(v13 - 20) & 0x7FFFFFF;
        if ( !*(_QWORD *)(v13 - 24 + 16) )
          goto LABEL_7;
LABEL_6:
        v10 = *(__int64 **)(v14 + 8);
        if ( sub_D97040((__int64)a4, (__int64)v10) && v143 )
        {
          v31 = 0;
          while ( 2 )
          {
            v32 = *(_QWORD *)(v14 - 8);
            v142 = v31;
            v33 = *(unsigned __int8 **)(v32 + 32 * v31);
            if ( *v33 <= 0x1Cu )
              goto LABEL_43;
            v34 = 8 * v31;
            v10 = *(__int64 **)(32LL * *(unsigned int *)(v14 + 72) + v32 + 8 * v31);
            v35 = *(_DWORD *)(v135 + 24);
            v36 = *(_QWORD *)(v135 + 8);
            if ( v35 )
            {
              v37 = v35 - 1;
              v38 = v37 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
              v39 = (__int64 **)(v36 + 16LL * v38);
              v40 = *v39;
              if ( v10 == *v39 )
              {
LABEL_47:
                v41 = v39[1];
LABEL_48:
                if ( (__int64 *)v15 == v41 )
                {
                  v10 = (__int64 *)*((_QWORD *)v33 + 5);
                  v133 = v15 + 56;
                  if ( *(_BYTE *)(v15 + 84) )
                  {
                    v42 = *(__int64 ***)(v15 + 64);
                    v43 = &v42[*(unsigned int *)(v15 + 76)];
                    if ( v42 == v43 )
                      goto LABEL_43;
                    while ( v10 != *v42 )
                    {
                      if ( v43 == ++v42 )
                        goto LABEL_43;
                    }
                  }
                  else if ( !sub_C8CA60(v133, (__int64)v10) )
                  {
                    goto LABEL_43;
                  }
                  if ( a8 != 3 )
                    goto LABEL_77;
                  v152 = 6;
                  v158 = v160;
                  v153 = 0;
                  v154 = 0;
                  LODWORD(v155) = 0;
                  v156 = 0;
                  v157 = 0;
                  v159 = 0x200000000LL;
                  v44 = *v33;
                  if ( (_BYTE)v44 == 84 )
                  {
                    if ( !sub_D4B130(v15)
                      || *((_QWORD *)v33 + 5) != **(_QWORD **)(v15 + 32)
                      || (v10 = (__int64 *)v15, !(unsigned __int8)sub_10238A0(v33, v15, a4, &v152, 0, 0)) )
                    {
                      v45 = v158;
                      goto LABEL_59;
                    }
                    v83 = *((_QWORD *)v33 + 2);
                    v10 = v157;
                    if ( v83 )
                    {
                      while ( 1 )
                      {
                        v84 = *(unsigned __int8 **)(v83 + 24);
                        v85 = *v84;
                        if ( (unsigned __int8)v85 <= 0x1Cu
                          || (_BYTE)v85 != 84 && ((unsigned int)(v85 - 42) > 0x11 || v84 != (unsigned __int8 *)v157) )
                        {
                          break;
                        }
                        v83 = *(_QWORD *)(v83 + 8);
                        if ( !v83 )
                          goto LABEL_139;
                      }
                      v45 = v158;
LABEL_59:
                      if ( v45 == v160 )
                      {
LABEL_127:
                        v46 = v154;
                        v47 = v154 != -8192;
                        v48 = v154 == -4096;
                      }
                      else
                      {
                        _libc_free(v45, v10);
                        v46 = v154;
                        v47 = v154 != -4096;
                        v48 = v154 == -8192;
                      }
                      if ( v46 != 0 && !v48 && v47 )
                        sub_BD60C0(&v152);
                      goto LABEL_43;
                    }
LABEL_139:
                    v53 = v158;
LABEL_72:
                    if ( v53 != v160 )
                      _libc_free(v53, v10);
                    if ( v154 != -4096 && v154 != 0 && v154 != -8192 )
                      sub_BD60C0(&v152);
LABEL_77:
                    v145 = sub_DDFBA0((__int64)a4, (__int64)v33, *(char **)v15);
                    if ( sub_D96A50((__int64)v145)
                      || !sub_DADE90((__int64)a4, (__int64)v145, v15)
                      || !(unsigned __int8)sub_F80610(a6, v145) )
                    {
                      v10 = (__int64 *)v15;
                      v55 = sub_DBA6E0(
                              (__int64)a4,
                              v15,
                              *(_QWORD *)(*(_QWORD *)(v14 - 8) + 32LL * *(unsigned int *)(v14 + 72) + v34),
                              0);
                      if ( sub_D96A50(v55) )
                        goto LABEL_43;
                      v10 = (__int64 *)v33;
                      v56 = sub_DD8400((__int64)a4, (__int64)v33);
                      if ( *((_WORD *)v56 + 12) == 8 && v15 == v56[6] )
                      {
                        v10 = (__int64 *)v55;
                        v145 = sub_DD0540((__int64)v56, v55, a4);
                        v57 = v145;
                      }
                      else
                      {
                        v57 = v145;
                      }
                      if ( sub_D96A50((__int64)v57) )
                        goto LABEL_43;
                      v10 = v145;
                      if ( !sub_DADE90((__int64)a4, (__int64)v145, v15) )
                        goto LABEL_43;
                      v10 = v145;
                      if ( !(unsigned __int8)sub_F80610(a6, v145) )
                        goto LABEL_43;
                    }
                    if ( a8 == 4 || (v58 = *((_WORD *)v145 + 12), v58 == 15) || !v58 )
                    {
LABEL_89:
                      v60 = sub_F6CE90(a6, (__int64 *)&v145, 1, v15, qword_4F8C268[8], a5, (__int64)v33);
                      if ( *v33 == 84 || *v33 == 95 )
                      {
                        v33 = (unsigned __int8 *)sub_AA5190(*((_QWORD *)v33 + 5));
                        if ( v33 )
                          v33 -= 24;
                      }
                      v61 = (unsigned int)v162;
                      v10 = v161;
                      v62 = v162;
                      v63 = (unsigned __int64)&v161[5 * (unsigned int)v162];
                      if ( (unsigned int)v162 >= (unsigned __int64)HIDWORD(v162) )
                      {
                        v154 = (__int64)v145;
                        v114 = (unsigned int)v162 + 1LL;
                        v155 = v33;
                        LODWORD(v153) = v142;
                        v152 = v14;
                        v115 = (const __m128i *)&v152;
                        LOBYTE(v156) = v60;
                        if ( HIDWORD(v162) < v114 )
                        {
                          if ( v161 > &v152 || v63 <= (unsigned __int64)&v152 )
                          {
                            sub_C8D5F0((__int64)&v161, v163, v114, 0x28u, v114, v59);
                            v10 = v161;
                            v61 = (unsigned int)v162;
                            v115 = (const __m128i *)&v152;
                          }
                          else
                          {
                            v123 = (char *)((char *)&v152 - (char *)v161);
                            sub_C8D5F0((__int64)&v161, v163, v114, 0x28u, v114, v59);
                            v10 = v161;
                            v61 = (unsigned int)v162;
                            v115 = (const __m128i *)&v123[(_QWORD)v161];
                          }
                        }
                        v116 = (__m128i *)&v10[5 * v61];
                        *v116 = _mm_loadu_si128(v115);
                        v117 = _mm_loadu_si128(v115 + 1);
                        LODWORD(v162) = v162 + 1;
                        v116[1] = v117;
                        v116[2].m128i_i64[0] = v115[2].m128i_i64[0];
                      }
                      else
                      {
                        if ( v63 )
                        {
                          *(_QWORD *)(v63 + 16) = v145;
                          *(_QWORD *)(v63 + 24) = v33;
                          *(_BYTE *)(v63 + 32) = v60;
                          *(_QWORD *)v63 = v14;
                          *(_DWORD *)(v63 + 8) = v142;
                          v62 = v162;
                        }
                        LODWORD(v162) = v62 + 1;
                      }
                      goto LABEL_43;
                    }
                    v93 = v151;
                    LODWORD(v155) = 0;
                    v94 = 1;
                    v153 = (__int64)&v156;
                    v154 = 0x100000008LL;
                    v149 = v151;
                    BYTE4(v155) = 1;
                    v156 = v33;
                    v152 = 1;
                    v151[0] = v33;
                    v150 = 0x800000001LL;
                    v128 = v33;
                    while ( 2 )
                    {
                      while ( 2 )
                      {
                        v95 = v94--;
                        v48 = *(_BYTE *)(v15 + 84) == 0;
                        v96 = v93[v95 - 1];
                        LODWORD(v150) = v94;
                        v10 = *(__int64 **)(v96 + 40);
                        if ( v48 )
                        {
                          if ( sub_C8CA60(v133, (__int64)v10) )
                            goto LABEL_164;
                          v94 = v150;
                          v93 = v149;
LABEL_183:
                          if ( !v94 )
                          {
                            v131 = 0;
                            v33 = v128;
                            goto LABEL_185;
                          }
                          continue;
                        }
                        break;
                      }
                      v97 = *(__int64 ***)(v15 + 64);
                      v98 = &v97[*(unsigned int *)(v15 + 76)];
                      if ( v97 == v98 )
                        goto LABEL_183;
                      while ( v10 != *v97 )
                      {
                        if ( v98 == ++v97 )
                          goto LABEL_183;
                      }
LABEL_164:
                      v131 = sub_B46970((unsigned __int8 *)v96);
                      if ( v131 )
                      {
                        v33 = v128;
                        v93 = v149;
                      }
                      else
                      {
                        v103 = *(_QWORD *)(v96 + 16);
                        if ( v103 )
                        {
                          while ( 1 )
                          {
                            v104 = *(_QWORD *)(v103 + 24);
                            if ( BYTE4(v155) )
                            {
                              v105 = (_QWORD *)v153;
                              v100 = HIDWORD(v154);
                              v99 = (_QWORD *)(v153 + 8LL * HIDWORD(v154));
                              if ( (_QWORD *)v153 != v99 )
                              {
                                while ( v104 != *v105 )
                                {
                                  if ( v99 == ++v105 )
                                    goto LABEL_179;
                                }
                                goto LABEL_171;
                              }
LABEL_179:
                              if ( HIDWORD(v154) < (unsigned int)v154 )
                              {
                                ++HIDWORD(v154);
                                *v99 = v104;
                                ++v152;
                                goto LABEL_175;
                              }
                            }
                            v10 = *(__int64 **)(v103 + 24);
                            sub_C8CC70((__int64)&v152, (__int64)v10, (__int64)v99, v100, v101, v102);
                            if ( (_BYTE)v99 )
                            {
LABEL_175:
                              v106 = (unsigned int)v150;
                              v100 = HIDWORD(v150);
                              v107 = (unsigned int)v150 + 1LL;
                              if ( v107 > HIDWORD(v150) )
                              {
                                v10 = v151;
                                sub_C8D5F0((__int64)&v149, v151, v107, 8u, v101, v102);
                                v106 = (unsigned int)v150;
                              }
                              v99 = v149;
                              v149[v106] = v104;
                              LODWORD(v150) = v150 + 1;
                              v103 = *(_QWORD *)(v103 + 8);
                              if ( !v103 )
                                break;
                            }
                            else
                            {
LABEL_171:
                              v103 = *(_QWORD *)(v103 + 8);
                              if ( !v103 )
                                break;
                            }
                          }
                        }
                        v94 = v150;
                        v93 = v149;
                        if ( (_DWORD)v150 )
                          continue;
                        v33 = v128;
                      }
                      break;
                    }
LABEL_185:
                    if ( v93 != v151 )
                      _libc_free(v93, v10);
                    if ( !BYTE4(v155) )
                      _libc_free(v153, v10);
                    if ( v131 )
                      goto LABEL_43;
                    goto LABEL_89;
                  }
                  if ( (unsigned int)(v44 - 42) <= 0x11 )
                  {
                    if ( !*((_QWORD *)v33 + 2) )
                      goto LABEL_127;
                    v129 = 8 * v31;
                    v130 = v31;
                    v50 = v33;
                    v51 = *((_QWORD *)v33 + 2);
                    do
                    {
                      v52 = *(_QWORD *)(v51 + 24);
                      if ( *(_BYTE *)v52 != 84 )
                        break;
                      if ( v52 != v14 )
                      {
                        if ( !sub_D4B130(v15) )
                          break;
                        if ( *(_QWORD *)(v52 + 40) != **(_QWORD **)(v15 + 32) )
                          break;
                        v10 = (__int64 *)v15;
                        if ( !(unsigned __int8)sub_10238A0(v52, v15, a4, &v152, 0, 0) )
                          break;
                      }
                      v51 = *(_QWORD *)(v51 + 8);
                    }
                    while ( v51 );
                    v53 = v158;
                    v54 = v51;
                    v34 = v129;
                    v33 = v50;
                    v31 = v130;
                    v45 = v158;
                    if ( v54 || v33 != (unsigned __int8 *)v157 )
                      goto LABEL_59;
                    goto LABEL_72;
                  }
                }
LABEL_43:
                if ( v143 == (_DWORD)++v31 )
                  goto LABEL_7;
                continue;
              }
              v49 = 1;
              while ( v40 != (__int64 *)-4096LL )
              {
                v64 = v49 + 1;
                v38 = v37 & (v49 + v38);
                v39 = (__int64 **)(v36 + 16LL * v38);
                v40 = *v39;
                if ( v10 == *v39 )
                  goto LABEL_47;
                v49 = v64;
              }
            }
            break;
          }
          v41 = 0;
          goto LABEL_48;
        }
LABEL_7:
        while ( 1 )
        {
          v14 = v138 - 24;
          if ( *(_BYTE *)(v138 - 24) != 84 )
            break;
          v138 = *(_QWORD *)(v138 + 8);
          if ( *(_QWORD *)(v14 + 16) )
            goto LABEL_6;
        }
        v12 = v15;
      }
      v136 += 8;
    }
    while ( v132 != v136 );
    v11 = v12;
    a2 = v135;
  }
  if ( !sub_D4B130(v11) )
  {
    v17 = 0;
    goto LABEL_20;
  }
  v149 = v151;
  v150 = 0x400000000LL;
  sub_D46D90(v11, (__int64)&v149);
  v10 = &v152;
  v152 = (__int64)&v154;
  v153 = 0x800000000LL;
  sub_D474A0(v11, (__int64)&v152);
  if ( (_DWORD)v153 == 1 && (_DWORD)v150 == 1 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)v152 + 56LL); ; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      if ( *(_BYTE *)(i - 24) != 84 )
        break;
      v87 = *(_QWORD *)(i - 32);
      v88 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
      {
        v89 = 0;
        v16 = v87 + 32LL * *(unsigned int *)(i + 48);
        do
        {
          if ( *v149 == *(_QWORD *)(v16 + 8 * v89) )
          {
            v88 = 32 * v89;
            goto LABEL_152;
          }
          ++v89;
        }
        while ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != (_DWORD)v89 );
        v88 = 0x1FFFFFFFE0LL;
      }
LABEL_152:
      v90 = *(__int64 **)(v87 + v88);
      v91 = v161;
      v92 = 5LL * (unsigned int)v162;
      v10 = &v161[5 * (unsigned int)v162];
      if ( v161 == v10 )
      {
LABEL_212:
        if ( *(_BYTE *)v90 > 0x1Cu )
        {
          v10 = v90;
          if ( !sub_D484B0(v11, (__int64)v90, v92, v16) )
            goto LABEL_15;
        }
      }
      else
      {
        v16 = i - 24;
        while ( 1 )
        {
          if ( v16 == *v91 )
          {
            v92 = *(_QWORD *)(i - 32) + 32LL * *((unsigned int *)v91 + 2);
            if ( *(__int64 **)v92 == v90 )
              break;
          }
          v91 += 5;
          if ( v10 == v91 )
            goto LABEL_212;
        }
      }
    }
    v109 = *(_QWORD *)(v11 + 32);
    if ( v109 == *(_QWORD *)(v11 + 40) )
    {
LABEL_208:
      v17 = 1;
      goto LABEL_16;
    }
    v140 = a2;
    v110 = *(_QWORD *)(v11 + 40);
    while ( 1 )
    {
      v111 = *(_QWORD *)(*(_QWORD *)v109 + 56LL);
      v112 = *(_QWORD *)v109 + 48LL;
      if ( v112 != v111 )
      {
        while ( 1 )
        {
          v113 = (unsigned __int8 *)(v111 - 24);
          if ( !v111 )
            v113 = 0;
          if ( (unsigned __int8)sub_B46970(v113) )
            break;
          v111 = *(_QWORD *)(v111 + 8);
          if ( v112 == v111 )
            goto LABEL_206;
        }
        if ( v112 != v111 )
          break;
      }
LABEL_206:
      v109 += 8;
      if ( v110 == v109 )
      {
        a2 = v140;
        goto LABEL_208;
      }
    }
    a2 = v140;
  }
LABEL_15:
  v17 = 0;
LABEL_16:
  if ( (__int64 *)v152 != &v154 )
    _libc_free(v152, v10);
  if ( v149 != v151 )
    _libc_free(v149, v10);
LABEL_20:
  v18 = v161;
  v141 = &v161[5 * (unsigned int)v162];
  if ( v141 != v161 )
  {
    v137 = a2;
    v144 = 0;
    v19 = v124;
    v20 = v17 | ((a8 & 0xFFFFFFFD) != 1);
    v21 = v161;
    v139 = v20;
    while ( 1 )
    {
      while ( 1 )
      {
        v29 = *v21;
        if ( v139 || !*((_BYTE *)v21 + 32) )
          break;
        v21 += 5;
        if ( v141 == v21 )
        {
LABEL_35:
          v18 = v161;
          goto LABEL_36;
        }
      }
      LOWORD(v19) = 0;
      ++v144;
      v22 = sub_F8DB90(a6, v21[2], *(_QWORD *)(v29 + 8), v21[3] + 24, v19);
      v23 = *(_QWORD *)(v29 - 8) + 32LL * *((unsigned int *)v21 + 2);
      v24 = *(unsigned __int8 **)v23;
      if ( *(_QWORD *)v23 )
      {
        v25 = *(_QWORD *)(v23 + 8);
        **(_QWORD **)(v23 + 16) = v25;
        if ( v25 )
          *(_QWORD *)(v25 + 16) = *(_QWORD *)(v23 + 16);
      }
      *(_QWORD *)v23 = v22;
      if ( v22 )
      {
        v26 = *(_QWORD *)(v22 + 16);
        *(_QWORD *)(v23 + 8) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 16) = v23 + 8;
        *(_QWORD *)(v23 + 16) = v22 + 16;
        *(_QWORD *)(v22 + 16) = v23;
      }
      sub_DAC8D0((__int64)a4, (_BYTE *)v29);
      v10 = a3;
      if ( !sub_F50EE0(v24, a3) )
        goto LABEL_30;
      v154 = (__int64)v24;
      v152 = 6;
      v153 = 0;
      if ( v24 != 0 && v24 + 4096 != 0 && v24 != (unsigned __int8 *)-8192LL )
        sub_BD73F0((__int64)&v152);
      v65 = &v152;
      v66 = *(unsigned int *)(a9 + 8);
      v67 = *(_QWORD *)a9;
      v10 = (__int64 *)(v66 + 1);
      v68 = *(_DWORD *)(a9 + 8);
      if ( v66 + 1 > (unsigned __int64)*(unsigned int *)(a9 + 12) )
      {
        if ( v67 > (unsigned __int64)&v152 || (unsigned __int64)&v152 >= v67 + 24 * v66 )
        {
          sub_F39130(a9, (unsigned __int64)v10, v66, v67, v27, v28);
          v66 = *(unsigned int *)(a9 + 8);
          v67 = *(_QWORD *)a9;
          v65 = &v152;
          v68 = *(_DWORD *)(a9 + 8);
        }
        else
        {
          v108 = (char *)&v152 - v67;
          sub_F39130(a9, (unsigned __int64)v10, v66, v67, v27, v28);
          v67 = *(_QWORD *)a9;
          v66 = *(unsigned int *)(a9 + 8);
          v65 = (__int64 *)&v108[*(_QWORD *)a9];
          v68 = *(_DWORD *)(a9 + 8);
        }
      }
      v69 = (unsigned __int64 *)(v67 + 24 * v66);
      if ( v69 )
      {
        *v69 = 6;
        v70 = v65[2];
        v69[1] = 0;
        v69[2] = v70;
        if ( v70 == 0 || v70 == -4096 || v70 == -8192 )
        {
          v68 = *(_DWORD *)(a9 + 8);
        }
        else
        {
          v10 = (__int64 *)(*v65 & 0xFFFFFFFFFFFFFFF8LL);
          sub_BD6050(v69, (unsigned __int64)v10);
          v68 = *(_DWORD *)(a9 + 8);
        }
      }
      *(_DWORD *)(a9 + 8) = v68 + 1;
      if ( v154 != 0 && v154 != -4096 && v154 != -8192 )
      {
        sub_BD60C0(&v152);
        if ( (*(_DWORD *)(v29 + 4) & 0x7FFFFFF) != 1 )
          goto LABEL_31;
      }
      else
      {
LABEL_30:
        if ( (*(_DWORD *)(v29 + 4) & 0x7FFFFFF) != 1 )
          goto LABEL_31;
      }
      if ( *(_BYTE *)v22 > 0x1Cu )
      {
        v71 = *(__int64 **)(v22 + 40);
        v10 = *(__int64 **)(v29 + 40);
        if ( v71 != v10 )
        {
          v72 = *(_QWORD *)(v137 + 8);
          v73 = *(_DWORD *)(v137 + 24);
          if ( v73 )
          {
            v74 = v73 - 1;
            v75 = (v73 - 1) & (((unsigned int)v71 >> 9) ^ ((unsigned int)v71 >> 4));
            v76 = (__int64 **)(v72 + 16LL * v75);
            v77 = *v76;
            if ( v71 == *v76 )
            {
LABEL_121:
              v78 = v76[1];
              if ( v78 )
              {
                v79 = v74 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
                v80 = (__int64 **)(v72 + 16LL * v79);
                v81 = *v80;
                if ( *v80 != v10 )
                {
                  v118 = 1;
                  while ( v81 != (__int64 *)-4096LL )
                  {
                    v119 = v118 + 1;
                    v79 = v74 & (v79 + v118);
                    v80 = (__int64 **)(v72 + 16LL * v79);
                    v81 = *v80;
                    if ( v10 == *v80 )
                      goto LABEL_123;
                    v118 = v119;
                  }
                  goto LABEL_31;
                }
LABEL_123:
                v82 = v80[1];
                if ( v78 != v82 )
                {
                  while ( v82 )
                  {
                    v82 = (__int64 *)*v82;
                    if ( v78 == v82 )
                      goto LABEL_126;
                  }
                  goto LABEL_31;
                }
              }
            }
            else
            {
              v120 = 1;
              while ( v77 != (__int64 *)-4096LL )
              {
                v121 = v120 + 1;
                v122 = v74 & (v75 + v120);
                v75 = v122;
                v76 = (__int64 **)(v72 + 16 * v122);
                v77 = *v76;
                if ( v71 == *v76 )
                  goto LABEL_121;
                v120 = v121;
              }
            }
          }
        }
      }
LABEL_126:
      v10 = (__int64 *)v22;
      sub_BD84D0(v29, v22);
      sub_B43D60((_QWORD *)v29);
LABEL_31:
      v21 += 5;
      if ( v141 == v21 )
        goto LABEL_35;
    }
  }
  v144 = 0;
LABEL_36:
  *(_QWORD *)(a6 + 568) = 0;
  *(_QWORD *)(a6 + 576) = 0;
  *(_WORD *)(a6 + 584) = 0;
  if ( v18 != v163 )
    _libc_free(v18, v10);
  if ( v146 != v148 )
    _libc_free(v146, v10);
  return v144;
}
