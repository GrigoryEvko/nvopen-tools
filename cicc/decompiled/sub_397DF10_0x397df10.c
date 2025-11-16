// Function: sub_397DF10
// Address: 0x397df10
//
void __fastcall sub_397DF10(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // rcx
  _BYTE *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rdi
  bool v7; // zf
  _BYTE *v8; // rax
  void (__fastcall *v9)(__int64, unsigned __int64 *, __int64); // rcx
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 i; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __m128i *v18; // rax
  char v19; // al
  char *v20; // rbx
  int v21; // r14d
  __int64 v22; // r8
  int j; // ecx
  int v24; // eax
  __m128i *v25; // rax
  _QWORD *v26; // rdi
  __m128i *v27; // rax
  __int64 v28; // rbx
  _BYTE *v29; // rsi
  __int64 v30; // r14
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rax
  __int64 *v34; // r15
  __int64 v35; // rbx
  char v36; // al
  unsigned int v37; // r12d
  __int64 v38; // rax
  __int64 v39; // rdi
  void (__fastcall *v40)(__int64, __m128i *, __int64, _QWORD); // rcx
  _BYTE *v41; // rax
  unsigned __int64 *v42; // r12
  unsigned __int64 *v43; // r15
  __m128i *v44; // rax
  char v45; // al
  char v46; // r9
  char *v47; // rax
  unsigned __int64 v48; // rdi
  char v49; // al
  __int64 v50; // rsi
  __int32 v51; // ecx
  __m128i *v52; // rax
  char v53; // dl
  char v54; // dl
  unsigned int v55; // esi
  unsigned int v56; // edx
  size_t v57; // rax
  __int64 v58; // r8
  size_t v59; // rdx
  _BYTE *v60; // rax
  _BYTE *v61; // rdi
  __int64 v62; // r8
  __m128i *v63; // rax
  __int64 v64; // rdi
  void (__fastcall *v65)(__int64, unsigned __int64 *, __int64); // rcx
  _BYTE *v66; // rax
  char v67; // al
  char *v68; // r13
  __int64 v69; // r14
  int v70; // ecx
  int v71; // eax
  int v72; // ecx
  int v73; // eax
  size_t v74; // rdx
  char *v75; // rsi
  __m128i *v76; // rdx
  char v77; // al
  _BYTE *v78; // r14
  char *v79; // rax
  unsigned __int64 v80; // rdi
  __m128i *v81; // rax
  char v82; // al
  __int64 v83; // rsi
  __int32 v84; // ecx
  __m128i *v85; // rax
  char v86; // dl
  unsigned int v87; // esi
  unsigned int v88; // edx
  size_t v89; // rax
  __int64 v90; // r8
  size_t v91; // rdx
  _BYTE *v92; // rax
  _BYTE *v93; // rdi
  __int64 v94; // r8
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // rcx
  __int64 (*v98)(); // rax
  __int64 v99; // rsi
  __int64 v100; // rax
  __int64 v101; // rcx
  __int64 v102; // rdx
  __int64 v103; // rax
  __int16 *v104; // r8
  char v105; // al
  __m128i *v106; // rax
  __m128i *v107; // rax
  __int64 (*v108)(); // rax
  __int16 *v109; // r8
  _BYTE *v110; // rax
  __m128i *v111; // rax
  char v112; // dl
  __int64 v113; // [rsp+0h] [rbp-2C0h]
  __int64 v114; // [rsp+8h] [rbp-2B8h]
  __int64 v115; // [rsp+8h] [rbp-2B8h]
  unsigned int v116; // [rsp+10h] [rbp-2B0h]
  __int64 v117; // [rsp+18h] [rbp-2A8h]
  int v118; // [rsp+28h] [rbp-298h]
  __int64 v119; // [rsp+28h] [rbp-298h]
  int v120; // [rsp+34h] [rbp-28Ch]
  _BYTE *v121; // [rsp+38h] [rbp-288h]
  __int64 v122; // [rsp+38h] [rbp-288h]
  char v123; // [rsp+38h] [rbp-288h]
  __int64 v124; // [rsp+38h] [rbp-288h]
  __int64 v125; // [rsp+38h] [rbp-288h]
  int v126; // [rsp+38h] [rbp-288h]
  size_t v127; // [rsp+38h] [rbp-288h]
  unsigned __int64 v128; // [rsp+40h] [rbp-280h]
  __int64 v129; // [rsp+40h] [rbp-280h]
  __int64 v130; // [rsp+40h] [rbp-280h]
  size_t v131; // [rsp+40h] [rbp-280h]
  char *s; // [rsp+48h] [rbp-278h]
  int v134; // [rsp+58h] [rbp-268h]
  __int16 v135; // [rsp+6Eh] [rbp-252h] BYREF
  __m128i v136[2]; // [rsp+70h] [rbp-250h] BYREF
  __m128i v137; // [rsp+90h] [rbp-230h] BYREF
  char v138; // [rsp+A0h] [rbp-220h]
  char v139; // [rsp+A1h] [rbp-21Fh]
  __m128i v140; // [rsp+B0h] [rbp-210h] BYREF
  __int16 v141; // [rsp+C0h] [rbp-200h]
  __m128i v142; // [rsp+D0h] [rbp-1F0h] BYREF
  _QWORD v143[2]; // [rsp+E0h] [rbp-1E0h] BYREF
  _QWORD v144[2]; // [rsp+F0h] [rbp-1D0h] BYREF
  unsigned __int64 v145; // [rsp+100h] [rbp-1C0h]
  __m128i *v146; // [rsp+108h] [rbp-1B8h]
  int v147; // [rsp+110h] [rbp-1B0h]
  unsigned __int64 *v148; // [rsp+118h] [rbp-1A8h]
  __m128i v149; // [rsp+120h] [rbp-1A0h] BYREF
  __int64 v150; // [rsp+130h] [rbp-190h] BYREF
  __int64 v151; // [rsp+138h] [rbp-188h] BYREF
  int v152; // [rsp+140h] [rbp-180h]
  __int64 v153[2]; // [rsp+148h] [rbp-178h] BYREF
  _QWORD v154[2]; // [rsp+158h] [rbp-168h] BYREF
  __int64 *v155; // [rsp+168h] [rbp-158h]
  __int64 *v156; // [rsp+170h] [rbp-150h]
  char *v157; // [rsp+178h] [rbp-148h]
  unsigned __int64 v158[2]; // [rsp+180h] [rbp-140h] BYREF
  _WORD v159[152]; // [rsp+190h] [rbp-130h] BYREF

  v3 = *(_BYTE **)(a2 + 32);
  v4 = v3;
  if ( !*v3 )
  {
    LODWORD(v5) = 0;
    do
    {
      if ( (v4[3] & 0x10) == 0 )
        break;
      v5 = (unsigned int)(v5 + 1);
      v4 = &v3[40 * v5];
    }
    while ( !*v4 );
  }
  v6 = a1[32];
  s = (char *)*((_QWORD *)v4 + 3);
  v7 = *s == 0;
  v8 = *(_BYTE **)(a1[30] + 128LL);
  v9 = *(void (__fastcall **)(__int64, unsigned __int64 *, __int64))(*(_QWORD *)v6 + 120LL);
  v159[0] = 257;
  if ( !v7 )
  {
    if ( *v8 )
    {
      v158[0] = (unsigned __int64)v8;
      LOBYTE(v159[0]) = 3;
    }
    v9(v6, v158, 1);
    v10 = *(_DWORD *)(a2 + 40);
    v120 = v10;
    if ( v10 )
    {
      v11 = *(_QWORD *)(a2 + 32);
      v12 = 0;
      for ( i = v11 + 40LL * (unsigned int)(v10 - 1); ; i -= 40 )
      {
        if ( *(_BYTE *)i == 14 )
        {
          v12 = *(_QWORD *)(i + 24);
          if ( v12 )
          {
            v14 = *(unsigned int *)(v12 + 8);
            if ( (_DWORD)v14 )
            {
              v15 = *(_QWORD *)(v12 - 8 * v14);
              if ( *(_BYTE *)v15 == 1 )
              {
                v16 = *(_QWORD *)(v15 + 136);
                if ( *(_BYTE *)(v16 + 16) == 13 )
                  break;
              }
            }
          }
        }
        if ( i == v11 )
        {
          v117 = v12;
          v120 = 0;
          goto LABEL_20;
        }
      }
      v17 = *(_QWORD **)(v16 + 24);
      v117 = v12;
      if ( *(_DWORD *)(v16 + 32) > 0x40u )
        v17 = (_QWORD *)*v17;
      v120 = (int)v17;
    }
    else
    {
      v117 = 0;
    }
LABEL_20:
    v158[0] = (unsigned __int64)v159;
    v158[1] = 0x10000000000LL;
    v148 = v158;
    v147 = 1;
    v144[0] = &unk_49EFC48;
    v146 = 0;
    v145 = 0;
    v144[1] = 0;
    sub_16E7A40((__int64)v144, 0, 0, 0);
    v134 = *(_DWORD *)(a1[30] + 168LL);
    v116 = sub_1E16470(a2);
    if ( !v116 )
    {
      v114 = a1[34];
      v118 = *(_DWORD *)(a2 + 40);
      v18 = v146;
      if ( (unsigned __int64)v146 >= v145 )
      {
        sub_16E7DE0((__int64)v144, 9);
      }
      else
      {
        v146 = (__m128i *)((char *)v146 + 1);
        v18->m128i_i8[0] = 9;
      }
      v19 = *s;
      if ( *s )
      {
        v20 = s;
        v21 = -1;
        while ( 1 )
        {
          v22 = (__int64)(v20 + 1);
          if ( v19 == 10 )
          {
            v44 = v146;
            if ( (unsigned __int64)v146 >= v145 )
            {
              sub_16E7DE0((__int64)v144, 10);
              ++v20;
            }
            else
            {
              ++v20;
              v146 = (__m128i *)((char *)v146 + 1);
              v44->m128i_i8[0] = 10;
            }
            goto LABEL_35;
          }
          if ( v19 != 36 )
          {
            for ( j = (unsigned __int8)v20[1]; ; j = *(unsigned __int8 *)++v22 )
            {
              v24 = j - 123;
              LOBYTE(v24) = (unsigned __int8)(j - 123) <= 2u;
              if ( (unsigned __int8)j <= 0x24u )
                v24 |= (0x1000000401uLL >> j) & 1;
              if ( (_BYTE)v24 )
                break;
            }
            if ( v21 == -1 || v134 == v21 )
            {
              v122 = v22;
              sub_16E7EE0((__int64)v144, v20, v22 - (_QWORD)v20);
              v20 = (char *)v122;
            }
            else
            {
              v20 = (char *)v22;
            }
            goto LABEL_35;
          }
          v45 = v20[1];
          if ( v45 == 41 )
          {
            v20 += 2;
            if ( v21 == -1 )
            {
              v107 = v146;
              if ( (unsigned __int64)v146 >= v145 )
              {
                sub_16E7DE0((__int64)v144, 125);
              }
              else
              {
                v146 = (__m128i *)((char *)v146 + 1);
                v107->m128i_i8[0] = 125;
              }
            }
            else
            {
              v21 = -1;
            }
            goto LABEL_35;
          }
          if ( v45 <= 41 )
            break;
          if ( v45 == 124 )
          {
            v20 += 2;
            if ( v21 == -1 )
            {
              v106 = v146;
              if ( (unsigned __int64)v146 >= v145 )
              {
                sub_16E7DE0((__int64)v144, 124);
              }
              else
              {
                v146 = (__m128i *)((char *)v146 + 1);
                v106->m128i_i8[0] = 124;
              }
            }
            else
            {
              ++v21;
            }
            goto LABEL_35;
          }
          if ( v45 != 123 )
            goto LABEL_82;
          v45 = v20[2];
          v22 = (__int64)(v20 + 2);
          v46 = 1;
          if ( v45 == 58 )
          {
            v121 = v20 + 3;
            v47 = strchr(v20 + 3, 125);
            if ( !v47 )
            {
              v7 = *s == 0;
              v52 = (__m128i *)"Unterminated ${:foo} operand in inline asm string: '";
              v149.m128i_i64[0] = (__int64)"Unterminated ${:foo} operand in inline asm string: '";
              if ( !v7 )
              {
                v53 = 2;
                LOWORD(v150) = 771;
                v149.m128i_i64[1] = (__int64)s;
                v52 = &v149;
                goto LABEL_90;
              }
LABEL_220:
              v53 = 3;
              LOWORD(v150) = 259;
              goto LABEL_90;
            }
            v149.m128i_i64[0] = (__int64)&v150;
            v20 = v47 + 1;
            sub_397D0D0(v149.m128i_i64, v121, (__int64)v47);
            (*(void (__fastcall **)(_QWORD *, __int64, _QWORD *, __int64))(*a1 + 352LL))(
              a1,
              a2,
              v144,
              v149.m128i_i64[0]);
            v48 = v149.m128i_i64[0];
            if ( (__int64 *)v149.m128i_i64[0] != &v150 )
              goto LABEL_79;
            goto LABEL_35;
          }
LABEL_83:
          v20 = (char *)v22;
          if ( (unsigned __int8)(v45 - 48) > 9u )
          {
            v50 = 0;
          }
          else
          {
            do
              v49 = *++v20;
            while ( (unsigned __int8)(v49 - 48) <= 9u );
            v50 = (__int64)&v20[-v22];
          }
          v123 = v46;
          if ( sub_16D2B80(v22, v50, 0xAu, (unsigned __int64 *)&v149)
            || (v51 = v149.m128i_i32[0], v149.m128i_i64[0] != v149.m128i_u32[0]) )
          {
            v7 = *s == 0;
            v52 = (__m128i *)"Bad $ operand number in inline asm string: '";
            v149.m128i_i64[0] = (__int64)"Bad $ operand number in inline asm string: '";
            if ( v7 )
              goto LABEL_220;
            goto LABEL_89;
          }
          v135 = 0;
          if ( v123 )
          {
            v54 = *v20;
            if ( *v20 == 58 )
            {
              if ( !v20[1] )
              {
                v139 = 1;
                v137.m128i_i64[0] = (__int64)"'";
                v138 = 3;
                LOWORD(v143[0]) = 257;
                if ( *s )
                {
                  v142.m128i_i64[0] = (__int64)s;
                  LOBYTE(v143[0]) = 3;
                }
                v149.m128i_i64[0] = (__int64)"Bad ${:} expression in inline asm string: '";
                LOWORD(v150) = 259;
                sub_14EC200(&v140, &v149, &v142);
                sub_14EC200(v136, &v140, &v137);
                sub_16BCFB0((__int64)v136, 1u);
              }
              LOBYTE(v135) = v20[1];
              v20 += 2;
              v54 = *v20;
            }
            if ( v54 != 125 )
            {
              v7 = *s == 0;
              v52 = (__m128i *)"Bad ${} expression in inline asm string: '";
              v149.m128i_i64[0] = (__int64)"Bad ${} expression in inline asm string: '";
              if ( v7 )
                goto LABEL_220;
LABEL_89:
              v53 = 2;
              LOWORD(v150) = 771;
              v149.m128i_i64[1] = (__int64)s;
              v52 = &v149;
LABEL_90:
              v142.m128i_i64[0] = (__int64)v52;
              v142.m128i_i64[1] = (__int64)"'";
              LOBYTE(v143[0]) = v53;
              BYTE1(v143[0]) = 3;
              sub_16BCFB0((__int64)&v142, 1u);
            }
            ++v20;
          }
          if ( (unsigned int)(v118 - 1) <= v149.m128i_i32[0] )
          {
            v7 = *s == 0;
            v52 = (__m128i *)"Invalid $ operand number in inline asm string: '";
            v149.m128i_i64[0] = (__int64)"Invalid $ operand number in inline asm string: '";
            if ( !v7 )
              goto LABEL_89;
            goto LABEL_220;
          }
          if ( v21 == -1 || v134 == v21 )
          {
            v55 = *(_DWORD *)(a2 + 40);
            v56 = 2;
            if ( v149.m128i_i32[0] )
            {
              while ( v56 < v55 )
              {
                v56 += (((unsigned int)*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v56 + 24) >> 3) & 0x1FFF) + 1;
                if ( !--v51 )
                  goto LABEL_184;
              }
            }
            else
            {
LABEL_184:
              if ( v56 < v55 )
              {
                v99 = *(_QWORD *)(a2 + 32);
                v100 = v99 + 40LL * v56;
                if ( *(_BYTE *)v100 != 14 )
                {
                  v101 = *(_QWORD *)(v100 + 24);
                  v102 = v56 + 1;
                  if ( (_BYTE)v135 == 108 )
                  {
                    v110 = (_BYTE *)sub_1DD5A70(*(_QWORD *)(v99 + 40 * v102 + 24));
                    sub_38E2490(v110, (__int64)v144, (_BYTE *)a1[30]);
                    goto LABEL_35;
                  }
                  v103 = *a1;
                  if ( (v101 & 7) != 6 )
                  {
                    v104 = &v135;
                    if ( !(_BYTE)v135 )
                      v104 = 0;
                    v105 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD, __int16 *, _QWORD *))(v103 + 360))(
                             a1,
                             a2,
                             v102,
                             0,
                             v104,
                             v144);
                    goto LABEL_191;
                  }
                  v108 = *(__int64 (**)())(v103 + 368);
                  v109 = &v135;
                  if ( !(_BYTE)v135 )
                    v109 = 0;
                  if ( v108 != sub_397CF50 )
                  {
                    v105 = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, __int16 *, _QWORD *))v108)(
                             a1,
                             a2,
                             v102,
                             0,
                             v109,
                             v144);
LABEL_191:
                    if ( !v105 )
                      goto LABEL_35;
                  }
                }
              }
            }
            v142.m128i_i64[1] = 0;
            v142.m128i_i64[0] = (__int64)v143;
            v149.m128i_i64[0] = (__int64)&unk_49EFBE0;
            LOBYTE(v143[0]) = 0;
            v152 = 1;
            v151 = 0;
            v150 = 0;
            v149.m128i_i64[1] = 0;
            v153[0] = (__int64)&v142;
            v124 = sub_16E7EE0((__int64)&v149, "invalid operand in inline asm: '", 0x20u);
            v57 = strlen(s);
            v58 = v124;
            v59 = v57;
            v60 = *(_BYTE **)(v124 + 16);
            v61 = *(_BYTE **)(v124 + 24);
            if ( v59 > v60 - v61 )
            {
              v58 = sub_16E7EE0(v124, s, v59);
              v60 = *(_BYTE **)(v58 + 16);
              v61 = *(_BYTE **)(v58 + 24);
            }
            else if ( v59 )
            {
              v113 = v124;
              v127 = v59;
              memcpy(v61, s, v59);
              v58 = v113;
              v61 = (_BYTE *)(*(_QWORD *)(v113 + 24) + v127);
              v60 = *(_BYTE **)(v113 + 16);
              *(_QWORD *)(v113 + 24) = v61;
            }
            if ( v61 == v60 )
            {
              sub_16E7EE0(v58, "'", 1u);
            }
            else
            {
              *v61 = 39;
              ++*(_QWORD *)(v58 + 24);
            }
            v62 = **(_QWORD **)(v114 + 1688);
            if ( v151 != v149.m128i_i64[1] )
            {
              v125 = **(_QWORD **)(v114 + 1688);
              sub_16E7BA0(v149.m128i_i64);
              v62 = v125;
            }
            v141 = 260;
            v140.m128i_i64[0] = v153[0];
            sub_1602B40(v62, v120, (__int64)&v140);
            sub_16E7BC0(v149.m128i_i64);
            v48 = v142.m128i_i64[0];
            if ( (_QWORD *)v142.m128i_i64[0] == v143 )
              goto LABEL_35;
LABEL_79:
            j_j___libc_free_0(v48);
          }
LABEL_35:
          v19 = *v20;
          if ( !*v20 )
            goto LABEL_36;
        }
        if ( v45 == 36 )
        {
          if ( v21 == -1 || v134 == v21 )
          {
            v63 = v146;
            if ( (unsigned __int64)v146 >= v145 )
            {
              sub_16E7DE0((__int64)v144, 36);
            }
            else
            {
              v146 = (__m128i *)((char *)v146 + 1);
              v63->m128i_i8[0] = 36;
            }
          }
          v20 += 2;
          goto LABEL_35;
        }
        if ( v45 == 40 )
        {
          v20 += 2;
          if ( v21 != -1 )
          {
            if ( *s )
            {
              v149.m128i_i64[1] = (__int64)s;
              v53 = 2;
              v52 = &v149;
              v149.m128i_i64[0] = (__int64)"Nested variants found in inline asm string: '";
              LOWORD(v150) = 771;
            }
            else
            {
              v52 = (__m128i *)"Nested variants found in inline asm string: '";
              v53 = 3;
              LOWORD(v150) = 259;
              v149.m128i_i64[0] = (__int64)"Nested variants found in inline asm string: '";
            }
            goto LABEL_90;
          }
          v21 = 0;
          goto LABEL_35;
        }
LABEL_82:
        v46 = 0;
        goto LABEL_83;
      }
LABEL_36:
      v25 = v146;
      if ( (unsigned __int64)v146 >= v145 )
      {
        v26 = (_QWORD *)sub_16E7DE0((__int64)v144, 10);
      }
      else
      {
        v26 = v144;
        v146 = (__m128i *)((char *)v146 + 1);
        v25->m128i_i8[0] = 10;
      }
      v27 = (__m128i *)v26[3];
      if ( (unsigned __int64)v27 < v26[2] )
        goto LABEL_39;
LABEL_141:
      sub_16E7DE0((__int64)v26, 0);
      goto LABEL_40;
    }
    v119 = a1[34];
    if ( v145 - (unsigned __int64)v146 <= 0xF )
      sub_16E7EE0((__int64)v144, "\t.intel_syntax\n\t", 0x10u);
    else
      *v146++ = _mm_load_si128((const __m128i *)&xmmword_44D4680);
    v126 = *(_DWORD *)(a2 + 40);
    v67 = *s;
    if ( !*s )
    {
LABEL_139:
      v76 = v146;
      if ( v145 - (unsigned __int64)v146 <= 0xD )
      {
        v26 = (_QWORD *)sub_16E7EE0((__int64)v144, "\n\t.att_syntax\n", 0xEu);
        v27 = (__m128i *)v26[3];
        if ( (unsigned __int64)v27 >= v26[2] )
          goto LABEL_141;
      }
      else
      {
        v146->m128i_i32[2] = 1635020409;
        v26 = v144;
        v76->m128i_i64[0] = 0x735F7474612E090ALL;
        v76->m128i_i16[6] = 2680;
        v27 = (__m128i *)((char *)&v146->m128i_u64[1] + 6);
        v146 = v27;
        if ( (unsigned __int64)v27 >= v145 )
          goto LABEL_141;
      }
LABEL_39:
      v26[3] = (char *)v27->m128i_i64 + 1;
      v27->m128i_i8[0] = 0;
LABEL_40:
      v28 = a1[29];
      v149.m128i_i16[0] = *(_WORD *)(v28 + 840) & 0x3FFF | v149.m128i_i16[0] & 0xC000;
      v149.m128i_i32[1] = *(_DWORD *)(v28 + 844);
      v149.m128i_i64[1] = (__int64)&v151;
      sub_397D180(&v149.m128i_i64[1], *(_BYTE **)(v28 + 848), *(_QWORD *)(v28 + 848) + *(_QWORD *)(v28 + 856));
      v153[0] = (__int64)v154;
      v29 = *(_BYTE **)(v28 + 880);
      sub_397D180(v153, v29, (__int64)&v29[*(_QWORD *)(v28 + 888)]);
      v30 = *(_QWORD *)(v28 + 920);
      v31 = *(_QWORD *)(v28 + 912);
      v155 = 0;
      v156 = 0;
      v157 = 0;
      v32 = v30 - v31;
      if ( v30 == v31 )
      {
        v34 = 0;
      }
      else
      {
        if ( v32 > 0x7FFFFFFFFFFFFFE0LL )
          sub_4261EA(v153, v29, v32);
        v128 = v30 - v31;
        v33 = sub_22077B0(v32);
        v30 = *(_QWORD *)(v28 + 920);
        v32 = v128;
        v34 = (__int64 *)v33;
        v31 = *(_QWORD *)(v28 + 912);
      }
      v155 = v34;
      v156 = v34;
      v157 = (char *)v34 + v32;
      if ( v31 != v30 )
      {
        v35 = v31;
        do
        {
          if ( v34 )
          {
            *v34 = (__int64)(v34 + 2);
            sub_397D180(v34, *(_BYTE **)v35, *(_QWORD *)v35 + *(_QWORD *)(v35 + 8));
          }
          v35 += 32;
          v34 += 4;
        }
        while ( v30 != v35 );
      }
      v156 = v34;
      v36 = sub_1560180(*(_QWORD *)a1[33] + 112LL, 42);
      v149.m128i_i8[0] = v36 & 1 | v149.m128i_i8[0] & 0xFE;
      v37 = sub_1E16470(a2);
      v38 = sub_396E580((__int64)a1);
      sub_397D680(a1, (char *)*v148, *((unsigned int *)v148 + 2), v38, (__int64)&v149, v117, v37);
      v39 = a1[32];
      v40 = *(void (__fastcall **)(__int64, __m128i *, __int64, _QWORD))(*(_QWORD *)v39 + 120LL);
      v41 = *(_BYTE **)(a1[30] + 136LL);
      LOWORD(v143[0]) = 257;
      if ( *v41 )
      {
        v142.m128i_i64[0] = (__int64)v41;
        LOBYTE(v143[0]) = 3;
      }
      v40(v39, &v142, 1, v40);
      v42 = (unsigned __int64 *)v156;
      v43 = (unsigned __int64 *)v155;
      if ( v156 != v155 )
      {
        do
        {
          if ( (unsigned __int64 *)*v43 != v43 + 2 )
            j_j___libc_free_0(*v43);
          v43 += 4;
        }
        while ( v42 != v43 );
        v43 = (unsigned __int64 *)v155;
      }
      if ( v43 )
        j_j___libc_free_0((unsigned __int64)v43);
      if ( (_QWORD *)v153[0] != v154 )
        j_j___libc_free_0(v153[0]);
      if ( (__int64 *)v149.m128i_i64[1] != &v151 )
        j_j___libc_free_0(v149.m128i_u64[1]);
      v144[0] = &unk_49EFD28;
      sub_16E7960((__int64)v144);
      if ( (_WORD *)v158[0] != v159 )
        _libc_free(v158[0]);
      return;
    }
    v68 = s;
    while ( 1 )
    {
      v69 = (__int64)(v68 + 1);
      if ( v67 == 10 )
      {
        v81 = v146;
        ++v68;
        if ( (unsigned __int64)v146 >= v145 )
        {
          sub_16E7DE0((__int64)v144, 10);
        }
        else
        {
          v146 = (__m128i *)((char *)v146 + 1);
          v81->m128i_i8[0] = 10;
        }
        goto LABEL_138;
      }
      if ( v67 != 36 )
      {
        v70 = (unsigned __int8)v68[1];
        v71 = v70 - 123;
        LOBYTE(v71) = (unsigned __int8)(v70 - 123) <= 2u;
        if ( (unsigned __int8)v70 <= 0x24u )
          v71 |= (0x1000000401uLL >> v70) & 1;
        if ( (_BYTE)v71 )
        {
          v74 = 1;
        }
        else
        {
          do
          {
            v72 = *(unsigned __int8 *)++v69;
            v73 = v72 - 123;
            LOBYTE(v73) = (unsigned __int8)(v72 - 123) <= 2u;
            if ( (unsigned __int8)v72 <= 0x24u )
              v73 |= (0x1000000401uLL >> v72) & 1;
          }
          while ( !(_BYTE)v73 );
          v74 = v69 - (_QWORD)v68;
        }
        v75 = v68;
        v68 = (char *)v69;
        sub_16E7EE0((__int64)v144, v75, v74);
        goto LABEL_138;
      }
      v77 = v68[1];
      if ( v77 == 36 )
      {
        v68 += 2;
        goto LABEL_138;
      }
      if ( v77 == 123 )
        break;
      ++v68;
      if ( (unsigned __int8)(v77 - 48) > 9u )
        goto LABEL_180;
      do
        v82 = *++v68;
      while ( (unsigned __int8)(v82 - 48) <= 9u );
      v83 = (__int64)&v68[-v69];
LABEL_153:
      if ( sub_16D2B80(v69, v83, 0xAu, (unsigned __int64 *)&v149)
        || (v84 = v149.m128i_i32[0], v149.m128i_i64[0] != v149.m128i_u32[0]) )
      {
        v7 = *s == 0;
        v85 = (__m128i *)"Bad $ operand number in inline asm string: '";
        v142.m128i_i64[0] = (__int64)"Bad $ operand number in inline asm string: '";
        if ( !v7 )
          goto LABEL_156;
LABEL_227:
        v86 = 3;
        LOWORD(v143[0]) = 259;
        goto LABEL_228;
      }
      if ( (unsigned int)(v126 - 1) <= v149.m128i_i32[0] )
      {
        v7 = *s == 0;
        v85 = (__m128i *)"Invalid $ operand number in inline asm string: '";
        v142.m128i_i64[0] = (__int64)"Invalid $ operand number in inline asm string: '";
        if ( v7 )
          goto LABEL_227;
LABEL_156:
        v86 = 2;
        LOWORD(v143[0]) = 771;
        v142.m128i_i64[1] = (__int64)s;
        v85 = &v142;
LABEL_228:
        v149.m128i_i64[0] = (__int64)v85;
        v149.m128i_i64[1] = (__int64)"'";
        LOBYTE(v150) = v86;
        BYTE1(v150) = 3;
        sub_16BCFB0((__int64)&v149, 1u);
      }
      v87 = *(_DWORD *)(a2 + 40);
      v88 = 2;
      if ( v149.m128i_i32[0] )
      {
        while ( v88 < v87 )
        {
          v88 += (((unsigned int)*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v88 + 24) >> 3) & 0x1FFF) + 1;
          if ( !--v84 )
            goto LABEL_173;
        }
      }
      else
      {
LABEL_173:
        if ( v88 < v87 )
        {
          v95 = *(_QWORD *)(a2 + 32) + 40LL * v88;
          if ( *(_BYTE *)v95 != 14 )
          {
            v96 = v88 + 1;
            v97 = *a1;
            if ( (*(_DWORD *)(v95 + 24) & 7) == 6 )
            {
              v98 = *(__int64 (**)())(v97 + 368);
              if ( v98 == sub_397CF50 )
                goto LABEL_162;
            }
            else
            {
              v98 = *(__int64 (**)())(v97 + 360);
              if ( (char *)v98 == (char *)&loc_397D030 )
                goto LABEL_162;
            }
            if ( !((unsigned __int8 (__fastcall *)(_QWORD *, __int64, __int64, _QWORD, _QWORD, _QWORD *))v98)(
                    a1,
                    a2,
                    v96,
                    v116,
                    0,
                    v144) )
              goto LABEL_138;
          }
        }
      }
LABEL_162:
      v142.m128i_i64[1] = 0;
      v142.m128i_i64[0] = (__int64)v143;
      v149.m128i_i64[0] = (__int64)&unk_49EFBE0;
      LOBYTE(v143[0]) = 0;
      v152 = 1;
      v151 = 0;
      v150 = 0;
      v149.m128i_i64[1] = 0;
      v153[0] = (__int64)&v142;
      v129 = sub_16E7EE0((__int64)&v149, "invalid operand in inline asm: '", 0x20u);
      v89 = strlen(s);
      v90 = v129;
      v91 = v89;
      v92 = *(_BYTE **)(v129 + 16);
      v93 = *(_BYTE **)(v129 + 24);
      if ( v91 > v92 - v93 )
      {
        v90 = sub_16E7EE0(v129, s, v91);
        v93 = *(_BYTE **)(v90 + 24);
        if ( *(_BYTE **)(v90 + 16) != v93 )
          goto LABEL_166;
      }
      else
      {
        if ( v91 )
        {
          v115 = v129;
          v131 = v91;
          memcpy(v93, s, v91);
          v90 = v115;
          v93 = (_BYTE *)(*(_QWORD *)(v115 + 24) + v131);
          v92 = *(_BYTE **)(v115 + 16);
          *(_QWORD *)(v115 + 24) = v93;
        }
        if ( v92 != v93 )
        {
LABEL_166:
          *v93 = 39;
          ++*(_QWORD *)(v90 + 24);
          goto LABEL_167;
        }
      }
      sub_16E7EE0(v90, "'", 1u);
LABEL_167:
      v94 = **(_QWORD **)(v119 + 1688);
      if ( v151 != v149.m128i_i64[1] )
      {
        v130 = **(_QWORD **)(v119 + 1688);
        sub_16E7BA0(v149.m128i_i64);
        v94 = v130;
      }
      v141 = 260;
      v140.m128i_i64[0] = v153[0];
      sub_1602B40(v94, v120, (__int64)&v140);
      sub_16E7BC0(v149.m128i_i64);
      v80 = v142.m128i_i64[0];
      if ( (_QWORD *)v142.m128i_i64[0] == v143 )
        goto LABEL_138;
LABEL_147:
      j_j___libc_free_0(v80);
LABEL_138:
      v67 = *v68;
      if ( !*v68 )
        goto LABEL_139;
    }
    if ( v68[2] == 58 )
    {
      v78 = v68 + 3;
      v79 = strchr(v68 + 3, 125);
      if ( !v79 )
      {
        v7 = *s == 0;
        v111 = (__m128i *)"Unterminated ${:foo} operand in inline asm string: '";
        v142.m128i_i64[0] = (__int64)"Unterminated ${:foo} operand in inline asm string: '";
        if ( v7 )
        {
          LOWORD(v143[0]) = 259;
          v112 = 3;
        }
        else
        {
          v112 = 2;
          LOWORD(v143[0]) = 771;
          v142.m128i_i64[1] = (__int64)s;
          v111 = &v142;
        }
        v149.m128i_i64[0] = (__int64)v111;
        v149.m128i_i64[1] = (__int64)"'";
        LOBYTE(v150) = v112;
        BYTE1(v150) = 3;
        sub_16BCFB0((__int64)&v149, 1u);
      }
      v68 = v79 + 1;
      v149.m128i_i64[0] = (__int64)&v150;
      sub_397D0D0(v149.m128i_i64, v78, (__int64)v79);
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD *, __int64))(*a1 + 352LL))(a1, a2, v144, v149.m128i_i64[0]);
      v80 = v149.m128i_i64[0];
      if ( (__int64 *)v149.m128i_i64[0] == &v150 )
        goto LABEL_138;
      goto LABEL_147;
    }
    ++v68;
LABEL_180:
    v83 = 0;
    goto LABEL_153;
  }
  if ( *v8 )
  {
    v158[0] = (unsigned __int64)v8;
    LOBYTE(v159[0]) = 3;
  }
  v9(v6, v158, 1);
  v64 = a1[32];
  v65 = *(void (__fastcall **)(__int64, unsigned __int64 *, __int64))(*(_QWORD *)v64 + 120LL);
  v66 = *(_BYTE **)(a1[30] + 136LL);
  v159[0] = 257;
  if ( *v66 )
  {
    v158[0] = (unsigned __int64)v66;
    LOBYTE(v159[0]) = 3;
  }
  v65(v64, v158, 1);
}
