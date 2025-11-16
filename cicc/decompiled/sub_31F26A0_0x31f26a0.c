// Function: sub_31F26A0
// Address: 0x31f26a0
//
void __fastcall sub_31F26A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char *v3; // rsi
  __int64 v5; // rdi
  bool v6; // zf
  char *v7; // rax
  void (__fastcall *v8)(__int64, char **, __int64); // rcx
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int8 v11; // al
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 (*v15)(void); // rax
  unsigned __int8 v16; // al
  unsigned __int8 *v17; // r10
  int v18; // r14d
  __int64 v19; // r13
  int i; // ecx
  int v21; // eax
  __m128i *v22; // rax
  _QWORD *v23; // rdi
  _BYTE *v24; // rax
  unsigned int v25; // ebx
  __int64 v26; // rax
  __m128i *v27; // r9
  _QWORD *v28; // r15
  unsigned int v29; // r14d
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 (*v33)(); // rax
  char v34; // al
  __int64 v35; // rax
  __int64 v36; // r8
  unsigned __int64 v37; // rdx
  char v38; // al
  __m128i *v39; // rax
  char v40; // r8
  char *v41; // rax
  char *v42; // r13
  unsigned __int8 *v43; // r10
  char v44; // al
  __int64 v45; // rsi
  __int32 v46; // ecx
  __m128i *v47; // rdx
  char v48; // al
  __int64 v49; // rsi
  unsigned __int8 v50; // dl
  unsigned int v51; // edx
  unsigned int v52; // esi
  __int64 *v53; // rax
  __int64 v54; // r13
  __m128i *v55; // rdx
  char v56; // al
  unsigned int v57; // r12d
  __int64 v58; // rbx
  __int64 v59; // rax
  __int64 v60; // rdi
  void (__fastcall *v61)(__int64, __m128i *, __int64, _QWORD); // rcx
  _BYTE *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rsi
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // rax
  __int16 *v68; // rcx
  char v69; // al
  __m128i *v70; // rax
  __m128i *v71; // rax
  __m128i *v72; // rax
  __int64 v73; // rdi
  void (__fastcall *v74)(__int64, char **, __int64); // rcx
  char *v75; // rax
  __int64 v76; // rax
  __m128i si128; // xmm0
  __m128i v78; // xmm0
  __int64 v79; // r13
  char *v80; // rsi
  unsigned int *v81; // rbx
  size_t v82; // rdx
  __int64 v83; // r12
  unsigned __int64 v84; // rdx
  char *v85; // r13
  const char *(__fastcall *v86)(__int64, unsigned int); // rax
  __int64 v87; // rax
  __int64 v88; // r13
  unsigned int *v89; // rbx
  __int64 (__fastcall *v90)(__int64); // r8
  __int16 *v91; // rcx
  __int64 (*v92)(); // rax
  __m128i *v93; // rax
  __m128i *v94; // rax
  __int64 v95; // r13
  __int64 v96; // rdi
  __int64 v97; // rax
  __int64 v98; // rdi
  __m128i *v99; // rdx
  char v100; // al
  __m128i *v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __m128i *v104; // rdx
  __m128i *v105; // rdx
  __int64 v106; // [rsp+8h] [rbp-328h]
  __int64 v107; // [rsp+10h] [rbp-320h]
  __int64 v108; // [rsp+18h] [rbp-318h]
  __int64 v109; // [rsp+20h] [rbp-310h]
  __int64 v110; // [rsp+28h] [rbp-308h]
  _QWORD *v111; // [rsp+30h] [rbp-300h]
  __int64 v112; // [rsp+40h] [rbp-2F0h]
  __int64 v113; // [rsp+48h] [rbp-2E8h]
  __int64 v114; // [rsp+50h] [rbp-2E0h]
  int v115; // [rsp+64h] [rbp-2CCh]
  int v116; // [rsp+68h] [rbp-2C8h]
  char *v118; // [rsp+78h] [rbp-2B8h]
  unsigned __int8 *v119; // [rsp+80h] [rbp-2B0h]
  unsigned __int8 v120; // [rsp+88h] [rbp-2A8h]
  __m128i *v121; // [rsp+90h] [rbp-2A0h]
  _BYTE *v122; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v123; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v124; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v125; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v126; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v127; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v128; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v129; // [rsp+90h] [rbp-2A0h]
  unsigned __int8 *v130; // [rsp+90h] [rbp-2A0h]
  unsigned int v131; // [rsp+90h] [rbp-2A0h]
  int v132; // [rsp+98h] [rbp-298h]
  unsigned int v133; // [rsp+98h] [rbp-298h]
  __int64 v134; // [rsp+98h] [rbp-298h]
  __int64 v135; // [rsp+98h] [rbp-298h]
  __m128i *v136; // [rsp+98h] [rbp-298h]
  __int16 v137; // [rsp+AEh] [rbp-282h] BYREF
  __m128i v138; // [rsp+B0h] [rbp-280h] BYREF
  _QWORD v139[2]; // [rsp+C0h] [rbp-270h] BYREF
  char v140; // [rsp+D0h] [rbp-260h]
  char v141; // [rsp+D1h] [rbp-25Fh]
  __m128i v142; // [rsp+E0h] [rbp-250h] BYREF
  __int64 v143; // [rsp+F0h] [rbp-240h] BYREF
  __int16 v144; // [rsp+100h] [rbp-230h]
  __m128i v145; // [rsp+110h] [rbp-220h] BYREF
  char *v146; // [rsp+120h] [rbp-210h]
  __int16 v147; // [rsp+130h] [rbp-200h]
  __m128i v148; // [rsp+140h] [rbp-1F0h] BYREF
  char *v149; // [rsp+150h] [rbp-1E0h]
  __int16 v150; // [rsp+160h] [rbp-1D0h]
  __m128i v151; // [rsp+170h] [rbp-1C0h] BYREF
  _QWORD v152[2]; // [rsp+180h] [rbp-1B0h] BYREF
  char v153; // [rsp+190h] [rbp-1A0h]
  char v154; // [rsp+191h] [rbp-19Fh]
  _QWORD v155[3]; // [rsp+1A0h] [rbp-190h] BYREF
  unsigned __int64 v156; // [rsp+1B8h] [rbp-178h]
  __m128i *v157; // [rsp+1C0h] [rbp-170h]
  __int64 v158; // [rsp+1C8h] [rbp-168h]
  char **v159; // [rsp+1D0h] [rbp-160h]
  char *v160; // [rsp+1E0h] [rbp-150h] BYREF
  unsigned __int64 v161; // [rsp+1E8h] [rbp-148h]
  __int64 v162; // [rsp+1F0h] [rbp-140h]
  char v163; // [rsp+1F8h] [rbp-138h] BYREF
  __int16 v164; // [rsp+200h] [rbp-130h]

  v2 = a2;
  v3 = *(char **)(*(_QWORD *)(a2 + 32) + 24LL);
  v5 = *(_QWORD *)(a1 + 224);
  v6 = *v3 == 0;
  v118 = v3;
  v7 = *(char **)(*(_QWORD *)(a1 + 208) + 136LL);
  v8 = *(void (__fastcall **)(__int64, char **, __int64))(*(_QWORD *)v5 + 136LL);
  v164 = 257;
  if ( v6 )
  {
    if ( *v7 )
    {
      v160 = v7;
      LOBYTE(v164) = 3;
    }
    v8(v5, &v160, 1);
    v73 = *(_QWORD *)(a1 + 224);
    v74 = *(void (__fastcall **)(__int64, char **, __int64))(*(_QWORD *)v73 + 136LL);
    v75 = *(char **)(*(_QWORD *)(a1 + 208) + 144LL);
    v164 = 257;
    if ( *v75 )
    {
      v160 = v75;
      LOBYTE(v164) = 3;
    }
    v74(v73, &v160, 1);
    return;
  }
  if ( *v7 )
  {
    v160 = v7;
    LOBYTE(v164) = 3;
  }
  v8(v5, &v160, 1);
  v9 = sub_2E8D910(v2);
  v114 = 0;
  v113 = v9;
  v10 = v9;
  if ( v9 )
  {
    v11 = *(_BYTE *)(v9 - 16);
    if ( (v11 & 2) != 0 )
      v12 = *(_QWORD *)(v10 - 32);
    else
      v12 = v113 - 8LL * ((v11 >> 2) & 0xF) - 16;
    v13 = *(_QWORD *)(*(_QWORD *)v12 + 136LL);
    if ( *(_DWORD *)(v13 + 32) <= 0x40u )
      v114 = *(_QWORD *)(v13 + 24);
    else
      v114 = **(_QWORD **)(v13 + 24);
  }
  v160 = &v163;
  v158 = 0x100000000LL;
  v161 = 0;
  v162 = 256;
  v155[0] = &unk_49DD288;
  v155[1] = 2;
  v155[2] = 0;
  v156 = 0;
  v157 = 0;
  v159 = &v160;
  sub_CB5980((__int64)v155, 0, 0, 0);
  v14 = *(_QWORD *)(a1 + 208);
  v111 = *(_QWORD **)(a1 + 240);
  v115 = sub_2E89090(v2);
  if ( v115 == 1 )
  {
    if ( v156 - (unsigned __int64)v157 <= 0xF )
      sub_CB6200((__int64)v155, "\t.intel_syntax\n\t", 0x10u);
    else
      *v157++ = _mm_load_si128((const __m128i *)&xmmword_44D4680);
    v116 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
    v16 = *v3;
    if ( !*v3 )
      goto LABEL_162;
    v132 = 1;
  }
  else
  {
    v132 = 0;
    v116 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
    v15 = *(__int64 (**)(void))(*(_QWORD *)*v111 + 240LL);
    if ( v15 != sub_23CE3E0 )
      v132 = v15();
    if ( !*(_BYTE *)(v14 + 22) )
    {
      v93 = v157;
      if ( (unsigned __int64)v157 >= v156 )
      {
        sub_CB5D20((__int64)v155, 9);
      }
      else
      {
        v157 = (__m128i *)((char *)v157 + 1);
        v93->m128i_i8[0] = 9;
      }
    }
    v16 = *v3;
    if ( !*v3 )
      goto LABEL_27;
  }
  v17 = (unsigned __int8 *)v3;
  v18 = -1;
  do
  {
    v19 = (__int64)(v17 + 1);
    if ( v16 == 10 )
    {
      v39 = v157;
      if ( (unsigned __int64)v157 >= v156 )
      {
        sub_CB5D20((__int64)v155, 10);
        v17 = (unsigned __int8 *)v19;
      }
      else
      {
        ++v17;
        v157 = (__m128i *)((char *)v157 + 1);
        v39->m128i_i8[0] = 10;
      }
      goto LABEL_25;
    }
    if ( v16 != 36 )
    {
      for ( i = v17[1]; ; i = *(unsigned __int8 *)++v19 )
      {
        v21 = i - 123;
        LOBYTE(v21) = (unsigned __int8)(i - 123) <= 2u;
        if ( (unsigned __int8)i <= 0x24u )
          v21 |= (0x1000000401uLL >> i) & 1;
        if ( (_BYTE)v21 )
          break;
      }
      if ( v18 == -1 || v18 == v132 )
      {
        sub_CB6200((__int64)v155, v17, v19 - (_QWORD)v17);
        v17 = (unsigned __int8 *)v19;
      }
      else
      {
        v17 = (unsigned __int8 *)v19;
      }
      goto LABEL_25;
    }
    v38 = v17[1];
    if ( v38 == 41 )
    {
      v17 += 2;
      if ( v18 == -1 )
      {
        v72 = v157;
        if ( (unsigned __int64)v157 >= v156 )
        {
          v128 = v17;
          sub_CB5D20((__int64)v155, 125);
          v17 = v128;
        }
        else
        {
          v157 = (__m128i *)((char *)v157 + 1);
          v72->m128i_i8[0] = 125;
        }
      }
      else
      {
        v18 = -1;
      }
      goto LABEL_25;
    }
    if ( v38 <= 41 )
    {
      if ( v38 == 36 )
      {
        if ( v115 != 1 && (v18 == -1 || v18 == v132) )
        {
          v70 = v157;
          if ( (unsigned __int64)v157 >= v156 )
          {
            v126 = v17;
            sub_CB5D20((__int64)v155, 36);
            v17 = v126;
          }
          else
          {
            v157 = (__m128i *)((char *)v157 + 1);
            v70->m128i_i8[0] = 36;
          }
        }
        v17 += 2;
        goto LABEL_25;
      }
      if ( v38 == 40 )
      {
        v17 += 2;
        if ( v18 != -1 )
        {
          if ( *v118 )
          {
            v149 = v118;
            v148.m128i_i64[0] = (__int64)"Nested variants found in inline asm string: '";
            v150 = 771;
          }
          else
          {
            v150 = 259;
            v148.m128i_i64[0] = (__int64)"Nested variants found in inline asm string: '";
          }
          if ( HIBYTE(v150) == 1 )
          {
            v107 = v148.m128i_i64[1];
            v99 = (__m128i *)v148.m128i_i64[0];
            v100 = 3;
          }
          else
          {
            v99 = &v148;
            v100 = 2;
          }
          v151.m128i_i64[0] = (__int64)v99;
          v151.m128i_i64[1] = v107;
          goto LABEL_181;
        }
        v18 = 0;
        goto LABEL_25;
      }
LABEL_62:
      v40 = 0;
      goto LABEL_63;
    }
    if ( v38 == 124 )
    {
      v17 += 2;
      if ( v18 == -1 )
      {
        v71 = v157;
        if ( (unsigned __int64)v157 >= v156 )
        {
          v127 = v17;
          sub_CB5D20((__int64)v155, 124);
          v17 = v127;
        }
        else
        {
          v157 = (__m128i *)((char *)v157 + 1);
          v71->m128i_i8[0] = 124;
        }
      }
      else
      {
        ++v18;
      }
      goto LABEL_25;
    }
    if ( v38 != 123 )
      goto LABEL_62;
    v38 = v17[2];
    v19 = (__int64)(v17 + 2);
    v40 = 1;
    if ( v38 == 58 )
    {
      v122 = v17 + 3;
      v41 = strchr((const char *)v17 + 3, 125);
      v42 = v41;
      if ( !v41 )
      {
        v6 = *v118 == 0;
        v148.m128i_i64[0] = (__int64)"Unterminated ${:foo} operand in inline asm string: '";
        if ( v6 )
        {
          v150 = 259;
        }
        else
        {
          v150 = 771;
          v149 = v118;
        }
        if ( HIBYTE(v150) == 1 )
        {
          v106 = v148.m128i_i64[1];
          v101 = (__m128i *)v148.m128i_i64[0];
          v100 = 3;
        }
        else
        {
          v101 = &v148;
          v100 = 2;
        }
        v151.m128i_i64[0] = (__int64)v101;
        v151.m128i_i64[1] = v106;
LABEL_181:
        v153 = v100;
        v152[0] = "'";
        v154 = 3;
        sub_C64D30((__int64)&v151, 1u);
      }
      if ( v18 == -1 || v18 == v132 )
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, _BYTE *, signed __int64))(*(_QWORD *)a1 + 512LL))(
          a1,
          v2,
          v155,
          v122,
          v41 - v122);
      v17 = (unsigned __int8 *)(v42 + 1);
      goto LABEL_25;
    }
LABEL_63:
    v43 = (unsigned __int8 *)v19;
    if ( (unsigned __int8)(v38 - 48) > 9u )
    {
      v45 = 0;
    }
    else
    {
      do
        v44 = *++v43;
      while ( (unsigned __int8)(v44 - 48) <= 9u );
      v45 = (__int64)&v43[-v19];
    }
    v119 = v43;
    v120 = v40;
    if ( sub_C93C90(v19, v45, 0xAu, (unsigned __int64 *)&v151)
      || (v46 = v151.m128i_i32[0], v151.m128i_i64[0] != v151.m128i_u32[0]) )
    {
      v6 = *v118 == 0;
      v148.m128i_i64[0] = (__int64)"Bad $ operand number in inline asm string: '";
      if ( v6 )
      {
        v150 = 259;
      }
      else
      {
        v150 = 771;
        v149 = v118;
      }
      v47 = &v148;
      v48 = 2;
      if ( HIBYTE(v150) == 1 )
      {
        v108 = v148.m128i_i64[1];
        v47 = (__m128i *)v148.m128i_i64[0];
        v48 = 3;
      }
      v151.m128i_i64[0] = (__int64)v47;
      v49 = v108;
      goto LABEL_199;
    }
    v17 = v119;
    if ( (unsigned int)(v116 - 1) <= v151.m128i_i32[0] )
    {
      v6 = *v118 == 0;
      v148.m128i_i64[0] = (__int64)"Invalid $ operand number in inline asm string: '";
      if ( v6 )
      {
        v150 = 259;
      }
      else
      {
        v150 = 771;
        v149 = v118;
      }
      v105 = &v148;
      v48 = 2;
      if ( HIBYTE(v150) == 1 )
      {
        v109 = v148.m128i_i64[1];
        v105 = (__m128i *)v148.m128i_i64[0];
        v48 = 3;
      }
      v151.m128i_i64[0] = (__int64)v105;
      v49 = v109;
LABEL_199:
      v151.m128i_i64[1] = v49;
      v152[0] = "'";
      v153 = v48;
      v154 = 3;
      sub_C64D30((__int64)&v151, 1u);
    }
    v137 = 0;
    if ( v120 )
    {
      v50 = *v119;
      if ( *v119 == 58 )
      {
        if ( !v119[1] )
        {
          v148.m128i_i64[0] = (__int64)"'";
          v150 = 259;
          v144 = 257;
          if ( *v118 )
          {
            v142.m128i_i64[0] = (__int64)v118;
            LOBYTE(v144) = 3;
          }
          v138.m128i_i64[0] = (__int64)"Bad ${:} expression in inline asm string: '";
          v141 = 1;
          v140 = 3;
          sub_9C6370(&v145, &v138, &v142, v151.m128i_u32[0], v120, (__int64)&v151);
          sub_9C6370(&v151, &v145, &v148, v102, v103, (__int64)&v151);
          sub_C64D30((__int64)&v151, 1u);
        }
        LOBYTE(v137) = v119[1];
        v17 = v119 + 2;
        v50 = v119[2];
      }
      if ( v50 != 125 )
      {
        v6 = *v118 == 0;
        v148.m128i_i64[0] = (__int64)"Bad ${} expression in inline asm string: '";
        if ( v6 )
        {
          v150 = 259;
        }
        else
        {
          v150 = 771;
          v149 = v118;
        }
        if ( HIBYTE(v150) == 1 )
        {
          v110 = v148.m128i_i64[1];
          v104 = (__m128i *)v148.m128i_i64[0];
          v48 = 3;
        }
        else
        {
          v104 = &v148;
          v48 = 2;
        }
        v151.m128i_i64[0] = (__int64)v104;
        v49 = v110;
        goto LABEL_199;
      }
      ++v17;
    }
    if ( v18 == -1 || v18 == v132 )
    {
      v51 = 2;
      v52 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
      if ( v151.m128i_i32[0] )
      {
        while ( v51 < v52 )
        {
          v51 += (((unsigned int)*(_QWORD *)(*(_QWORD *)(v2 + 32) + 40LL * v51 + 24) >> 3) & 0x1FFF) + 1;
          if ( !--v46 )
            goto LABEL_104;
        }
        goto LABEL_84;
      }
LABEL_104:
      if ( v52 <= v51 )
        goto LABEL_84;
      v63 = *(_QWORD *)(v2 + 32);
      v64 = v63 + 40LL * v51;
      if ( *(_BYTE *)v64 == 14 )
        goto LABEL_84;
      v65 = v51 + 1;
      v66 = v63 + 40 * v65;
      if ( *(_BYTE *)v66 != 11 )
      {
        if ( *(_BYTE *)v66 == 4 )
        {
          v130 = v17;
          v97 = sub_2E309C0(*(_QWORD *)(v66 + 24), v64, v65, v66, v120);
          sub_EA12C0(v97, (__int64)v155, *(_BYTE **)(a1 + 208));
          v17 = v130;
          goto LABEL_25;
        }
        v67 = *(_QWORD *)a1;
        if ( (*(_DWORD *)(v64 + 24) & 7) != 6 )
        {
          v68 = &v137;
          v124 = v17;
          if ( !(_BYTE)v137 )
            v68 = 0;
          v69 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int16 *, _QWORD *, __m128i *))(v67 + 528))(
                  a1,
                  v2,
                  v65,
                  v68,
                  v155,
                  &v151);
          v17 = v124;
          goto LABEL_112;
        }
        v91 = &v137;
        v92 = *(__int64 (**)())(v67 + 536);
        if ( !(_BYTE)v137 )
          v91 = 0;
        if ( v92 != sub_31F17A0 )
        {
          v125 = v17;
          v69 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int16 *, _QWORD *))v92)(a1, v2, v65, v91, v155);
          v17 = v125;
LABEL_112:
          if ( !v69 )
            goto LABEL_25;
        }
LABEL_84:
        v123 = v17;
        v53 = (__int64 *)sub_2E88D60(v2);
        v54 = sub_B2BE50(*v53);
        if ( *v118 )
        {
          v145.m128i_i64[0] = (__int64)"invalid operand in inline asm: '";
          v146 = v118;
          v147 = 771;
        }
        else
        {
          v145.m128i_i64[0] = (__int64)"invalid operand in inline asm: '";
          v147 = 259;
        }
        v55 = &v145;
        v56 = 2;
        if ( HIBYTE(v147) == 1 )
        {
          v112 = v145.m128i_i64[1];
          v55 = (__m128i *)v145.m128i_i64[0];
          v56 = 3;
        }
        v148.m128i_i64[0] = (__int64)v55;
        v148.m128i_i64[1] = v112;
        v149 = "'";
        LOBYTE(v150) = v56;
        HIBYTE(v150) = 3;
        sub_B156D0((__int64)&v151, v114, (__int64)&v148, 0);
        sub_B6EB20(v54, (__int64)&v151);
        v17 = v123;
        goto LABEL_25;
      }
      v129 = v17;
      v95 = sub_31E0E80(a1, *(_QWORD *)(v66 + 24));
      sub_EA12C0(v95, (__int64)v155, *(_BYTE **)(a1 + 208));
      v96 = v111[310];
      if ( !v96 )
        v96 = (__int64)(v111 + 1);
      sub_E6D160(v96, v95);
      v17 = v129;
    }
LABEL_25:
    v16 = *v17;
  }
  while ( *v17 );
  if ( v115 != 1 )
    goto LABEL_27;
LABEL_162:
  v94 = v157;
  if ( v156 - (unsigned __int64)v157 <= 0xC )
  {
    sub_CB6200((__int64)v155, "\n\t.att_syntax", 0xDu);
LABEL_27:
    v22 = v157;
    goto LABEL_28;
  }
  v157->m128i_i32[2] = 1635020409;
  v94->m128i_i64[0] = 0x735F7474612E090ALL;
  v94->m128i_i8[12] = 120;
  v22 = (__m128i *)((char *)&v157->m128i_u64[1] + 5);
  v157 = (__m128i *)((char *)v157 + 13);
LABEL_28:
  if ( (unsigned __int64)v22 >= v156 )
  {
    v23 = (_QWORD *)sub_CB5D20((__int64)v155, 10);
  }
  else
  {
    v23 = v155;
    v157 = (__m128i *)&v22->m128i_i8[1];
    v22->m128i_i8[0] = 10;
  }
  v24 = (_BYTE *)v23[4];
  if ( (unsigned __int64)v24 >= v23[3] )
  {
    sub_CB5D20((__int64)v23, 0);
  }
  else
  {
    v23[4] = v24 + 1;
    *v24 = 0;
  }
  v25 = 2;
  v151.m128i_i64[0] = (__int64)v152;
  v151.m128i_i64[1] = 0x800000000LL;
  v26 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 232) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 232) + 16LL));
  v27 = &v151;
  v28 = (_QWORD *)v26;
  v29 = *(_DWORD *)(v2 + 40) & 0xFFFFFF;
  if ( v29 > 2 )
  {
    do
    {
      v30 = *(_QWORD *)(v2 + 32);
      v31 = v30 + 40LL * v25;
      if ( *(_BYTE *)v31 == 1 )
      {
        v32 = *(_QWORD *)(v31 + 24);
        if ( (*(_DWORD *)(v31 + 24) & 7) == 4 )
        {
          v33 = *(__int64 (**)())(*v28 + 152LL);
          if ( v33 != sub_2FF51D0 )
          {
            v121 = v27;
            v133 = *(_DWORD *)(v30 + 40LL * (v25 + 1) + 8);
            v34 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, _QWORD))v33)(v28, *(_QWORD *)(a1 + 232), v133);
            v27 = v121;
            if ( !v34 )
            {
              v35 = v151.m128i_u32[2];
              v36 = v133;
              v37 = v151.m128i_u32[2] + 1LL;
              if ( v37 > v151.m128i_u32[3] )
              {
                v98 = (__int64)v121;
                v131 = v133;
                v136 = v27;
                sub_C8D5F0(v98, v152, v37, 4u, v36, (__int64)v27);
                v35 = v151.m128i_u32[2];
                LODWORD(v36) = v131;
                v27 = v136;
              }
              *(_DWORD *)(v151.m128i_i64[0] + 4 * v35) = v36;
              ++v151.m128i_i32[2];
            }
          }
        }
        v25 += (unsigned __int16)v32 >> 3;
      }
      ++v25;
    }
    while ( v25 < v29 );
  }
  if ( v151.m128i_i32[2] )
  {
    v148.m128i_i64[0] = 53;
    v138.m128i_i64[0] = (__int64)v139;
    v76 = sub_22409D0((__int64)&v138, (unsigned __int64 *)&v148, 0);
    v138.m128i_i64[0] = v76;
    v139[0] = v148.m128i_i64[0];
    *(__m128i *)v76 = _mm_load_si128((const __m128i *)&xmmword_44D4690);
    si128 = _mm_load_si128((const __m128i *)&xmmword_44D46A0);
    *(_DWORD *)(v76 + 48) = 980644453;
    *(__m128i *)(v76 + 16) = si128;
    v78 = _mm_load_si128((const __m128i *)&xmmword_44D46B0);
    *(_BYTE *)(v76 + 52) = 32;
    *(__m128i *)(v76 + 32) = v78;
    v138.m128i_i64[1] = v148.m128i_i64[0];
    *(_BYTE *)(v138.m128i_i64[0] + v148.m128i_i64[0]) = 0;
    if ( v151.m128i_i64[0] + 4LL * v151.m128i_u32[2] != v151.m128i_i64[0] )
    {
      v134 = v2;
      v79 = *(unsigned int *)v151.m128i_i64[0];
      v80 = 0;
      v81 = (unsigned int *)(v151.m128i_i64[0] + 4);
      v82 = 0;
      v83 = v151.m128i_i64[0] + 4LL * v151.m128i_u32[2];
      while ( 1 )
      {
        sub_2241490((unsigned __int64 *)&v138, v80, v82);
        v86 = *(const char *(__fastcall **)(__int64, unsigned int))(*v28 + 632LL);
        if ( v86 == sub_2FF5340 )
        {
          v84 = 0;
          v85 = (char *)(v28[9] + *(unsigned int *)(v28[1] + 24 * v79));
          if ( v85 )
            v84 = strlen(v85);
        }
        else
        {
          v85 = (char *)v86((__int64)v28, v79);
        }
        if ( v84 > 0x3FFFFFFFFFFFFFFFLL - v138.m128i_i64[1] )
LABEL_175:
          sub_4262D8((__int64)"basic_string::append");
        sub_2241490((unsigned __int64 *)&v138, v85, v84);
        if ( (unsigned int *)v83 == v81 )
          break;
        v79 = *v81;
        v82 = 2;
        ++v81;
        v80 = ", ";
        if ( v138.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL || v138.m128i_i64[1] == 4611686018427387902LL )
          goto LABEL_175;
      }
      v2 = v134;
    }
    v87 = sub_B2BE50(**(_QWORD **)(a1 + 232));
    v145.m128i_i64[0] = (__int64)&v138;
    v88 = v87;
    v147 = 260;
    sub_B156D0((__int64)&v148, v114, (__int64)&v145, 1);
    sub_B6EB20(v88, (__int64)&v148);
    v145.m128i_i64[0] = (__int64)"Reserved registers on the clobber list may not be preserved across the asm statement, a"
                                 "nd clobbering them may lead to undefined behaviour.";
    v147 = 259;
    sub_B156D0((__int64)&v148, v114, (__int64)&v145, 3);
    sub_B6EB20(v88, (__int64)&v148);
    v135 = v151.m128i_i64[0] + 4LL * v151.m128i_u32[2];
    if ( v135 != v151.m128i_i64[0] )
    {
      v89 = (unsigned int *)v151.m128i_i64[0];
      do
      {
        while ( 1 )
        {
          v90 = *(__int64 (__fastcall **)(__int64))(*v28 + 144LL);
          if ( v90 != sub_2FF51B0 )
          {
            ((void (__fastcall *)(__m128i *, _QWORD *, _QWORD, _QWORD))v90)(&v142, v28, *(_QWORD *)(a1 + 232), *v89);
            if ( (_BYTE)v144 )
            {
              v147 = 260;
              v145.m128i_i64[0] = (__int64)&v142;
              sub_B156D0((__int64)&v148, v114, (__int64)&v145, 3);
              sub_B6EB20(v88, (__int64)&v148);
              if ( (_BYTE)v144 )
              {
                LOBYTE(v144) = 0;
                if ( (__int64 *)v142.m128i_i64[0] != &v143 )
                  break;
              }
            }
          }
          if ( (unsigned int *)v135 == ++v89 )
            goto LABEL_148;
        }
        ++v89;
        j_j___libc_free_0(v142.m128i_u64[0]);
      }
      while ( (unsigned int *)v135 != v89 );
    }
LABEL_148:
    if ( (_QWORD *)v138.m128i_i64[0] != v139 )
      j_j___libc_free_0(v138.m128i_u64[0]);
  }
  v57 = sub_2E89090(v2);
  v58 = *(_QWORD *)(a1 + 200) + 976LL;
  v59 = sub_31DB000(a1);
  sub_31F20D0(a1, v160, v161, v59, v58, v113, v57);
  v60 = *(_QWORD *)(a1 + 224);
  v61 = *(void (__fastcall **)(__int64, __m128i *, __int64, _QWORD))(*(_QWORD *)v60 + 136LL);
  v62 = *(_BYTE **)(*(_QWORD *)(a1 + 208) + 144LL);
  v150 = 257;
  if ( *v62 )
  {
    v148.m128i_i64[0] = (__int64)v62;
    LOBYTE(v150) = 3;
  }
  v61(v60, &v148, 1, v61);
  if ( (_QWORD *)v151.m128i_i64[0] != v152 )
    _libc_free(v151.m128i_u64[0]);
  v155[0] = &unk_49DD388;
  sub_CB5840((__int64)v155);
  if ( v160 != &v163 )
    _libc_free((unsigned __int64)v160);
}
