// Function: sub_2EAE5A0
// Address: 0x2eae5a0
//
void __fastcall sub_2EAE5A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        char a7,
        char a8,
        unsigned int a9,
        __int64 a10)
{
  void *v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r14
  __int64 v17; // rsi
  int v18; // r12d
  _BYTE *v19; // rax
  __int64 v20; // rsi
  _BYTE *v21; // r12
  void **v22; // rdi
  __int64 v23; // rdx
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // rax
  int *v25; // rax
  unsigned __int8 *v26; // rsi
  int *v27; // r12
  int *v28; // rbx
  size_t i; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdi
  int v32; // r14d
  size_t v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 (*v37)(); // rax
  _QWORD *v38; // r12
  _QWORD *(__fastcall *v39)(__int64); // rax
  __int64 v40; // rdi
  signed __int64 v41; // rsi
  __int64 (__fastcall *v42)(__int64, __int64, __int64, __int64, signed __int64); // rax
  char v43; // al
  int v44; // r12d
  const char *v45; // rsi
  unsigned __int8 v46; // al
  bool v47; // dl
  char v48; // al
  __int64 v49; // r8
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // r15
  __int64 v56; // r14
  unsigned int v57; // ebx
  char v58; // dl
  int v59; // r15d
  int v60; // eax
  unsigned int v61; // r15d
  int v62; // r12d
  __int64 v63; // rax
  __int64 v64; // rax
  _BYTE *v65; // rdi
  __int64 v66; // rax
  _QWORD *v67; // rcx
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  const char *v71; // rsi
  __int64 v72; // rdi
  _BYTE *v73; // rax
  int v74; // r12d
  const char *v75; // rsi
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rdi
  _BYTE *v79; // rax
  unsigned int v80; // r12d
  __int64 v81; // r13
  unsigned __int8 *v82; // rax
  size_t v83; // rdx
  void *v84; // rdi
  unsigned __int8 *v85; // r14
  size_t v86; // r12
  _BYTE *v87; // rax
  __int64 v88; // rax
  __int64 v89; // rax
  __int64 v90; // rax
  const char *v91; // rsi
  const char *v92; // rsi
  _WORD *v93; // rdx
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // rax
  __int64 v97; // r14
  char v98; // al
  __int64 v99; // rax
  const char *v100; // rsi
  const char *v101; // rsi
  __int64 v102; // rax
  __int64 v103; // rax
  __int64 v104; // rax
  unsigned int v105; // eax
  __int64 v106; // rdi
  _BYTE *v107; // rax
  unsigned int v108; // eax
  __int64 v109; // rdi
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rax
  __int64 v114; // rax
  __int64 v115; // rdx
  __int64 v116; // rax
  __int64 v117; // rax
  __int64 v118; // rdx
  unsigned __int8 *v119; // rax
  size_t v120; // rdx
  __int64 v121; // rax
  __int64 v122; // rax
  __int64 v123; // rax
  unsigned __int64 v124; // r8
  void (__fastcall *v125)(unsigned __int64); // rax
  char v126; // al
  __int64 v127; // [rsp+0h] [rbp-170h]
  int v128; // [rsp+10h] [rbp-160h]
  __int64 v129; // [rsp+10h] [rbp-160h]
  unsigned __int16 v130; // [rsp+10h] [rbp-160h]
  __int64 v131; // [rsp+18h] [rbp-158h]
  unsigned int v132; // [rsp+18h] [rbp-158h]
  __int64 v133; // [rsp+18h] [rbp-158h]
  __int64 v134; // [rsp+18h] [rbp-158h]
  __int64 v135; // [rsp+18h] [rbp-158h]
  size_t v136; // [rsp+18h] [rbp-158h]
  __int64 v138; // [rsp+28h] [rbp-148h] BYREF
  _BYTE v139[16]; // [rsp+30h] [rbp-140h] BYREF
  void (__fastcall *v140)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-130h]
  void (__fastcall *v141)(_BYTE *, __int64); // [rsp+48h] [rbp-128h]
  _BYTE v142[16]; // [rsp+50h] [rbp-120h] BYREF
  void (__fastcall *v143)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-110h]
  void (__fastcall *v144)(_BYTE *, __int64); // [rsp+68h] [rbp-108h]
  _QWORD v145[2]; // [rsp+70h] [rbp-100h] BYREF
  void (__fastcall *v146)(_BYTE *, _BYTE *, __int64); // [rsp+80h] [rbp-F0h]
  void (__fastcall *v147)(_QWORD *, __int64); // [rsp+88h] [rbp-E8h]
  _BYTE v148[16]; // [rsp+90h] [rbp-E0h] BYREF
  void (__fastcall *v149)(_BYTE *, _BYTE *, __int64); // [rsp+A0h] [rbp-D0h]
  void (__fastcall *v150)(_BYTE *, __int64); // [rsp+A8h] [rbp-C8h]
  _BYTE v151[16]; // [rsp+B0h] [rbp-C0h] BYREF
  void (__fastcall *v152)(_BYTE *, _BYTE *, __int64); // [rsp+C0h] [rbp-B0h]
  void (__fastcall *v153)(_BYTE *, __int64); // [rsp+C8h] [rbp-A8h]
  void *v154; // [rsp+D0h] [rbp-A0h] BYREF
  const char *v155; // [rsp+D8h] [rbp-98h]
  void (__fastcall *v156)(void **, void **, __int64); // [rsp+E0h] [rbp-90h]
  void (__fastcall *v157)(void **, __int64); // [rsp+E8h] [rbp-88h]

  v138 = a4;
  sub_2EABA60(a2, a1);
  switch ( *(_BYTE *)a1 )
  {
    case 0:
      v43 = *(_BYTE *)(a1 + 3);
      v44 = *(_DWORD *)(a1 + 8);
      if ( (v43 & 0x20) != 0 )
      {
        v45 = "implicit-def ";
        if ( (v43 & 0x10) == 0 )
          v45 = "implicit ";
        sub_904010(a2, v45);
      }
      else if ( a6 && (v43 & 0x10) != 0 )
      {
        sub_904010(a2, "def ");
      }
      if ( (*(_BYTE *)(a1 + 4) & 2) != 0 )
        sub_904010(a2, "internal ");
      v46 = *(_BYTE *)(a1 + 3);
      v47 = (v46 & 0x40) != 0;
      if ( (v47 & (v46 >> 4)) != 0 )
      {
        sub_904010(a2, "dead ");
        v46 = *(_BYTE *)(a1 + 3);
        v47 = (v46 & 0x40) != 0;
      }
      if ( (v47 & ((v46 >> 4) ^ 1) & 1) != 0 )
        sub_904010(a2, "killed ");
      v48 = *(_BYTE *)(a1 + 4);
      if ( (v48 & 1) != 0 )
      {
        sub_904010(a2, "undef ");
        v48 = *(_BYTE *)(a1 + 4);
      }
      if ( (v48 & 4) != 0 )
        sub_904010(a2, "early-clobber ");
      if ( (unsigned int)(*(_DWORD *)(a1 + 8) - 1) <= 0x3FFFFFFE && (unsigned __int8)sub_2EAB300(a1) )
        sub_904010(a2, "renamable ");
      v49 = 0;
      if ( v44 < 0 )
      {
        v49 = *(_QWORD *)(a1 + 16);
        if ( v49 )
        {
          v49 = *(_QWORD *)(v49 + 24);
          if ( v49 )
          {
            v49 = *(_QWORD *)(v49 + 32);
            if ( v49 )
              v49 = *(_QWORD *)(v49 + 32);
          }
        }
      }
      v20 = (unsigned int)v44;
      sub_2FF6320(v139, (unsigned int)v44, a10, 0, v49);
      v22 = (void **)v139;
      if ( !v140 )
        goto LABEL_234;
      v141(v139, a2);
      if ( v140 )
        v140(v139, v139, 3);
      v128 = (*(_DWORD *)a1 >> 8) & 0xFFF;
      if ( v128 )
      {
        if ( a10 )
        {
          v50 = sub_A51310(a2, 0x2Eu);
          sub_904010(v50, *(const char **)(*(_QWORD *)(a10 + 256) + 8LL * (unsigned int)(v128 - 1)));
        }
        else
        {
          v130 = (*(_DWORD *)a1 >> 8) & 0xFFF;
          v121 = sub_904010(a2, ".subreg");
          sub_CB59D0(v121, v130);
        }
      }
      if ( v44 >= 0 )
        goto LABEL_59;
      v112 = *(_QWORD *)(a1 + 16);
      if ( !v112 )
        goto LABEL_59;
      v113 = *(_QWORD *)(v112 + 24);
      if ( !v113 )
        goto LABEL_59;
      v114 = *(_QWORD *)(v113 + 32);
      if ( !v114 )
        goto LABEL_59;
      v115 = *(_QWORD *)(v114 + 32);
      if ( a6 == 1 && !a7 )
      {
        v116 = *(_QWORD *)(*(_QWORD *)(v115 + 56) + 16LL * (v44 & 0x7FFFFFFF) + 8);
        if ( v116 )
        {
          if ( (*(_BYTE *)(v116 + 3) & 0x10) != 0 )
            goto LABEL_59;
          v117 = *(_QWORD *)(v116 + 32);
          if ( v117 )
          {
            if ( (*(_BYTE *)(v117 + 3) & 0x10) != 0 )
              goto LABEL_59;
          }
        }
      }
      v134 = v115;
      sub_A51310(a2, 0x3Au);
      v20 = (unsigned int)v44;
      v22 = (void **)v142;
      sub_2FF63B0(v142, (unsigned int)v44, v134, a10);
      if ( !v143 )
        goto LABEL_234;
      v144(v142, a2);
      if ( v143 )
        v143(v142, v142, 3);
LABEL_59:
      if ( a8 && (*(_WORD *)(a1 + 2) & 0xFF0) != 0 && (*(_BYTE *)(a1 + 3) & 0x10) == 0 )
      {
        v122 = sub_904010(a2, "(tied-def ");
        v123 = sub_CB59D0(v122, a9);
        sub_904010(v123, ")");
      }
      if ( (v138 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
      {
        v51 = sub_A51310(a2, 0x28u);
        sub_34B2640(&v138, v51);
        sub_A51310(v51, 0x29u);
      }
      return;
    case 1:
      v34 = *(_QWORD *)(a1 + 16);
      if ( !v34 )
        goto LABEL_181;
      v35 = *(_QWORD *)(v34 + 24);
      if ( !v35 )
        goto LABEL_181;
      v36 = *(_QWORD *)(v35 + 32);
      if ( !v36 )
        goto LABEL_181;
      v37 = *(__int64 (**)())(**(_QWORD **)(v36 + 16) + 128LL);
      if ( v37 == sub_2DAC790 )
        BUG();
      v38 = (_QWORD *)v37();
      v39 = *(_QWORD *(__fastcall **)(__int64))(*v38 + 1480LL);
      if ( v39 == sub_2EAAED0 )
      {
        v40 = v38[7];
        if ( v40 )
          goto LABEL_34;
        v40 = sub_22077B0(8u);
        if ( v40 )
          *(_QWORD *)v40 = &unk_4A29780;
        v124 = v38[7];
        v38[7] = v40;
        if ( v124 )
        {
          v125 = *(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v124 + 8LL);
          if ( v125 == sub_2EAAD20 )
            j_j___libc_free_0(v124);
          else
            v125(v124);
          v40 = v38[7];
        }
      }
      else
      {
        v40 = (__int64)v39((__int64)v38);
      }
      if ( !v40 )
      {
LABEL_181:
        v41 = *(_QWORD *)(a1 + 24);
        goto LABEL_35;
      }
LABEL_34:
      v41 = *(_QWORD *)(a1 + 24);
      v42 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, signed __int64))(*(_QWORD *)v40 + 16LL);
      if ( v42 == sub_2EAAD30 )
      {
LABEL_35:
        sub_CB59F0(a2, v41);
        return;
      }
      v42(v40, a2, *(_QWORD *)(a1 + 16), a5, *(_QWORD *)(a1 + 24));
      return;
    case 2:
    case 3:
      sub_A5C020(*(_BYTE **)(a1 + 24), a2, 1, a3);
      return;
    case 4:
      v20 = *(_QWORD *)(a1 + 24);
      v21 = v145;
      v22 = (void **)v145;
      sub_2E31000(v145, v20);
      if ( !v146 )
        goto LABEL_234;
      v147(v145, a2);
      v24 = v146;
      if ( v146 )
        goto LABEL_17;
      return;
    case 5:
      v67 = *(_QWORD **)(a1 + 16);
      if ( v67 )
      {
        v67 = (_QWORD *)v67[3];
        if ( v67 )
        {
          v67 = (_QWORD *)v67[4];
          if ( v67 )
            v67 = (_QWORD *)v67[6];
        }
      }
      sub_2EAC020(a2, *(_DWORD *)(a1 + 24), 0, (__int64)v67);
      return;
    case 6:
      v66 = sub_904010(a2, "%const.");
      sub_CB59F0(v66, *(int *)(a1 + 24));
      goto LABEL_12;
    case 7:
      sub_904010(a2, "target-index(");
      v68 = *(_QWORD *)(a1 + 16);
      if ( !v68
        || (v69 = *(_QWORD *)(v68 + 24)) == 0
        || (v70 = *(_QWORD *)(v69 + 32)) == 0
        || (v71 = (const char *)sub_2EAAD50(*(_QWORD *)(v70 + 16), *(_DWORD *)(a1 + 24))) == 0 )
      {
        v71 = "<unknown>";
      }
      v72 = sub_904010(a2, v71);
      v73 = *(_BYTE **)(v72 + 32);
      if ( (unsigned __int64)v73 >= *(_QWORD *)(v72 + 24) )
      {
        sub_CB5D20(v72, 41);
      }
      else
      {
        *(_QWORD *)(v72 + 32) = v73 + 1;
        *v73 = 41;
      }
      goto LABEL_12;
    case 8:
      v20 = *(unsigned int *)(a1 + 24);
      v21 = v148;
      v22 = (void **)v148;
      sub_2E79E20((__int64)v148, v20);
      if ( !v149 )
        goto LABEL_234;
      v150(v148, a2);
      v24 = v149;
      if ( !v149 )
        return;
LABEL_17:
      v24(v21, v21, 3);
      return;
    case 9:
      v85 = *(unsigned __int8 **)(a1 + 24);
      v86 = 0;
      if ( v85 )
        v86 = strlen(*(const char **)(a1 + 24));
      v87 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v87 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 38);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v87 + 1;
        *v87 = 38;
      }
      if ( v86 )
        sub_A54F00(a2, v85, v86);
      else
        sub_904010(a2, "\"\"");
      goto LABEL_12;
    case 0xA:
      v65 = *(_BYTE **)(a1 + 24);
      if ( v65 )
        sub_A5C020(v65, a2, 0, a3);
      else
        sub_904010(a2, "globaladdress(null)");
      goto LABEL_12;
    case 0xB:
      sub_904010(a2, "blockaddress(");
      sub_A5C020(*(_BYTE **)(*(_QWORD *)(a1 + 24) - 64LL), a2, 0, a3);
      sub_904010(a2, ", ");
      v14 = *(void **)(a2 + 32);
      v15 = *(_QWORD *)(*(_QWORD *)(a1 + 24) - 32LL);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v14 <= 9u )
      {
        v135 = *(_QWORD *)(*(_QWORD *)(a1 + 24) - 32LL);
        sub_CB6200(a2, "%ir-block.", 0xAu);
        v15 = v135;
      }
      else
      {
        qmemcpy(v14, "%ir-block.", 10);
        *(_QWORD *)(a2 + 32) += 10LL;
      }
      if ( (*(_BYTE *)(v15 + 7) & 0x10) != 0 )
      {
        v119 = (unsigned __int8 *)sub_BD5D20(v15);
        sub_A54F00(a2, v119, v120);
        v19 = *(_BYTE **)(a2 + 32);
      }
      else
      {
        v16 = *(_QWORD *)(v15 + 72);
        if ( v16 )
        {
          if ( v16 == *(_QWORD *)(a3 + 32) )
          {
            v18 = sub_A5A720(a3, v15);
            goto LABEL_9;
          }
          v17 = *(_QWORD *)(v16 + 40);
          v131 = v15;
          if ( v17 )
          {
            sub_A558A0((__int64)&v154, v17, 0);
            sub_A564B0((__int64)&v154, v16);
            v18 = sub_A5A720((__int64)&v154, v131);
            sub_A55520(&v154, v131);
LABEL_9:
            sub_2EAC190(a2, v18);
            v19 = *(_BYTE **)(a2 + 32);
            goto LABEL_10;
          }
        }
        v118 = *(_QWORD *)(a2 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v118) <= 8 )
        {
          sub_CB6200(a2, "<unknown>", 9u);
          v19 = *(_BYTE **)(a2 + 32);
        }
        else
        {
          *(_BYTE *)(v118 + 8) = 62;
          *(_QWORD *)v118 = 0x6E776F6E6B6E753CLL;
          v19 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 9LL);
          *(_QWORD *)(a2 + 32) = v19;
        }
      }
LABEL_10:
      if ( (unsigned __int64)v19 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 41);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v19 + 1;
        *v19 = 41;
      }
LABEL_12:
      sub_2EAC0D0(a2, *(unsigned int *)(a1 + 8) | (unsigned __int64)((__int64)*(int *)(a1 + 32) << 32));
      return;
    case 0xC:
      sub_904010(a2, "<regmask");
      if ( a10 )
      {
        v61 = 0;
        v62 = 0;
        v132 = 0;
        if ( *(_DWORD *)(a10 + 16) )
        {
          do
          {
            if ( (*(_DWORD *)(*(_QWORD *)(a1 + 24) + 4LL * (v61 >> 5)) & (1 << v61)) != 0 )
            {
              if ( (int)qword_5020968 < 0 || (unsigned int)qword_5020968 >= v132 )
              {
                v20 = v61;
                v22 = (void **)v151;
                v129 = sub_904010(a2, " ");
                sub_2FF6320(v151, v61, a10, 0, 0);
                if ( !v152 )
                  goto LABEL_234;
                v153(v151, v129);
                if ( v152 )
                  v152(v151, v151, 3);
                ++v132;
              }
              ++v62;
            }
            ++v61;
          }
          while ( v61 < *(_DWORD *)(a10 + 16) );
          if ( v62 != v132 )
          {
            v63 = sub_904010(a2, " and ");
            v64 = sub_CB59D0(v63, v62 - v132);
            sub_904010(v64, " more...");
          }
        }
      }
      else
      {
        sub_904010(a2, " ...");
      }
      sub_904010(a2, ">");
      return;
    case 0xD:
      v56 = *(_QWORD *)(a1 + 24);
      sub_904010(a2, "liveout(");
      if ( a10 )
      {
        v57 = 0;
        v58 = 0;
        v59 = *(_DWORD *)(a10 + 16);
        if ( v59 )
        {
          do
          {
            v60 = *(_DWORD *)(v56 + 4LL * (v57 >> 5));
            if ( _bittest(&v60, v57) )
            {
              if ( v58 )
                sub_904010(a2, ", ");
              v20 = v57;
              v22 = &v154;
              sub_2FF6320(&v154, v57, a10, 0, 0);
              if ( !v156 )
LABEL_234:
                sub_4263D6(v22, v20, v23);
              v157(&v154, a2);
              if ( v156 )
                v156(&v154, &v154, 3);
              v58 = 1;
            }
            ++v57;
          }
          while ( v57 != v59 );
        }
      }
      else
      {
        sub_904010(a2, "<unknown>");
      }
      sub_904010(a2, ")");
      return;
    case 0xE:
      sub_A61DC0(*(const char **)(a1 + 24), a2, a3, 0);
      return;
    case 0xF:
      sub_2EABE30(a2, *(_QWORD *)(a1 + 24));
      return;
    case 0x10:
      v52 = *(_QWORD *)(a1 + 16);
      if ( v52 && (v53 = *(_QWORD *)(v52 + 24)) != 0 && (v54 = *(_QWORD *)(v53 + 32)) != 0 )
      {
        v55 = *(_QWORD *)(v54 + 360) + 104LL * *(unsigned int *)(a1 + 24);
        switch ( *(_BYTE *)(v55 + 32) )
        {
          case 0:
            v92 = "same_value ";
            goto LABEL_138;
          case 1:
            v91 = "remember_state ";
            goto LABEL_131;
          case 2:
            v91 = "restore_state ";
            goto LABEL_131;
          case 3:
            v101 = "offset ";
            goto LABEL_156;
          case 4:
            sub_904010(a2, "llvm_def_aspace_cfa ");
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            sub_2EAAF50(*(_DWORD *)(v55 + 8), a2, a10);
            v103 = sub_904010(a2, ", ");
            sub_CB59F0(v103, *(_QWORD *)(v55 + 16));
            v104 = sub_904010(a2, ", ");
            sub_CB59D0(v104, *(unsigned int *)(v55 + 24));
            return;
          case 5:
            v92 = "def_cfa_register ";
            goto LABEL_138;
          case 6:
            v100 = "def_cfa_offset ";
            goto LABEL_152;
          case 7:
            v101 = "def_cfa ";
            goto LABEL_156;
          case 8:
            v101 = "rel_offset ";
LABEL_156:
            sub_904010(a2, v101);
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            sub_2EAAF50(*(_DWORD *)(v55 + 8), a2, a10);
            v102 = sub_904010(a2, ", ");
            sub_CB59F0(v102, *(_QWORD *)(v55 + 16));
            return;
          case 9:
            v100 = "adjust_cfa_offset ";
LABEL_152:
            sub_904010(a2, v100);
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            sub_CB59F0(a2, *(_QWORD *)(v55 + 16));
            return;
          case 0xA:
            sub_904010(a2, "escape ");
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            v96 = *(_QWORD *)(v55 + 56) - *(_QWORD *)(v55 + 48);
            v127 = v96;
            if ( v96 )
            {
              v133 = v96 - 1;
              if ( v96 != 1 )
              {
                v97 = 0;
                do
                {
                  v98 = *(_BYTE *)(*(_QWORD *)(v55 + 48) + v97);
                  v155 = "0x%02x";
                  v154 = &unk_49DD0D8;
                  LOBYTE(v156) = v98;
                  v99 = sub_CB6620(a2, (__int64)&v154, (__int64)v93, (__int64)&unk_49DD0D8, v94, v95);
                  v93 = *(_WORD **)(v99 + 32);
                  if ( *(_QWORD *)(v99 + 24) - (_QWORD)v93 > 1u )
                  {
                    *v93 = 8236;
                    *(_QWORD *)(v99 + 32) += 2LL;
                  }
                  else
                  {
                    sub_CB6200(v99, (unsigned __int8 *)", ", 2u);
                  }
                  ++v97;
                }
                while ( v133 != v97 );
              }
              v126 = *(_BYTE *)(*(_QWORD *)(v55 + 48) + v127 - 1);
              v155 = "0x%02x";
              LOBYTE(v156) = v126;
              v154 = &unk_49DD0D8;
              sub_CB6620(a2, (__int64)&v154, (__int64)&unk_49DD0D8, (__int64)&unk_49DD0C8, v94, v95);
            }
            return;
          case 0xB:
            v92 = "restore ";
            goto LABEL_138;
          case 0xC:
            v92 = "undefined ";
LABEL_138:
            sub_904010(a2, v92);
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            sub_2EAAF50(*(_DWORD *)(v55 + 8), a2, a10);
            return;
          case 0xD:
            sub_904010(a2, "register ");
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            sub_2EAAF50(*(_DWORD *)(v55 + 8), a2, a10);
            sub_904010(a2, ", ");
            sub_2EAAF50(*(_DWORD *)(v55 + 12), a2, a10);
            return;
          case 0xE:
            v91 = "window_save ";
            goto LABEL_131;
          case 0xF:
            v91 = "negate_ra_sign_state ";
            goto LABEL_131;
          case 0x10:
            v91 = "negate_ra_sign_state_with_pc ";
LABEL_131:
            sub_904010(a2, v91);
            if ( *(_QWORD *)v55 )
              sub_2EABE30(a2, *(_QWORD *)v55);
            break;
          default:
            sub_904010(a2, "<unserializable cfi directive>");
            break;
        }
      }
      else
      {
        sub_904010(a2, "<cfi directive>");
      }
      return;
    case 0x11:
      v80 = *(_DWORD *)(a1 + 24);
      if ( v80 > 0x3EEF )
      {
        v110 = sub_904010(a2, "intrinsic(");
        v111 = sub_CB59D0(v110, v80);
        sub_A51310(v111, 0x29u);
      }
      else
      {
        v81 = sub_904010(a2, "intrinsic(@");
        v82 = (unsigned __int8 *)sub_B60B70(v80);
        v84 = *(void **)(v81 + 32);
        if ( v83 > *(_QWORD *)(v81 + 24) - (_QWORD)v84 )
        {
          v81 = sub_CB6200(v81, v82, v83);
        }
        else if ( v83 )
        {
          v136 = v83;
          memcpy(v84, v82, v83);
          *(_QWORD *)(v81 + 32) += v136;
        }
        sub_A51310(v81, 0x29u);
      }
      return;
    case 0x12:
      v74 = *(_DWORD *)(a1 + 24);
      v75 = "int";
      if ( (unsigned int)(v74 - 32) >= 0xA )
        v75 = "float";
      v76 = sub_904010(a2, v75);
      v77 = sub_904010(v76, "pred(");
      v78 = sub_B52E10(v77, v74);
      v79 = *(_BYTE **)(v78 + 32);
      if ( (unsigned __int64)v79 < *(_QWORD *)(v78 + 24) )
        goto LABEL_116;
      goto LABEL_129;
    case 0x13:
      sub_904010(a2, "shufflemask(");
      v25 = *(int **)(a1 + 24);
      v26 = 0;
      v27 = &v25[*(_QWORD *)(a1 + 32)];
      v28 = v25;
      for ( i = 0; v27 != v28; i = 2 )
      {
        v30 = *(_QWORD *)(a2 + 32);
        v31 = a2;
        v32 = *v28;
        v33 = *(_QWORD *)(a2 + 24) - v30;
        if ( *v28 == -1 )
        {
          if ( i > v33 )
          {
            v31 = sub_CB6200(a2, v26, i);
          }
          else if ( i )
          {
            v108 = 0;
            do
            {
              v109 = v108++;
              *(_BYTE *)(v30 + v109) = v26[v109];
            }
            while ( v108 < (unsigned int)i );
            *(_QWORD *)(a2 + 32) += i;
            v31 = a2;
          }
          sub_904010(v31, "undef");
        }
        else
        {
          if ( i > v33 )
          {
            v31 = sub_CB6200(a2, v26, i);
          }
          else if ( i )
          {
            v105 = 0;
            do
            {
              v106 = v105++;
              *(_BYTE *)(v30 + v106) = v26[v106];
            }
            while ( v105 < (unsigned int)i );
            *(_QWORD *)(a2 + 32) += i;
            v31 = a2;
          }
          sub_CB59F0(v31, v32);
        }
        ++v28;
        v26 = (unsigned __int8 *)", ";
      }
      v107 = *(_BYTE **)(a2 + 32);
      if ( (unsigned __int64)v107 >= *(_QWORD *)(a2 + 24) )
      {
        sub_CB5D20(a2, 41);
      }
      else
      {
        *(_QWORD *)(a2 + 32) = v107 + 1;
        *v107 = 41;
      }
      return;
    case 0x14:
      v88 = sub_904010(a2, "dbg-instr-ref(");
      v89 = sub_CB59D0(v88, *(unsigned int *)(a1 + 24));
      v90 = sub_904010(v89, ", ");
      v78 = sub_CB59D0(v90, *(unsigned int *)(a1 + 28));
      v79 = *(_BYTE **)(v78 + 32);
      if ( (unsigned __int64)v79 < *(_QWORD *)(v78 + 24) )
      {
LABEL_116:
        *(_QWORD *)(v78 + 32) = v79 + 1;
        *v79 = 41;
      }
      else
      {
LABEL_129:
        sub_CB5D20(v78, 41);
      }
      return;
    default:
      return;
  }
}
