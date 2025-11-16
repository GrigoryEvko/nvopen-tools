// Function: sub_1188BC0
// Address: 0x1188bc0
//
unsigned __int8 *__fastcall sub_1188BC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // r15
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // rcx
  _BYTE *v15; // rdx
  unsigned int **v16; // rdi
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  bool v22; // cl
  bool v23; // zf
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rdx
  _BYTE *v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 *v32; // r11
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  bool v37; // al
  bool v38; // zf
  bool v39; // al
  unsigned __int8 *v40; // rdx
  unsigned int **v41; // rdi
  __int64 v42; // rax
  __int64 *v43; // r13
  const char *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // r15
  _BYTE **v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rdx
  _BYTE **v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rdx
  __int64 *v54; // r13
  const char *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  __int64 v58; // r12
  __int64 v59; // r14
  _BYTE *v60; // r15
  __int64 v61; // rax
  __int64 v62; // rcx
  __int64 v63; // rsi
  __int64 v64; // rdx
  _QWORD **v65; // r8
  unsigned int **v66; // rdi
  __int64 v67; // rdx
  __int64 v68; // r12
  __int64 v69; // r13
  __int64 v70; // rdx
  unsigned int v71; // esi
  _BYTE **v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rdx
  _BYTE *v75; // rsi
  unsigned int **v76; // r15
  __int64 v77; // rax
  _BYTE *v78; // r14
  __int64 v79; // r12
  __int64 v80; // rax
  __int64 v81; // r12
  __int64 v82; // r13
  __int64 v83; // rdx
  unsigned int v84; // esi
  char v85; // dl
  __int64 *v86; // rcx
  __int64 v87; // rax
  __int64 v88; // rax
  unsigned int **v89; // rdi
  __int64 v90; // rax
  unsigned int *v91; // r15
  __int64 v92; // rbx
  __int64 v93; // rdx
  unsigned int v94; // esi
  char v95; // al
  __int64 v96; // rsi
  __int64 v97; // r14
  __int64 *v98; // rdx
  char *v99; // rax
  char *v100; // r12
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  char *v105; // r14
  __int64 v106; // rdx
  __int64 v107; // rcx
  __int64 v108; // r8
  __int64 v109; // r9
  __int64 v110; // rax
  __int64 v111; // r12
  __int64 v112; // rdx
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // rax
  __int64 v117; // rax
  unsigned int **v118; // rdi
  __int64 v119; // rax
  char v120; // dl
  __int64 *v121; // rcx
  __int64 v122; // rax
  __int64 v123; // rax
  unsigned int **v124; // rdi
  __int64 v125; // rax
  bool v126; // al
  __int64 v127; // rax
  unsigned __int16 v128; // ax
  unsigned __int16 v129; // ax
  unsigned __int16 v130; // ax
  unsigned __int16 v131; // ax
  __int64 *v132; // rax
  __int64 *v133; // rax
  unsigned __int8 *v134; // [rsp+8h] [rbp-118h]
  unsigned __int8 *v135; // [rsp+8h] [rbp-118h]
  __int64 v136; // [rsp+10h] [rbp-110h]
  unsigned __int8 *v137; // [rsp+10h] [rbp-110h]
  unsigned __int8 *v138; // [rsp+10h] [rbp-110h]
  unsigned __int8 *v139; // [rsp+10h] [rbp-110h]
  _BYTE *v140; // [rsp+10h] [rbp-110h]
  __int64 *v141; // [rsp+10h] [rbp-110h]
  char v142; // [rsp+10h] [rbp-110h]
  _BYTE *v143; // [rsp+20h] [rbp-100h]
  _BYTE *v144; // [rsp+20h] [rbp-100h]
  __int64 v146; // [rsp+28h] [rbp-F8h]
  __int64 v147; // [rsp+28h] [rbp-F8h]
  _BYTE *v148; // [rsp+30h] [rbp-F0h] BYREF
  _BYTE *v149; // [rsp+38h] [rbp-E8h] BYREF
  _BYTE *v150; // [rsp+40h] [rbp-E0h] BYREF
  _BYTE *v151; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v152; // [rsp+50h] [rbp-D0h] BYREF
  _BYTE *v153; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v154[4]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v155; // [rsp+80h] [rbp-A0h]
  __int64 *v156; // [rsp+90h] [rbp-90h] BYREF
  _BYTE **v157; // [rsp+98h] [rbp-88h]
  const char *v158; // [rsp+A0h] [rbp-80h]
  __int64 v159; // [rsp+A8h] [rbp-78h]
  __int16 v160; // [rsp+B0h] [rbp-70h]
  char *v161; // [rsp+C0h] [rbp-60h] BYREF
  __int64 *v162; // [rsp+C8h] [rbp-58h] BYREF
  __int64 *v163; // [rsp+D0h] [rbp-50h] BYREF
  __int64 *v164; // [rsp+D8h] [rbp-48h]
  _BYTE **v165; // [rsp+E0h] [rbp-40h]
  __int64 *v166; // [rsp+E8h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 8);
  v4 = *(_QWORD *)(a2 - 96);
  v5 = *(_QWORD *)(a2 - 64);
  v6 = *(_QWORD *)(a2 - 32);
  v7 = v3;
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v7 = **(_QWORD **)(v3 + 16);
  if ( !sub_BCAC40(v7, 1) || *(_BYTE *)v4 <= 0x15u || *(_QWORD *)(v5 + 8) != *(_QWORD *)(v4 + 8) )
    return 0;
  v148 = (_BYTE *)sub_AD6400(v3);
  v8 = sub_AD6450(v3);
  v161 = 0;
  v149 = (_BYTE *)v8;
  if ( (unsigned __int8)sub_993A50((_QWORD **)&v161, v5) )
  {
    if ( (unsigned __int8)sub_98EF70(v6, v4) || *(_BYTE *)v6 == 82 && sub_117DB50(v6, (_BYTE *)v4, 0) )
    {
      LOWORD(v165) = 257;
      return (unsigned __int8 *)sub_B504D0(29, v4, v6, (__int64)&v161, 0, 0);
    }
    v162 = 0;
    v161 = (char *)&v150;
    v163 = (__int64 *)&v151;
    v9 = *(_QWORD *)(v4 + 16);
    if ( v9 )
    {
      if ( !*(_QWORD *)(v9 + 8) )
      {
        if ( (unsigned __int8)sub_1181510((_QWORD **)&v161, v4) )
        {
          v140 = v151;
          if ( (unsigned __int8)sub_98EF70(v6, (__int64)v151) || *(_BYTE *)v6 == 82 && sub_117DB50(v6, v140, 0) )
          {
            v160 = 257;
            v58 = *(_QWORD *)(a1 + 32);
            v155 = 257;
            v143 = v151;
            v59 = (*(__int64 (__fastcall **)(_QWORD, __int64, _BYTE *, __int64))(**(_QWORD **)(v58 + 80) + 16LL))(
                    *(_QWORD *)(v58 + 80),
                    29,
                    v151,
                    v6);
            if ( !v59 )
            {
              LOWORD(v165) = 257;
              v59 = sub_B504D0(29, (__int64)v143, v6, (__int64)&v161, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v58 + 88) + 16LL))(
                *(_QWORD *)(v58 + 88),
                v59,
                v154,
                *(_QWORD *)(v58 + 56),
                *(_QWORD *)(v58 + 64));
              v91 = *(unsigned int **)v58;
              v92 = *(_QWORD *)v58 + 16LL * *(unsigned int *)(v58 + 8);
              if ( *(_QWORD *)v58 != v92 )
              {
                do
                {
                  v93 = *((_QWORD *)v91 + 1);
                  v94 = *v91;
                  v91 += 4;
                  sub_B99FD0(v59, v94, v93);
                }
                while ( (unsigned int *)v92 != v91 );
              }
            }
            v60 = v150;
            v61 = sub_AD62B0(*(_QWORD *)(v59 + 8));
            v62 = v59;
            v63 = (__int64)v60;
            v64 = v61;
            v65 = &v156;
            v66 = (unsigned int **)v58;
            goto LABEL_150;
          }
        }
      }
    }
    v156 = (__int64 *)&v150;
    v157 = &v151;
    if ( sub_1181310(&v156, v4) )
    {
      v161 = (char *)&v152;
      v162 = (__int64 *)&v153;
      if ( sub_1181310((_QWORD **)&v161, v6)
        && ((v35 = *(_QWORD *)(v4 + 16)) != 0 && !*(_QWORD *)(v35 + 8)
         || (v36 = *(_QWORD *)(v6 + 16)) != 0 && !*(_QWORD *)(v36 + 8)) )
      {
        v37 = *(_BYTE *)v4 == 86;
        LOBYTE(v154[0]) = v37;
        v38 = *(_BYTE *)v6 == 86;
        v161 = (char *)a1;
        v162 = (__int64 *)&v148;
        v164 = v154;
        v163 = (__int64 *)&v156;
        v165 = &v150;
        v166 = (__int64 *)&v149;
        LOBYTE(v156) = v38;
        if ( v150 == (_BYTE *)v152 )
          return (unsigned __int8 *)sub_1179150((__int64)&v161, (__int64)v150, (__int64)v151, (__int64)v153, 0);
        if ( v150 == v153 )
          return (unsigned __int8 *)sub_1179150((__int64)&v161, (__int64)v150, (__int64)v151, v152, 0);
        if ( (_BYTE *)v152 == v151 )
          return (unsigned __int8 *)sub_1179150((__int64)&v161, v152, (__int64)v150, (__int64)v153, 0);
        if ( v153 == v151 )
          return (unsigned __int8 *)sub_1179150(
                                      (__int64)&v161,
                                      (__int64)v153,
                                      (__int64)v150,
                                      v152,
                                      v38 & (unsigned __int8)v37);
      }
    }
  }
  if ( (unsigned __int8)sub_1178DE0(v6) )
  {
    if ( (unsigned __int8)sub_98EF70(v5, v4) || *(_BYTE *)v5 == 82 && sub_117DB50(v5, (_BYTE *)v4, 1) )
    {
      LOWORD(v165) = 257;
      return (unsigned __int8 *)sub_B504D0(28, v4, v5, (__int64)&v161, 0, 0);
    }
    v19 = *(_QWORD *)(v4 + 16);
    if ( v19 && !*(_QWORD *)(v19 + 8) && *(_BYTE *)v4 == 86 )
    {
      v72 = (*(_BYTE *)(v4 + 7) & 0x40) != 0
          ? *(_BYTE ***)(v4 - 8)
          : (_BYTE **)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
      if ( *v72 )
      {
        v150 = *v72;
        v73 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)(v73 + 32) )
        {
          v151 = *(_BYTE **)(v73 + 32);
          v74 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
          if ( (unsigned __int8)sub_1178DE0(*(_QWORD *)(v74 + 64)) )
          {
            v144 = v151;
            if ( (unsigned __int8)sub_98EF70(v5, (__int64)v151) || *(_BYTE *)v5 == 82 && sub_117DB50(v5, v144, 1) )
            {
              LOWORD(v165) = 257;
              v75 = v151;
              v76 = *(unsigned int ***)(a1 + 32);
              v160 = 257;
              v77 = sub_A82350(v76, v151, (_BYTE *)v5, (__int64)&v156);
              v78 = v150;
              v79 = v77;
              v80 = sub_AD6530(*(_QWORD *)(v77 + 8), (__int64)v75);
              v65 = (_QWORD **)&v161;
              v64 = v79;
              v62 = v80;
              v63 = (__int64)v78;
              v66 = v76;
LABEL_150:
              v67 = sub_B36550(v66, v63, v64, v62, (__int64)v65, 0);
              return sub_F162A0(a1, a2, v67);
            }
          }
        }
      }
    }
    v156 = (__int64 *)&v150;
    v157 = &v151;
    if ( sub_1181410(&v156, (unsigned __int8 *)v4) )
    {
      v161 = (char *)&v152;
      v162 = (__int64 *)&v153;
      if ( sub_1181410((_QWORD **)&v161, (unsigned __int8 *)v5)
        && ((v20 = *(_QWORD *)(v4 + 16)) != 0 && !*(_QWORD *)(v20 + 8)
         || (v21 = *(_QWORD *)(v5 + 16)) != 0 && !*(_QWORD *)(v21 + 8)) )
      {
        v22 = *(_BYTE *)v4 == 86;
        LOBYTE(v154[0]) = v22;
        v23 = *(_BYTE *)v5 == 86;
        v161 = (char *)a1;
        v162 = (__int64 *)&v149;
        v164 = v154;
        v163 = (__int64 *)&v156;
        v165 = &v150;
        v166 = (__int64 *)&v148;
        LOBYTE(v156) = v23;
        if ( v150 == (_BYTE *)v152 )
          return (unsigned __int8 *)sub_1178F40((__int64)&v161, (__int64)v150, (__int64)v151, (__int64)v153, 0);
        if ( v150 == v153 )
          return (unsigned __int8 *)sub_1178F40((__int64)&v161, (__int64)v150, (__int64)v151, v152, 0);
        if ( (_BYTE *)v152 == v151 )
          return (unsigned __int8 *)sub_1178F40((__int64)&v161, v152, (__int64)v150, (__int64)v153, 0);
        if ( v153 == v151 )
          return (unsigned __int8 *)sub_1178F40(
                                      (__int64)&v161,
                                      (__int64)v153,
                                      (__int64)v150,
                                      v152,
                                      v22 & (unsigned __int8)v23);
      }
    }
  }
  if ( v149 == (_BYTE *)v5 )
  {
    v54 = *(__int64 **)(a1 + 32);
    v55 = sub_BD5D20(v4);
    v160 = 1283;
    v156 = (__int64 *)"not.";
    v159 = v56;
    v158 = v55;
    v147 = sub_AD62B0(*(_QWORD *)(v4 + 8));
    v57 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v54[10] + 16LL))(
            v54[10],
            30,
            v4,
            v147);
    if ( !v57 )
    {
      LOWORD(v165) = 257;
      v57 = sub_B504D0(30, v4, v147, (__int64)&v161, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, __int64 **, __int64, __int64))(*(_QWORD *)v54[11] + 16LL))(
        v54[11],
        v57,
        &v156,
        v54[7],
        v54[8]);
      v81 = *v54;
      v82 = *v54 + 16LL * *((unsigned int *)v54 + 2);
      while ( v82 != v81 )
      {
        v83 = *(_QWORD *)(v81 + 8);
        v84 = *(_DWORD *)v81;
        v81 += 16;
        sub_B99FD0(v57, v84, v83);
      }
    }
    LOWORD(v165) = 257;
    return sub_109FEA0(v57, v6, (__int64)v149, (const char **)&v161, 0, 0, 0);
  }
  if ( v148 == (_BYTE *)v6 )
  {
    v43 = *(__int64 **)(a1 + 32);
    v44 = sub_BD5D20(v4);
    v160 = 1283;
    v156 = (__int64 *)"not.";
    v159 = v45;
    v158 = v44;
    v146 = sub_AD62B0(*(_QWORD *)(v4 + 8));
    v46 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v43[10] + 16LL))(
            v43[10],
            30,
            v4,
            v146);
    if ( !v46 )
    {
      LOWORD(v165) = 257;
      v46 = sub_B504D0(30, v4, v146, (__int64)&v161, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, __int64 **, __int64, __int64))(*(_QWORD *)v43[11] + 16LL))(
        v43[11],
        v46,
        &v156,
        v43[7],
        v43[8]);
      v68 = *v43;
      v69 = *v43 + 16LL * *((unsigned int *)v43 + 2);
      while ( v69 != v68 )
      {
        v70 = *(_QWORD *)(v68 + 8);
        v71 = *(_DWORD *)v68;
        v68 += 16;
        sub_B99FD0(v46, v71, v70);
      }
    }
    LOWORD(v165) = 257;
    return sub_109FEA0(v46, (__int64)v148, v5, (const char **)&v161, 0, 0, 0);
  }
  v10 = *(_QWORD *)(a2 + 8);
  v161 = 0;
  v162 = (__int64 *)&v150;
  v163 = 0;
  v164 = (__int64 *)&v151;
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
    v10 = **(_QWORD **)(v10 + 16);
  if ( !sub_BCAC40(v10, 1) )
    goto LABEL_52;
  if ( *(_BYTE *)a2 == 57 )
  {
    v24 = sub_986520(a2);
    v137 = *(unsigned __int8 **)(v24 + 32);
    if ( (unsigned __int8)sub_996420((_QWORD **)&v161, 30, *(unsigned __int8 **)v24)
      && (unsigned __int8)sub_996420(&v163, 30, v137) )
    {
LABEL_24:
      v12 = *(_QWORD *)(v4 + 16);
      if ( (v12 && !*(_QWORD *)(v12 + 8) || (v13 = *(_QWORD *)(v5 + 16)) != 0 && !*(_QWORD *)(v13 + 8))
        && (*v150 > 0x15u || *v150 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v150)) )
      {
        v14 = v151;
        if ( *v151 > 0x15u )
        {
LABEL_34:
          v15 = v148;
          LOWORD(v165) = 257;
          v16 = *(unsigned int ***)(a1 + 32);
          v160 = 257;
LABEL_35:
          v17 = sub_B36550(v16, (__int64)v150, (__int64)v15, (__int64)v14, (__int64)&v156, 0);
          return (unsigned __int8 *)sub_B50640(v17, (__int64)&v161, 0, 0);
        }
        if ( *v151 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v151) )
        {
          v14 = v151;
          goto LABEL_34;
        }
      }
    }
LABEL_52:
    v11 = *(_QWORD *)(a2 + 8);
    goto LABEL_53;
  }
  v11 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)a2 == 86 )
  {
    v136 = *(_QWORD *)(a2 - 96);
    if ( *(_QWORD *)(v136 + 8) == v11 && **(_BYTE **)(a2 - 32) <= 0x15u )
    {
      v134 = *(unsigned __int8 **)(a2 - 64);
      if ( sub_AC30F0(*(_QWORD *)(a2 - 32))
        && (unsigned __int8)sub_996420((_QWORD **)&v161, 30, (unsigned __int8 *)v136)
        && (unsigned __int8)sub_996420(&v163, 30, v134) )
      {
        goto LABEL_24;
      }
      goto LABEL_52;
    }
  }
LABEL_53:
  v161 = 0;
  v163 = 0;
  v162 = (__int64 *)&v150;
  v164 = (__int64 *)&v151;
  if ( (unsigned int)*(unsigned __int8 *)(v11 + 8) - 17 <= 1 )
    v11 = **(_QWORD **)(v11 + 16);
  if ( sub_BCAC40(v11, 1) )
  {
    if ( *(_BYTE *)a2 == 58 )
    {
      v31 = sub_986520(a2);
      v139 = *(unsigned __int8 **)(v31 + 32);
      if ( !(unsigned __int8)sub_996420((_QWORD **)&v161, 30, *(unsigned __int8 **)v31)
        || !(unsigned __int8)sub_996420(&v163, 30, v139) )
      {
        goto LABEL_76;
      }
    }
    else
    {
      if ( *(_BYTE *)a2 != 86 )
        goto LABEL_76;
      v27 = *(_QWORD *)(a2 - 96);
      v138 = (unsigned __int8 *)v27;
      if ( *(_QWORD *)(v27 + 8) != *(_QWORD *)(a2 + 8) )
        goto LABEL_76;
      v28 = *(_BYTE **)(a2 - 64);
      if ( *v28 > 0x15u )
        goto LABEL_76;
      v135 = *(unsigned __int8 **)(a2 - 32);
      if ( !sub_AD7A80(v28, 1, v27, v25, v26)
        || !(unsigned __int8)sub_996420((_QWORD **)&v161, 30, v138)
        || !(unsigned __int8)sub_996420(&v163, 30, v135) )
      {
        goto LABEL_76;
      }
    }
    v29 = *(_QWORD *)(v4 + 16);
    if ( (v29 && !*(_QWORD *)(v29 + 8) || (v30 = *(_QWORD *)(v6 + 16)) != 0 && !*(_QWORD *)(v30 + 8))
      && (*v150 > 0x15u || *v150 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v150)) )
    {
      v15 = v151;
      if ( *v151 > 0x15u )
      {
LABEL_73:
        v14 = v149;
        LOWORD(v165) = 257;
        v16 = *(unsigned int ***)(a1 + 32);
        v160 = 257;
        goto LABEL_35;
      }
      if ( *v151 != 5 && !(unsigned __int8)sub_AD6CA0((__int64)v151) )
      {
        v15 = v151;
        goto LABEL_73;
      }
    }
  }
LABEL_76:
  v162 = 0;
  v161 = (char *)&v150;
  v163 = (__int64 *)&v151;
  if ( *(_BYTE *)v4 == 86 )
  {
    v47 = (*(_BYTE *)(v4 + 7) & 0x40) != 0
        ? *(_BYTE ***)(v4 - 8)
        : (_BYTE **)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
    if ( *v47 )
    {
      v150 = *v47;
      v48 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
      if ( (unsigned __int8)sub_993A50(&v162, *(_QWORD *)(v48 + 32)) )
      {
        v49 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
        v50 = *(_QWORD *)(v49 + 64);
        if ( v50 )
        {
          *v163 = v50;
          v156 = 0;
          if ( (unsigned __int8)sub_993A50(&v156, v5) )
          {
            if ( v151 == (_BYTE *)v6 )
              return (unsigned __int8 *)sub_F20660(a1, a2, 0, (__int64)v150);
          }
        }
      }
    }
  }
  if ( *(_BYTE *)v4 == 86 )
  {
    v51 = (*(_BYTE *)(v4 + 7) & 0x40) != 0
        ? *(_BYTE ***)(v4 - 8)
        : (_BYTE **)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
    if ( *v51 )
    {
      v150 = *v51;
      v52 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)(v52 + 32) )
      {
        v151 = *(_BYTE **)(v52 + 32);
        v53 = (*(_BYTE *)(v4 + 7) & 0x40) != 0 ? *(_QWORD *)(v4 - 8) : v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF);
        if ( (unsigned __int8)sub_1178DE0(*(_QWORD *)(v53 + 64))
          && v151 == (_BYTE *)v5
          && (unsigned __int8)sub_1178DE0(v6) )
        {
          return (unsigned __int8 *)sub_F20660(a1, a2, 0, (__int64)v150);
        }
      }
    }
  }
  v161 = 0;
  v162 = (__int64 *)&v150;
  v163 = (__int64 *)&v151;
  v164 = (__int64 *)&v150;
  v165 = &v151;
  if ( (unsigned __int8)sub_1187090((_QWORD **)&v161, a2) )
  {
    LOWORD(v165) = 257;
    return (unsigned __int8 *)sub_B504D0(30, (__int64)v150, (__int64)v151, (__int64)&v161, 0, 0);
  }
  v32 = &v152;
  v161 = 0;
  v162 = (__int64 *)v5;
  v163 = &v152;
  v33 = *(_QWORD *)(v4 + 16);
  if ( v33 && !*(_QWORD *)(v33 + 8) && *(_BYTE *)v4 == 58 )
  {
    v39 = sub_9987C0((__int64)&v161, 30, *(unsigned __int8 **)(v4 - 64));
    v40 = *(unsigned __int8 **)(v4 - 32);
    if ( v39 && v40 )
    {
      *v163 = (__int64)v40;
LABEL_112:
      v41 = *(unsigned int ***)(a1 + 32);
      LOWORD(v165) = 257;
      v42 = sub_B36550(v41, v152, (__int64)v148, v6, (__int64)&v161, 0);
      LOWORD(v165) = 257;
      return sub_109FEA0(v5, v42, (__int64)v149, (const char **)&v161, 0, 0, 0);
    }
    v126 = sub_9987C0((__int64)&v161, 30, v40);
    v32 = &v152;
    if ( v126 )
    {
      v127 = *(_QWORD *)(v4 - 64);
      if ( v127 )
      {
        *v163 = v127;
        goto LABEL_112;
      }
    }
    v33 = *(_QWORD *)(v4 + 16);
  }
  if ( v33 )
  {
    if ( !*(_QWORD *)(v33 + 8)
      && *(_BYTE *)v4 == 57
      && (*(_QWORD *)(v4 - 64) && (v152 = *(_QWORD *)(v4 - 64), v6 == *(_QWORD *)(v4 - 32))
       || *(_QWORD *)(v4 - 32) && (v152 = *(_QWORD *)(v4 - 32), v6 == *(_QWORD *)(v4 - 64))) )
    {
      v120 = 0;
      v121 = *(__int64 **)(a1 + 32);
      v122 = *(_QWORD *)(v152 + 16);
      if ( v122 )
        v120 = *(_QWORD *)(v122 + 8) == 0;
      LOBYTE(v161) = 0;
      v123 = sub_F13D80((__int64 *)a1, v152, v120, v121, &v161, 0);
      v32 = &v152;
      if ( v123 )
      {
        v124 = *(unsigned int ***)(a1 + 32);
        LOWORD(v165) = 257;
        v125 = sub_B36550(v124, v123, (__int64)v148, v5, (__int64)&v161, 0);
        LOWORD(v165) = 257;
        return sub_109FEA0(v6, v125, (__int64)v149, (const char **)&v161, 0, 0, 0);
      }
      v33 = *(_QWORD *)(v4 + 16);
      v161 = (char *)v5;
      v162 = &v152;
      if ( !v33 )
        goto LABEL_84;
    }
    else
    {
      v161 = (char *)v5;
      v162 = &v152;
    }
    if ( !*(_QWORD *)(v33 + 8) && *(_BYTE *)v4 == 58 && (unsigned __int8)sub_11783C0((__int64)&v161, v4) )
    {
      v85 = 0;
      v86 = *(__int64 **)(a1 + 32);
      v87 = *(_QWORD *)(v152 + 16);
      if ( v87 )
        v85 = *(_QWORD *)(v87 + 8) == 0;
      v141 = v32;
      LOBYTE(v161) = 0;
      v88 = sub_F13D80((__int64 *)a1, v152, v85, v86, &v161, 0);
      v32 = v141;
      if ( v88 )
      {
        v89 = *(unsigned int ***)(a1 + 32);
        LOWORD(v165) = 257;
        v90 = sub_B36550(v89, v88, v6, (__int64)v149, (__int64)&v161, 0);
        LOWORD(v165) = 257;
        return sub_109FEA0(v5, (__int64)v148, v90, (const char **)&v161, 0, 0, 0);
      }
    }
  }
LABEL_84:
  v161 = (char *)v32;
  v162 = 0;
  v163 = (__int64 *)v6;
  v34 = *(_QWORD *)(v4 + 16);
  if ( v34 && !*(_QWORD *)(v34 + 8) && *(_BYTE *)v4 == 57 )
  {
    if ( *(_QWORD *)(v4 - 64)
      && (v152 = *(_QWORD *)(v4 - 64), sub_9987C0((__int64)&v162, 30, *(unsigned __int8 **)(v4 - 32)))
      || (v117 = *(_QWORD *)(v4 - 32)) != 0
      && (*(_QWORD *)v161 = v117, sub_9987C0((__int64)&v162, 30, *(unsigned __int8 **)(v4 - 64))) )
    {
      v118 = *(unsigned int ***)(a1 + 32);
      LOWORD(v165) = 257;
      v119 = sub_B36550(v118, v152, v5, (__int64)v149, (__int64)&v161, 0);
      LOWORD(v165) = 257;
      return sub_109FEA0(v6, (__int64)v148, v119, (const char **)&v161, 0, 0, 0);
    }
  }
  if ( (unsigned __int8)sub_1178DE0(v6) || (v161 = 0, (unsigned __int8)sub_993A50((_QWORD **)&v161, v5)) )
  {
    v156 = 0;
    v95 = sub_1178DE0(v6);
    v96 = v6;
    if ( v95 )
      v96 = v5;
    v142 = v95;
    if ( (unsigned __int8)sub_104A1B0((_BYTE *)v4, v96, v95, &v156) )
    {
      v97 = *v156;
      v161 = (char *)sub_BD5D20(*v156);
      v163 = (__int64 *)".fr";
      LOWORD(v165) = 773;
      v162 = v98;
      v99 = (char *)sub_BD2C40(72, unk_3F10A14);
      v100 = v99;
      if ( v99 )
        sub_B549F0((__int64)v99, v97, (__int64)&v161, 0, 0);
      sub_B44220(v100, v156[3] + 24, 0);
      v161 = v100;
      sub_1187E30(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v161, v101, v102, v103, v104);
      v105 = (char *)*v156;
      sub_AC2B30((__int64)v156, (__int64)v100);
      if ( (unsigned __int8)*v105 > 0x1Cu )
      {
        v110 = *(_QWORD *)(a1 + 40);
        v161 = v105;
        v111 = v110 + 2096;
        sub_1187E30(v110 + 2096, (__int64 *)&v161, v106, v107, v108, v109);
        v116 = *((_QWORD *)v105 + 2);
        if ( v116 )
        {
          if ( !*(_QWORD *)(v116 + 8) )
          {
            v161 = *(char **)(v116 + 24);
            sub_1187E30(v111, (__int64 *)&v161, v112, v113, v114, v115);
          }
        }
      }
      return sub_F162A0(a1, a2, v96);
    }
    v67 = sub_10CF840((const __m128i *)a1, v4, v96, a2, v142, 1);
    if ( v67 )
      return sub_F162A0(a1, a2, v67);
  }
  v161 = (char *)&v150;
  v162 = (__int64 *)&v151;
  if ( sub_1181410((_QWORD **)&v161, (unsigned __int8 *)v4) )
  {
    if ( (unsigned __int8)sub_1178DE0(v6) )
    {
      v130 = sub_9A18B0(v5, v151, *(_QWORD *)(a1 + 88), 1u, 0);
      if ( HIBYTE(v130) )
      {
        if ( !(_BYTE)v130 )
          return (unsigned __int8 *)sub_F20660(a1, a2, 0, (__int64)v150);
      }
    }
  }
  v161 = (char *)&v150;
  v162 = (__int64 *)&v151;
  if ( sub_1181410((_QWORD **)&v161, (unsigned __int8 *)v5) )
  {
    if ( (unsigned __int8)sub_1178DE0(v6) )
    {
      v129 = sub_9A18B0(v4, v151, *(_QWORD *)(a1 + 88), 1u, 0);
      if ( HIBYTE(v129) )
      {
        if ( !(_BYTE)v129 )
          return (unsigned __int8 *)sub_F20660(a1, a2, 1u, (__int64)v150);
      }
    }
  }
  v156 = 0;
  if ( (unsigned __int8)sub_993A50(&v156, v5) )
  {
    v161 = (char *)&v150;
    v162 = (__int64 *)&v151;
    if ( sub_1181310((_QWORD **)&v161, v6) )
    {
      v131 = sub_9A18B0(v4, v151, *(_QWORD *)(a1 + 88), 0, 0);
      if ( HIBYTE(v131) )
      {
        if ( (_BYTE)v131 )
          return (unsigned __int8 *)sub_F20660(a1, a2, 2u, (__int64)v150);
      }
    }
  }
  v161 = (char *)&v150;
  v162 = (__int64 *)&v151;
  if ( sub_1181310((_QWORD **)&v161, v4) )
  {
    v156 = 0;
    if ( (unsigned __int8)sub_993A50(&v156, v5) )
    {
      v128 = sub_9A18B0(v6, v151, *(_QWORD *)(a1 + 88), 0, 0);
      if ( HIBYTE(v128) )
      {
        if ( (_BYTE)v128 )
          return (unsigned __int8 *)sub_F20660(a1, a2, 0, (__int64)v150);
      }
    }
  }
  v161 = 0;
  if ( !(unsigned __int8)sub_993A50((_QWORD **)&v161, v5) )
    return 0;
  v161 = 0;
  v162 = v154;
  v163 = (__int64 *)&v151;
  if ( !(unsigned __int8)sub_1184DC0((_QWORD **)&v161, v6)
    || (v156 = (__int64 *)v154[0], v157 = &v150, !sub_11811E0((__int64)&v156, v4)) )
  {
    v161 = 0;
    v162 = v154;
    v163 = (__int64 *)&v150;
    if ( (unsigned __int8)sub_1184DC0((_QWORD **)&v161, v4) )
    {
      v156 = (__int64 *)v154[0];
      v157 = &v151;
      if ( sub_11811E0((__int64)&v156, v6) )
      {
        if ( *(_BYTE *)v4 == 86 && *(_BYTE *)v6 == 86 )
        {
          v132 = *(__int64 **)(v6 - 64);
          v161 = 0;
          v162 = v132;
          if ( sub_9987C0((__int64)&v161, 30, *(unsigned __int8 **)(v4 - 64)) )
          {
            LOWORD(v165) = 257;
            v154[0] = sub_1156690(*(__int64 **)(a1 + 32), v154[0], (__int64)&v161);
          }
        }
        LOWORD(v165) = 257;
        return sub_109FEA0(v154[0], (__int64)v151, (__int64)v150, (const char **)&v161, 0, 0, 0);
      }
    }
    return 0;
  }
  if ( *(_BYTE *)v4 == 86 && *(_BYTE *)v6 == 86 )
  {
    v133 = *(__int64 **)(v4 - 64);
    v161 = 0;
    v162 = v133;
    if ( sub_9987C0((__int64)&v161, 30, *(unsigned __int8 **)(v6 - 64)) )
    {
      LOWORD(v165) = 257;
      v154[0] = sub_1156690(*(__int64 **)(a1 + 32), v154[0], (__int64)&v161);
    }
  }
  LOWORD(v165) = 257;
  return sub_109FEA0(v154[0], (__int64)v150, (__int64)v151, (const char **)&v161, 0, 0, 0);
}
