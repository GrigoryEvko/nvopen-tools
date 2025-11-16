// Function: sub_274BEC0
// Address: 0x274bec0
//
__int64 __fastcall sub_274BEC0(unsigned __int8 *a1, __int64 *a2)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int *v11; // rax
  unsigned int v12; // r12d
  unsigned __int8 *v13; // rbx
  unsigned __int8 *v14; // r14
  _BYTE *v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // r12d
  unsigned int v21; // eax
  unsigned int v22; // r12d
  __int64 v23; // r14
  __int64 *v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // r14d
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned int *v35; // rsi
  __int64 v36; // r8
  __int64 v37; // rdi
  unsigned __int8 *v38; // r8
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rdi
  unsigned int *v42; // r8
  int v43; // eax
  __int64 *v44; // rax
  __int64 v45; // rdx
  unsigned __int8 *v46; // r10
  __int64 v47; // rax
  __int64 v48; // rax
  const char *v49; // rax
  __int64 v50; // rdx
  _BYTE *v51; // rax
  unsigned __int8 *v52; // r12
  int v53; // edx
  __int64 v54; // r14
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rbx
  int v58; // ebx
  __int64 v59; // rax
  __int64 v60; // rdx
  unsigned __int8 *v61; // rcx
  int v62; // r14d
  unsigned __int8 *v63; // rbx
  __int64 *v64; // r12
  __int64 v65; // rax
  __int64 *v66; // rsi
  unsigned int *v67; // rdi
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // r8
  __int64 v71; // r9
  __int64 v72; // rax
  unsigned __int64 v73; // rdx
  int v74; // edx
  __int64 *v75; // rdx
  unsigned __int8 *v76; // rdx
  __int64 v77; // rax
  unsigned int v78; // eax
  __int64 v79; // rax
  unsigned int v80; // eax
  unsigned int v81; // eax
  __int64 v82; // rdi
  unsigned __int8 v83; // dl
  __int64 v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rsi
  unsigned __int8 *v87; // rsi
  unsigned int v88; // edx
  unsigned int v89; // edi
  int v90; // r14d
  __int64 *v91; // rdx
  unsigned __int8 *v92; // rdx
  unsigned int v93; // eax
  __int64 v94; // rax
  unsigned int v95; // eax
  __int64 v96; // rax
  unsigned int v97; // eax
  int v98; // edi
  unsigned int v99; // r13d
  int v100; // eax
  __int64 v101; // rax
  __int64 v102; // rcx
  __int64 v103; // rdx
  __int64 v104; // rax
  __int64 v105; // rax
  int v106; // eax
  bool v107; // bl
  __int64 v108; // rax
  __int64 v109; // rdx
  __int64 v110; // rbx
  _BYTE *v111; // r11
  __int64 (__fastcall *v112)(__int64, _BYTE *, unsigned __int8 *, unsigned __int64 *, __int64, __int64); // rax
  __int64 v113; // rax
  __int64 v114; // r10
  _QWORD *v115; // rax
  _QWORD *v116; // r10
  __int64 v117; // r11
  __int64 v118; // r14
  unsigned int *v119; // rbx
  __int64 v120; // rdx
  unsigned int v121; // esi
  __int64 v122; // rax
  __int64 v123; // [rsp-10h] [rbp-190h]
  unsigned int *v124; // [rsp+0h] [rbp-180h]
  int v125; // [rsp+8h] [rbp-178h]
  __int64 v126; // [rsp+10h] [rbp-170h]
  __int64 v127; // [rsp+18h] [rbp-168h]
  char v128; // [rsp+18h] [rbp-168h]
  unsigned __int8 *v129; // [rsp+18h] [rbp-168h]
  unsigned int v130; // [rsp+18h] [rbp-168h]
  __int64 v131; // [rsp+18h] [rbp-168h]
  char v132; // [rsp+20h] [rbp-160h]
  char v133; // [rsp+20h] [rbp-160h]
  _BYTE *v134; // [rsp+20h] [rbp-160h]
  unsigned int v135; // [rsp+20h] [rbp-160h]
  unsigned __int8 *v136; // [rsp+20h] [rbp-160h]
  __int64 v137; // [rsp+28h] [rbp-158h]
  __int64 v138; // [rsp+28h] [rbp-158h]
  __int64 v139; // [rsp+28h] [rbp-158h]
  int v140; // [rsp+28h] [rbp-158h]
  __int64 v141; // [rsp+28h] [rbp-158h]
  __int64 v142; // [rsp+28h] [rbp-158h]
  char v143; // [rsp+28h] [rbp-158h]
  _BYTE *v144; // [rsp+28h] [rbp-158h]
  _BYTE *v145; // [rsp+28h] [rbp-158h]
  __int64 v146; // [rsp+28h] [rbp-158h]
  __int64 v147; // [rsp+28h] [rbp-158h]
  __int64 v148; // [rsp+28h] [rbp-158h]
  _BYTE *v149; // [rsp+28h] [rbp-158h]
  __int64 v150; // [rsp+38h] [rbp-148h]
  unsigned __int64 v151; // [rsp+40h] [rbp-140h] BYREF
  unsigned int v152; // [rsp+48h] [rbp-138h]
  unsigned __int64 v153; // [rsp+50h] [rbp-130h]
  unsigned int v154; // [rsp+58h] [rbp-128h]
  unsigned __int64 v155; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v156; // [rsp+68h] [rbp-118h]
  unsigned __int64 v157; // [rsp+70h] [rbp-110h]
  unsigned int v158; // [rsp+78h] [rbp-108h]
  __int16 v159; // [rsp+80h] [rbp-100h]
  unsigned __int64 v160; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v161; // [rsp+98h] [rbp-E8h]
  unsigned __int64 v162; // [rsp+A0h] [rbp-E0h]
  unsigned int v163; // [rsp+A8h] [rbp-D8h]
  __int16 v164; // [rsp+B0h] [rbp-D0h]
  __int64 v165; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v166; // [rsp+C8h] [rbp-B8h]
  unsigned __int64 v167; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v168; // [rsp+D8h] [rbp-A8h]
  __int16 v169; // [rsp+E0h] [rbp-A0h]
  __int64 v170; // [rsp+F8h] [rbp-88h]
  __int64 v171; // [rsp+100h] [rbp-80h]
  __int64 v172; // [rsp+110h] [rbp-70h]
  __int64 v173; // [rsp+118h] [rbp-68h]
  void *v174; // [rsp+140h] [rbp-40h]

  if ( (unsigned int)sub_B49240((__int64)a1) == 1 )
  {
    v127 = *(_QWORD *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
    v19 = *(_QWORD *)&a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
    v20 = *(_DWORD *)(v19 + 32);
    if ( v20 <= 0x40 )
      v132 = *(_QWORD *)(v19 + 24) == 1;
    else
      v132 = v20 - 1 == (unsigned int)sub_C444A0(v19 + 24);
    v21 = sub_BCB060(*(_QWORD *)(v127 + 8));
    v22 = v21 - 1;
    v152 = v21;
    v23 = 1LL << ((unsigned __int8)v21 - 1);
    if ( v21 > 0x40 )
    {
      sub_C43690((__int64)&v151, 0, 0);
      if ( v152 > 0x40 )
      {
        *(_QWORD *)(v151 + 8LL * (v22 >> 6)) |= v23;
LABEL_29:
        if ( (a1[7] & 0x40) != 0 )
          v24 = (__int64 *)*((_QWORD *)a1 - 1);
        else
          v24 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        sub_22CEA30((__int64)&v155, a2, v24, v132);
        LODWORD(v161) = v152;
        if ( v152 > 0x40 )
          sub_C43780((__int64)&v160, (const void **)&v151);
        else
          v160 = v151;
        sub_AADBC0((__int64)&v165, (__int64 *)&v160);
        LOBYTE(v25) = sub_ABB410((__int64 *)&v155, 37, &v165);
        v12 = v25;
        if ( v168 > 0x40 && v167 )
          j_j___libc_free_0_0(v167);
        if ( (unsigned int)v166 > 0x40 && v165 )
          j_j___libc_free_0_0(v165);
        if ( (unsigned int)v161 > 0x40 && v160 )
          j_j___libc_free_0_0(v160);
        if ( (_BYTE)v12 )
        {
          sub_BD84D0((__int64)a1, v127);
          sub_B43D60(a1);
        }
        else
        {
          sub_AB13A0((__int64)&v165, (__int64)&v155);
          v26 = 1LL << ((unsigned __int8)v166 - 1);
          if ( (unsigned int)v166 > 0x40 )
          {
            v42 = (unsigned int *)v165;
            if ( (*(_QWORD *)(v165 + 8LL * ((unsigned int)(v166 - 1) >> 6)) & v26) != 0
              || (v124 = (unsigned int *)v165, v125 = v166, v43 = sub_C444A0((__int64)&v165), v42 = v124, v125 == v43) )
            {
              if ( v42 )
                j_j___libc_free_0_0((unsigned __int64)v42);
              goto LABEL_100;
            }
            if ( v124 )
              j_j___libc_free_0_0((unsigned __int64)v124);
          }
          else if ( (v165 & v26) != 0 || !v165 )
          {
LABEL_100:
            sub_23D0AB0((__int64)&v165, (__int64)a1, 0, 0, 0);
            v49 = sub_BD5D20((__int64)a1);
            v164 = 261;
            v161 = v50;
            v160 = (unsigned __int64)v49;
            v51 = (_BYTE *)sub_AD6530(*(_QWORD *)(v127 + 8), (__int64)a1);
            v52 = (unsigned __int8 *)sub_929DE0((unsigned int **)&v165, v51, (_BYTE *)v127, (__int64)&v160, 0, v132);
            sub_BD84D0((__int64)a1, (__int64)v52);
            sub_B43D60(a1);
            if ( (unsigned __int8)(*v52 - 42) <= 0x11u )
              sub_274B8C0(v52, a2);
            nullsub_61();
            v174 = &unk_49DA100;
            nullsub_63();
            if ( (unsigned __int64 *)v165 != &v167 )
              _libc_free(v165);
LABEL_97:
            v12 = 1;
            goto LABEL_73;
          }
          if ( !v132 && !sub_AB1B10((__int64)&v155, (__int64)&v151) )
          {
            v44 = (__int64 *)sub_BD5C60((__int64)a1);
            v45 = sub_ACD6D0(v44);
            v46 = &a1[32 * (1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
            if ( *(_QWORD *)v46 )
            {
              v47 = *((_QWORD *)v46 + 1);
              **((_QWORD **)v46 + 2) = v47;
              if ( v47 )
                *(_QWORD *)(v47 + 16) = *((_QWORD *)v46 + 2);
            }
            *(_QWORD *)v46 = v45;
            v12 = 1;
            if ( v45 )
            {
              v48 = *(_QWORD *)(v45 + 16);
              *((_QWORD *)v46 + 1) = v48;
              if ( v48 )
                *(_QWORD *)(v48 + 16) = v46 + 8;
              *((_QWORD *)v46 + 2) = v45 + 16;
              *(_QWORD *)(v45 + 16) = v46;
              goto LABEL_97;
            }
          }
        }
LABEL_73:
        if ( v158 > 0x40 && v157 )
          j_j___libc_free_0_0(v157);
        if ( v156 <= 0x40 || (v39 = v155) == 0 )
        {
LABEL_79:
          if ( v152 <= 0x40 )
            return v12;
          v40 = v151;
          if ( !v151 )
            return v12;
          goto LABEL_81;
        }
LABEL_78:
        j_j___libc_free_0_0(v39);
        goto LABEL_79;
      }
    }
    else
    {
      v151 = 0;
    }
    v151 |= v23;
    goto LABEL_29;
  }
  if ( *a1 != 85 )
    goto LABEL_3;
  v27 = *((_QWORD *)a1 - 4);
  v28 = v27;
  if ( !v27 )
    goto LABEL_3;
  if ( !*(_BYTE *)v27 && *(_QWORD *)(v27 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v27 + 33) & 0x20) != 0 )
  {
    v74 = *(_DWORD *)(v27 + 36);
    if ( v74 == 313 || v74 == 362 )
    {
      if ( (a1[7] & 0x40) != 0 )
        v75 = (__int64 *)*((_QWORD *)a1 - 1);
      else
        v75 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      sub_22CEA30((__int64)&v160, a2, v75, 0);
      if ( (a1[7] & 0x40) != 0 )
        v76 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v76 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      sub_22CEA30((__int64)&v165, a2, (__int64 *)v76 + 4, 0);
      v77 = *((_QWORD *)a1 - 4);
      if ( !v77 || *(_BYTE *)v77 || *(_QWORD *)(v77 + 24) != *((_QWORD *)a1 + 10) )
        BUG();
      LOBYTE(v78) = sub_ABB410((__int64 *)&v160, 4 * (*(_DWORD *)(v77 + 36) == 313) + 34, &v165);
      v12 = v78;
      if ( (_BYTE)v78 )
      {
        v82 = *((_QWORD *)a1 + 1);
        v83 = 0;
        v84 = 1;
      }
      else
      {
        v79 = *((_QWORD *)a1 - 4);
        if ( !v79 || *(_BYTE *)v79 || *(_QWORD *)(v79 + 24) != *((_QWORD *)a1 + 10) )
          BUG();
        LOBYTE(v80) = sub_ABB410((__int64 *)&v160, 4 * (*(_DWORD *)(v79 + 36) == 313) + 36, &v165);
        v12 = v80;
        if ( (_BYTE)v80 )
        {
          v82 = *((_QWORD *)a1 + 1);
          v83 = 1;
          v84 = -1;
        }
        else
        {
          LOBYTE(v81) = sub_ABB410((__int64 *)&v160, 32, &v165);
          v12 = v81;
          if ( !(_BYTE)v81 )
          {
LABEL_148:
            if ( v168 > 0x40 && v167 )
              j_j___libc_free_0_0(v167);
            if ( (unsigned int)v166 > 0x40 && v165 )
              j_j___libc_free_0_0(v165);
            if ( v163 > 0x40 && v162 )
              j_j___libc_free_0_0(v162);
            if ( (unsigned int)v161 <= 0x40 )
              return v12;
            v40 = v160;
            if ( !v160 )
              return v12;
LABEL_81:
            j_j___libc_free_0_0(v40);
            return v12;
          }
          v82 = *((_QWORD *)a1 + 1);
          v83 = 0;
          v84 = 0;
        }
      }
      v85 = sub_AD64C0(v82, v84, v83);
      sub_BD84D0((__int64)a1, v85);
      sub_B43D60(a1);
      goto LABEL_148;
    }
  }
  else
  {
    v28 = *((_QWORD *)a1 - 4);
  }
  if ( !*(_BYTE *)v27 && *(_QWORD *)(v27 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v27 + 33) & 0x20) != 0 )
  {
    v88 = *(_DWORD *)(v27 + 36);
    if ( v88 > 0x14A )
    {
      if ( v88 - 365 <= 1 )
      {
        v89 = 2 * (v88 != 365) + 34;
        goto LABEL_179;
      }
    }
    else if ( v88 > 0x148 )
    {
      v89 = 38;
      if ( v88 != 329 )
        v89 = 40;
LABEL_179:
      v90 = sub_B531B0(v89);
      if ( (a1[7] & 0x40) != 0 )
        v91 = (__int64 *)*((_QWORD *)a1 - 1);
      else
        v91 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      sub_22CEA30((__int64)&v151, a2, v91, 0);
      if ( (a1[7] & 0x40) != 0 )
        v92 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v92 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      sub_22CEA30((__int64)&v155, a2, (__int64 *)v92 + 4, 0);
      LOBYTE(v93) = sub_ABB410((__int64 *)&v151, v90, (__int64 *)&v155);
      v12 = v93;
      if ( (_BYTE)v93 )
      {
        v94 = -(__int64)(*((_DWORD *)a1 + 1) & 0x7FFFFFF);
LABEL_185:
        sub_BD84D0((__int64)a1, *(_QWORD *)&a1[32 * v94]);
        sub_B43D60(a1);
        goto LABEL_186;
      }
      LOBYTE(v95) = sub_ABB410((__int64 *)&v155, v90, (__int64 *)&v151);
      v12 = v95;
      if ( (_BYTE)v95 )
      {
        v94 = 1LL - (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
        goto LABEL_185;
      }
      v96 = *((_QWORD *)a1 - 4);
      if ( !v96 || *(_BYTE *)v96 || *(_QWORD *)(v96 + 24) != *((_QWORD *)a1 + 10) )
        BUG();
      v97 = *(_DWORD *)(v96 + 36);
      if ( v97 == 365 )
      {
        v98 = 34;
        goto LABEL_208;
      }
      if ( v97 > 0x16D )
      {
        if ( v97 == 366 )
        {
          v98 = 36;
          goto LABEL_208;
        }
      }
      else
      {
        if ( v97 == 329 )
        {
          v98 = 38;
LABEL_208:
          if ( sub_B532B0(v98) )
          {
            v99 = sub_AB07B0((__int64)&v151, (__int64)&v155);
            if ( (_BYTE)v99 )
            {
              sub_23D0AB0((__int64)&v165, (__int64)a1, 0, 0, 0);
              v100 = *((_DWORD *)a1 + 1);
              v164 = 257;
              HIDWORD(v150) = 0;
              v101 = v100 & 0x7FFFFFF;
              v102 = *(_QWORD *)&a1[32 * (1 - v101)];
              v103 = *(_QWORD *)&a1[-32 * v101];
              v104 = *((_QWORD *)a1 - 4);
              if ( !v104 || *(_BYTE *)v104 || *(_QWORD *)(v104 + 24) != *((_QWORD *)a1 + 10) )
                BUG();
              v105 = sub_B33C40(
                       (__int64)&v165,
                       (unsigned int)(*(_DWORD *)(v104 + 36) == 330) + 365,
                       v103,
                       v102,
                       v150,
                       (__int64)&v160);
              sub_BD84D0((__int64)a1, v105);
              sub_B43D60(a1);
              nullsub_61();
              v174 = &unk_49DA100;
              nullsub_63();
              if ( (unsigned __int64 *)v165 != &v167 )
                _libc_free(v165);
              v12 = v99;
            }
          }
LABEL_186:
          if ( v158 > 0x40 && v157 )
            j_j___libc_free_0_0(v157);
          if ( v156 > 0x40 && v155 )
            j_j___libc_free_0_0(v155);
          if ( v154 <= 0x40 )
            goto LABEL_79;
          v39 = v153;
          if ( !v153 )
            goto LABEL_79;
          goto LABEL_78;
        }
        if ( v97 == 330 )
        {
          v98 = 40;
          goto LABEL_208;
        }
      }
LABEL_253:
      BUG();
    }
  }
  else
  {
    v28 = *((_QWORD *)a1 - 4);
  }
  if ( !*(_BYTE *)v27 && *(_QWORD *)(v27 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v27 + 33) & 0x20) != 0 )
  {
    v106 = *(_DWORD *)(v27 + 36);
    if ( v106 != 312 )
    {
      switch ( v106 )
      {
        case 333:
        case 339:
        case 360:
        case 369:
        case 372:
          break;
        case 334:
        case 335:
        case 336:
        case 337:
        case 338:
        case 340:
        case 341:
        case 342:
        case 343:
        case 344:
        case 345:
        case 346:
        case 347:
        case 348:
        case 349:
        case 350:
        case 351:
        case 352:
        case 353:
        case 354:
        case 355:
        case 356:
        case 357:
        case 358:
        case 359:
        case 361:
        case 362:
        case 363:
        case 364:
        case 365:
        case 366:
        case 367:
        case 368:
        case 370:
        case 371:
          goto LABEL_228;
        default:
          goto LABEL_57;
      }
    }
    v12 = sub_274B690((__int64)a1, a2);
    if ( !(_BYTE)v12 )
    {
      if ( *a1 != 85 )
        goto LABEL_3;
      v28 = *((_QWORD *)a1 - 4);
LABEL_228:
      if ( !v28 )
        goto LABEL_3;
      goto LABEL_57;
    }
    sub_23D0AB0((__int64)&v165, (__int64)a1, 0, 0, 0);
    v135 = sub_B5B5E0((__int64)a1);
    v143 = sub_B5B640((__int64)a1);
    v107 = sub_B5B640((__int64)a1);
    v160 = (unsigned __int64)sub_BD5D20((__int64)a1);
    v108 = *((_DWORD *)a1 + 1) & 0x7FFFFFF;
    v161 = v109;
    v164 = 261;
    v130 = v135;
    v136 = (unsigned __int8 *)sub_274BB90(
                                &v165,
                                v135,
                                *(unsigned __int8 **)&a1[-32 * v108],
                                *(unsigned __int8 **)&a1[32 * (1 - v108)],
                                v155,
                                0,
                                (__int64)&v160,
                                0);
    sub_274B030(v136, v130, v143, !v107);
    v110 = *((_QWORD *)a1 + 1);
    v160 = sub_ACADE0(**(__int64 ****)(v110 + 16));
    v161 = sub_AD6450(*(_QWORD *)(*(_QWORD *)(v110 + 16) + 8LL));
    v159 = 257;
    v111 = (_BYTE *)sub_AD24A0((__int64 **)v110, (__int64 *)&v160, 2);
    LODWORD(v151) = 0;
    v112 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v172 + 88LL);
    if ( (char *)v112 == (char *)sub_9482E0 )
    {
      if ( *v111 > 0x15u || *v136 > 0x15u )
      {
LABEL_239:
        v145 = v111;
        v164 = 257;
        v115 = sub_BD2C40(104, unk_3F148BC);
        v116 = v115;
        if ( v115 )
        {
          v117 = (__int64)v145;
          v146 = (__int64)v115;
          v131 = v117;
          sub_B44260((__int64)v115, *(_QWORD *)(v117 + 8), 65, 2u, 0, 0);
          *(_QWORD *)(v146 + 80) = 0x400000000LL;
          *(_QWORD *)(v146 + 72) = v146 + 88;
          sub_B4FD20(v146, v131, (__int64)v136, &v151, 1, (__int64)&v160);
          v116 = (_QWORD *)v146;
        }
        v147 = (__int64)v116;
        (*(void (__fastcall **)(__int64, _QWORD *, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v173 + 16LL))(
          v173,
          v116,
          &v155,
          v170,
          v171);
        v118 = v165;
        v114 = v147;
        v119 = (unsigned int *)(v165 + 16LL * (unsigned int)v166);
        if ( (unsigned int *)v165 != v119 )
        {
          do
          {
            v120 = *(_QWORD *)(v118 + 8);
            v121 = *(_DWORD *)v118;
            v118 += 16;
            v148 = v114;
            sub_B99FD0(v114, v121, v120);
            v114 = v148;
          }
          while ( v119 != (unsigned int *)v118 );
        }
LABEL_235:
        sub_BD84D0((__int64)a1, v114);
        sub_B43D60(a1);
        if ( (unsigned __int8)(*v136 - 42) <= 0x11u )
          sub_274B8C0(v136, a2);
        nullsub_61();
        v174 = &unk_49DA100;
        nullsub_63();
        v67 = (unsigned int *)v165;
        if ( (unsigned __int64 *)v165 != &v167 )
          goto LABEL_124;
        return v12;
      }
      v144 = v111;
      v113 = sub_AAAE30((__int64)v111, (__int64)v136, &v151, 1);
      v111 = v144;
      v114 = v113;
    }
    else
    {
      v149 = v111;
      v122 = v112(v172, v111, v136, &v151, 1, v123);
      v111 = v149;
      v114 = v122;
    }
    if ( v114 )
      goto LABEL_235;
    goto LABEL_239;
  }
LABEL_57:
  if ( !*(_BYTE *)v28 && *(_QWORD *)(v28 + 24) == *((_QWORD *)a1 + 10) && (*(_BYTE *)(v28 + 33) & 0x20) != 0 )
  {
    v29 = (unsigned int)(*(_DWORD *)(v28 + 36) - 311);
    if ( (unsigned int)v29 <= 0x3C )
    {
      v30 = 0x1001000008000001LL;
      if ( _bittest64(&v30, v29) )
      {
        v12 = sub_274B690((__int64)a1, a2);
        if ( (_BYTE)v12 )
        {
          v31 = sub_B5B5E0((__int64)a1);
          v133 = sub_B5B640((__int64)a1);
          v128 = !sub_B5B640((__int64)a1);
          v165 = (__int64)sub_BD5D20((__int64)a1);
          v32 = *((_DWORD *)a1 + 1);
          v166 = v33;
          v169 = 261;
          v34 = sub_B504D0(
                  v31,
                  *(_QWORD *)&a1[-32 * (v32 & 0x7FFFFFF)],
                  *(_QWORD *)&a1[32 * (1LL - (v32 & 0x7FFFFFF))],
                  (__int64)&v165,
                  (__int64)(a1 + 24),
                  0);
          v35 = (unsigned int *)*((_QWORD *)a1 + 6);
          v36 = v34;
          v165 = (__int64)v35;
          if ( v35 )
          {
            v137 = v34;
            sub_B96E90((__int64)&v165, (__int64)v35, 1);
            v36 = v137;
            v37 = v137 + 48;
            if ( (__int64 *)(v137 + 48) == &v165 )
            {
              if ( v165 )
              {
                sub_B91220(v37, v165);
                v36 = v137;
              }
LABEL_67:
              v138 = v36;
              sub_274B030((unsigned __int8 *)v36, v31, v133, v128);
              sub_BD84D0((__int64)a1, v138);
              sub_B43D60(a1);
              v38 = (unsigned __int8 *)v138;
              if ( !v138 )
                return v12;
LABEL_68:
              sub_274B8C0(v38, a2);
              return v12;
            }
          }
          else
          {
            v37 = v34 + 48;
            if ( (__int64 *)(v34 + 48) == &v165 )
              goto LABEL_67;
          }
          v86 = *(_QWORD *)(v36 + 48);
          if ( v86 )
          {
            v126 = v36;
            sub_B91220(v37, v86);
            v36 = v126;
          }
          v87 = (unsigned __int8 *)v165;
          *(_QWORD *)(v36 + 48) = v165;
          if ( v87 )
          {
            v141 = v36;
            sub_B976B0((__int64)&v165, v87, v37);
            v36 = v141;
          }
          v142 = v36;
          sub_274B030((unsigned __int8 *)v36, v31, v133, v128);
          sub_BD84D0((__int64)a1, v142);
          sub_B43D60(a1);
          v38 = (unsigned __int8 *)v142;
          goto LABEL_68;
        }
      }
    }
  }
LABEL_3:
  if ( (a1[7] & 0x80u) == 0 )
    goto LABEL_104;
  v4 = sub_BD2BC0((__int64)a1);
  v6 = v4 + v5;
  if ( (a1[7] & 0x80u) != 0 )
    v6 -= sub_BD2BC0((__int64)a1);
  v7 = v6 >> 4;
  if ( !(_DWORD)v7 )
    goto LABEL_104;
  v8 = 0;
  v9 = 16LL * (unsigned int)v7;
  while ( 1 )
  {
    v10 = 0;
    if ( (a1[7] & 0x80u) != 0 )
      v10 = sub_BD2BC0((__int64)a1);
    v11 = (unsigned int *)(v8 + v10);
    if ( !*(_DWORD *)(*(_QWORD *)v11 + 8LL) )
      break;
    v8 += 16;
    if ( v8 == v9 )
      goto LABEL_104;
  }
  v12 = 0;
  v13 = &a1[32LL * v11[2] - 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v14 = &v13[32LL * v11[3] - 32LL * v11[2]];
  if ( v14 != v13 )
  {
    do
    {
      v15 = *(_BYTE **)v13;
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v13 + 8LL) + 8LL) - 17 > 1 && *v15 > 0x15u )
      {
        v16 = sub_22CE0E0(a2, (__int64)v15, (__int64)a1);
        if ( v16 )
        {
          if ( *(_QWORD *)v13 )
          {
            v17 = *((_QWORD *)v13 + 1);
            **((_QWORD **)v13 + 2) = v17;
            if ( v17 )
              *(_QWORD *)(v17 + 16) = *((_QWORD *)v13 + 2);
          }
          *(_QWORD *)v13 = v16;
          v18 = *(_QWORD *)(v16 + 16);
          *((_QWORD *)v13 + 1) = v18;
          if ( v18 )
            *(_QWORD *)(v18 + 16) = v13 + 8;
          *((_QWORD *)v13 + 2) = v16 + 16;
          v12 = 1;
          *(_QWORD *)(v16 + 16) = v13;
        }
      }
      v13 += 32;
    }
    while ( v14 != v13 );
  }
  else
  {
LABEL_104:
    v12 = 0;
  }
  v53 = *a1;
  v165 = (__int64)&v167;
  v166 = 0x400000000LL;
  if ( v53 == 40 )
  {
    v54 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v54 = -32;
    if ( v53 != 85 )
    {
      v54 = -96;
      if ( v53 != 34 )
        goto LABEL_253;
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v55 = sub_BD2BC0((__int64)a1);
    v57 = v55 + v56;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v57 >> 4) )
        goto LABEL_116;
    }
    else
    {
      if ( !(unsigned int)((v57 - sub_BD2BC0((__int64)a1)) >> 4) )
        goto LABEL_116;
      if ( (a1[7] & 0x80u) != 0 )
      {
        v58 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v59 = sub_BD2BC0((__int64)a1);
        v54 -= 32LL * (unsigned int)(*(_DWORD *)(v59 + v60 - 4) - v58);
        goto LABEL_116;
      }
    }
    BUG();
  }
LABEL_116:
  v61 = &a1[v54];
  v62 = 0;
  v129 = v61;
  v63 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  if ( v63 != v61 )
  {
    do
    {
      v134 = *(_BYTE **)v63;
      v139 = *(_QWORD *)(*(_QWORD *)v63 + 8LL);
      if ( *(_BYTE *)(v139 + 8) == 14 && !(unsigned __int8)sub_B49B80((__int64)a1, v62, 43) && *v134 > 0x15u )
      {
        v68 = sub_AC9EC0((__int64 **)v139);
        v69 = sub_22CF7C0(a2, 0x20u, (__int64)v134, v68, (__int64)a1, 0);
        if ( v69 )
        {
          if ( *(_BYTE *)v69 == 17 )
          {
            if ( *(_DWORD *)(v69 + 32) <= 0x40u )
            {
              if ( !*(_QWORD *)(v69 + 24) )
              {
LABEL_129:
                v72 = (unsigned int)v166;
                v73 = (unsigned int)v166 + 1LL;
                if ( v73 > HIDWORD(v166) )
                {
                  sub_C8D5F0((__int64)&v165, &v167, v73, 4u, v70, v71);
                  v72 = (unsigned int)v166;
                }
                *(_DWORD *)(v165 + 4 * v72) = v62;
                LODWORD(v166) = v166 + 1;
              }
            }
            else
            {
              v140 = *(_DWORD *)(v69 + 32);
              if ( v140 == (unsigned int)sub_C444A0(v69 + 24) )
                goto LABEL_129;
            }
          }
        }
      }
      ++v62;
      v63 += 32;
    }
    while ( v129 != v63 );
  }
  if ( (_DWORD)v166 )
  {
    v160 = *((_QWORD *)a1 + 9);
    v64 = (__int64 *)sub_BD5C60((__int64)a1);
    v65 = sub_A778C0(v64, 43, 0);
    v66 = v64;
    v12 = 1;
    v160 = sub_A7B660((__int64 *)&v160, v66, (_DWORD *)v165, (unsigned int)v166, v65);
    *((_QWORD *)a1 + 9) = v160;
  }
  v67 = (unsigned int *)v165;
  if ( (unsigned __int64 *)v165 != &v167 )
LABEL_124:
    _libc_free((unsigned __int64)v67);
  return v12;
}
