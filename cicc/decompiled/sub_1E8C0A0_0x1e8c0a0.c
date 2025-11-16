// Function: sub_1E8C0A0
// Address: 0x1e8c0a0
//
void __fastcall sub_1E8C0A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 *v4; // rdi
  __int64 *v5; // rbx
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 *v10; // rax
  void *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rbx
  __int64 *v14; // rax
  __int64 *v15; // rdi
  __int64 *v16; // rcx
  void *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // rdi
  __int64 (*v24)(); // rax
  __int64 v25; // rax
  unsigned int v26; // esi
  unsigned __int64 v27; // rdi
  __int64 v29; // rdx
  __int64 v31; // rdx
  unsigned __int16 v32; // r15
  __int64 v33; // rdx
  _WORD *v34; // rbx
  __int64 v35; // r12
  int v36; // eax
  int *v37; // r14
  unsigned int v38; // edi
  unsigned int v39; // edx
  int *v40; // rsi
  int v41; // ecx
  __int16 v42; // ax
  unsigned int v43; // edx
  unsigned int v44; // r11d
  unsigned int v45; // esi
  int v46; // ecx
  __int64 v47; // rdx
  unsigned __int64 v48; // r9
  unsigned __int64 v49; // r9
  int v52; // r9d
  int *v53; // r8
  int v54; // esi
  int v55; // ecx
  unsigned __int64 v56; // rdx
  unsigned __int64 v57; // rax
  _DWORD *v58; // rax
  __int64 v59; // rdx
  int *v60; // rcx
  _DWORD *m; // rdx
  int *v62; // r12
  __int64 v63; // r14
  int *v64; // r13
  int *v65; // rsi
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rax
  __int64 v68; // rax
  _DWORD *v69; // rax
  __int64 v70; // rdx
  int *v71; // rcx
  _DWORD *j; // rdx
  int *v73; // r12
  int *v74; // r13
  __int64 v75; // rdx
  _DWORD *n; // rdx
  __int64 v77; // rdx
  _DWORD *k; // rdx
  unsigned __int16 *v79; // r14
  unsigned __int16 v80; // r15
  __int64 v81; // rdx
  __int64 v82; // rdx
  __int16 *v83; // rbx
  char v84; // al
  _DWORD *v85; // rdx
  __int16 v86; // ax
  unsigned int v87; // esi
  int v88; // eax
  int v89; // eax
  _QWORD *v90; // rcx
  __int64 v91; // rdx
  _WORD *v92; // r12
  _WORD *i; // rbx
  unsigned int v94; // eax
  __int64 v95; // rdx
  __int64 *v96; // rcx
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // rcx
  _QWORD *v100; // rdx
  __int64 v101; // rax
  bool v102; // al
  __int64 v103; // rsi
  __int64 v104; // rax
  __int64 v105; // r15
  __int64 (*v106)(); // r12
  __int64 v107; // rdi
  __int64 *v108; // rax
  __int64 *v109; // rcx
  __int64 v110; // rdx
  __int64 *v111; // r8
  unsigned __int16 v112; // [rsp+16h] [rbp-19Ah]
  int *v113; // [rsp+18h] [rbp-198h]
  __int64 v114; // [rsp+20h] [rbp-190h]
  __int64 v115; // [rsp+20h] [rbp-190h]
  _WORD *v116; // [rsp+20h] [rbp-190h]
  unsigned __int16 *v117; // [rsp+20h] [rbp-190h]
  __int64 v119; // [rsp+30h] [rbp-180h]
  __int64 *v120; // [rsp+38h] [rbp-178h]
  __int64 *v121; // [rsp+38h] [rbp-178h]
  unsigned int v122; // [rsp+38h] [rbp-178h]
  int v123; // [rsp+44h] [rbp-16Ch] BYREF
  __int64 v124; // [rsp+48h] [rbp-168h] BYREF
  __int64 v125; // [rsp+50h] [rbp-160h] BYREF
  int *v126; // [rsp+58h] [rbp-158h] BYREF
  unsigned __int64 v127[2]; // [rsp+60h] [rbp-150h] BYREF
  int v128; // [rsp+70h] [rbp-140h]
  __int64 v129; // [rsp+80h] [rbp-130h] BYREF
  __int64 *v130; // [rsp+88h] [rbp-128h]
  __int64 *v131; // [rsp+90h] [rbp-120h]
  __int64 v132; // [rsp+98h] [rbp-118h]
  int v133; // [rsp+A0h] [rbp-110h]
  _BYTE v134[40]; // [rsp+A8h] [rbp-108h] BYREF
  _QWORD *v135; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v136; // [rsp+D8h] [rbp-D8h]
  _QWORD v137[26]; // [rsp+E0h] [rbp-D0h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 64) = 0;
  if ( (**(_BYTE **)(v3 + 352) & 2) == 0 && (**(_BYTE **)(**(_QWORD **)(a1 + 48) + 352LL) & 4) != 0 )
  {
    v92 = *(_WORD **)(a2 + 160);
    for ( i = (_WORD *)sub_1DD77D0(a2); v92 != i; i += 4 )
    {
      v94 = (unsigned __int16)*i;
      v95 = *(_QWORD *)(a1 + 40);
      if ( v94 < *(_DWORD *)(v95 + 16)
        && *(_BYTE *)(*(_QWORD *)(v95 + 232) + 8LL * (unsigned __int16)v94 + 4)
        && (*(_QWORD *)(*(_QWORD *)(a1 + 176) + 8LL * (v94 >> 6)) & (1LL << *i)) == 0
        && !*(_BYTE *)(a2 + 180)
        && a2 != *(_QWORD *)(*(_QWORD *)(a2 + 56) + 328LL) )
      {
        sub_1E869F0(a1, "MBB has allocatable live-in, but isn't entry or landing-pad.", a2);
      }
    }
  }
  v129 = 0;
  v130 = (__int64 *)v134;
  v131 = (__int64 *)v134;
  v132 = 4;
  v4 = *(__int64 **)(a2 + 96);
  v5 = *(__int64 **)(a2 + 88);
  v133 = 0;
  v120 = v4;
  if ( v5 == v4 )
  {
    v13 = *(__int64 **)(a2 + 64);
    v121 = *(__int64 **)(a2 + 72);
    if ( v121 != v13 )
    {
      v6 = v2 + 72;
      v7 = v2 + 528;
      goto LABEL_18;
    }
    goto LABEL_37;
  }
  v6 = v2 + 72;
  v7 = v2 + 528;
  do
  {
    while ( 1 )
    {
      v9 = *v5;
      if ( !*(_BYTE *)(*v5 + 180) )
        goto LABEL_5;
      v10 = v130;
      if ( v131 == v130 )
      {
        v15 = &v130[HIDWORD(v132)];
        if ( v130 != v15 )
        {
          v16 = 0;
          while ( v9 != *v10 )
          {
            if ( *v10 == -2 )
              v16 = v10;
            if ( v15 == ++v10 )
            {
              if ( !v16 )
                goto LABEL_165;
              *v16 = v9;
              --v133;
              ++v129;
              v9 = *v5;
              break;
            }
          }
LABEL_5:
          if ( !sub_1DA1810(v6, v9) )
            goto LABEL_11;
          goto LABEL_6;
        }
LABEL_165:
        if ( HIDWORD(v132) < (unsigned int)v132 )
        {
          ++HIDWORD(v132);
          *v15 = v9;
          ++v129;
          v9 = *v5;
          goto LABEL_5;
        }
      }
      sub_16CCBA0((__int64)&v129, v9);
      if ( !sub_1DA1810(v6, *v5) )
LABEL_11:
        sub_1E869F0(v2, "MBB has successor that isn't part of the function.", a2);
LABEL_6:
      v8 = sub_1E855C0(v7, v5);
      if ( !sub_1DA1810((__int64)(v8 + 22), a2) )
      {
        sub_1E869F0(v2, "Inconsistent CFG", a2);
        v11 = sub_16E8CB0();
        v114 = sub_1263B40((__int64)v11, "MBB is not in the predecessor list of the successor ");
        sub_1DD5B60(&v135, *v5);
        sub_1E869D0((__int64)&v135, v114, v12);
        sub_1263B40(v114, ".\n");
        if ( v137[0] )
          break;
      }
      if ( v120 == ++v5 )
        goto LABEL_14;
    }
    ++v5;
    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v137[0])(&v135, &v135, 3);
  }
  while ( v120 != v5 );
LABEL_14:
  v13 = *(__int64 **)(a2 + 64);
  v121 = *(__int64 **)(a2 + 72);
  while ( v121 != v13 )
  {
    while ( 1 )
    {
LABEL_18:
      if ( !sub_1DA1810(v6, *v13) )
        sub_1E869F0(v2, "MBB has predecessor that isn't part of the function.", a2);
      v14 = sub_1E855C0(v7, v13);
      if ( !sub_1DA1810((__int64)(v14 + 35), a2) )
      {
        sub_1E869F0(v2, "Inconsistent CFG", a2);
        v17 = sub_16E8CB0();
        v115 = sub_1263B40((__int64)v17, "MBB is not in the successor list of the predecessor ");
        sub_1DD5B60(&v135, *v13);
        sub_1E869D0((__int64)&v135, v115, v18);
        sub_1263B40(v115, ".\n");
        if ( v137[0] )
          break;
      }
      if ( v121 == ++v13 )
        goto LABEL_30;
    }
    ++v13;
    ((void (__fastcall *)(_QWORD **, _QWORD **, __int64))v137[0])(&v135, &v135, 3);
  }
LABEL_30:
  if ( (unsigned int)(HIDWORD(v132) - v133) > 1 )
  {
    v19 = *(_QWORD *)(*(_QWORD *)(v2 + 24) + 608LL);
    if ( !v19
      || (v20 = *(_QWORD *)(a2 + 40), *(_DWORD *)(v19 + 348) != 2)
      || !v20
      || *(_BYTE *)(sub_157EBA0(v20) + 16) != 27 )
    {
      v21 = sub_15E38F0(**(_QWORD **)(v2 + 16));
      v22 = sub_14DD7D0(v21);
      if ( v22 > 10 )
      {
        if ( v22 == 12 )
          goto LABEL_37;
      }
      else if ( v22 > 6 )
      {
        goto LABEL_37;
      }
      sub_1E869F0(v2, "MBB has more than one landing pad successor", a2);
    }
  }
LABEL_37:
  v23 = *(_QWORD *)(v2 + 32);
  v124 = 0;
  v135 = v137;
  v136 = 0x400000000LL;
  v125 = 0;
  v24 = *(__int64 (**)())(*(_QWORD *)v23 + 264LL);
  if ( v24 == sub_1D820E0
    || ((unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *, __int64 *, _QWORD **, _QWORD))v24)(
         v23,
         a2,
         &v124,
         &v125,
         &v135,
         0) )
  {
    goto LABEL_38;
  }
  if ( !v124 )
  {
    if ( v125 )
    {
      sub_1E869F0(v2, "AnalyzeBranch returned invalid data!", a2);
    }
    else
    {
      v103 = *(_QWORD *)(a2 + 8);
      if ( v103 != *(_QWORD *)(v2 + 16) + 320LL )
      {
        v104 = (__int64)(*(_QWORD *)(a2 + 96) - *(_QWORD *)(a2 + 88)) >> 3;
        if ( HIDWORD(v132) - v133 != (_DWORD)v104 )
        {
          if ( HIDWORD(v132) - v133 + 1 == (_DWORD)v104 )
          {
            if ( !sub_1DD6970(a2, v103) )
              sub_1E869F0(
                v2,
                "MBB exits via unconditional fall-through but its successor differs from its CFG successor!",
                a2);
          }
          else
          {
            sub_1E869F0(v2, "MBB exits via unconditional fall-through but doesn't have exactly one CFG successor!", a2);
          }
        }
      }
      if ( a2 + 24 != (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v127[0] = a2 + 24;
        sub_1E89700(v127);
        if ( (unsigned __int8)sub_1DF81C0(v127[0], 5, 1) )
        {
          v105 = *(_QWORD *)(v2 + 32);
          v106 = *(__int64 (**)())(*(_QWORD *)v105 + 656LL);
          v127[0] = a2 + 24;
          sub_1E89700(v127);
          if ( v106 == sub_1D918C0 || !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64))v106)(v105, v127[0]) )
            sub_1E869F0(v2, "MBB exits via unconditional fall-through but ends with a barrier instruction!", a2);
        }
      }
      if ( (_DWORD)v136 )
        sub_1E869F0(v2, "MBB exits via unconditional fall-through but has a condition!", a2);
    }
    goto LABEL_38;
  }
  if ( !v125 )
  {
    if ( !(_DWORD)v136 )
    {
      v96 = *(__int64 **)(a2 + 88);
      v97 = HIDWORD(v132);
      v98 = (__int64)(*(_QWORD *)(a2 + 96) - (_QWORD)v96) >> 3;
      if ( HIDWORD(v132) - v133 + 1 == (_DWORD)v98 )
        goto LABEL_248;
      if ( HIDWORD(v132) - v133 != 1 || (_DWORD)v98 != 1 )
        goto LABEL_179;
      v107 = *v96;
      v108 = v131;
      if ( v131 != v130 )
        v97 = (unsigned int)v132;
      v109 = &v131[v97];
      v110 = *v131;
      if ( v131 != v109 )
      {
        while ( 1 )
        {
          v110 = *v108;
          v111 = v108;
          if ( (unsigned __int64)*v108 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v109 == ++v108 )
          {
            v110 = v111[1];
            break;
          }
        }
      }
      if ( v107 == v110 )
      {
LABEL_248:
        if ( !sub_1DD6970(a2, v124) )
          sub_1E869F0(
            v2,
            "MBB exits via unconditional branch but the CFG successor doesn't match the actual successor!",
            a2);
      }
      else
      {
LABEL_179:
        sub_1E869F0(v2, "MBB exits via unconditional branch but doesn't have exactly one CFG successor!", a2);
      }
      if ( a2 + 24 == (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
      {
        sub_1E869F0(v2, "MBB exits via unconditional branch but doesn't contain any instructions!", a2);
      }
      else
      {
        v127[0] = a2 + 24;
        sub_1E89700(v127);
        if ( (unsigned __int8)sub_1DF81C0(v127[0], 5, 1) )
        {
          v127[0] = a2 + 24;
          sub_1E89700(v127);
          if ( !(unsigned __int8)sub_1DF81C0(v127[0], 6, 1) )
            sub_1E869F0(v2, "MBB exits via unconditional branch but the branch isn't a terminator instruction!", a2);
        }
        else
        {
          sub_1E869F0(v2, "MBB exits via unconditional branch but doesn't end with a barrier instruction!", a2);
        }
      }
      goto LABEL_38;
    }
    v99 = *(_QWORD *)(a2 + 8);
    if ( v99 == *(_QWORD *)(v2 + 16) + 320LL )
    {
      sub_1E869F0(v2, "MBB conditionally falls through out of function!", a2);
    }
    else
    {
      v100 = *(_QWORD **)(a2 + 88);
      v101 = (__int64)(*(_QWORD *)(a2 + 96) - (_QWORD)v100) >> 3;
      if ( (_DWORD)v101 == 1 )
      {
        if ( v124 == v99 )
        {
          if ( v124 != *v100 )
            sub_1E869F0(
              v2,
              "MBB exits via conditional branch/fall-through but the CFG successor don't match the actual successor!",
              a2);
        }
        else
        {
          sub_1E869F0(v2, "MBB exits via conditional branch/fall-through but only has one CFG successor!", a2);
        }
        goto LABEL_193;
      }
      if ( (_DWORD)v101 != 2 )
      {
        sub_1E869F0(
          v2,
          "MBB exits via conditional branch/fall-through but doesn't have exactly two CFG successors!",
          a2);
        goto LABEL_193;
      }
      if ( v124 == *v100 )
      {
        if ( v99 != v100[1] )
LABEL_229:
          sub_1E869F0(
            v2,
            "MBB exits via conditional branch/fall-through but the CFG successors don't match the actual successors!",
            a2);
      }
      else if ( v99 != *v100 || v124 != v100[1] )
      {
        goto LABEL_229;
      }
    }
LABEL_193:
    if ( a2 + 24 == (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
    {
      sub_1E869F0(v2, "MBB exits via conditional branch/fall-through but doesn't contain any instructions!", a2);
    }
    else
    {
      v127[0] = a2 + 24;
      sub_1E89700(v127);
      if ( (unsigned __int8)sub_1DF81C0(v127[0], 5, 1) )
      {
        sub_1E869F0(v2, "MBB exits via conditional branch/fall-through but ends with a barrier instruction!", a2);
      }
      else
      {
        v127[0] = a2 + 24;
        sub_1E89700(v127);
        if ( !(unsigned __int8)sub_1DF81C0(v127[0], 6, 1) )
          sub_1E869F0(
            v2,
            "MBB exits via conditional branch/fall-through but the branch isn't a terminator instruction!",
            a2);
      }
    }
    goto LABEL_38;
  }
  v90 = *(_QWORD **)(a2 + 88);
  v91 = (__int64)(*(_QWORD *)(a2 + 96) - (_QWORD)v90) >> 3;
  if ( (_DWORD)v91 == 1 )
  {
    if ( v124 == v125 )
    {
      if ( v124 != *v90 )
        sub_1E869F0(
          v2,
          "MBB exits via conditional branch/branch through but the CFG successor don't match the actual successor!",
          a2);
    }
    else
    {
      sub_1E869F0(v2, "MBB exits via conditional branch/branch through but only has one CFG successor!", a2);
    }
  }
  else
  {
    if ( (_DWORD)v91 != 2 )
    {
      sub_1E869F0(v2, "MBB exits via conditional branch/branch but doesn't have exactly two CFG successors!", a2);
      goto LABEL_151;
    }
    if ( v124 == *v90 )
    {
      v102 = v90[1] == v125;
    }
    else
    {
      if ( *v90 != v125 )
        goto LABEL_185;
      v102 = v90[1] == v124;
    }
    if ( !v102 )
LABEL_185:
      sub_1E869F0(
        v2,
        "MBB exits via conditional branch/branch but the CFG successors don't match the actual successors!",
        a2);
  }
LABEL_151:
  if ( a2 + 24 == (*(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    sub_1E869F0(v2, "MBB exits via conditional branch/branch but doesn't contain any instructions!", a2);
  }
  else
  {
    v127[0] = a2 + 24;
    sub_1E89700(v127);
    if ( (unsigned __int8)sub_1DF81C0(v127[0], 5, 1) )
    {
      v127[0] = a2 + 24;
      sub_1E89700(v127);
      if ( !(unsigned __int8)sub_1DF81C0(v127[0], 6, 1) )
        sub_1E869F0(v2, "MBB exits via conditional branch/branch but the branch isn't a terminator instruction!", a2);
    }
    else
    {
      sub_1E869F0(v2, "MBB exits via conditional branch/branch but doesn't end with a barrier instruction!", a2);
    }
  }
  if ( !(_DWORD)v136 )
    sub_1E869F0(v2, "MBB exits via conditinal branch/branch but there's no condition!", a2);
LABEL_38:
  v119 = v2 + 200;
  sub_1E89740(v2 + 200);
  if ( (**(_BYTE **)(**(_QWORD **)(v2 + 48) + 352LL) & 4) != 0 )
  {
    v79 = *(unsigned __int16 **)(a2 + 160);
    v117 = (unsigned __int16 *)sub_1DD77D0(a2);
    if ( v79 != v117 )
    {
      while ( 2 )
      {
        v80 = *v117;
        if ( *v117 )
        {
          v81 = *(_QWORD *)(v2 + 40);
          if ( !v81 )
            BUG();
          v82 = *(_QWORD *)(v81 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v81 + 8) + 24LL * v80 + 4);
LABEL_129:
          v83 = (__int16 *)v82;
          if ( v82 )
          {
            while ( 1 )
            {
              LODWORD(v126) = v80;
              v84 = sub_1DF91F0(v119, (int *)&v126, v127);
              v85 = (_DWORD *)v127[0];
              if ( !v84 )
                break;
LABEL_131:
              v86 = *v83;
              v82 = 0;
              ++v83;
              if ( !v86 )
                goto LABEL_129;
              v80 += v86;
              if ( !v83 )
                goto LABEL_133;
            }
            v87 = *(_DWORD *)(v2 + 224);
            v88 = *(_DWORD *)(v2 + 216);
            ++*(_QWORD *)(v2 + 200);
            v89 = v88 + 1;
            if ( 4 * v89 >= 3 * v87 )
            {
              v87 *= 2;
            }
            else if ( v87 - *(_DWORD *)(v2 + 220) - v89 > v87 >> 3 )
            {
LABEL_138:
              *(_DWORD *)(v2 + 216) = v89;
              if ( *v85 != -1 )
                --*(_DWORD *)(v2 + 220);
              *v85 = (_DWORD)v126;
              goto LABEL_131;
            }
            sub_136B240(v119, v87);
            sub_1DF91F0(v119, (int *)&v126, v127);
            v85 = (_DWORD *)v127[0];
            v89 = *(_DWORD *)(v2 + 216) + 1;
            goto LABEL_138;
          }
        }
        else
        {
          sub_1E869F0(v2, "MBB live-in list contains non-physical register", a2);
        }
LABEL_133:
        v117 += 4;
        if ( v79 == v117 )
          break;
        continue;
      }
    }
  }
  sub_1E08750((__int64)v127, *(_QWORD *)(*(_QWORD *)(v2 + 16) + 56LL), *(_QWORD *)(v2 + 16));
  if ( v128 )
  {
    v25 = 0;
    v26 = (unsigned int)(v128 - 1) >> 6;
    v27 = v127[0];
    while ( 1 )
    {
      _RDX = *(_QWORD *)(v127[0] + 8 * v25);
      if ( v26 == (_DWORD)v25 )
        _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v128) & *(_QWORD *)(v127[0] + 8 * v25);
      if ( _RDX )
        break;
      if ( v26 + 1 == ++v25 )
        goto LABEL_45;
    }
    __asm { tzcnt   rdx, rdx }
    v122 = ((_DWORD)v25 << 6) + _RDX;
    if ( v122 != -1 )
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v2 + 40);
        if ( !v31 )
          BUG();
        v32 = v122;
        v33 = *(_QWORD *)(v31 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v31 + 8) + 24LL * v122 + 4);
LABEL_57:
        v34 = (_WORD *)v33;
        if ( v33 )
          break;
LABEL_62:
        v27 = v127[0];
        v43 = v122 + 1;
        if ( v128 != v122 + 1 )
        {
          v44 = v43 >> 6;
          v45 = (unsigned int)(v128 - 1) >> 6;
          if ( v43 >> 6 <= v45 )
          {
            v46 = 64 - (v43 & 0x3F);
            v47 = v44;
            v48 = 0xFFFFFFFFFFFFFFFFLL >> v46;
            if ( v46 == 64 )
              v48 = (unsigned __int64)v34;
            v49 = ~v48;
            while ( 1 )
            {
              _RAX = *(_QWORD *)(v127[0] + 8 * v47);
              if ( v44 == (_DWORD)v47 )
                _RAX = v49 & *(_QWORD *)(v127[0] + 8 * v47);
              if ( (_DWORD)v47 == v45 )
                _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v128;
              if ( _RAX )
                break;
              if ( v45 < (unsigned int)++v47 )
                goto LABEL_45;
            }
            __asm { tzcnt   rax, rax }
            v122 = ((_DWORD)v47 << 6) + _RAX;
            if ( v122 != -1 )
              continue;
          }
        }
        goto LABEL_45;
      }
      while ( 2 )
      {
        v35 = *(unsigned int *)(v2 + 224);
        v36 = v32;
        v37 = *(int **)(v2 + 208);
        v123 = v32;
        if ( (_DWORD)v35 )
        {
          v38 = v35 - 1;
          v39 = (v35 - 1) & (37 * v32);
          v40 = &v37[v39];
          v41 = *v40;
          if ( v32 == *v40 )
            goto LABEL_60;
          v52 = 1;
          v53 = 0;
          while ( v41 != -1 )
          {
            if ( v53 || v41 != -2 )
              v40 = v53;
            v39 = v38 & (v52 + v39);
            v41 = v37[v39];
            if ( v32 == v41 )
              goto LABEL_60;
            ++v52;
            v53 = v40;
            v40 = &v37[v39];
          }
          if ( !v53 )
            v53 = v40;
          v54 = *(_DWORD *)(v2 + 216);
          ++*(_QWORD *)(v2 + 200);
          v55 = v54 + 1;
          if ( 4 * (v54 + 1) < (unsigned int)(3 * v35) )
          {
            if ( (int)v35 - *(_DWORD *)(v2 + 220) - v55 > (unsigned int)v35 >> 3 )
            {
LABEL_81:
              *(_DWORD *)(v2 + 216) = v55;
              if ( *v53 != -1 )
                --*(_DWORD *)(v2 + 220);
              *v53 = v36;
LABEL_60:
              v42 = *v34;
              v33 = 0;
              ++v34;
              if ( !v42 )
                goto LABEL_57;
              v32 += v42;
              if ( !v34 )
                goto LABEL_62;
              continue;
            }
            v66 = ((((unsigned __int64)v38 >> 1) | v38) >> 2) | ((unsigned __int64)v38 >> 1) | v38;
            v67 = (((v66 >> 4) | v66) >> 8) | (v66 >> 4) | v66;
            v68 = ((v67 >> 16) | v67) + 1;
            if ( (unsigned int)v68 < 0x40 )
              LODWORD(v68) = 64;
            *(_DWORD *)(v2 + 224) = v68;
            v69 = (_DWORD *)sub_22077B0(4LL * (unsigned int)v68);
            *(_QWORD *)(v2 + 208) = v69;
            if ( v37 )
            {
              v70 = *(unsigned int *)(v2 + 224);
              v71 = &v37[v35];
              *(_QWORD *)(v2 + 216) = 0;
              for ( j = &v69[v70]; j != v69; ++v69 )
              {
                if ( v69 )
                  *v69 = -1;
              }
              v116 = v34;
              v73 = v37;
              v113 = v37;
              v63 = v2;
              v74 = v71;
              v112 = v32;
              do
              {
                if ( (unsigned int)*v73 <= 0xFFFFFFFD )
                {
                  sub_1DF91F0(v119, v73, &v126);
                  *v126 = *v73;
                  ++*(_DWORD *)(v63 + 216);
                }
                ++v73;
              }
              while ( v74 != v73 );
              goto LABEL_97;
            }
            v77 = *(unsigned int *)(v2 + 224);
            *(_QWORD *)(v2 + 216) = 0;
            for ( k = &v69[v77]; k != v69; ++v69 )
            {
              if ( v69 )
                *v69 = -1;
            }
            goto LABEL_99;
          }
        }
        else
        {
          ++*(_QWORD *)(v2 + 200);
        }
        break;
      }
      v56 = ((((((((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v35 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v35 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 8)
           | (((((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v35 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 4)
           | (((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v35 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 16;
      v57 = (v56
           | (((((((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v35 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v35 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 8)
           | (((((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v35 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 4)
           | (((unsigned int)(2 * v35 - 1) | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1)) >> 2)
           | (unsigned int)(2 * v35 - 1)
           | ((unsigned __int64)(unsigned int)(2 * v35 - 1) >> 1))
          + 1;
      if ( (unsigned int)v57 < 0x40 )
        LODWORD(v57) = 64;
      *(_DWORD *)(v2 + 224) = v57;
      v58 = (_DWORD *)sub_22077B0(4LL * (unsigned int)v57);
      *(_QWORD *)(v2 + 208) = v58;
      if ( v37 )
      {
        v59 = *(unsigned int *)(v2 + 224);
        v60 = &v37[v35];
        *(_QWORD *)(v2 + 216) = 0;
        for ( m = &v58[v59]; m != v58; ++v58 )
        {
          if ( v58 )
            *v58 = -1;
        }
        v62 = v37;
        if ( v37 != v60 )
        {
          v116 = v34;
          v113 = v37;
          v63 = v2;
          v64 = v60;
          v112 = v32;
          do
          {
            while ( (unsigned int)*v62 > 0xFFFFFFFD )
            {
              if ( v64 == ++v62 )
                goto LABEL_97;
            }
            v65 = v62++;
            sub_1DF91F0(v119, v65, &v126);
            *v126 = *(v62 - 1);
            ++*(_DWORD *)(v63 + 216);
          }
          while ( v64 != v62 );
LABEL_97:
          v2 = v63;
          v34 = v116;
          v37 = v113;
          v32 = v112;
        }
        j___libc_free_0(v37);
      }
      else
      {
        v75 = *(unsigned int *)(v2 + 224);
        *(_QWORD *)(v2 + 216) = 0;
        for ( n = &v58[v75]; n != v58; ++v58 )
        {
          if ( v58 )
            *v58 = -1;
        }
      }
LABEL_99:
      sub_1DF91F0(v119, &v123, &v126);
      v53 = v126;
      v36 = v123;
      v55 = *(_DWORD *)(v2 + 216) + 1;
      goto LABEL_81;
    }
  }
  else
  {
    v27 = v127[0];
  }
LABEL_45:
  v29 = *(_QWORD *)(v2 + 584);
  *(_DWORD *)(v2 + 400) = 0;
  *(_DWORD *)(v2 + 240) = 0;
  if ( v29 )
    *(_QWORD *)(v2 + 520) = *(_QWORD *)(*(_QWORD *)(v29 + 392) + 16LL * *(unsigned int *)(a2 + 48));
  _libc_free(v27);
  if ( v135 != v137 )
    _libc_free((unsigned __int64)v135);
  if ( v131 != v130 )
    _libc_free((unsigned __int64)v131);
}
