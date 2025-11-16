// Function: sub_1701250
// Address: 0x1701250
//
__int64 __fastcall sub_1701250(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v12; // rax
  unsigned int v13; // r14d
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r9
  _QWORD *v20; // r12
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rcx
  unsigned __int8 v26; // al
  unsigned int v27; // ebx
  char v28; // bl
  unsigned int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int8 *v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 v35; // r9
  __int64 v36; // r10
  unsigned __int8 v37; // al
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rbx
  double v42; // xmm4_8
  double v43; // xmm5_8
  __int64 **v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r10
  __int64 v48; // r9
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  int v52; // edx
  __int64 i; // rbx
  __int64 v54; // rdi
  char v56; // cl
  __int64 v57; // rcx
  char v58; // cl
  __int64 v59; // rcx
  __int64 v60; // rdx
  char v61; // dl
  __int64 v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rdx
  char v65; // dl
  unsigned int v66; // ebx
  bool v67; // al
  int v68; // eax
  bool v69; // al
  __int64 v70; // rax
  __int64 *v71; // r14
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rsi
  unsigned __int8 *v75; // rsi
  __int64 v76; // rcx
  __int64 v77; // rax
  unsigned int v78; // ebx
  __int64 v79; // rax
  unsigned int v80; // ebx
  unsigned int v81; // ebx
  __int64 v82; // rax
  char v83; // dl
  bool v84; // al
  int v85; // r10d
  unsigned int v86; // ebx
  __int64 v87; // rax
  char v88; // dl
  bool v89; // al
  unsigned __int8 v91; // [rsp+17h] [rbp-1D9h]
  __int64 v93; // [rsp+28h] [rbp-1C8h]
  __int64 v94; // [rsp+30h] [rbp-1C0h]
  unsigned int v95; // [rsp+30h] [rbp-1C0h]
  __int64 *v96; // [rsp+38h] [rbp-1B8h]
  __int64 v97; // [rsp+38h] [rbp-1B8h]
  __int64 v98; // [rsp+38h] [rbp-1B8h]
  __int64 v99; // [rsp+38h] [rbp-1B8h]
  int v100; // [rsp+38h] [rbp-1B8h]
  int v101; // [rsp+38h] [rbp-1B8h]
  __int64 v102; // [rsp+40h] [rbp-1B0h]
  _BYTE *v103; // [rsp+40h] [rbp-1B0h]
  __int64 v104; // [rsp+40h] [rbp-1B0h]
  __int64 v105; // [rsp+40h] [rbp-1B0h]
  __int64 v106; // [rsp+40h] [rbp-1B0h]
  __int64 v107; // [rsp+40h] [rbp-1B0h]
  __int64 v108; // [rsp+40h] [rbp-1B0h]
  int v109; // [rsp+40h] [rbp-1B0h]
  int v110; // [rsp+40h] [rbp-1B0h]
  __int64 v111; // [rsp+48h] [rbp-1A8h]
  _QWORD *v112; // [rsp+60h] [rbp-190h]
  __int64 v113; // [rsp+68h] [rbp-188h]
  unsigned __int8 *v114; // [rsp+78h] [rbp-178h] BYREF
  __int64 v115[2]; // [rsp+80h] [rbp-170h] BYREF
  __int16 v116; // [rsp+90h] [rbp-160h]
  unsigned __int8 *v117[2]; // [rsp+A0h] [rbp-150h] BYREF
  __int16 v118; // [rsp+B0h] [rbp-140h]
  __int64 v119; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v120; // [rsp+C8h] [rbp-128h] BYREF
  unsigned int v121; // [rsp+D0h] [rbp-120h]
  char v122; // [rsp+D8h] [rbp-118h]
  char v123; // [rsp+D9h] [rbp-117h]
  unsigned __int8 *v124; // [rsp+E0h] [rbp-110h] BYREF
  __int64 v125; // [rsp+E8h] [rbp-108h]
  __int64 *v126; // [rsp+F0h] [rbp-100h]
  __int64 v127; // [rsp+F8h] [rbp-F8h]
  __int64 v128; // [rsp+100h] [rbp-F0h]
  int v129; // [rsp+108h] [rbp-E8h]
  __int64 v130; // [rsp+110h] [rbp-E0h]
  __int64 v131; // [rsp+118h] [rbp-D8h]
  _QWORD v132[3]; // [rsp+130h] [rbp-C0h] BYREF
  _BYTE *v133; // [rsp+148h] [rbp-A8h]
  __int64 v134; // [rsp+150h] [rbp-A0h]
  _BYTE v135[32]; // [rsp+158h] [rbp-98h] BYREF
  __int64 v136; // [rsp+178h] [rbp-78h]
  __int64 v137; // [rsp+180h] [rbp-70h]
  __int64 v138; // [rsp+188h] [rbp-68h]
  __int64 v139; // [rsp+190h] [rbp-60h]
  int v140; // [rsp+198h] [rbp-58h]
  __int64 v141; // [rsp+1A0h] [rbp-50h]
  __int64 v142; // [rsp+1A8h] [rbp-48h]
  __int64 v143; // [rsp+1B0h] [rbp-40h]

  v12 = sub_1632FA0(*(_QWORD *)(a1 + 40));
  v132[0] = a2;
  v132[1] = v12;
  v133 = v135;
  v132[2] = a3;
  v134 = 0x400000000LL;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v91 = sub_17040B0(v132, a1);
  v93 = a1 + 72;
  v111 = *(_QWORD *)(a1 + 80);
  if ( v111 == a1 + 72 )
  {
    v13 = v91;
    goto LABEL_73;
  }
  v13 = 0;
  do
  {
    v14 = v111 - 24;
    v15 = *(unsigned int *)(a3 + 48);
    if ( !v111 )
      v14 = 0;
    if ( (_DWORD)v15 )
    {
      v16 = *(_QWORD *)(a3 + 32);
      v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v14 == *v18 )
      {
LABEL_7:
        if ( v18 == (__int64 *)(v16 + 16 * v15) )
          goto LABEL_67;
        if ( !v18[1] )
          goto LABEL_67;
        v112 = (_QWORD *)(v14 + 40);
        v20 = (_QWORD *)(*(_QWORD *)(v14 + 40) & 0xFFFFFFFFFFFFFFF8LL);
        if ( (_QWORD *)(v14 + 40) == v20 )
          goto LABEL_67;
        while ( 1 )
        {
          if ( !v20 )
            BUG();
          v113 = (__int64)(v20 - 3);
          v21 = *((_BYTE *)v20 - 8);
          if ( v21 == 50 )
            break;
          if ( v21 == 5 && *((_WORD *)v20 - 3) == 26 )
          {
            v14 = *(_QWORD *)(v113 - 24LL * (*((_DWORD *)v20 - 1) & 0xFFFFFFF));
            v57 = *(_QWORD *)(v14 + 8);
            if ( v57
              && !*(_QWORD *)(v57 + 8)
              && ((v58 = *(_BYTE *)(v14 + 16), v58 == 50) || v58 == 5 && *(_WORD *)(v14 + 18) == 26)
              || (v59 = *(_QWORD *)(v113 + 24 * (1LL - (*((_DWORD *)v20 - 1) & 0xFFFFFFF))),
                  (v60 = *(_QWORD *)(v59 + 8)) != 0)
              && !*(_QWORD *)(v60 + 8)
              && ((v61 = *(_BYTE *)(v59 + 16), v61 == 50) || v61 == 5 && *(_WORD *)(v59 + 18) == 26) )
            {
LABEL_98:
              v28 = 1;
LABEL_29:
              v29 = sub_16431D0(*(v20 - 3));
              v119 = 0;
              v121 = v29;
              if ( v29 > 0x40 )
                sub_16A4EF0((__int64)&v120, 0, 0);
              else
                v120 = 0;
              v122 = v28;
              v123 = 0;
              if ( !v28 )
              {
                v14 = (__int64)&v119;
                if ( sub_1700A80(*(v20 - 9), (__int64)&v119) )
                  goto LABEL_33;
LABEL_50:
                if ( v121 > 0x40 && v120 )
                  j_j___libc_free_0_0(v120);
                goto LABEL_13;
              }
              v14 = (__int64)&v119;
              if ( !sub_1700A80(v113, (__int64)&v119) || !v123 )
                goto LABEL_50;
LABEL_33:
              v30 = sub_16498A0(v113);
              v128 = 0;
              v127 = v30;
              v129 = 0;
              v130 = 0;
              v131 = 0;
              v31 = v20[2];
              v32 = (unsigned __int8 *)v20[3];
              v124 = 0;
              v125 = v31;
              v126 = v20;
              v117[0] = v32;
              if ( v32 )
              {
                sub_1623A60((__int64)v117, (__int64)v32, 2);
                if ( v124 )
                  sub_161E7C0((__int64)&v124, (__int64)v124);
                v124 = v117[0];
                if ( v117[0] )
                  sub_1623210((__int64)v117, v117[0], (__int64)&v124);
              }
              v33 = (unsigned __int8 *)&v120;
              v34 = sub_15A1070(*(v20 - 3), (__int64)&v120);
              v35 = v119;
              v36 = v34;
              v116 = 257;
              v37 = *(_BYTE *)(v34 + 16);
              if ( v37 <= 0x10u )
              {
                if ( v37 == 13 )
                {
                  v39 = *(unsigned int *)(v36 + 32);
                  if ( (unsigned int)v39 <= 0x40 )
                  {
                    v40 = (unsigned int)(64 - v39);
                    v69 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v39) == *(_QWORD *)(v36 + 24);
                  }
                  else
                  {
                    v95 = *(_DWORD *)(v36 + 32);
                    v99 = v119;
                    v108 = v36;
                    v68 = sub_16A58F0(v36 + 24);
                    v39 = v95;
                    v36 = v108;
                    v35 = v99;
                    v69 = v95 == v68;
                  }
                  if ( v69 )
                    goto LABEL_42;
                }
                if ( *(_BYTE *)(v35 + 16) <= 0x10u )
                {
                  v33 = (unsigned __int8 *)v36;
                  v102 = v36;
                  v38 = sub_15A2CF0((__int64 *)v35, v36, *(double *)a4.m128_u64, a5, a6);
                  v36 = v102;
                  v35 = v38;
LABEL_42:
                  v118 = 257;
                  if ( v28 )
                  {
                    v41 = sub_12AA0C0((__int64 *)&v124, 0x20u, (_BYTE *)v35, v36, (__int64)v117);
                  }
                  else
                  {
                    v103 = (_BYTE *)v35;
                    v45 = sub_15A06D0(*(__int64 ***)v35, (__int64)v33, v39, v40);
                    v41 = sub_12AA0C0((__int64 *)&v124, 0x21u, v103, v45, (__int64)v117);
                  }
                  v116 = 257;
                  v44 = (__int64 **)*(v20 - 3);
                  if ( v44 != *(__int64 ***)v41 )
                  {
                    if ( *(_BYTE *)(v41 + 16) > 0x10u )
                    {
                      v118 = 257;
                      v70 = sub_15FDBD0(37, v41, (__int64)v44, (__int64)v117, 0);
                      v41 = v70;
                      if ( v125 )
                      {
                        v71 = v126;
                        sub_157E9D0(v125 + 40, v70);
                        v72 = *(_QWORD *)(v41 + 24);
                        v73 = *v71;
                        *(_QWORD *)(v41 + 32) = v71;
                        v73 &= 0xFFFFFFFFFFFFFFF8LL;
                        *(_QWORD *)(v41 + 24) = v73 | v72 & 7;
                        *(_QWORD *)(v73 + 8) = v41 + 24;
                        *v71 = *v71 & 7 | (v41 + 24);
                      }
                      sub_164B780(v41, v115);
                      if ( v124 )
                      {
                        v114 = v124;
                        sub_1623A60((__int64)&v114, (__int64)v124, 2);
                        v74 = *(_QWORD *)(v41 + 48);
                        if ( v74 )
                          sub_161E7C0(v41 + 48, v74);
                        v75 = v114;
                        *(_QWORD *)(v41 + 48) = v114;
                        if ( v75 )
                          sub_1623210((__int64)&v114, v75, v41 + 48);
                      }
                    }
                    else
                    {
                      v41 = sub_15A46C0(37, (__int64 ***)v41, v44, 0);
                    }
                  }
                  sub_164D160(v113, v41, a4, a5, a6, a7, v42, v43, a10, a11);
                  v14 = (__int64)v124;
                  if ( v124 )
                    sub_161E7C0((__int64)&v124, (__int64)v124);
                  v13 = 1;
                  goto LABEL_50;
                }
              }
              v118 = 257;
              v104 = v36;
              v46 = sub_15FB440(26, (__int64 *)v35, v36, (__int64)v117, 0);
              v47 = v104;
              v48 = v46;
              if ( v125 )
              {
                v94 = v104;
                v105 = v46;
                v96 = v126;
                sub_157E9D0(v125 + 40, v46);
                v48 = v105;
                v47 = v94;
                v49 = *v96;
                v50 = *(_QWORD *)(v105 + 24);
                *(_QWORD *)(v105 + 32) = v96;
                v49 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v105 + 24) = v49 | v50 & 7;
                *(_QWORD *)(v49 + 8) = v105 + 24;
                *v96 = *v96 & 7 | (v105 + 24);
              }
              v97 = v47;
              v106 = v48;
              sub_164B780(v48, v115);
              v33 = v124;
              v35 = v106;
              v36 = v97;
              if ( v124 )
              {
                v114 = v124;
                sub_1623A60((__int64)&v114, (__int64)v124, 2);
                v35 = v106;
                v36 = v97;
                v51 = *(_QWORD *)(v106 + 48);
                v39 = v106 + 48;
                if ( v51 )
                {
                  sub_161E7C0(v106 + 48, v51);
                  v35 = v106;
                  v36 = v97;
                  v39 = v106 + 48;
                }
                v33 = v114;
                *(_QWORD *)(v35 + 48) = v114;
                if ( v33 )
                {
                  v98 = v35;
                  v107 = v36;
                  sub_1623210((__int64)&v114, v33, v39);
                  v35 = v98;
                  v36 = v107;
                }
              }
              goto LABEL_42;
            }
            v62 = *((_DWORD *)v20 - 1) & 0xFFFFFFF;
            v63 = *(_QWORD *)(v113 - 24 * v62);
            v64 = *(_QWORD *)(v63 + 8);
            if ( v64 )
            {
              if ( !*(_QWORD *)(v64 + 8) )
              {
                v65 = *(_BYTE *)(v63 + 16);
                if ( v65 == 51 || v65 == 5 && *(_WORD *)(v63 + 18) == 27 )
                {
                  v24 = *(_QWORD *)(v113 + 24 * (1 - v62));
                  if ( *(_BYTE *)(v24 + 16) != 13 )
                  {
                    if ( *(_BYTE *)(*(_QWORD *)v24 + 8LL) != 16 )
                      goto LABEL_13;
                    v79 = sub_15A1020((_BYTE *)v24, v14, 1 - v62, v63);
                    if ( !v79 || *(_BYTE *)(v79 + 16) != 13 )
                    {
                      v110 = *(_QWORD *)(*(_QWORD *)v24 + 32LL);
                      if ( v110 )
                      {
                        v86 = 0;
                        do
                        {
                          v14 = v86;
                          v87 = sub_15A0A60(v24, v86);
                          if ( !v87 )
                            goto LABEL_13;
                          v88 = *(_BYTE *)(v87 + 16);
                          if ( v88 != 9 )
                          {
                            if ( v88 != 13 )
                              goto LABEL_13;
                            if ( *(_DWORD *)(v87 + 32) <= 0x40u )
                            {
                              v89 = *(_QWORD *)(v87 + 24) == 1;
                            }
                            else
                            {
                              v101 = *(_DWORD *)(v87 + 32);
                              v89 = v101 - 1 == (unsigned int)sub_16A57B0(v87 + 24);
                            }
                            if ( !v89 )
                              goto LABEL_13;
                          }
                        }
                        while ( v110 != ++v86 );
                      }
LABEL_28:
                      v28 = 0;
                      goto LABEL_29;
                    }
                    v80 = *(_DWORD *)(v79 + 32);
                    if ( v80 <= 0x40 )
                      v67 = *(_QWORD *)(v79 + 24) == 1;
                    else
                      v67 = v80 - 1 == (unsigned int)sub_16A57B0(v79 + 24);
                    goto LABEL_107;
                  }
                  v66 = *(_DWORD *)(v24 + 32);
                  if ( v66 > 0x40 )
                  {
                    v67 = v66 - 1 == (unsigned int)sub_16A57B0(v24 + 24);
LABEL_107:
                    if ( v67 )
                      goto LABEL_28;
                    goto LABEL_13;
                  }
                  goto LABEL_131;
                }
              }
            }
          }
LABEL_13:
          v20 = (_QWORD *)(*v20 & 0xFFFFFFFFFFFFFFF8LL);
          if ( v112 == v20 )
            goto LABEL_67;
        }
        v22 = *(v20 - 9);
        v23 = *(_QWORD *)(v22 + 8);
        if ( v23 )
        {
          if ( *(_QWORD *)(v23 + 8) )
          {
            v24 = *(v20 - 6);
            v14 = *(_QWORD *)(v24 + 8);
            if ( !v14 )
              goto LABEL_13;
            if ( *(_QWORD *)(v14 + 8) )
              goto LABEL_20;
          }
          else
          {
            v25 = *(unsigned __int8 *)(v22 + 16);
            if ( (_BYTE)v25 == 50 || (_BYTE)v25 == 5 && *(_WORD *)(v22 + 18) == 26 )
              goto LABEL_98;
            v24 = *(v20 - 6);
            v14 = *(_QWORD *)(v24 + 8);
            if ( !v14 )
            {
LABEL_22:
              if ( (_BYTE)v25 != 51 && ((_BYTE)v25 != 5 || *(_WORD *)(v22 + 18) != 27) )
                goto LABEL_13;
              v26 = *(_BYTE *)(v24 + 16);
              if ( v26 == 13 )
              {
                v27 = *(_DWORD *)(v24 + 32);
                if ( v27 > 0x40 )
                {
                  if ( (unsigned int)sub_16A57B0(v24 + 24) == v27 - 1 )
                    goto LABEL_28;
                  goto LABEL_13;
                }
LABEL_131:
                if ( *(_QWORD *)(v24 + 24) == 1 )
                  goto LABEL_28;
                goto LABEL_13;
              }
              if ( *(_BYTE *)(*(_QWORD *)v24 + 8LL) == 16 && v26 <= 0x10u )
              {
                v77 = sub_15A1020((_BYTE *)v24, v14, *(_QWORD *)v24, v25);
                if ( !v77 || *(_BYTE *)(v77 + 16) != 13 )
                {
                  v109 = *(_QWORD *)(*(_QWORD *)v24 + 32LL);
                  if ( v109 )
                  {
                    v81 = 0;
                    while ( 1 )
                    {
                      v14 = v81;
                      v82 = sub_15A0A60(v24, v81);
                      if ( !v82 )
                        goto LABEL_13;
                      v83 = *(_BYTE *)(v82 + 16);
                      if ( v83 != 9 )
                      {
                        if ( v83 != 13 )
                          goto LABEL_13;
                        if ( *(_DWORD *)(v82 + 32) <= 0x40u )
                        {
                          v84 = *(_QWORD *)(v82 + 24) == 1;
                        }
                        else
                        {
                          v100 = *(_DWORD *)(v82 + 32);
                          v84 = v100 - 1 == (unsigned int)sub_16A57B0(v82 + 24);
                        }
                        if ( !v84 )
                          goto LABEL_13;
                      }
                      if ( v109 == ++v81 )
                        goto LABEL_28;
                    }
                  }
                  goto LABEL_28;
                }
                v78 = *(_DWORD *)(v77 + 32);
                if ( v78 <= 0x40 )
                {
                  if ( *(_QWORD *)(v77 + 24) == 1 )
                    goto LABEL_28;
                }
                else if ( (unsigned int)sub_16A57B0(v77 + 24) == v78 - 1 )
                {
                  goto LABEL_28;
                }
              }
              goto LABEL_13;
            }
            if ( *(_QWORD *)(v14 + 8) )
            {
LABEL_20:
              if ( *(_QWORD *)(v23 + 8) )
                goto LABEL_13;
              v25 = *(unsigned __int8 *)(v22 + 16);
              goto LABEL_22;
            }
          }
        }
        else
        {
          v24 = *(v20 - 6);
          v76 = *(_QWORD *)(v24 + 8);
          if ( !v76 || *(_QWORD *)(v76 + 8) )
            goto LABEL_13;
        }
        v56 = *(_BYTE *)(v24 + 16);
        if ( v56 == 50 || v56 == 5 && *(_WORD *)(v24 + 18) == 26 )
          goto LABEL_98;
        if ( !v23 )
          goto LABEL_13;
        goto LABEL_20;
      }
      v52 = 1;
      while ( v19 != -8 )
      {
        v85 = v52 + 1;
        v17 = (v15 - 1) & (v52 + v17);
        v18 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v14 == *v18 )
          goto LABEL_7;
        v52 = v85;
      }
    }
LABEL_67:
    v111 = *(_QWORD *)(v111 + 8);
  }
  while ( v93 != v111 );
  if ( (_BYTE)v13 )
  {
    for ( i = *(_QWORD *)(a1 + 80); v111 != i; i = *(_QWORD *)(i + 8) )
    {
      v54 = i - 24;
      if ( !i )
        v54 = 0;
      sub_1AF47C0(v54, 0);
    }
  }
  else
  {
    v13 |= v91;
  }
LABEL_73:
  if ( v141 )
    j_j___libc_free_0(v141, v143 - v141);
  j___libc_free_0(v138);
  if ( v133 != v135 )
    _libc_free((unsigned __int64)v133);
  return v13;
}
