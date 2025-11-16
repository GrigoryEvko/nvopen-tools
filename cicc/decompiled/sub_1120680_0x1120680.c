// Function: sub_1120680
// Address: 0x1120680
//
unsigned __int8 *__fastcall sub_1120680(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  int v6; // r12d
  unsigned int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // rbx
  int v10; // eax
  __int64 *v11; // rdx
  bool v12; // al
  bool v13; // al
  bool v14; // al
  __int64 v15; // r13
  __int64 v16; // r14
  _QWORD *v17; // rax
  _QWORD *v18; // r15
  __int64 v20; // rdi
  unsigned int v21; // ebx
  _QWORD *v22; // rdx
  __int64 v23; // rax
  int v24; // edi
  __int64 v25; // rdx
  unsigned int v26; // r14d
  unsigned int v27; // ebx
  __int64 v28; // rdi
  __int64 v29; // r8
  bool v30; // al
  __int64 v31; // rdx
  _BYTE *v32; // rax
  int v33; // eax
  unsigned __int64 v34; // rax
  __int64 v35; // rcx
  int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // rbx
  __int64 v39; // rax
  int v40; // r15d
  int v41; // r12d
  unsigned int v42; // r14d
  bool v43; // al
  unsigned int v44; // r13d
  bool v45; // al
  unsigned int v46; // r13d
  unsigned __int64 v47; // rdx
  unsigned int v48; // eax
  int v49; // eax
  __int64 v50; // r13
  _QWORD *v51; // rax
  __int64 v52; // rdx
  _BYTE *v53; // rax
  int v54; // eax
  unsigned int v55; // eax
  unsigned int v56; // eax
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  unsigned int v60; // eax
  __int64 *v61; // rax
  __int64 v62; // r13
  __int64 v63; // r13
  __int64 v64; // rax
  __int64 v65; // rbx
  _QWORD *v66; // rax
  _QWORD *v67; // rax
  _BYTE *v68; // r14
  __int64 v69; // rdx
  __int64 v70; // r14
  __int64 v71; // r13
  _QWORD *v72; // rax
  unsigned int v73; // eax
  __int64 v74; // rax
  __int64 v75; // r12
  _QWORD *v76; // rax
  unsigned int v77; // eax
  unsigned int v78; // r14d
  __int64 *v79; // r13
  int v80; // edx
  int v81; // eax
  __int64 v82; // r14
  char v83; // al
  __int64 v84; // r13
  _QWORD *v85; // rax
  unsigned int v86; // r12d
  __int64 v87; // r12
  unsigned int v88; // r15d
  __int64 v89; // rax
  bool v90; // r13
  _QWORD *v91; // rax
  __int64 v92; // r13
  __int64 v93; // r12
  _QWORD *v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // r12
  _QWORD *v98; // rax
  __int64 v99; // rdx
  _BYTE *v100; // rax
  unsigned int v101; // eax
  __int64 v102; // r12
  _QWORD *v103; // rax
  _QWORD *v104; // rax
  _BYTE *v105; // r12
  unsigned int **v106; // r13
  __int64 v107; // rdx
  __int64 v108; // r12
  __int64 v109; // r13
  __int16 v110; // si
  unsigned int v111; // ebx
  __int64 *v112; // r14
  __int64 v113; // rax
  _QWORD *v114; // rax
  __int64 *v115; // r15
  __int64 v116; // rdx
  __int64 v117; // rcx
  __int64 v118; // r8
  unsigned int v119; // eax
  __int64 *v120; // rax
  __int64 v121; // r13
  __int64 v122; // r13
  int v123; // eax
  unsigned __int8 *v124; // rax
  char *v125; // rax
  char v126; // [rsp+Fh] [rbp-111h]
  char v127; // [rsp+10h] [rbp-110h]
  _BYTE *v128; // [rsp+18h] [rbp-108h]
  __int64 **v129; // [rsp+20h] [rbp-100h]
  unsigned int v130; // [rsp+28h] [rbp-F8h]
  __int64 *v132; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v133; // [rsp+38h] [rbp-E8h]
  __int64 v134; // [rsp+38h] [rbp-E8h]
  int v135; // [rsp+38h] [rbp-E8h]
  __int16 v136; // [rsp+40h] [rbp-E0h]
  __int64 v137; // [rsp+40h] [rbp-E0h]
  __int64 v138; // [rsp+40h] [rbp-E0h]
  __int64 *v139; // [rsp+40h] [rbp-E0h]
  unsigned int v140; // [rsp+48h] [rbp-D8h]
  __int16 v141; // [rsp+48h] [rbp-D8h]
  unsigned int **v142; // [rsp+48h] [rbp-D8h]
  unsigned int **v143; // [rsp+48h] [rbp-D8h]
  char *v144; // [rsp+48h] [rbp-D8h]
  char v145; // [rsp+5Fh] [rbp-C1h] BYREF
  char *v146; // [rsp+60h] [rbp-C0h] BYREF
  unsigned int v147; // [rsp+68h] [rbp-B8h]
  char *v148; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int v149; // [rsp+78h] [rbp-A8h]
  unsigned __int64 v150; // [rsp+80h] [rbp-A0h] BYREF
  unsigned int v151; // [rsp+88h] [rbp-98h]
  unsigned __int64 v152; // [rsp+90h] [rbp-90h] BYREF
  unsigned int v153; // [rsp+98h] [rbp-88h]
  __int16 v154; // [rsp+B0h] [rbp-70h]
  const char *v155; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v156; // [rsp+C8h] [rbp-58h]
  char *v157; // [rsp+D0h] [rbp-50h]
  __int16 v158; // [rsp+E0h] [rbp-40h]

  v6 = *(_WORD *)(a2 + 2) & 0x3F;
  v136 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( (unsigned int)(v6 - 32) > 1 )
  {
LABEL_2:
    v7 = *(_DWORD *)(a4 + 8);
    v8 = *(__int64 **)a4;
    v140 = v7;
    if ( v7 <= 0x40 )
    {
      if ( !v7 )
      {
LABEL_22:
        if ( sub_B448F0((__int64)a3) )
        {
          if ( sub_B44900((__int64)a3) )
            goto LABEL_17;
          v130 = v6 - 32;
          if ( (unsigned int)(v6 - 32) > 1 )
            goto LABEL_25;
          goto LABEL_50;
        }
LABEL_7:
        v130 = v6 - 32;
        if ( (unsigned int)(v6 - 32) > 1 )
        {
          if ( !sub_B44900((__int64)a3) || ((v136 - 38) & 0xFFFD) != 0 )
            goto LABEL_25;
          if ( v140 <= 0x40 )
            v13 = *(_QWORD *)a4 == 0;
          else
            v13 = v140 == (unsigned int)sub_C444A0(a4);
          if ( v13 )
            goto LABEL_17;
          if ( v136 != 38 )
          {
            if ( v140 <= 0x40 )
              v14 = *(_QWORD *)a4 == 1;
            else
              v14 = v140 - 1 == (unsigned int)sub_C444A0(a4);
            if ( v14 )
              goto LABEL_17;
LABEL_25:
            v20 = *((_QWORD *)a3 - 4);
            if ( *(_BYTE *)v20 == 17 )
            {
              v133 = v20 + 24;
LABEL_27:
              v21 = *(_DWORD *)(v133 + 8);
              if ( v21 <= 0x40 )
              {
                v22 = *(_QWORD **)v133;
              }
              else
              {
                if ( v21 - (unsigned int)sub_C444A0(v133) > 0x40 )
                  return 0;
                v22 = **(_QWORD ***)v133;
              }
              if ( v140 > (unsigned __int64)v22 )
              {
                v128 = (_BYTE *)*((_QWORD *)a3 - 8);
                v129 = (__int64 **)*((_QWORD *)a3 + 1);
                if ( sub_B44900((__int64)a3) )
                {
                  if ( v136 == 38 )
                  {
                    sub_9865C0((__int64)&v152, a4);
                    sub_C44D10((__int64)&v152, v133);
                    v93 = sub_AD8D80((__int64)v129, (__int64)&v152);
                    v158 = 257;
                    v94 = sub_BD2C40(72, unk_3F10FD0);
                    v18 = v94;
                    if ( v94 )
                      sub_1113300((__int64)v94, 38, (__int64)v128, v93, (__int64)&v155);
                    goto LABEL_158;
                  }
                  if ( v130 <= 1 )
                  {
                    sub_9865C0((__int64)&v152, a4);
                    sub_C44D10((__int64)&v152, v133);
                    sub_9865C0((__int64)&v155, (__int64)&v152);
                    sub_C47AC0((__int64)&v155, v133);
                    v126 = sub_AAD8B0((__int64)&v155, (_QWORD *)a4);
                    sub_969240((__int64 *)&v155);
                    sub_969240((__int64 *)&v152);
                    if ( v126 )
                    {
                      sub_9865C0((__int64)&v152, a4);
                      sub_C44D10((__int64)&v152, v133);
                      v92 = sub_AD8D80((__int64)v129, (__int64)&v152);
                      v158 = 257;
                      v18 = sub_BD2C40(72, unk_3F10FD0);
                      if ( v18 )
LABEL_157:
                        sub_1113300((__int64)v18, v6, (__int64)v128, v92, (__int64)&v155);
LABEL_158:
                      sub_969240((__int64 *)&v152);
                      return (unsigned __int8 *)v18;
                    }
                  }
                  else if ( v6 == 40 )
                  {
                    sub_9865C0((__int64)&v150, a4);
                    sub_C46F20((__int64)&v150, 1u);
                    v101 = v151;
                    v151 = 0;
                    v153 = v101;
                    v152 = v150;
                    sub_9865C0((__int64)&v155, (__int64)&v152);
                    sub_C44D10((__int64)&v155, v133);
                    sub_C46A40((__int64)&v155, 1);
                    v149 = v156;
                    v148 = (char *)v155;
                    sub_969240((__int64 *)&v152);
                    sub_969240((__int64 *)&v150);
                    v102 = sub_AD8D80((__int64)v129, (__int64)&v148);
                    v158 = 257;
                    v103 = sub_BD2C40(72, unk_3F10FD0);
                    v18 = v103;
                    if ( v103 )
                      sub_1113300((__int64)v103, 40, (__int64)v128, v102, (__int64)&v155);
                    goto LABEL_133;
                  }
                }
                if ( !sub_B448F0((__int64)a3) )
                  goto LABEL_39;
                if ( v136 == 34 )
                {
                  sub_9865C0((__int64)&v152, a4);
                  sub_C48380((__int64)&v152, v133);
                  v96 = sub_AD8D80((__int64)v129, (__int64)&v152);
                  v158 = 257;
                  v97 = v96;
                  v98 = sub_BD2C40(72, unk_3F10FD0);
                  v18 = v98;
                  if ( v98 )
                    sub_1113300((__int64)v98, 34, (__int64)v128, v97, (__int64)&v155);
                  goto LABEL_158;
                }
                if ( v130 > 1 )
                {
                  if ( v6 != 36 )
                    goto LABEL_39;
                  sub_9865C0((__int64)&v150, a4);
                  sub_C46F20((__int64)&v150, 1u);
                  v73 = v151;
                  v151 = 0;
                  v153 = v73;
                  v152 = v150;
                  sub_9865C0((__int64)&v155, (__int64)&v152);
                  sub_C48380((__int64)&v155, v133);
                  sub_C46A40((__int64)&v155, 1);
                  v149 = v156;
                  v148 = (char *)v155;
                  sub_969240((__int64 *)&v152);
                  sub_969240((__int64 *)&v150);
                  v74 = sub_AD8D80((__int64)v129, (__int64)&v148);
                  v158 = 257;
                  v75 = v74;
                  v76 = sub_BD2C40(72, unk_3F10FD0);
                  v18 = v76;
                  if ( v76 )
                    sub_1113300((__int64)v76, 36, (__int64)v128, v75, (__int64)&v155);
LABEL_133:
                  sub_969240((__int64 *)&v148);
                  return (unsigned __int8 *)v18;
                }
                sub_9865C0((__int64)&v152, a4);
                sub_C48380((__int64)&v152, v133);
                sub_9865C0((__int64)&v155, (__int64)&v152);
                sub_C47AC0((__int64)&v155, v133);
                v127 = sub_AAD8B0((__int64)&v155, (_QWORD *)a4);
                sub_969240((__int64 *)&v155);
                sub_969240((__int64 *)&v152);
                if ( !v127 )
                {
LABEL_39:
                  v23 = *((_QWORD *)a3 + 2);
                  v24 = *(_WORD *)(a2 + 2) & 0x3F;
                  if ( (unsigned int)(v24 - 32) <= 1 )
                  {
                    if ( v23 && !*(_QWORD *)(v23 + 8) )
                    {
                      v67 = *(_QWORD **)v133;
                      if ( *(_DWORD *)(v133 + 8) > 0x40u )
                        v67 = (_QWORD *)*v67;
                      sub_F0A5D0((__int64)&v155, v140, v140 - (_DWORD)v67);
                      v68 = (_BYTE *)sub_AD8D80((__int64)v129, (__int64)&v155);
                      sub_969240((__int64 *)&v155);
                      v142 = *(unsigned int ***)(a1 + 32);
                      v155 = sub_BD5D20((__int64)a3);
                      v156 = v69;
                      v158 = 773;
                      v157 = ".mask";
                      v70 = sub_A82350(v142, v128, v68, (__int64)&v155);
                      sub_9865C0((__int64)&v155, a4);
                      sub_C48380((__int64)&v155, v133);
                      v71 = sub_AD8D80((__int64)v129, (__int64)&v155);
                      sub_969240((__int64 *)&v155);
                      v158 = 257;
                      v72 = sub_BD2C40(72, unk_3F10FD0);
                      v18 = v72;
                      if ( v72 )
                        sub_1113300((__int64)v72, v6, v70, v71, (__int64)&v155);
                      return (unsigned __int8 *)v18;
                    }
                    v145 = 0;
                  }
                  else
                  {
                    v145 = 0;
                    if ( v23 && !*(_QWORD *)(v23 + 8) )
                    {
                      if ( sub_9893F0(v6, a4, &v145) )
                      {
                        v104 = *(_QWORD **)v133;
                        if ( *(_DWORD *)(v133 + 8) > 0x40u )
                          v104 = (_QWORD *)*v104;
                        sub_9866F0((__int64)&v155, v140, v140 - 1 - (_DWORD)v104);
                        v105 = (_BYTE *)sub_AD8D80((__int64)v129, (__int64)&v155);
                        sub_969240((__int64 *)&v155);
                        v106 = *(unsigned int ***)(a1 + 32);
                        v155 = sub_BD5D20((__int64)a3);
                        v156 = v107;
                        v158 = 773;
                        v157 = ".mask";
                        v108 = sub_A82350(v106, v128, v105, (__int64)&v155);
                        v109 = sub_AD6530((__int64)v129, (__int64)v128);
                        v158 = 257;
                        v18 = sub_BD2C40(72, unk_3F10FD0);
                        if ( !v18 )
                          return (unsigned __int8 *)v18;
                        v110 = 32 - ((v145 == 0) - 1);
                        goto LABEL_192;
                      }
                      v24 = *(_WORD *)(a2 + 2) & 0x3F;
                    }
                  }
                  if ( !sub_B532A0(v24) )
                  {
LABEL_100:
                    v25 = *((_QWORD *)a3 + 2);
                    goto LABEL_43;
                  }
                  v25 = *((_QWORD *)a3 + 2);
                  if ( !v25 || *(_QWORD *)(v25 + 8) )
                  {
LABEL_43:
                    v26 = *(_DWORD *)(v133 + 8);
                    v27 = v140 - 1;
                    if ( v26 > 0x40 )
                    {
                      v138 = v25;
                      v54 = sub_C444A0(v133);
                      v25 = v138;
                      if ( v26 - v54 <= 0x40 && (unsigned __int64)(v140 - 1) >= **(_QWORD **)v133 )
                        v27 = **(_QWORD **)v133;
                    }
                    else if ( (unsigned __int64)(v140 - 1) >= *(_QWORD *)v133 )
                    {
                      v27 = *(_QWORD *)v133;
                    }
                    if ( v25 )
                    {
                      if ( !*(_QWORD *)(v25 + 8) )
                      {
                        if ( v27 )
                        {
                          v77 = sub_BCB060((__int64)v129);
                          v78 = v140 - v27;
                          if ( (unsigned __int8)sub_F0C790(a1, v77, v140 - v27) )
                          {
                            sub_9865C0((__int64)&v150, a4);
                            if ( (unsigned int)sub_D949C0((__int64)&v150) >= v27 )
                              goto LABEL_138;
                            if ( (unsigned __int8)sub_B530E0(v6) )
                            {
                              v124 = (unsigned __int8 *)sub_ACCFD0(*v129, a4);
                              sub_98FF80((__int64)&v155, v6, v124);
                              if ( (_BYTE)v157 )
                              {
                                LOWORD(v6) = (_WORD)v155;
                                if ( v151 <= 0x40 && *(_DWORD *)(v156 + 32) <= 0x40u )
                                {
                                  v125 = *(char **)(v156 + 24);
                                  v151 = *(_DWORD *)(v156 + 32);
                                  v150 = (unsigned __int64)v125;
                                }
                                else
                                {
                                  sub_C43990((__int64)&v150, v156 + 24);
                                }
                              }
                            }
                            if ( (unsigned int)sub_D949C0((__int64)&v150) >= v27 )
                            {
LABEL_138:
                              v79 = (__int64 *)sub_BCD140(*v129, v78);
                              v80 = *((unsigned __int8 *)v129 + 8);
                              if ( (unsigned int)(v80 - 17) <= 1 )
                              {
                                v81 = *((_DWORD *)v129 + 8);
                                BYTE4(v148) = (_BYTE)v80 == 18;
                                LODWORD(v148) = v81;
                                v79 = (__int64 *)sub_BCE1B0(v79, (__int64)v148);
                              }
                              sub_9865C0((__int64)&v152, (__int64)&v150);
                              sub_C44D10((__int64)&v152, v133);
                              sub_C44740((__int64)&v155, (char **)&v152, v78);
                              v82 = sub_AD8D80((__int64)v79, (__int64)&v155);
                              sub_969240((__int64 *)&v155);
                              sub_969240((__int64 *)&v152);
                              v143 = *(unsigned int ***)(a1 + 32);
                              v83 = sub_B44900((__int64)a3);
                              v154 = 257;
                              v84 = sub_A82DA0(v143, (__int64)v128, (__int64)v79, (__int64)&v152, 0, v83);
                              v158 = 257;
                              v85 = sub_BD2C40(72, unk_3F10FD0);
                              v18 = v85;
                              if ( v85 )
                                sub_1113300((__int64)v85, v6, v84, v82, (__int64)&v155);
                              sub_969240((__int64 *)&v150);
                              return (unsigned __int8 *)v18;
                            }
                            sub_969240((__int64 *)&v150);
                          }
                        }
                      }
                    }
                    return 0;
                  }
                  sub_9865C0((__int64)&v152, a4);
                  sub_C46A40((__int64)&v152, 1);
                  v55 = v153;
                  v153 = 0;
                  LODWORD(v156) = v55;
                  v155 = (const char *)v152;
                  if ( !sub_986BA0((__int64)&v155) || v6 != 37 && v6 != 34 )
                  {
                    sub_969240((__int64 *)&v155);
                    sub_969240((__int64 *)&v152);
                    if ( sub_986BA0(a4) && (unsigned int)(v6 - 35) <= 1 )
                    {
                      v158 = 257;
                      v139 = *(__int64 **)(a1 + 32);
                      sub_9865C0((__int64)&v146, a4);
                      sub_C46F20((__int64)&v146, 1u);
                      v56 = v147;
                      v147 = 0;
                      v149 = v56;
                      v148 = v146;
                      sub_987160((__int64)&v148, 1, v57, v58, v59);
                      v60 = v149;
                      v149 = 0;
                      v151 = v60;
                      v150 = (unsigned __int64)v148;
                      v61 = (__int64 *)v133;
                      if ( *(_DWORD *)(v133 + 8) > 0x40u )
                        v61 = *(__int64 **)v133;
                      v62 = *v61;
                      v135 = *v61;
                      sub_9865C0((__int64)&v152, (__int64)&v150);
                      if ( v153 > 0x40 )
                      {
                        sub_C482E0((__int64)&v152, v62);
                      }
                      else if ( (_DWORD)v62 == v153 )
                      {
                        v152 = 0;
                      }
                      else
                      {
                        v152 >>= v135;
                      }
                      v63 = sub_10BC480(v139, (__int64)v128, (__int64)&v152, (__int64)&v155);
                      sub_969240((__int64 *)&v152);
                      sub_969240((__int64 *)&v150);
                      sub_969240((__int64 *)&v148);
                      sub_969240((__int64 *)&v146);
                      v64 = sub_AD6530((__int64)v129, (__int64)v128);
                      v158 = 257;
                      v65 = v64;
                      v66 = sub_BD2C40(72, unk_3F10FD0);
                      v18 = v66;
                      if ( v66 )
                        sub_1113300((__int64)v66, (v6 != 36) + 32, v63, v65, (__int64)&v155);
                      return (unsigned __int8 *)v18;
                    }
                    goto LABEL_100;
                  }
                  sub_969240((__int64 *)&v155);
                  sub_969240((__int64 *)&v152);
                  v158 = 257;
                  v115 = *(__int64 **)(a1 + 32);
                  sub_9865C0((__int64)&v148, a4);
                  sub_987160((__int64)&v148, a4, v116, v117, v118);
                  v119 = v149;
                  v149 = 0;
                  v151 = v119;
                  v150 = (unsigned __int64)v148;
                  v120 = (__int64 *)v133;
                  if ( *(_DWORD *)(v133 + 8) > 0x40u )
                    v120 = *(__int64 **)v133;
                  v121 = *v120;
                  sub_9865C0((__int64)&v152, (__int64)&v150);
                  if ( v153 > 0x40 )
                  {
                    sub_C482E0((__int64)&v152, v121);
                  }
                  else if ( (_DWORD)v121 == v153 )
                  {
                    v152 = 0;
                  }
                  else
                  {
                    v152 >>= v121;
                  }
                  v108 = sub_10BC480(v115, (__int64)v128, (__int64)&v152, (__int64)&v155);
                  sub_969240((__int64 *)&v152);
                  sub_969240((__int64 *)&v150);
                  sub_969240((__int64 *)&v148);
                  v109 = sub_AD6530((__int64)v129, (__int64)v128);
                  v158 = 257;
                  v18 = sub_BD2C40(72, unk_3F10FD0);
                  if ( !v18 )
                    return (unsigned __int8 *)v18;
                  v110 = (v136 != 37) + 32;
LABEL_192:
                  sub_1113300((__int64)v18, v110, v108, v109, (__int64)&v155);
                  return (unsigned __int8 *)v18;
                }
                sub_9865C0((__int64)&v152, a4);
                sub_C48380((__int64)&v152, v133);
                v95 = sub_AD8D80((__int64)v129, (__int64)&v152);
                v158 = 257;
                v92 = v95;
                v18 = sub_BD2C40(72, unk_3F10FD0);
                if ( v18 )
                  goto LABEL_157;
                goto LABEL_158;
              }
              return 0;
            }
            v31 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v20 + 8) + 8LL) - 17;
            if ( (unsigned int)v31 <= 1 && *(_BYTE *)v20 <= 0x15u )
            {
              v32 = sub_AD7630(v20, 0, v31);
              if ( v32 )
              {
                if ( *v32 == 17 )
                {
                  v133 = (unsigned __int64)(v32 + 24);
                  v140 = *(_DWORD *)(a4 + 8);
                  goto LABEL_27;
                }
              }
            }
            v34 = *a3;
            if ( (unsigned __int8)v34 > 0x36u )
              return 0;
            v35 = 0x40540000000000LL;
            if ( !_bittest64(&v35, v34) )
              return 0;
            v36 = (unsigned __int8)v34 <= 0x1Cu ? *((unsigned __int16 *)a3 + 1) : (unsigned __int8)v34 - 29;
            if ( v36 != 25 || (a3[1] & 2) == 0 )
              return 0;
            v37 = *((_QWORD *)a3 - 8);
            v38 = v37 + 24;
            if ( *(_BYTE *)v37 != 17 )
            {
              v99 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v37 + 8) + 8LL) - 17;
              if ( (unsigned int)v99 > 1 )
                return 0;
              if ( *(_BYTE *)v37 > 0x15u )
                return 0;
              v100 = sub_AD7630(v37, 0, v99);
              if ( !v100 || *v100 != 17 )
                return 0;
              v38 = (__int64)(v100 + 24);
            }
            v134 = *((_QWORD *)a3 - 4);
            if ( !v134 )
              return 0;
            v39 = *((_QWORD *)a3 + 1);
            v40 = *(_DWORD *)(a4 + 8);
            v137 = v39;
            v41 = *(_WORD *)(a2 + 2) & 0x3F;
            v141 = *(_WORD *)(a2 + 2) & 0x3F;
            if ( sub_B532A0(v41) )
            {
              v42 = *(_DWORD *)(v38 + 8);
              if ( v42 <= 0x40 )
                v43 = *(_QWORD *)v38 == 0;
              else
                v43 = v42 == (unsigned int)sub_C444A0(v38);
              v18 = 0;
              if ( v43 || (int)sub_C49970(v38, (unsigned __int64 *)a4) > 0 )
                return (unsigned __int8 *)v18;
              v151 = 1;
              v150 = 0;
              v153 = 1;
              v152 = 0;
              sub_C4BFE0(a4, v38, &v150, &v152);
              v44 = v153;
              if ( v153 <= 0x40 )
                v45 = v152 == 0;
              else
                v45 = v44 == (unsigned int)sub_C444A0((__int64)&v152);
              v46 = v151;
              if ( v45 )
              {
                if ( v151 > 0x40 )
                {
                  if ( (unsigned int)sub_C44630((__int64)&v150) == 1 )
                    goto LABEL_184;
                  if ( v141 == 36 )
                  {
                    LOWORD(v41) = 37;
                    goto LABEL_184;
                  }
                  if ( v41 != 35 )
                    goto LABEL_184;
                }
                else
                {
                  v47 = v150;
                  if ( v150 )
                  {
                    if ( (v150 & (v150 - 1)) == 0 )
                    {
                      v48 = v151 - 64;
                      goto LABEL_85;
                    }
                    if ( v141 == 36 )
                    {
                      v48 = v151 - 64;
                      LOWORD(v41) = 37;
                      goto LABEL_85;
                    }
                  }
                  else if ( v141 == 36 )
                  {
                    v49 = v151;
                    LOWORD(v41) = 37;
                    goto LABEL_86;
                  }
                  if ( v41 != 35 )
                  {
LABEL_181:
                    v48 = v46 - 64;
                    if ( !v47 )
                    {
                      v49 = v46;
                      goto LABEL_86;
                    }
LABEL_85:
                    _BitScanReverse64(&v47, v47);
                    v49 = (v47 ^ 0x3F) + v48;
LABEL_86:
                    v50 = sub_AD64C0(v137, v46 - 1 - v49, 0);
                    v158 = 257;
                    v51 = sub_BD2C40(72, unk_3F10FD0);
                    v18 = v51;
                    if ( v51 )
                      sub_1113300((__int64)v51, v41, v134, v50, (__int64)&v155);
                    if ( v153 > 0x40 && v152 )
                      j_j___libc_free_0_0(v152);
                    if ( v151 > 0x40 && v150 )
                      j_j___libc_free_0_0(v150);
                    return (unsigned __int8 *)v18;
                  }
                }
                LOWORD(v41) = 34;
              }
              else if ( v141 == 36 )
              {
                LOWORD(v41) = 37;
              }
              else if ( v41 == 35 )
              {
                LOWORD(v41) = 34;
              }
              if ( v46 <= 0x40 )
              {
                v47 = v150;
                goto LABEL_181;
              }
LABEL_184:
              v49 = sub_C444A0((__int64)&v150);
              goto LABEL_86;
            }
            if ( !sub_B532B0(*(_WORD *)(a2 + 2) & 0x3F) )
              return 0;
            v86 = *(_DWORD *)(v38 + 8);
            if ( v86 <= 0x40 )
            {
              if ( *(_QWORD *)v38 != 1 )
                return 0;
            }
            else if ( (unsigned int)sub_C444A0(v38) != v86 - 1 )
            {
              return 0;
            }
            v87 = sub_AD64C0(v137, (unsigned int)(v40 - 1), 0);
            if ( v141 != 38 )
            {
              if ( v141 != 40 )
                return 0;
              sub_9865C0((__int64)&v152, a4);
              sub_C46F20((__int64)&v152, 1u);
              v88 = v153;
              v153 = 0;
              LODWORD(v156) = v88;
              v155 = (const char *)v152;
              if ( v88 > 0x40 )
              {
                v144 = (char *)v152;
                v122 = *(_QWORD *)(v152 + 8LL * ((v88 - 1) >> 6)) & (1LL << ((unsigned __int8)v88 - 1));
                if ( v122 )
                  v123 = sub_C44500((__int64)&v155);
                else
                  v123 = sub_C444A0((__int64)&v155);
                if ( v88 + 1 - v123 > 0x40 )
                {
                  v90 = v122 == 0;
LABEL_152:
                  sub_969240((__int64 *)&v155);
                  sub_969240((__int64 *)&v152);
                  if ( !v90 )
                  {
                    v158 = 257;
                    v91 = sub_BD2C40(72, unk_3F10FD0);
                    v18 = v91;
                    if ( v91 )
                      sub_1113300((__int64)v91, 32, v134, v87, (__int64)&v155);
                    return (unsigned __int8 *)v18;
                  }
                  return 0;
                }
                v89 = *(_QWORD *)v144;
              }
              else if ( v88 )
              {
                v89 = (__int64)(v152 << (64 - (unsigned __int8)v88)) >> (64 - (unsigned __int8)v88);
              }
              else
              {
                v89 = 0;
              }
              v90 = v89 > 0;
              goto LABEL_152;
            }
            v111 = *(_DWORD *)(a4 + 8);
            v112 = *(__int64 **)a4;
            if ( v111 > 0x40 )
            {
              if ( (v112[(v111 - 1) >> 6] & (1LL << ((unsigned __int8)v111 - 1))) != 0 )
              {
                if ( v111 + 1 - (unsigned int)sub_C44500(a4) > 0x40 )
                  goto LABEL_201;
              }
              else if ( v111 + 1 - (unsigned int)sub_C444A0(a4) > 0x40 )
              {
                return 0;
              }
              v113 = *v112;
            }
            else
            {
              if ( !v111 )
                goto LABEL_201;
              v113 = (__int64)((_QWORD)v112 << (64 - (unsigned __int8)v111)) >> (64 - (unsigned __int8)v111);
            }
            if ( v113 > 0 )
              return 0;
LABEL_201:
            v158 = 257;
            v114 = sub_BD2C40(72, unk_3F10FD0);
            v18 = v114;
            if ( v114 )
              sub_1113300((__int64)v114, 33, v134, v87, (__int64)&v155);
            return (unsigned __int8 *)v18;
          }
          if ( !sub_986760(a4) )
            goto LABEL_25;
LABEL_17:
          v15 = *((_QWORD *)a3 - 8);
          v158 = 257;
          v16 = *(_QWORD *)(a2 - 32);
          v17 = sub_BD2C40(72, unk_3F10FD0);
          v18 = v17;
          if ( v17 )
            sub_1113300((__int64)v17, v6, v15, v16, (__int64)&v155);
          return (unsigned __int8 *)v18;
        }
LABEL_50:
        if ( v140 <= 0x40 )
          v30 = *(_QWORD *)a4 == 0;
        else
          v30 = v140 == (unsigned int)sub_C444A0(a4);
        if ( v30 && (sub_B448F0((__int64)a3) || sub_B44900((__int64)a3)) )
          goto LABEL_17;
        goto LABEL_25;
      }
      v12 = (__int64)((_QWORD)v8 << (64 - (unsigned __int8)v7)) >> (64 - (unsigned __int8)v7) > 0;
    }
    else
    {
      v132 = *(__int64 **)a4;
      v9 = v8[(v7 - 1) >> 6] & (1LL << ((unsigned __int8)v7 - 1));
      if ( v9 )
      {
        v33 = sub_C44500(a4);
        v11 = v132;
        if ( v140 + 1 - v33 > 0x40 )
          goto LABEL_5;
      }
      else
      {
        v10 = sub_C444A0(a4);
        v11 = v132;
        if ( v140 + 1 - v10 > 0x40 )
        {
LABEL_5:
          v12 = v9 == 0;
          goto LABEL_6;
        }
      }
      v12 = *v11 > 0;
    }
LABEL_6:
    if ( v12 )
      goto LABEL_7;
    goto LABEL_22;
  }
  v28 = *((_QWORD *)a3 - 8);
  v29 = v28 + 24;
  if ( *(_BYTE *)v28 != 17 )
  {
    v52 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v28 + 8) + 8LL) - 17;
    if ( (unsigned int)v52 > 1 || *(_BYTE *)v28 > 0x15u )
      goto LABEL_2;
    v53 = sub_AD7630(v28, 0, v52);
    if ( !v53 || *v53 != 17 )
    {
      v6 = *(_WORD *)(a2 + 2) & 0x3F;
      v136 = *(_WORD *)(a2 + 2) & 0x3F;
      goto LABEL_2;
    }
    v29 = (__int64)(v53 + 24);
  }
  return sub_1120150(a1, a2, *((_QWORD *)a3 - 4), a4, v29);
}
