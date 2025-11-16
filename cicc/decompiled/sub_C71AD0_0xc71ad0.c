// Function: sub_C71AD0
// Address: 0xc71ad0
//
unsigned __int64 *__fastcall sub_C71AD0(unsigned __int64 *a1, char a2, char a3, __int64 a4, __int64 a5)
{
  char v5; // r10
  char v9; // r11
  char v10; // r14
  unsigned int v11; // edx
  __int64 v12; // rsi
  unsigned int v13; // ecx
  __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rdx
  char v21; // r10
  char v22; // r11
  unsigned int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // esi
  unsigned __int64 v26; // r8
  unsigned int v27; // edi
  __int64 v28; // r14
  unsigned int v29; // r11d
  __int64 v30; // r9
  unsigned int v31; // r8d
  unsigned __int64 v32; // rcx
  unsigned int v33; // r8d
  __int64 v34; // rcx
  unsigned int v35; // esi
  __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  unsigned int v38; // edx
  __int64 v39; // rcx
  unsigned __int64 v40; // rcx
  unsigned int v41; // edx
  __int64 v42; // rcx
  unsigned __int64 v43; // rcx
  __int64 v45; // rsi
  unsigned int v46; // r9d
  unsigned __int64 v47; // r8
  unsigned int v48; // ecx
  __int64 v49; // rsi
  unsigned __int64 v50; // rcx
  unsigned int v51; // r8d
  unsigned __int64 v52; // rcx
  unsigned int v53; // r9d
  unsigned int v54; // esi
  unsigned int v55; // r8d
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 v58; // rdx
  unsigned __int64 v59; // rdx
  unsigned int v60; // edx
  unsigned int v61; // eax
  unsigned __int64 v62; // rax
  unsigned int v63; // ebx
  __int64 v64; // r13
  unsigned __int64 v65; // r13
  bool v66; // cc
  __int64 v67; // rdi
  unsigned int v68; // eax
  __int64 v69; // r9
  unsigned __int64 v70; // rdx
  unsigned int v71; // edi
  __int64 v72; // r8
  unsigned int v73; // edx
  unsigned int v74; // esi
  unsigned int v75; // ebx
  __int64 v76; // rax
  unsigned int v77; // esi
  __int64 v78; // rax
  __int64 v79; // rbx
  unsigned __int64 v80; // rax
  unsigned int v81; // edx
  unsigned int v82; // esi
  unsigned int v83; // ebx
  __int64 v84; // rax
  unsigned int v85; // edi
  unsigned __int64 v86; // r8
  __int64 v87; // r8
  __int64 v88; // r9
  unsigned __int64 v89; // rdx
  unsigned int v90; // edi
  unsigned __int64 v91; // r8
  __int64 v92; // rbx
  unsigned __int64 v93; // rdx
  unsigned int v94; // r9d
  __int64 v95; // rdx
  unsigned int v96; // eax
  unsigned int v97; // edi
  __int64 v98; // r8
  unsigned int v99; // edi
  unsigned __int64 v100; // rdx
  __int64 v101; // rdx
  unsigned int v102; // [rsp+4h] [rbp-ECh]
  _QWORD *v103; // [rsp+8h] [rbp-E8h]
  __int64 v104; // [rsp+10h] [rbp-E0h]
  unsigned int v105; // [rsp+18h] [rbp-D8h]
  __int64 v106; // [rsp+18h] [rbp-D8h]
  char v107; // [rsp+20h] [rbp-D0h]
  _QWORD *v108; // [rsp+20h] [rbp-D0h]
  __int64 v109; // [rsp+20h] [rbp-D0h]
  char v110; // [rsp+28h] [rbp-C8h]
  bool v111; // [rsp+28h] [rbp-C8h]
  __int64 v112; // [rsp+28h] [rbp-C8h]
  char v113; // [rsp+28h] [rbp-C8h]
  char v114; // [rsp+28h] [rbp-C8h]
  char v115; // [rsp+30h] [rbp-C0h]
  char v116; // [rsp+30h] [rbp-C0h]
  char v117; // [rsp+30h] [rbp-C0h]
  char v118; // [rsp+30h] [rbp-C0h]
  char v119; // [rsp+30h] [rbp-C0h]
  unsigned int v120; // [rsp+34h] [rbp-BCh]
  __int64 v121; // [rsp+38h] [rbp-B8h]
  unsigned int v122; // [rsp+38h] [rbp-B8h]
  char v123; // [rsp+38h] [rbp-B8h]
  char v124; // [rsp+40h] [rbp-B0h]
  char v125; // [rsp+40h] [rbp-B0h]
  char v126; // [rsp+4Bh] [rbp-A5h]
  unsigned int v127; // [rsp+4Ch] [rbp-A4h]
  bool v128; // [rsp+5Fh] [rbp-91h] BYREF
  unsigned __int64 v129; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v130; // [rsp+68h] [rbp-88h]
  __int64 v131; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v132; // [rsp+78h] [rbp-78h]
  unsigned __int64 v133; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v134; // [rsp+88h] [rbp-68h]
  __int64 v135; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v136; // [rsp+98h] [rbp-58h]
  __int64 v137; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v138; // [rsp+A8h] [rbp-48h]
  __int64 v139; // [rsp+B0h] [rbp-40h]
  unsigned int v140; // [rsp+B8h] [rbp-38h]

  v5 = a2;
  v126 = a2;
  v127 = *(_DWORD *)(a4 + 8);
  if ( a3 )
  {
    v9 = a3;
    v10 = a3;
    v11 = *(_DWORD *)(a4 + 24);
    v12 = *(_QWORD *)(a4 + 16);
    v13 = v11 - 1;
    v14 = 1LL << ((unsigned __int8)v11 - 1);
    if ( v5 )
    {
      if ( v11 > 0x40 )
      {
        if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & v14) == 0 )
          goto LABEL_5;
      }
      else if ( (v12 & v14) == 0 )
      {
LABEL_5:
        v15 = *(_QWORD *)a4;
        v120 = v127 - 1;
        v16 = 1LL << ((unsigned __int8)v127 - 1);
        if ( v127 <= 0x40 )
        {
          if ( (v15 & v16) == 0 )
          {
LABEL_7:
            v130 = v127;
LABEL_8:
            v129 = v15;
            goto LABEL_9;
          }
LABEL_96:
          v53 = *(_DWORD *)(a5 + 24);
          v47 = *(_QWORD *)(a5 + 16);
          v48 = v53 - 1;
          v49 = 1LL << ((unsigned __int8)v53 - 1);
          if ( v53 <= 0x40 )
            goto LABEL_97;
          goto LABEL_80;
        }
        if ( (*(_QWORD *)(v15 + 8LL * (v120 >> 6)) & v16) != 0 )
          goto LABEL_96;
LABEL_114:
        v130 = v127;
        goto LABEL_115;
      }
      v51 = *(_DWORD *)(a5 + 8);
      if ( v51 > 0x40 )
        v52 = *(_QWORD *)(*(_QWORD *)a5 + 8LL * ((v51 - 1) >> 6));
      else
        v52 = *(_QWORD *)a5;
      if ( (v52 & (1LL << ((unsigned __int8)v51 - 1))) != 0 )
        goto LABEL_38;
      goto LABEL_5;
    }
    if ( v11 > 0x40 )
    {
      if ( (*(_QWORD *)(v12 + 8LL * (v13 >> 6)) & v14) == 0 )
        goto LABEL_77;
    }
    else if ( (v12 & v14) == 0 )
    {
      goto LABEL_77;
    }
    v33 = *(_DWORD *)(a5 + 24);
    if ( v33 > 0x40 )
      v34 = *(_QWORD *)(*(_QWORD *)(a5 + 16) + 8LL * ((v33 - 1) >> 6));
    else
      v34 = *(_QWORD *)(a5 + 16);
    if ( (v34 & (1LL << ((unsigned __int8)v33 - 1))) != 0 )
      goto LABEL_38;
LABEL_77:
    v15 = *(_QWORD *)a4;
    v120 = v127 - 1;
    v45 = 1LL << ((unsigned __int8)v127 - 1);
    if ( v127 > 0x40 )
    {
      if ( (*(_QWORD *)(v15 + 8LL * (v120 >> 6)) & v45) == 0 )
        goto LABEL_114;
    }
    else if ( (v45 & v15) == 0 )
    {
      goto LABEL_7;
    }
    v46 = *(_DWORD *)(a5 + 8);
    v47 = *(_QWORD *)a5;
    v48 = v46 - 1;
    v49 = 1LL << ((unsigned __int8)v46 - 1);
    if ( v46 <= 0x40 )
    {
LABEL_97:
      v50 = v47;
LABEL_81:
      if ( (v50 & v49) == 0 )
      {
        v130 = v127;
        if ( v127 <= 0x40 )
          goto LABEL_8;
LABEL_115:
        v116 = v9;
        v124 = v5;
        sub_C43780((__int64)&v129, (const void **)a4);
        v11 = *(_DWORD *)(a4 + 24);
        v9 = v116;
        v5 = v124;
LABEL_9:
        v132 = v11;
        if ( v11 > 0x40 )
        {
          v118 = v9;
          v125 = v5;
          sub_C43780((__int64)&v131, (const void **)(a4 + 16));
          v9 = v118;
          v5 = v125;
        }
        else
        {
          v131 = *(_QWORD *)(a4 + 16);
        }
        v134 = *(_DWORD *)(a5 + 8);
        if ( v134 > 0x40 )
        {
          v113 = v9;
          v117 = v5;
          sub_C43780((__int64)&v133, (const void **)a5);
          v9 = v113;
          v5 = v117;
        }
        else
        {
          v133 = *(_QWORD *)a5;
        }
        v136 = *(_DWORD *)(a5 + 24);
        if ( v136 > 0x40 )
        {
          v114 = v9;
          v119 = v5;
          sub_C43780((__int64)&v135, (const void **)(a5 + 16));
          v9 = v114;
          v5 = v119;
        }
        else
        {
          v135 = *(_QWORD *)(a5 + 16);
        }
        v17 = ~(1LL << ((unsigned __int8)v132 - 1));
        if ( v132 > 0x40 )
          *(_QWORD *)(v131 + 8LL * ((v132 - 1) >> 6)) &= v17;
        else
          v131 &= v17;
        v18 = 1LL << ((unsigned __int8)v130 - 1);
        if ( v130 > 0x40 )
          *(_QWORD *)(v129 + 8LL * ((v130 - 1) >> 6)) |= v18;
        else
          v129 |= v18;
        v19 = ~(1LL << ((unsigned __int8)v136 - 1));
        if ( v136 > 0x40 )
          *(_QWORD *)(v135 + 8LL * ((v136 - 1) >> 6)) &= v19;
        else
          v135 &= v19;
        v20 = 1LL << ((unsigned __int8)v134 - 1);
        if ( v134 > 0x40 )
          *(_QWORD *)(v133 + 8LL * ((v134 - 1) >> 6)) |= v20;
        else
          v133 |= v20;
        v107 = v9;
        v110 = v5;
        v115 = v5;
        sub_C70430((__int64)&v137, v5, 0, 0, (__int64)&v129, (__int64)&v133);
        v21 = v110;
        v22 = v107;
        v23 = v140 - 1;
        if ( v110 )
        {
          v24 = v139;
          if ( v140 > 0x40 )
            v24 = *(_QWORD *)(v139 + 8LL * (v23 >> 6));
          v25 = *(_DWORD *)(a4 + 8);
          v105 = v25 - 1;
          v121 = 1LL << ((unsigned __int8)v25 - 1);
          v108 = *(_QWORD **)a4;
          if ( (v24 & (1LL << v23)) != 0 )
          {
            v26 = *(_QWORD *)a4;
            if ( v25 > 0x40 )
              v26 = v108[v105 >> 6];
            v27 = *(_DWORD *)(a4 + 24);
            v28 = *(_QWORD *)(a4 + 16);
            v29 = v27 - 1;
            v30 = 1LL << ((unsigned __int8)v27 - 1);
            if ( (v26 & v121) != 0 )
            {
              v31 = *(_DWORD *)(a5 + 8);
              if ( v31 > 0x40 )
                v32 = *(_QWORD *)(*(_QWORD *)a5 + 8LL * ((v31 - 1) >> 6));
              else
                v32 = *(_QWORD *)a5;
              v111 = (v32 & (1LL << ((unsigned __int8)v31 - 1))) != 0;
              v21 = 0;
            }
            else
            {
              v111 = 0;
              v21 = 0;
            }
          }
          else
          {
            v87 = v137;
            if ( v138 > 0x40 )
              v87 = *(_QWORD *)(v137 + 8LL * ((v138 - 1) >> 6));
            v27 = *(_DWORD *)(a4 + 24);
            v28 = *(_QWORD *)(a4 + 16);
            v29 = v27 - 1;
            v30 = 1LL << ((unsigned __int8)v27 - 1);
            if ( (v87 & (1LL << ((unsigned __int8)v138 - 1))) != 0 )
            {
              if ( v27 > 0x40 )
              {
                if ( (*(_QWORD *)(v28 + 8LL * (v29 >> 6)) & v30) == 0 )
                {
                  v126 = 0;
                  v111 = 0;
                  goto LABEL_307;
                }
              }
              else if ( (v30 & v28) == 0 )
              {
                v111 = 0;
                v126 = 0;
                goto LABEL_282;
              }
              v101 = *(_QWORD *)(a5 + 16);
              if ( *(_DWORD *)(a5 + 24) > 0x40u )
                v101 = *(_QWORD *)(v101 + 8LL * ((unsigned int)(*(_DWORD *)(a5 + 24) - 1) >> 6));
              v126 = 0;
              v111 = (v101 & (1LL << (*(_BYTE *)(a5 + 24) - 1))) != 0;
            }
            else
            {
              v111 = 0;
            }
          }
          if ( v27 > 0x40 )
          {
            v88 = *(_QWORD *)(v28 + 8LL * (v29 >> 6)) & v30;
            goto LABEL_258;
          }
LABEL_282:
          v88 = v28 & v30;
LABEL_258:
          v10 = 0;
          if ( v88 )
          {
LABEL_259:
            if ( v25 > 0x40 )
              v89 = v108[v105 >> 6];
            else
              v89 = *(_QWORD *)a4;
            if ( (v89 & v121) != 0 )
              goto LABEL_303;
            v90 = *(_DWORD *)(a5 + 8);
            v91 = *(_QWORD *)a5;
            if ( v90 > 0x40 )
              v91 = *(_QWORD *)(v91 + 8LL * ((v90 - 1) >> 6));
            if ( (v91 & (1LL << ((unsigned __int8)v90 - 1))) != 0 )
            {
LABEL_303:
              v123 = v10;
              v126 = 0;
            }
            else
            {
              v126 = v21;
              v123 = v21 | v10;
            }
            goto LABEL_192;
          }
LABEL_307:
          v97 = *(_DWORD *)(a5 + 24);
          v98 = *(_QWORD *)(a5 + 16);
          if ( v97 > 0x40 )
            v98 = *(_QWORD *)(v98 + 8LL * ((v97 - 1) >> 6));
          v10 = 0;
          if ( (v98 & (1LL << ((unsigned __int8)v97 - 1))) == 0 )
            v10 = v126;
          goto LABEL_259;
        }
        if ( v140 > 0x40 )
          v112 = *(_QWORD *)(v139 + 8LL * (v23 >> 6));
        else
          v112 = v139;
        v54 = *(_DWORD *)(a4 + 8);
        v55 = *(_DWORD *)(a4 + 24);
        v122 = v54 - 1;
        v102 = v55 - 1;
        v109 = 1LL << ((unsigned __int8)v55 - 1);
        v106 = *(_QWORD *)(a4 + 16);
        v104 = 1LL << ((unsigned __int8)v54 - 1);
        v103 = *(_QWORD **)a4;
        if ( ((1LL << v23) & v112) != 0 )
        {
          if ( v55 > 0x40 )
          {
            if ( (*(_QWORD *)(v106 + 8LL * (v102 >> 6)) & v109) == 0 )
            {
              v111 = 0;
              v10 = v21;
              goto LABEL_246;
            }
          }
          else if ( (v109 & v106) == 0 )
          {
            v111 = 0;
            v10 = v21;
            goto LABEL_104;
          }
          v99 = *(_DWORD *)(a5 + 8);
          v100 = *(_QWORD *)a5;
          if ( v99 > 0x40 )
            v100 = *(_QWORD *)(v100 + 8LL * ((v99 - 1) >> 6));
          v10 = v21;
          v111 = (v100 & (1LL << ((unsigned __int8)v99 - 1))) != 0;
LABEL_181:
          if ( v55 > 0x40 )
          {
            v56 = *(_QWORD *)(v106 + 8LL * (v102 >> 6)) & v109;
LABEL_183:
            if ( v56 )
            {
              v10 = v21;
LABEL_185:
              if ( v54 > 0x40 )
                v70 = v103[v122 >> 6];
              else
                v70 = *(_QWORD *)a4;
              v123 = v10;
              if ( (v70 & v104) == 0 )
              {
                v71 = *(_DWORD *)(a5 + 24);
                v72 = *(_QWORD *)(a5 + 16);
                if ( v71 > 0x40 )
                  v72 = *(_QWORD *)(v72 + 8LL * ((v71 - 1) >> 6));
                v123 = v10;
                if ( (v72 & (1LL << ((unsigned __int8)v71 - 1))) == 0 )
                {
                  v126 = v22;
                  v123 = v22 | v10;
                }
              }
LABEL_192:
              if ( v140 > 0x40 && v139 )
                j_j___libc_free_0_0(v139);
              if ( v138 > 0x40 && v137 )
                j_j___libc_free_0_0(v137);
              if ( v136 > 0x40 && v135 )
                j_j___libc_free_0_0(v135);
              if ( v134 > 0x40 && v133 )
                j_j___libc_free_0_0(v133);
              if ( v132 > 0x40 && v131 )
                j_j___libc_free_0_0(v131);
              if ( v130 > 0x40 && v129 )
                j_j___libc_free_0_0(v129);
              if ( !v123 )
              {
                sub_C70430((__int64)a1, v115, 1, 0, a4, a5);
                return a1;
              }
              sub_C70430((__int64)a1, v115, 1, 0, a4, a5);
              if ( !v111 )
              {
                if ( !v10 )
                  goto LABEL_214;
                v81 = *((_DWORD *)a1 + 2);
                v138 = v81;
                v82 = v120;
                v83 = v120 - v81;
                if ( v81 > 0x40 )
                {
                  sub_C43690((__int64)&v137, 0, 0);
                  v81 = v138;
                  v82 = v83 + v138;
                  if ( v138 == v83 + v138 )
                  {
LABEL_237:
                    if ( *((_DWORD *)a1 + 2) > 0x40u )
                    {
                      sub_C43B90(a1, &v137);
                      v81 = v138;
                      goto LABEL_240;
                    }
                    v84 = v137;
                    v81 = v138;
LABEL_239:
                    *a1 &= v84;
LABEL_240:
                    if ( v81 > 0x40 && v137 )
                      j_j___libc_free_0_0(v137);
LABEL_214:
                    if ( !v126 )
                      return a1;
                    v73 = *((_DWORD *)a1 + 6);
                    v138 = v73;
                    v74 = v120;
                    v75 = v120 - v73;
                    if ( v73 > 0x40 )
                    {
                      sub_C43690((__int64)&v137, 0, 0);
                      v73 = v138;
                      v74 = v75 + v138;
                      if ( v138 == v75 + v138 )
                      {
LABEL_220:
                        if ( *((_DWORD *)a1 + 6) > 0x40u )
                        {
                          sub_C43B90(a1 + 2, &v137);
                          v73 = v138;
                          goto LABEL_223;
                        }
                        v76 = v137;
                        v73 = v138;
LABEL_222:
                        a1[2] &= v76;
LABEL_223:
                        if ( v73 <= 0x40 )
                          return a1;
                        v67 = v137;
                        if ( !v137 )
                          return a1;
LABEL_153:
                        j_j___libc_free_0_0(v67);
                        return a1;
                      }
                    }
                    else
                    {
                      v137 = 0;
                      if ( v120 == v73 )
                      {
                        v76 = 0;
                        goto LABEL_222;
                      }
                    }
                    if ( v74 > 0x3F || v73 > 0x40 )
                      sub_C43C90(&v137, v74, v73);
                    else
                      v137 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v75 + 64) << v74;
                    goto LABEL_220;
                  }
                }
                else
                {
                  v137 = 0;
                  if ( v120 == v81 )
                  {
                    v84 = 0;
                    goto LABEL_239;
                  }
                }
                if ( v82 > 0x3F || v81 > 0x40 )
                  sub_C43C90(&v137, v82, v81);
                else
                  v137 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v83 + 64) << v82;
                goto LABEL_237;
              }
              v77 = *(_DWORD *)(a4 + 24);
              v134 = 1;
              v133 = 0;
              v78 = *(_QWORD *)(a4 + 16);
              if ( v77 > 0x40 )
                v78 = *(_QWORD *)(v78 + 8LL * ((v77 - 1) >> 6));
              v79 = 1LL << v120;
              v80 = (1LL << ((unsigned __int8)v77 - 1)) & v78;
              if ( v80 )
              {
                v138 = v127;
                if ( v127 <= 0x40 )
                {
                  v137 = 1LL << v120;
LABEL_231:
                  v133 = v137;
                  v134 = v138;
                  goto LABEL_139;
                }
                sub_C43690((__int64)&v137, 0, 0);
                if ( v138 <= 0x40 )
                  v137 |= v79;
                else
                  *(_QWORD *)(v137 + 8LL * (v120 >> 6)) |= v79;
                v96 = v134;
              }
              else
              {
                v92 = ~v79;
                v138 = v127;
                if ( v127 <= 0x40 )
                {
                  if ( v127 )
                    v80 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v127;
                  v137 = v92 & v80;
                  goto LABEL_231;
                }
                sub_C43690((__int64)&v137, -1, 1);
                if ( v138 <= 0x40 )
                  v137 &= v92;
                else
                  *(_QWORD *)(v137 + 8LL * (v120 >> 6)) &= v92;
                v96 = v134;
              }
              if ( v96 > 0x40 && v133 )
                j_j___libc_free_0_0(v133);
              goto LABEL_231;
            }
LABEL_246:
            v85 = *(_DWORD *)(a5 + 8);
            v86 = *(_QWORD *)a5;
            if ( v85 > 0x40 )
              v86 = *(_QWORD *)(v86 + 8LL * ((v85 - 1) >> 6));
            if ( (v86 & (1LL << ((unsigned __int8)v85 - 1))) != 0 )
              v10 = v21;
            goto LABEL_185;
          }
LABEL_104:
          v56 = v109 & v106;
          goto LABEL_183;
        }
        v69 = v137;
        if ( v138 > 0x40 )
          v69 = *(_QWORD *)(v137 + 8LL * ((v138 - 1) >> 6));
        if ( (v69 & (1LL << ((unsigned __int8)v138 - 1))) != 0 )
        {
          if ( v54 > 0x40 )
            v93 = v103[v122 >> 6];
          else
            v93 = *(_QWORD *)a4;
          if ( (v93 & v104) != 0 )
          {
            v94 = *(_DWORD *)(a5 + 24);
            v95 = *(_QWORD *)(a5 + 16);
            if ( v94 > 0x40 )
              v95 = *(_QWORD *)(v95 + 8LL * ((v94 - 1) >> 6));
            v22 = v21;
            v111 = (v95 & (1LL << ((unsigned __int8)v94 - 1))) != 0;
            goto LABEL_181;
          }
          v22 = v21;
        }
        v111 = 0;
        goto LABEL_181;
      }
LABEL_38:
      sub_C70430((__int64)a1, v5, 1, 0, a4, a5);
      return a1;
    }
LABEL_80:
    v50 = *(_QWORD *)(v47 + 8LL * (v48 >> 6));
    goto LABEL_81;
  }
  if ( !a2 )
  {
    v130 = *(_DWORD *)(a4 + 24);
    if ( v130 > 0x40 )
      sub_C43780((__int64)&v129, (const void **)(a4 + 16));
    else
      v129 = *(_QWORD *)(a4 + 16);
    v41 = *(_DWORD *)(a5 + 8);
    v138 = v41;
    if ( v41 > 0x40 )
    {
      sub_C43780((__int64)&v137, (const void **)a5);
      v41 = v138;
      if ( v138 > 0x40 )
      {
        sub_C43D10((__int64)&v137);
        v41 = v138;
        v43 = v137;
        goto LABEL_65;
      }
      v42 = v137;
    }
    else
    {
      v42 = *(_QWORD *)a5;
    }
    v43 = ~v42 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v41);
    if ( !v41 )
      v43 = 0;
LABEL_65:
    v134 = v41;
    v133 = v43;
    sub_C499A0((__int64)&v137, (__int64)&v129, (__int64 *)&v133, &v128);
    if ( v138 > 0x40 && v137 )
      j_j___libc_free_0_0(v137);
    if ( v134 > 0x40 && v133 )
      j_j___libc_free_0_0(v133);
    if ( v130 > 0x40 && v129 )
      j_j___libc_free_0_0(v129);
    if ( !v128 )
    {
      sub_C70430((__int64)a1, 0, 0, 1, a4, a5);
      return a1;
    }
    v57 = *(_DWORD *)(a4 + 8);
    v138 = v57;
    if ( v57 > 0x40 )
    {
      sub_C43780((__int64)&v137, (const void **)a4);
      v57 = v138;
      if ( v138 > 0x40 )
      {
        sub_C43D10((__int64)&v137);
        v57 = v138;
        v59 = v137;
        goto LABEL_123;
      }
      v58 = v137;
    }
    else
    {
      v58 = *(_QWORD *)a4;
    }
    v59 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v57) & ~v58;
    if ( !v57 )
      v59 = 0;
LABEL_123:
    v129 = v59;
    v60 = *(_DWORD *)(a5 + 24);
    v130 = v57;
    v134 = v60;
    if ( v60 > 0x40 )
      sub_C43780((__int64)&v133, (const void **)(a5 + 16));
    else
      v133 = *(_QWORD *)(a5 + 16);
    sub_C499A0((__int64)&v137, (__int64)&v129, (__int64 *)&v133, &v128);
    if ( v138 > 0x40 && v137 )
      j_j___libc_free_0_0(v137);
    if ( v134 > 0x40 && v133 )
      j_j___libc_free_0_0(v133);
    if ( v130 > 0x40 && v129 )
      j_j___libc_free_0_0(v129);
    if ( !v128 )
    {
      sub_C70430((__int64)a1, 0, 0, 1, a4, a5);
      v61 = *((_DWORD *)a1 + 6);
      if ( v61 > 0x40 )
        memset((void *)a1[2], 0, 8 * (((unsigned __int64)v61 + 63) >> 6));
      else
        a1[2] = 0;
      return a1;
    }
    sub_C70430((__int64)a1, 0, 0, 1, a4, a5);
    v134 = 1;
    v62 = 0;
    v133 = 0;
    v138 = v127;
    if ( v127 <= 0x40 )
      goto LABEL_138;
    sub_C43690((__int64)&v137, 0, 0);
    if ( v134 <= 0x40 )
      goto LABEL_287;
    goto LABEL_285;
  }
  v35 = *(_DWORD *)(a4 + 8);
  v138 = v35;
  if ( v127 > 0x40 )
  {
    sub_C43780((__int64)&v137, (const void **)a4);
    v35 = v138;
    if ( v138 > 0x40 )
    {
      sub_C43D10((__int64)&v137);
      v35 = v138;
      v37 = v137;
      goto LABEL_44;
    }
    v36 = v137;
  }
  else
  {
    v36 = *(_QWORD *)a4;
  }
  v37 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v35) & ~v36;
  if ( !v35 )
    v37 = 0;
LABEL_44:
  v129 = v37;
  v38 = *(_DWORD *)(a5 + 8);
  v130 = v35;
  v138 = v38;
  if ( v38 > 0x40 )
  {
    sub_C43780((__int64)&v137, (const void **)a5);
    v38 = v138;
    if ( v138 > 0x40 )
    {
      sub_C43D10((__int64)&v137);
      v38 = v138;
      v40 = v137;
      goto LABEL_48;
    }
    v39 = v137;
  }
  else
  {
    v39 = *(_QWORD *)a5;
  }
  v40 = ~v39 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v38);
  if ( !v38 )
    v40 = 0;
LABEL_48:
  v134 = v38;
  v133 = v40;
  sub_C49AB0((__int64)&v137, (__int64)&v129, (__int64 *)&v133, &v128);
  if ( v138 > 0x40 && v137 )
    j_j___libc_free_0_0(v137);
  if ( v134 > 0x40 && v133 )
    j_j___libc_free_0_0(v133);
  if ( v130 > 0x40 && v129 )
    j_j___libc_free_0_0(v129);
  if ( !v128 )
  {
    sub_C70430((__int64)a1, 1, 0, 1, a4, a5);
    return a1;
  }
  v130 = *(_DWORD *)(a4 + 24);
  if ( v130 > 0x40 )
    sub_C43780((__int64)&v129, (const void **)(a4 + 16));
  else
    v129 = *(_QWORD *)(a4 + 16);
  v134 = *(_DWORD *)(a5 + 24);
  if ( v134 > 0x40 )
    sub_C43780((__int64)&v133, (const void **)(a5 + 16));
  else
    v133 = *(_QWORD *)(a5 + 16);
  sub_C49AB0((__int64)&v137, (__int64)&v129, (__int64 *)&v133, &v128);
  if ( v138 > 0x40 && v137 )
    j_j___libc_free_0_0(v137);
  if ( v134 > 0x40 && v133 )
    j_j___libc_free_0_0(v133);
  if ( v130 > 0x40 && v129 )
    j_j___libc_free_0_0(v129);
  if ( !v128 )
  {
    sub_C70430((__int64)a1, 1, 0, 1, a4, a5);
    v68 = *((_DWORD *)a1 + 2);
    if ( v68 > 0x40 )
      memset((void *)*a1, 0, 8 * (((unsigned __int64)v68 + 63) >> 6));
    else
      *a1 = 0;
    return a1;
  }
  sub_C70430((__int64)a1, 1, 0, 1, a4, a5);
  v134 = 1;
  v133 = 0;
  v138 = v127;
  if ( v127 <= 0x40 )
  {
    v62 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v127;
    if ( !v127 )
      v62 = 0;
    goto LABEL_138;
  }
  sub_C43690((__int64)&v137, -1, 1);
  if ( v134 > 0x40 )
  {
LABEL_285:
    if ( v133 )
      j_j___libc_free_0_0(v133);
  }
LABEL_287:
  v62 = v137;
  v127 = v138;
LABEL_138:
  v133 = v62;
  v134 = v127;
LABEL_139:
  if ( *((_DWORD *)a1 + 6) <= 0x40u && (v63 = v134, v134 <= 0x40) )
  {
    v64 = v133;
    *((_DWORD *)a1 + 6) = v134;
    a1[2] = v64;
  }
  else
  {
    sub_C43990((__int64)(a1 + 2), (__int64)&v133);
    v63 = v134;
    v138 = v134;
    if ( v134 > 0x40 )
    {
      sub_C43780((__int64)&v137, (const void **)&v133);
      v63 = v138;
      if ( v138 > 0x40 )
      {
        sub_C43D10((__int64)&v137);
        v63 = v138;
        v65 = v137;
        goto LABEL_146;
      }
      v64 = v137;
    }
    else
    {
      v64 = v133;
    }
  }
  v65 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v63) & ~v64;
  if ( !v63 )
    v65 = 0;
  v137 = v65;
LABEL_146:
  v66 = *((_DWORD *)a1 + 2) <= 0x40u;
  v138 = 0;
  if ( v66 || !*a1 )
  {
    *a1 = v65;
    *((_DWORD *)a1 + 2) = v63;
  }
  else
  {
    j_j___libc_free_0_0(*a1);
    v66 = v138 <= 0x40;
    *a1 = v65;
    *((_DWORD *)a1 + 2) = v63;
    if ( !v66 && v137 )
      j_j___libc_free_0_0(v137);
  }
  if ( v134 > 0x40 )
  {
    v67 = v133;
    if ( v133 )
      goto LABEL_153;
  }
  return a1;
}
