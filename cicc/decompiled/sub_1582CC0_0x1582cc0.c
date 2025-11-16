// Function: sub_1582CC0
// Address: 0x1582cc0
//
__int64 __fastcall sub_1582CC0(unsigned int a1, unsigned __int64 a2, _QWORD *a3)
{
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rcx
  int v9; // eax
  __int64 v10; // rdx
  int v11; // r13d
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned int v14; // eax
  __int64 v15; // r14
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rsi
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __int64 v30; // rdx
  int v31; // r13d
  unsigned int v32; // r14d
  __int64 v33; // rax
  unsigned __int64 v34; // rsi
  __int64 v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rcx
  char v38; // al
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 *v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rcx
  bool v44; // bl
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rdx
  char v49; // bl
  __int64 v50; // r13
  int v51; // eax
  int v52; // r14d
  int v53; // ebx
  __int64 v54; // r15
  int v55; // ebx
  __int64 v56; // r14
  __int64 v57; // rax
  __int64 v58; // r9
  unsigned int v59; // eax
  __int64 v60; // rax
  __int64 v61; // rsi
  __int64 v62; // r12
  unsigned __int64 v63; // rsi
  _BOOL8 v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // rbx
  __int64 v68; // rcx
  __int64 v69; // rax
  __int64 v70; // r15
  __int64 v71; // rcx
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // rax
  unsigned __int64 v77; // rdx
  __int64 v78; // r12
  __int64 v79; // rcx
  __int64 v80; // rax
  __int64 v81; // r15
  __int64 v82; // rcx
  __int64 v83; // rdx
  unsigned __int64 v84; // r14
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r15
  __int64 v88; // rbx
  __int64 v89; // r12
  __int64 v90; // rcx
  __int64 v91; // rax
  __int64 v92; // r15
  __int64 v93; // rcx
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // r13
  __int64 v97; // rax
  __int64 v98; // rbx
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // r9
  __int64 v102; // rax
  __int64 v103; // rax
  unsigned __int64 v104; // rsi
  __int64 v105; // rcx
  __int64 v106; // rdi
  __int64 v107; // rax
  __int64 v108; // r14
  __int64 v109; // rdi
  __int64 v110; // r15
  char v111; // dl
  __int64 v112; // rdx
  __int64 v113; // rax
  __int64 v114; // r14
  __int64 v115; // r13
  unsigned int v116; // eax
  __int64 v117; // rax
  char v118; // cl
  unsigned int v119; // eax
  __int64 v120; // r14
  __int64 v121; // rax
  __int64 v122; // [rsp+0h] [rbp-100h]
  __int64 v123; // [rsp+8h] [rbp-F8h]
  __int64 v124; // [rsp+8h] [rbp-F8h]
  __int64 v125; // [rsp+8h] [rbp-F8h]
  __int64 v126; // [rsp+8h] [rbp-F8h]
  __int64 v127; // [rsp+10h] [rbp-F0h]
  __int64 v128; // [rsp+10h] [rbp-F0h]
  __int64 v129; // [rsp+10h] [rbp-F0h]
  __int64 v130; // [rsp+18h] [rbp-E8h]
  __int64 v131; // [rsp+18h] [rbp-E8h]
  __int64 v132; // [rsp+18h] [rbp-E8h]
  __int64 v133; // [rsp+18h] [rbp-E8h]
  __int64 v134; // [rsp+18h] [rbp-E8h]
  __int64 v135; // [rsp+18h] [rbp-E8h]
  __int64 v136; // [rsp+18h] [rbp-E8h]
  __int64 v137; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v138; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v139; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v140; // [rsp+38h] [rbp-C8h]
  _QWORD *v141; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v142; // [rsp+48h] [rbp-B8h] BYREF
  _QWORD v143[22]; // [rsp+50h] [rbp-B0h] BYREF

  if ( *(_BYTE *)(a2 + 16) != 9 )
  {
    v6 = a2;
    v7 = a2;
    if ( (unsigned __int8)sub_1593BB0(a2) && *((_BYTE *)a3 + 8) != 9 && a1 != 48 )
      return sub_15A06D0(a3);
    v9 = *(unsigned __int8 *)(a2 + 16);
    if ( (_BYTE)v9 == 5 )
    {
      v6 = a2;
      if ( (unsigned __int8)sub_1594510(a2) )
      {
        v11 = *(unsigned __int16 *)(a2 + 18);
        v130 = *(_QWORD *)a2;
        v12 = **(_QWORD **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        v13 = sub_1643360(*a3);
        a2 = a1;
        v14 = sub_15FB960(v11, a1, v12, v130, (_DWORD)a3, 0, v13, 0);
        v6 = v14;
        if ( v14 )
          return sub_15A46C0(v14, *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)), a3, 0);
      }
      else if ( *(_WORD *)(a2 + 18) == 32 && (*(_BYTE *)(a2 + 17) & 0xFC) == 0 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 16 )
      {
        v30 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v31 = v30;
        if ( (_DWORD)v30 == 1 )
          return sub_15A4A70(*(_QWORD *)(a2 - 24 * v30), a3);
        v32 = 1;
        while ( 1 )
        {
          v6 = *(_QWORD *)(a2 + 24 * (v32 - v30));
          if ( !(unsigned __int8)sub_1593BB0(v6) )
            break;
          ++v32;
          v30 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          if ( v32 == v31 )
            return sub_15A4A70(*(_QWORD *)(a2 - 24 * v30), a3);
        }
      }
      v9 = *(unsigned __int8 *)(v7 + 16);
    }
    v10 = v9 & 0xFFFFFFFB;
    if ( (v9 & 0xFB) == 8 )
    {
      if ( *((_BYTE *)a3 + 8) != 16 || (v8 = *((unsigned int *)a3 + 8), *(_DWORD *)(*(_QWORD *)v7 + 32LL) != (_DWORD)v8) )
      {
        v10 = a1 - 36;
        switch ( a1 )
        {
          case '$':
            goto LABEL_66;
          case '%':
          case '&':
          case '0':
            return 0;
          case '\'':
          case '(':
            goto LABEL_56;
          case ')':
          case '*':
            goto LABEL_52;
          case '+':
          case ',':
            goto LABEL_40;
          case '-':
            goto LABEL_76;
          case '.':
            goto LABEL_64;
          case '/':
            goto LABEL_70;
          default:
LABEL_248:
            JUMPOUT(0x419F4A);
        }
      }
      v16 = a3[3];
      v141 = v143;
      v142 = 0x1000000000LL;
      v17 = sub_16498A0(v7);
      v18 = sub_1644900(v17, 32);
      v19 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
      if ( (_DWORD)v19 )
      {
        v20 = (unsigned int)v19;
        v21 = 0;
        v131 = v20;
        do
        {
          v22 = sub_15A0680(v18, v21, 0);
          v23 = sub_15A37D0(v7, v22, 0);
          v24 = sub_15A46C0(a1, v23, v16, 0);
          v25 = (unsigned int)v142;
          if ( (unsigned int)v142 >= HIDWORD(v142) )
          {
            v122 = v24;
            sub_16CD150(&v141, v143, 0, 8);
            v25 = (unsigned int)v142;
            v24 = v122;
          }
          ++v21;
          v141[v25] = v24;
          v26 = v142 + 1;
          LODWORD(v142) = v142 + 1;
        }
        while ( v131 != v21 );
      }
      else
      {
        v26 = v142;
      }
      v27 = v26;
      goto LABEL_31;
    }
    v8 = a1 - 36;
    switch ( a1 )
    {
      case '$':
LABEL_66:
        if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
          return 0;
        if ( (_BYTE)v9 == 13 )
        {
          sub_16A5A50(&v141, v7 + 24);
          v46 = sub_16498A0(v7);
          v15 = sub_159C0E0(v46, &v141);
          if ( (unsigned int)v142 <= 0x40 )
            return v15;
          goto LABEL_62;
        }
        if ( ((*((_DWORD *)a3 + 2) >> 8) & 7) != 0 || (*(_BYTE *)(*(_QWORD *)v7 + 9LL) & 7) != 0 )
          return 0;
        return sub_1581390((_BYTE *)v7, 0, *((_DWORD *)a3 + 2) >> 11);
      case '%':
        if ( (_BYTE)v9 != 13 )
          return 0;
        sub_16A5C50(&v141, v7 + 24, *((_DWORD *)a3 + 2) >> 8);
        goto LABEL_60;
      case '&':
        if ( (_BYTE)v9 != 13 )
          return 0;
        sub_16A5B10(&v141, v7 + 24, *((_DWORD *)a3 + 2) >> 8);
        goto LABEL_60;
      case '\'':
      case '(':
LABEL_56:
        if ( (_BYTE)v9 != 14 )
          return 0;
        LODWORD(v142) = *((_DWORD *)a3 + 2) >> 8;
        v44 = a1 == 39;
        if ( (unsigned int)v142 > 0x40 )
          sub_16A4EF0(&v141, 0, 0);
        else
          v141 = 0;
        BYTE4(v142) = v44;
        if ( (unsigned int)sub_169E1A0(v7 + 24, &v141, 3, &v139) != 1 )
          goto LABEL_60;
        v15 = sub_1599EF0(a3);
        goto LABEL_61;
      case ')':
      case '*':
LABEL_52:
        if ( (_BYTE)v9 != 13 )
          return 0;
        v41 = a3;
        v140 = sub_1643030(a3);
        if ( v140 > 0x40 )
        {
          v41 = &v139;
          a2 = 0;
          sub_16A4EF0(&v139, 0, 0);
        }
        else
        {
          v139 = 0;
        }
        switch ( *((_BYTE *)a3 + 8) )
        {
          case 1:
            v73 = sub_1698260(v41, a2, v42, v43);
            goto LABEL_124;
          case 2:
            v73 = sub_1698270(v41, a2);
            goto LABEL_124;
          case 3:
            v73 = sub_1698280(v41);
            goto LABEL_124;
          case 4:
            v73 = sub_16982A0();
            goto LABEL_124;
          case 5:
            v73 = sub_1698290();
LABEL_124:
            v134 = v73;
            v76 = sub_16982C0(v41, a2, v74, v75);
            v61 = v134;
            v62 = v76;
            if ( v134 == v76 )
              goto LABEL_100;
            sub_169D050(&v142, v134, &v139);
            goto LABEL_101;
          case 6:
            v61 = sub_16982C0(v41, a2, v42, v43);
            v62 = v61;
LABEL_100:
            sub_169D060(&v142, v61, &v139);
LABEL_101:
            if ( v140 > 0x40 && v139 )
              j_j___libc_free_0_0(v139);
            v63 = v7 + 24;
            v64 = a1 == 42;
            if ( v142 == v62 )
              sub_169E6C0(&v142, v63, v64, 0);
            else
              sub_169A290(&v142, v63, v64, 0);
            v65 = sub_16498A0(v7);
            v15 = sub_159CCF0(v65, &v141);
            if ( v142 != v62 )
              goto LABEL_51;
            v66 = v143[0];
            if ( !v143[0] )
              return v15;
            v67 = v143[0] + 32LL * *(_QWORD *)(v143[0] - 8LL);
            if ( v143[0] != v67 )
            {
              do
              {
                v67 -= 32;
                if ( *(_QWORD *)(v67 + 8) == v62 )
                {
                  v68 = *(_QWORD *)(v67 + 16);
                  v123 = v68;
                  if ( v68 )
                  {
                    v69 = 32LL * *(_QWORD *)(v68 - 8);
                    v70 = v68 + v69;
                    if ( v68 != v68 + v69 )
                    {
                      do
                      {
                        v70 -= 32;
                        if ( *(_QWORD *)(v70 + 8) == v62 )
                        {
                          v71 = *(_QWORD *)(v70 + 16);
                          if ( v71 )
                          {
                            v72 = v71 + 32LL * *(_QWORD *)(v71 - 8);
                            if ( v71 != v72 )
                            {
                              do
                              {
                                v127 = v71;
                                v133 = v72 - 32;
                                sub_127D120((_QWORD *)(v72 - 24));
                                v72 = v133;
                                v71 = v127;
                              }
                              while ( v127 != v133 );
                            }
                            j_j_j___libc_free_0_0(v71 - 8);
                          }
                        }
                        else
                        {
                          sub_1698460(v70 + 8);
                        }
                      }
                      while ( v123 != v70 );
                    }
                    j_j_j___libc_free_0_0(v123 - 8);
                  }
                }
                else
                {
                  sub_1698460(v67 + 8);
                }
              }
              while ( v66 != v67 );
            }
            break;
          default:
            goto LABEL_248;
        }
        goto LABEL_147;
      case '+':
      case ',':
LABEL_40:
        if ( (_BYTE)v9 != 14 )
          return 0;
        v33 = sub_16982C0(v6, a2, v10, v8);
        v34 = v7 + 32;
        v35 = v33;
        if ( *(_QWORD *)(v7 + 32) == v33 )
          sub_169C6E0(&v142, v34);
        else
          sub_16986C0(&v142, v34);
        v38 = *((_BYTE *)a3 + 8);
        switch ( v38 )
        {
          case 1:
            v39 = sub_1698260(&v142, v34, v36, v37);
            break;
          case 2:
            v39 = sub_1698270(&v142, v34);
            break;
          case 3:
            v39 = sub_1698280(&v142);
            break;
          case 4:
            v39 = sub_16982A0();
            break;
          case 5:
            v39 = sub_1698290();
            break;
          default:
            v39 = v35;
            if ( v38 != 6 )
              v39 = sub_16982B0(&v142, v35);
            break;
        }
        sub_16A3360(&v141, v39, 0, &v139);
        v40 = sub_16498A0(v7);
        v15 = sub_159CCF0(v40, &v141);
        if ( v35 != v142 )
        {
LABEL_51:
          sub_1698460(&v142);
          return v15;
        }
        v66 = v143[0];
        if ( !v143[0] )
          return v15;
        v78 = v143[0] + 32LL * *(_QWORD *)(v143[0] - 8LL);
        if ( v143[0] != v78 )
        {
          do
          {
            v78 -= 32;
            if ( v35 == *(_QWORD *)(v78 + 8) )
            {
              v79 = *(_QWORD *)(v78 + 16);
              v124 = v79;
              if ( v79 )
              {
                v80 = 32LL * *(_QWORD *)(v79 - 8);
                v81 = v79 + v80;
                if ( v79 != v79 + v80 )
                {
                  do
                  {
                    v81 -= 32;
                    if ( v35 == *(_QWORD *)(v81 + 8) )
                    {
                      v82 = *(_QWORD *)(v81 + 16);
                      if ( v82 )
                      {
                        v83 = v82 + 32LL * *(_QWORD *)(v82 - 8);
                        if ( v82 != v83 )
                        {
                          do
                          {
                            v128 = v82;
                            v135 = v83 - 32;
                            sub_127D120((_QWORD *)(v83 - 24));
                            v83 = v135;
                            v82 = v128;
                          }
                          while ( v128 != v135 );
                        }
                        j_j_j___libc_free_0_0(v82 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v81 + 8);
                    }
                  }
                  while ( v124 != v81 );
                }
                j_j_j___libc_free_0_0(v124 - 8);
              }
            }
            else
            {
              sub_1698460(v78 + 8);
            }
          }
          while ( v66 != v78 );
        }
        goto LABEL_147;
      case '-':
LABEL_76:
        v49 = sub_1593BB0(v7);
        if ( v49 )
          return sub_15A0680(a3, 0, 0);
        if ( *(_BYTE *)(v7 + 16) != 5
          || *(_WORD *)(v7 + 18) != 32
          || !(unsigned __int8)sub_1593BB0(*(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)))
          || *((_BYTE *)a3 + 8) == 16 )
        {
          return 0;
        }
        v50 = sub_16348C0(v7);
        v51 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
        if ( v51 == 2 )
        {
          v114 = *(_QWORD *)(v7 - 24);
          if ( *(_BYTE *)(v114 + 16) == 13 )
            v49 = sub_1455000(v114 + 24);
          v115 = sub_1580D00(v50, (__int64)a3, v49 ^ 1u);
          if ( v115 )
          {
            v116 = sub_15FBEB0(v114, 1, a3, 0);
            v117 = sub_15A46C0(v116, v114, a3, 0);
            return sub_15A2C20(v115, v117, 0, 0);
          }
          return 0;
        }
        if ( v51 != 3 || !(unsigned __int8)sub_1593BB0(*(_QWORD *)(v7 - 48)) )
          return 0;
        if ( *(_BYTE *)(v50 + 8) != 13 )
          goto LABEL_241;
        v52 = *(_DWORD *)(v50 + 8);
        v53 = *(_DWORD *)(v7 + 20);
        if ( (v52 & 0x200) == 0
          && sub_1455000(*(_QWORD *)(v7 + 24 * (2LL - (v53 & 0xFFFFFFF))) + 24LL)
          && *(_DWORD *)(v50 + 12) == 2 )
        {
          if ( (unsigned __int8)sub_1642F90(**(_QWORD **)(v50 + 16), 1) )
            return sub_1580EE0(*(_QWORD *)(*(_QWORD *)(v50 + 16) + 8LL), (__int64)a3, 0);
LABEL_241:
          v118 = *(_BYTE *)(v50 + 8);
          if ( (unsigned __int8)(v118 - 13) > 1u )
            return 0;
          v54 = *(_QWORD *)(v7 + 24 * (2LL - (*(_DWORD *)(v7 + 20) & 0xFFFFFFF)));
          if ( v118 == 14 )
          {
            v119 = sub_15FBEB0(v54, 0, a3, 0);
            v120 = sub_15A46C0(v119, v54, a3, 0);
            v121 = sub_1580D00(*(_QWORD *)(v50 + 24), (__int64)a3, 1);
            return sub_15A2C20(v121, v120, 1, 0);
          }
          v52 = *(_DWORD *)(v50 + 8);
        }
        else
        {
          v54 = *(_QWORD *)(v7 + 24 * (2LL - (v53 & 0xFFFFFFF)));
        }
        if ( (v52 & 0x200) != 0 )
          return 0;
        v55 = *(_DWORD *)(v50 + 12);
        if ( !v55 )
          return 0;
        v56 = sub_1580D00(**(_QWORD **)(v50 + 16), (__int64)a3, 1);
        v57 = 0;
        while ( v55 != (_DWORD)v57 + 1 )
        {
          v132 = v57 + 1;
          v58 = sub_1580D00(*(_QWORD *)(*(_QWORD *)(v50 + 16) + 8 * (v57 + 1)), (__int64)a3, 1);
          v57 = v132;
          if ( v56 != v58 )
            return 0;
        }
        v59 = sub_15FBEB0(v54, 0, a3, 0);
        v60 = sub_15A46C0(v59, v54, a3, 0);
        return sub_15A2C20(v56, v60, 1, 0);
      case '.':
LABEL_64:
        if ( !(unsigned __int8)sub_1593BB0(v7) )
          return 0;
        return sub_1599A20(a3);
      case '/':
LABEL_70:
        v47 = *(_QWORD *)v7;
        v138 = v7;
        if ( a3 == (_QWORD *)v47 )
          return v7;
        if ( *(_BYTE *)(v47 + 8) != 15 )
          goto LABEL_135;
        v48 = *((unsigned __int8 *)a3 + 8);
        if ( (_BYTE)v48 == 15 )
        {
          if ( *(_DWORD *)(v47 + 8) >> 8 != *((_DWORD *)a3 + 2) >> 8 )
            goto LABEL_74;
          v6 = *(_QWORD *)(v47 + 24);
          v77 = *(unsigned __int8 *)(v6 + 8);
          if ( (unsigned __int8)v77 <= 0xFu )
          {
            v105 = 35454;
            if ( _bittest64(&v105, v77) )
              goto LABEL_217;
          }
          if ( (unsigned int)(v77 - 13) > 1 && (_DWORD)v77 != 16 )
          {
LABEL_74:
            if ( (_BYTE)v9 == 15 )
              return sub_1599A20(a3);
            return 0;
          }
          a2 = 0;
          if ( (unsigned __int8)sub_16435F0(v6, 0) )
          {
LABEL_217:
            v106 = *a3;
            v141 = v143;
            v142 = 0x800000001LL;
            v107 = sub_1643350(v106);
            v108 = sub_15A06D0(v107);
            v143[0] = v108;
            v109 = *(_QWORD *)(v47 + 24);
            v110 = v109;
            if ( v109 != a3[3] )
            {
              do
              {
                v111 = *(_BYTE *)(v110 + 8);
                if ( v111 == 13 )
                {
                  a2 = *(unsigned int *)(v110 + 12);
                  if ( !(_DWORD)a2 )
                    goto LABEL_238;
                  v112 = (unsigned int)v142;
                  v110 = **(_QWORD **)(v110 + 16);
                  if ( (unsigned int)v142 >= HIDWORD(v142) )
                  {
                    a2 = (unsigned __int64)v143;
                    sub_16CD150(&v141, v143, 0, 8);
                    v112 = (unsigned int)v142;
                  }
                  v8 = (unsigned __int64)v141;
                  v141[v112] = v108;
                  LODWORD(v142) = v142 + 1;
                }
                else
                {
                  if ( ((v111 - 14) & 0xFD) != 0 )
                  {
LABEL_238:
                    v6 = (unsigned __int64)v141;
                    if ( v141 != v143 )
                      _libc_free((unsigned __int64)v141);
                    goto LABEL_134;
                  }
                  v110 = *(_QWORD *)(v110 + 24);
                  if ( (unsigned int)v142 >= HIDWORD(v142) )
                  {
                    a2 = (unsigned __int64)v143;
                    sub_16CD150(&v141, v143, 0, 8);
                  }
                  v8 = (unsigned int)v142;
                  v141[(unsigned int)v142] = v108;
                  LODWORD(v142) = v142 + 1;
                }
              }
              while ( v110 != a3[3] );
              v109 = *(_QWORD *)(v47 + 24);
            }
            BYTE4(v139) = 0;
            v113 = sub_15A2E80(v109, v138, (_DWORD)v141, v142, 1, (unsigned int)&v139, 0);
            v29 = (unsigned __int64)v141;
            v15 = v113;
            if ( v141 == v143 )
              return v15;
            goto LABEL_32;
          }
LABEL_134:
          v7 = v138;
          LOBYTE(v9) = *(_BYTE *)(v138 + 16);
LABEL_135:
          v48 = *((unsigned __int8 *)a3 + 8);
          if ( (_BYTE)v48 == 16 )
          {
            if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 )
            {
              if ( (_BYTE)v9 == 10 )
                return sub_15A06D0(a3);
              if ( (unsigned __int8)sub_1596070(v7) )
                return sub_15A04A0(a3);
              if ( (unsigned __int8)sub_1593BB0(v7) )
                return sub_15A06D0(a3);
              v15 = 0;
              v137 = a3[4];
              if ( *(_DWORD *)(*(_QWORD *)v7 + 32LL) != (_DWORD)v137 )
                return v15;
              v96 = a3[3];
              v141 = v143;
              v142 = 0x1000000000LL;
              v97 = sub_16498A0(v7);
              v98 = sub_1644900(v97, 32);
              if ( (_DWORD)v137 )
              {
                do
                {
                  v99 = sub_15A0680(v98, v15, 0);
                  v100 = sub_15A37D0(v7, v99, 0);
                  v101 = sub_15A4510(v100, v96, 0);
                  v102 = (unsigned int)v142;
                  if ( (unsigned int)v142 >= HIDWORD(v142) )
                  {
                    v126 = v101;
                    sub_16CD150(&v141, v143, 0, 8);
                    v102 = (unsigned int)v142;
                    v101 = v126;
                  }
                  ++v15;
                  v141[v102] = v101;
                  LODWORD(v142) = v142 + 1;
                }
                while ( (unsigned int)v137 != v15 );
              }
              v27 = (unsigned int)v142;
LABEL_31:
              v28 = sub_15A01B0(v141, v27);
              v29 = (unsigned __int64)v141;
              v15 = v28;
              if ( v141 == v143 )
                return v15;
LABEL_32:
              _libc_free(v29);
              return v15;
            }
LABEL_20:
            if ( (unsigned __int8)(v9 - 13) <= 1u )
            {
              v95 = sub_15A01B0(&v138, 1);
              return sub_15A4510(v95, a3, 0);
            }
            if ( *(_BYTE *)(v138 + 16) == 15 )
              return sub_1599A20(a3);
            return 0;
          }
        }
        else if ( (_BYTE)v48 == 16 )
        {
          goto LABEL_20;
        }
        if ( (_BYTE)v9 == 15 )
          return sub_1599A20(a3);
        if ( (_BYTE)v9 == 13 )
        {
          if ( (_BYTE)v48 == 11 )
            return v7;
          if ( (unsigned __int8)(v48 - 1) <= 4u )
          {
            v84 = v7 + 24;
            switch ( (char)v48 )
            {
              case 1:
                v87 = sub_1698260(v6, a2, v48, v8);
                goto LABEL_174;
              case 2:
                v87 = sub_1698270(v6, a2);
                goto LABEL_174;
              case 3:
                v87 = sub_1698280(v6);
                goto LABEL_174;
              case 4:
                v87 = sub_16982A0();
                goto LABEL_174;
              case 5:
                v87 = sub_1698290();
LABEL_174:
                v88 = sub_16982C0(v6, a2, v85, v86);
                if ( v88 == v87 )
                  sub_169D060(&v142, v87, v84);
                else
                  sub_169D050(&v142, v87, v84);
                v15 = sub_159CCF0(*a3, &v141);
                if ( v142 != v88 )
                  goto LABEL_51;
                v66 = v143[0];
                if ( !v143[0] )
                  return v15;
                v89 = v143[0] + 32LL * *(_QWORD *)(v143[0] - 8LL);
                if ( v143[0] != v89 )
                {
                  do
                  {
                    v89 -= 32;
                    if ( v88 == *(_QWORD *)(v89 + 8) )
                    {
                      v90 = *(_QWORD *)(v89 + 16);
                      v136 = v90;
                      if ( v90 )
                      {
                        v91 = 32LL * *(_QWORD *)(v90 - 8);
                        v92 = v90 + v91;
                        if ( v90 != v90 + v91 )
                        {
                          do
                          {
                            v92 -= 32;
                            if ( v88 == *(_QWORD *)(v92 + 8) )
                            {
                              v93 = *(_QWORD *)(v92 + 16);
                              if ( v93 )
                              {
                                v94 = v93 + 32LL * *(_QWORD *)(v93 - 8);
                                if ( v93 != v94 )
                                {
                                  do
                                  {
                                    v125 = v93;
                                    v129 = v94 - 32;
                                    sub_127D120((_QWORD *)(v94 - 24));
                                    v94 = v129;
                                    v93 = v125;
                                  }
                                  while ( v125 != v129 );
                                }
                                j_j_j___libc_free_0_0(v93 - 8);
                              }
                            }
                            else
                            {
                              sub_1698460(v92 + 8);
                            }
                          }
                          while ( v136 != v92 );
                        }
                        j_j_j___libc_free_0_0(v136 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v89 + 8);
                    }
                  }
                  while ( v66 != v89 );
                }
                break;
              default:
                goto LABEL_248;
            }
LABEL_147:
            j_j_j___libc_free_0_0(v66 - 8);
            return v15;
          }
        }
        else if ( (_BYTE)v9 == 14 && *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 6 && (_BYTE)v48 == 11 )
        {
          v103 = sub_16982C0(v6, a2, v48, v8);
          v104 = v7 + 32;
          if ( *(_QWORD *)(v7 + 32) == v103 )
            sub_169D930(&v141, v104);
          else
            sub_169D7E0(&v141, v104);
LABEL_60:
          v45 = sub_16498A0(v7);
          v15 = sub_159C0E0(v45, &v141);
LABEL_61:
          if ( (unsigned int)v142 > 0x40 )
          {
LABEL_62:
            if ( v141 )
              j_j___libc_free_0_0(v141);
          }
          return v15;
        }
        return 0;
      case '0':
        return 0;
    }
  }
  if ( ((a1 - 37) & 0xFFFFFFFA) != 0 )
    return sub_1599EF0(a3);
  else
    return sub_15A06D0(a3);
}
