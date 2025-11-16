// Function: sub_29BDD80
// Address: 0x29bdd80
//
__int64 __fastcall sub_29BDD80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int8 a6)
{
  unsigned int v6; // r11d
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v11; // rax
  __int64 v12; // r15
  int v13; // eax
  char v15; // al
  unsigned __int8 v16; // r9
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  char v21; // al
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  _BYTE **v25; // rcx
  _BYTE **v26; // r14
  _BYTE *v27; // rsi
  char v28; // al
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 v31; // rdx
  int v32; // r14d
  __int64 v33; // r15
  unsigned int v34; // eax
  __int64 v35; // rbx
  __int64 v36; // rax
  __int64 v37; // r11
  __int64 *v38; // rax
  int v39; // r12d
  __int64 *v40; // rcx
  __int64 *v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rdi
  unsigned int v45; // edx
  unsigned int v46; // r9d
  __int64 v47; // rsi
  __int64 v48; // rdi
  unsigned int v49; // edx
  __int64 *v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rdx
  unsigned __int8 v53; // r14
  int v54; // eax
  __int64 v55; // r9
  unsigned __int8 v56; // r15
  __int64 v57; // r14
  __int64 v58; // r13
  __int64 v59; // r12
  __int64 *v60; // rax
  __int64 *v61; // rsi
  __int64 *v62; // rax
  __int64 *v63; // rax
  __int64 *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 *v67; // rax
  __int64 v68; // r12
  __int64 v69; // r8
  __int64 v70; // r13
  unsigned int v71; // eax
  unsigned __int64 v72; // rax
  unsigned __int64 *v73; // r14
  __int64 v74; // rdx
  __int64 *v75; // r12
  unsigned __int8 *v76; // r15
  _BYTE **v77; // r12
  __int64 v78; // rdx
  _BYTE **v79; // r15
  _BYTE *v80; // rcx
  int v81; // eax
  unsigned int v82; // r13d
  int v83; // r14d
  __int64 v84; // rsi
  __int64 v85; // r9
  __int64 *v86; // rax
  char v87; // al
  unsigned __int64 v88; // rdi
  void (*v89)(void); // rdx
  unsigned __int64 *v90; // rax
  unsigned __int8 **v91; // r14
  unsigned __int8 *v92; // r12
  unsigned __int64 *v93; // r15
  __int64 v94; // rdi
  char v95; // al
  _QWORD *v96; // r8
  char v97; // al
  unsigned __int64 *v98; // rax
  int v99; // eax
  __int64 v100; // rax
  __int64 v101; // r13
  unsigned __int8 v102; // r12
  __int64 v103; // r14
  unsigned int v104; // ebx
  __int64 v105; // rsi
  __int64 v106; // r8
  __int64 v107; // r9
  __int64 *v108; // rax
  __int64 v109; // rax
  __int64 *v110; // rax
  unsigned int v111; // eax
  __int64 v112; // [rsp-320h] [rbp-320h]
  __int64 v113; // [rsp-320h] [rbp-320h]
  __int64 v114; // [rsp-320h] [rbp-320h]
  __int64 v115; // [rsp-318h] [rbp-318h]
  __int64 v116; // [rsp-318h] [rbp-318h]
  _BYTE *v117; // [rsp-310h] [rbp-310h]
  __int64 v118; // [rsp-310h] [rbp-310h]
  __int64 v119; // [rsp-308h] [rbp-308h]
  unsigned __int8 v120; // [rsp-308h] [rbp-308h]
  unsigned __int8 v121; // [rsp-308h] [rbp-308h]
  unsigned __int8 v122; // [rsp-2F9h] [rbp-2F9h]
  unsigned __int8 v124; // [rsp-2F8h] [rbp-2F8h]
  int v125; // [rsp-2F8h] [rbp-2F8h]
  _BYTE **v126; // [rsp-2F0h] [rbp-2F0h]
  __int64 v127; // [rsp-2F0h] [rbp-2F0h]
  unsigned __int8 v128; // [rsp-2F0h] [rbp-2F0h]
  char v129; // [rsp-2F0h] [rbp-2F0h]
  __int64 v130; // [rsp-2E8h] [rbp-2E8h] BYREF
  __int64 *v131; // [rsp-2E0h] [rbp-2E0h]
  __int64 v132; // [rsp-2D8h] [rbp-2D8h]
  int v133; // [rsp-2D0h] [rbp-2D0h]
  char v134; // [rsp-2CCh] [rbp-2CCh]
  __int64 v135; // [rsp-2C8h] [rbp-2C8h] BYREF
  __int64 v136; // [rsp-248h] [rbp-248h] BYREF
  __int64 *v137; // [rsp-240h] [rbp-240h]
  __int64 v138; // [rsp-238h] [rbp-238h] BYREF
  __int64 v139; // [rsp-230h] [rbp-230h]
  _QWORD v140[69]; // [rsp-228h] [rbp-228h] BYREF

  if ( a4 )
  {
    v7 = a5;
    v8 = a2;
    v9 = (__int64)a1;
    if ( a5 == 0 || a1 == (_QWORD *)a2 )
      return 0;
    v11 = a1[4];
    v12 = a3;
    LOBYTE(v6) = v11 != 0 && v11 != a1[5] + 48LL;
    if ( (_BYTE)v6 && a2 == v11 - 24 )
      return v6;
    v13 = *(unsigned __int8 *)a1;
    if ( (_BYTE)v13 == 84 )
      return 0;
    if ( *(_BYTE *)a2 == 84 )
      return 0;
    if ( (unsigned int)(v13 - 30) <= 0xA )
      return 0;
    v122 = sub_29BD9B0((__int64)a1, a2, a3, a4);
    if ( !v122 )
      return 0;
    v15 = sub_29BDD50((__int64)a1, a2, v12, a4);
    v16 = a6;
    if ( v15 && a1[2] )
    {
      v17 = a1[2];
      do
      {
        v18 = *(_QWORD *)(v17 + 24);
        if ( *(_BYTE *)v18 > 0x1Cu )
        {
          v19 = a1[5];
          if ( *(_QWORD *)(a2 + 40) == v19 )
          {
            v20 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v20 != v19 + 48 )
            {
              if ( !v20 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 <= 0xA && v18 == v20 - 24 )
                return 0;
            }
          }
          if ( a2 != v18
            && !(unsigned __int8)sub_B19F20(v12, (char *)a2, v17)
            && (!a6 || *(_QWORD *)(v18 + 40) != a1[5] || !(unsigned __int8)sub_B19DB0(v12, (__int64)a1, v18)) )
          {
            return 0;
          }
        }
        v17 = *(_QWORD *)(v17 + 8);
      }
      while ( v17 );
      v9 = (__int64)a1;
      v8 = a2;
      v16 = a6;
    }
    v124 = v16;
    v21 = sub_29BDD50(v8, v9, v12, a4);
    v22 = 0;
    v23 = v124;
    if ( v21 )
    {
      v24 = 4LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
      {
        v25 = *(_BYTE ***)(v9 - 8);
        v126 = &v25[v24];
      }
      else
      {
        v126 = (_BYTE **)v9;
        v25 = (_BYTE **)(v9 - v24 * 8);
      }
      v26 = v25;
      if ( v25 != v126 )
      {
        do
        {
          v27 = *v26;
          if ( **v26 > 0x1Cu )
          {
            if ( (_BYTE *)v8 == v27 )
              return 0;
            if ( !v124
              || *((_QWORD *)v27 + 5) != *(_QWORD *)(v9 + 40)
              || (v117 = *v26, v28 = sub_B19DB0(v12, (__int64)v27, v9), v27 = v117, !v28) )
            {
              if ( !(unsigned __int8)sub_B19DB0(v12, (__int64)v27, v8) )
                return 0;
            }
          }
          v26 += 4;
        }
        while ( v126 != v26 );
        v22 = 0;
      }
    }
    if ( *(_BYTE *)(v12 + 112) )
    {
      *(_DWORD *)(v12 + 116) = 0;
    }
    else
    {
      v29 = *(_QWORD *)(v12 + 96);
      v30 = &v138;
      HIDWORD(v137) = 32;
      v136 = (__int64)&v138;
      if ( v29 )
      {
        v31 = *(_QWORD *)(v29 + 24);
        v138 = v29;
        LODWORD(v137) = 1;
        v32 = 1;
        v139 = v31;
        v118 = v12;
        v33 = v9;
        *(_DWORD *)(v29 + 72) = 0;
        v34 = 1;
        v119 = v8;
        do
        {
          v39 = v32++;
          v40 = &v30[2 * v34 - 2];
          v41 = (__int64 *)v40[1];
          if ( v41 == (__int64 *)(*(_QWORD *)(*v40 + 24) + 8LL * *(unsigned int *)(*v40 + 32)) )
          {
            --v34;
            *(_DWORD *)(*v40 + 76) = v39;
            LODWORD(v137) = v34;
          }
          else
          {
            v35 = *v41;
            v40[1] = (__int64)(v41 + 1);
            v36 = (unsigned int)v137;
            v37 = *(_QWORD *)(v35 + 24);
            if ( (unsigned __int64)(unsigned int)v137 + 1 > HIDWORD(v137) )
            {
              v114 = *(_QWORD *)(v35 + 24);
              sub_C8D5F0((__int64)&v136, &v138, (unsigned int)v137 + 1LL, 0x10u, v22, v23);
              v30 = (__int64 *)v136;
              v36 = (unsigned int)v137;
              v37 = v114;
            }
            v38 = &v30[2 * v36];
            *v38 = v35;
            v38[1] = v37;
            v34 = (_DWORD)v137 + 1;
            LODWORD(v137) = (_DWORD)v137 + 1;
            *(_DWORD *)(v35 + 72) = v39;
            v30 = (__int64 *)v136;
          }
        }
        while ( v34 );
        v9 = v33;
        v12 = v118;
        v22 = 0;
        v8 = v119;
        *(_DWORD *)(v118 + 116) = 0;
        *(_BYTE *)(v118 + 112) = 1;
        if ( v30 != &v138 )
        {
          _libc_free((unsigned __int64)v30);
          v22 = 0;
        }
      }
    }
    v42 = *(_QWORD *)(v9 + 40);
    v43 = *(_QWORD *)(v8 + 40);
    if ( v42 == v43 )
    {
      LOBYTE(v111) = sub_B445A0(v9, v8);
      v22 = v111;
      if ( (_BYTE)v111 )
      {
        v127 = v8;
        v43 = *(_QWORD *)(v9 + 40);
        v50 = (__int64 *)v9;
      }
      else
      {
        v127 = v9;
        v43 = *(_QWORD *)(v8 + 40);
        v50 = (__int64 *)v8;
      }
    }
    else
    {
      if ( v42 )
      {
        v44 = (unsigned int)(*(_DWORD *)(v42 + 44) + 1);
        v45 = *(_DWORD *)(v42 + 44) + 1;
      }
      else
      {
        v44 = 0;
        v45 = 0;
      }
      v46 = *(_DWORD *)(v12 + 32);
      v47 = 0;
      if ( v45 < v46 )
        v47 = *(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v44);
      if ( v43 )
      {
        v48 = (unsigned int)(*(_DWORD *)(v43 + 44) + 1);
        v49 = *(_DWORD *)(v43 + 44) + 1;
      }
      else
      {
        v48 = 0;
        v49 = 0;
      }
      if ( v46 <= v49 )
        BUG();
      if ( *(_DWORD *)(v47 + 16) >= *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 24) + 8 * v48) + 16LL) )
      {
        v127 = v9;
        v50 = (__int64 *)v8;
      }
      else
      {
        v127 = v8;
        v22 = v122;
        v43 = *(_QWORD *)(v9 + 40);
        v50 = (__int64 *)v9;
      }
    }
    v51 = v50[4];
    v130 = 0;
    v131 = &v135;
    v52 = (__int64)v140;
    v137 = v140;
    v132 = 16;
    v133 = 0;
    v134 = 1;
    LOBYTE(v52) = v51 != v43 + 48 && v51 != 0;
    v136 = 0;
    v138 = 16;
    v53 = v52;
    LODWORD(v139) = 0;
    BYTE4(v139) = 1;
    if ( (_BYTE)v52 )
    {
      v54 = 0;
      v55 = 1;
      HIDWORD(v138) = 1;
      v140[0] = v51 - 24;
      v136 = 1;
      goto LABEL_65;
    }
    v121 = v22;
    v113 = (__int64)v50;
    v99 = sub_B46E30((__int64)v50);
    v53 = v122;
    v22 = v121;
    v125 = v99;
    if ( v99 )
    {
      v100 = v7;
      v101 = v8;
      v102 = v122;
      v116 = v9;
      v103 = v100;
      v104 = 0;
      while ( 1 )
      {
        v105 = *(_QWORD *)(sub_B46EC0(v113, v104) + 56);
        if ( v105 )
          v105 -= 24;
        if ( !v102 )
          goto LABEL_185;
        v108 = v137;
        v52 = (__int64)&v137[HIDWORD(v138)];
        if ( v137 != (__int64 *)v52 )
        {
          while ( v105 != *v108 )
          {
            if ( (__int64 *)v52 == ++v108 )
              goto LABEL_186;
          }
          goto LABEL_183;
        }
LABEL_186:
        if ( HIDWORD(v138) < (unsigned int)v138 )
        {
          ++HIDWORD(v138);
          *(_QWORD *)v52 = v105;
          v102 = BYTE4(v139);
          ++v136;
        }
        else
        {
LABEL_185:
          sub_C8CC70((__int64)&v136, v105, v52, (__int64)v50, v106, v107);
          v102 = BYTE4(v139);
        }
LABEL_183:
        if ( v125 == ++v104 )
        {
          v109 = v103;
          v22 = v121;
          v53 = v102;
          v8 = v101;
          v9 = v116;
          v7 = v109;
          v55 = HIDWORD(v138);
          v54 = v139;
          goto LABEL_65;
        }
      }
    }
    v54 = 0;
    v55 = 0;
LABEL_65:
    v120 = v22;
    v56 = v53;
    v57 = v7;
    v58 = v8;
    while ( 1 )
    {
      if ( v54 == (_DWORD)v55 )
      {
        v68 = v58;
        v69 = v120;
        v70 = v57;
        if ( !v56 )
        {
          _libc_free((unsigned __int64)v137);
          v69 = v120;
        }
        if ( (_BYTE)v69 )
          goto LABEL_107;
        if ( v134 )
        {
          v110 = v131;
          v52 = HIDWORD(v132);
          v50 = &v131[HIDWORD(v132)];
          if ( v131 != v50 )
          {
            while ( v68 != *v110 )
            {
              if ( v50 == ++v110 )
                goto LABEL_193;
            }
LABEL_107:
            LOBYTE(v71) = sub_991A70((unsigned __int8 *)v9, 0, 0, 0, 0, 1u, 0);
            v6 = v71;
            v72 = (unsigned __int64)v131;
            if ( !(_BYTE)v6 )
            {
              v73 = (unsigned __int64 *)v131;
              v74 = v134 ? HIDWORD(v132) : (unsigned int)v132;
              v75 = &v131[v74];
              if ( v75 != v131 )
              {
                while ( 1 )
                {
                  v76 = (unsigned __int8 *)*v73;
                  if ( *v73 < 0xFFFFFFFFFFFFFFFELL )
                    break;
                  if ( v75 == (__int64 *)++v73 )
                    goto LABEL_113;
                }
                if ( v75 != (__int64 *)v73 )
                {
                  v90 = v73;
                  v91 = (unsigned __int8 **)&v131[v74];
                  v92 = v76;
                  v93 = v90;
                  while ( !(unsigned __int8)sub_B46790(v92, 0) )
                  {
                    if ( (unsigned __int8)(*v92 - 34) <= 0x33u )
                    {
                      v94 = 0x8000000000041LL;
                      if ( _bittest64(&v94, (unsigned int)*v92 - 34) )
                      {
                        v95 = sub_A73ED0((_QWORD *)v92 + 9, 76);
                        v96 = v92 + 72;
                        if ( !v95 )
                        {
                          v97 = sub_B49560((__int64)v92, 76);
                          v96 = v92 + 72;
                          if ( !v97 )
                            break;
                        }
                        if ( !(unsigned __int8)sub_A73ED0(v96, 39) && !(unsigned __int8)sub_B49560((__int64)v92, 39) )
                          break;
                      }
                    }
                    v98 = v93 + 1;
                    if ( v91 != (unsigned __int8 **)(v93 + 1) )
                    {
                      while ( 1 )
                      {
                        v92 = (unsigned __int8 *)*v98;
                        v93 = v98;
                        if ( *v98 < 0xFFFFFFFFFFFFFFFELL )
                          break;
                        if ( v91 == (unsigned __int8 **)++v98 )
                          goto LABEL_162;
                      }
                      if ( v91 != (unsigned __int8 **)v98 )
                        continue;
                    }
                    goto LABEL_162;
                  }
                  v6 = 0;
                  if ( v91 != (unsigned __int8 **)v93 )
                    goto LABEL_119;
LABEL_162:
                  v72 = (unsigned __int64)v131;
                }
              }
            }
LABEL_113:
            v77 = (_BYTE **)v72;
            if ( v134 )
              v78 = HIDWORD(v132);
            else
              v78 = (unsigned int)v132;
            v79 = (_BYTE **)(v72 + 8 * v78);
            if ( (_BYTE **)v72 != v79 )
            {
              while ( 1 )
              {
                v80 = *v77;
                if ( (unsigned __int64)*v77 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v79 == ++v77 )
                  goto LABEL_118;
              }
              if ( v79 != v77 )
              {
LABEL_136:
                sub_2297CA0(&v136, v70, v9, v80);
                if ( !v136 )
                  goto LABEL_146;
                if ( !(unsigned __int8)sub_228CC50(v136) && !(unsigned __int8)sub_228CC90(v136) )
                {
                  v87 = sub_228CCD0(v136);
                  if ( !v87 )
                  {
                    v88 = v136;
                    if ( !v136 )
                      goto LABEL_146;
                    goto LABEL_141;
                  }
                }
                v88 = v136;
                if ( v136 )
                {
                  v87 = v122;
LABEL_141:
                  v129 = v87;
                  v89 = *(void (**)(void))(*(_QWORD *)v88 + 8LL);
                  if ( (char *)v89 == (char *)sub_228A6E0 )
                    j_j___libc_free_0(v88);
                  else
                    v89();
                  if ( !v129 )
                  {
LABEL_146:
                    while ( v79 != ++v77 )
                    {
                      v80 = *v77;
                      if ( (unsigned __int64)*v77 < 0xFFFFFFFFFFFFFFFELL )
                      {
                        if ( v77 != v79 )
                          goto LABEL_136;
                        break;
                      }
                    }
                  }
                }
              }
            }
LABEL_118:
            LOBYTE(v6) = v77 == v79;
LABEL_119:
            if ( !v134 )
            {
              v128 = v6;
              _libc_free((unsigned __int64)v131);
              return v128;
            }
            return v6;
          }
LABEL_193:
          if ( HIDWORD(v132) < (unsigned int)v132 )
          {
            ++HIDWORD(v132);
            *v50 = v68;
            ++v130;
            goto LABEL_107;
          }
        }
        sub_C8CC70((__int64)&v130, v68, v52, (__int64)v50, v69, v55);
        goto LABEL_107;
      }
      v59 = *v137;
      if ( !v56 )
        break;
      v52 = (__int64)&v137[(unsigned int)v55];
      if ( v137 != (__int64 *)v52 )
        goto LABEL_69;
LABEL_79:
      if ( v127 == v59 )
      {
        v55 = HIDWORD(v138);
        v54 = v139;
        v56 = BYTE4(v139);
      }
      else
      {
        if ( !v134 )
          goto LABEL_91;
        v63 = v131;
        v52 = (__int64)&v131[HIDWORD(v132)];
        if ( v131 != (__int64 *)v52 )
        {
          while ( *v63 != v59 )
          {
            if ( (__int64 *)v52 == ++v63 )
              goto LABEL_101;
          }
          v56 = BYTE4(v139);
          goto LABEL_86;
        }
LABEL_101:
        if ( HIDWORD(v132) < (unsigned int)v132 )
        {
          ++HIDWORD(v132);
          *(_QWORD *)v52 = v59;
          v56 = BYTE4(v139);
          ++v130;
        }
        else
        {
LABEL_91:
          sub_C8CC70((__int64)&v130, v59, v52, (__int64)v50, v22, v55);
          v56 = BYTE4(v139);
          if ( !(_BYTE)v52 )
            goto LABEL_86;
        }
        v65 = *(_QWORD *)(v59 + 32);
        if ( v65 == *(_QWORD *)(v59 + 40) + 48LL || !v65 )
        {
          v81 = sub_B46E30(v59);
          if ( v81 )
          {
            v115 = v58;
            v82 = 0;
            v112 = v57;
            v83 = v81;
            while ( 1 )
            {
              v84 = *(_QWORD *)(sub_B46EC0(v59, v82) + 56);
              if ( v84 )
                v84 -= 24;
              if ( !v56 )
                goto LABEL_132;
              v86 = v137;
              v52 = (__int64)&v137[HIDWORD(v138)];
              if ( v137 != (__int64 *)v52 )
              {
                while ( v84 != *v86 )
                {
                  if ( (__int64 *)v52 == ++v86 )
                    goto LABEL_133;
                }
                goto LABEL_130;
              }
LABEL_133:
              if ( HIDWORD(v138) < (unsigned int)v138 )
              {
                ++HIDWORD(v138);
                *(_QWORD *)v52 = v84;
                v56 = BYTE4(v139);
                ++v136;
              }
              else
              {
LABEL_132:
                sub_C8CC70((__int64)&v136, v84, v52, (__int64)v50, v22, v85);
                v56 = BYTE4(v139);
              }
LABEL_130:
              if ( ++v82 == v83 )
              {
                v58 = v115;
                v57 = v112;
                v55 = HIDWORD(v138);
                goto LABEL_87;
              }
            }
          }
LABEL_86:
          v55 = HIDWORD(v138);
          goto LABEL_87;
        }
        v66 = v65 - 24;
        if ( !v56 )
          goto LABEL_103;
        v67 = v137;
        v55 = HIDWORD(v138);
        v52 = (__int64)&v137[HIDWORD(v138)];
        if ( v137 == (__int64 *)v52 )
        {
LABEL_98:
          if ( HIDWORD(v138) < (unsigned int)v138 )
          {
            ++HIDWORD(v138);
            *(_QWORD *)v52 = v66;
            v55 = HIDWORD(v138);
            ++v136;
            v56 = BYTE4(v139);
            goto LABEL_87;
          }
LABEL_103:
          sub_C8CC70((__int64)&v136, v66, v52, (__int64)v50, v22, v55);
          v56 = BYTE4(v139);
          v55 = HIDWORD(v138);
          goto LABEL_87;
        }
        while ( v66 != *v67 )
        {
          if ( (__int64 *)v52 == ++v67 )
            goto LABEL_98;
        }
LABEL_87:
        v54 = v139;
      }
    }
    v52 = (__int64)&v137[(unsigned int)v138];
    if ( v137 != (__int64 *)v52 )
    {
LABEL_69:
      v60 = v137;
      while ( 1 )
      {
        v59 = *v60;
        v61 = v60;
        if ( (unsigned __int64)*v60 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( (__int64 *)v52 == ++v60 )
        {
          v59 = v61[1];
          break;
        }
      }
      if ( v56 )
      {
        v52 = (__int64)&v137[(unsigned int)v55];
        if ( v137 != (__int64 *)v52 )
        {
          v62 = v137;
          while ( v59 != *v62 )
          {
            if ( (__int64 *)v52 == ++v62 )
              goto LABEL_79;
          }
          v55 = (unsigned int)(v55 - 1);
          HIDWORD(v138) = v55;
          v52 = v137[v55];
          *v62 = v52;
          ++v136;
        }
        goto LABEL_79;
      }
    }
    v64 = sub_C8CA60((__int64)&v136, v59);
    if ( v64 )
    {
      *v64 = -2;
      LODWORD(v139) = v139 + 1;
      ++v136;
    }
    goto LABEL_79;
  }
  return 0;
}
