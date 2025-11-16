// Function: sub_19141C0
// Address: 0x19141c0
//
__int64 __fastcall sub_19141C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r12
  int *v5; // r15
  unsigned int v6; // ebx
  unsigned __int64 v7; // rdi
  int *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rdx
  int *v11; // r13
  int *v12; // rcx
  unsigned int v13; // edx
  int *v14; // rax
  _BOOL4 v15; // r10d
  __int64 v16; // rax
  int v17; // r15d
  __int64 v18; // r13
  int v19; // r14d
  __int64 v20; // r15
  __int64 v21; // rbx
  _BYTE *v22; // rdi
  _BYTE *v23; // r12
  _QWORD **v24; // rbx
  _QWORD *v25; // rdi
  __int64 v27; // r8
  int *v28; // rdx
  int *v29; // r9
  unsigned __int64 v30; // rcx
  int *v31; // rax
  __int64 v32; // rdi
  __int64 v33; // rsi
  int *v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 v39; // rax
  int *v40; // rdi
  __int64 v41; // rcx
  __int64 v42; // rdx
  _BYTE *v43; // rsi
  _QWORD ***v44; // rbx
  char v45; // al
  char v46; // r14
  unsigned int v47; // r13d
  int v48; // r12d
  char v49; // al
  _QWORD *v50; // rax
  __int64 v51; // r12
  unsigned __int64 **v52; // rbx
  unsigned __int64 *v53; // rsi
  __int64 v54; // r15
  __int64 v55; // rax
  int v56; // r8d
  int v57; // r9d
  __int64 v58; // r12
  __int64 v59; // rax
  __int64 v60; // rax
  int v61; // r8d
  int v62; // r9d
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 *v65; // r14
  __int64 v66; // r13
  unsigned int v67; // r8d
  __int64 v68; // rax
  _QWORD *v69; // r12
  __int64 v70; // rax
  __int64 *v71; // rax
  __int64 *v72; // rax
  int v73; // r8d
  __int64 v74; // rcx
  __int64 *v75; // r10
  __int64 *v76; // rsi
  __int64 *v77; // rax
  __int64 v78; // rdx
  __int64 *v79; // rax
  __int64 v80; // rsi
  __int64 *v81; // r13
  unsigned int v82; // r13d
  _QWORD *v83; // rax
  __int64 v84; // r14
  __int64 *v85; // r12
  __int64 v86; // rsi
  unsigned __int64 *v87; // rdi
  __int64 v88; // rsi
  unsigned __int8 *v89; // rsi
  __int64 v90; // rsi
  unsigned __int8 *v91; // rsi
  __int64 v92; // rsi
  unsigned __int8 *v93; // rsi
  __int64 *v94; // rax
  unsigned int *v95; // r14
  __int64 v96; // r12
  __int64 v97; // rax
  int v98; // r8d
  int v99; // r9d
  __int64 v100; // r15
  __int64 v101; // rax
  unsigned int *v102; // r12
  unsigned int v103; // eax
  __int64 v104; // r15
  __int64 v105; // rax
  int v106; // r8d
  int v107; // r9d
  __int64 v108; // r15
  __int64 v109; // rax
  __int64 v110; // r15
  __int64 *v111; // r14
  __int64 v112; // rsi
  __int64 v113; // rax
  _QWORD *v114; // rax
  _QWORD *v115; // r12
  __int64 v116; // rax
  __int64 *v117; // rax
  __int64 *v118; // rdi
  __int64 *v119; // rcx
  __int64 *v120; // rax
  __int64 v121; // rdx
  __int64 *v122; // r14
  __int64 v123; // rsi
  __int64 v124; // rsi
  unsigned __int8 *v125; // rsi
  unsigned int v126; // r15d
  _QWORD *v127; // rax
  __int64 v128; // r14
  __int64 *v129; // r12
  __int64 v130; // rsi
  __int64 v131; // rsi
  unsigned __int8 *v132; // rsi
  __int64 v133; // [rsp+8h] [rbp-178h]
  int v134; // [rsp+1Ch] [rbp-164h]
  __int64 v135; // [rsp+20h] [rbp-160h]
  __int64 v136; // [rsp+20h] [rbp-160h]
  _QWORD *v137; // [rsp+28h] [rbp-158h]
  int v138; // [rsp+30h] [rbp-150h]
  int v139; // [rsp+30h] [rbp-150h]
  unsigned int v140; // [rsp+38h] [rbp-148h]
  int v141; // [rsp+38h] [rbp-148h]
  __int64 v142; // [rsp+38h] [rbp-148h]
  __int64 v144; // [rsp+50h] [rbp-130h]
  __int64 v145; // [rsp+50h] [rbp-130h]
  __int64 v146; // [rsp+58h] [rbp-128h]
  unsigned int v147; // [rsp+58h] [rbp-128h]
  __int64 v148; // [rsp+60h] [rbp-120h]
  __int64 v149; // [rsp+60h] [rbp-120h]
  int v150; // [rsp+68h] [rbp-118h]
  __int64 v151; // [rsp+70h] [rbp-110h]
  __int64 v152; // [rsp+78h] [rbp-108h]
  _BOOL4 v153; // [rsp+80h] [rbp-100h]
  __int64 v154; // [rsp+80h] [rbp-100h]
  __int64 v155; // [rsp+80h] [rbp-100h]
  __int64 v156; // [rsp+80h] [rbp-100h]
  int *v157; // [rsp+88h] [rbp-F8h]
  __int64 v158; // [rsp+88h] [rbp-F8h]
  __int64 v159; // [rsp+90h] [rbp-F0h]
  __int64 v160; // [rsp+98h] [rbp-E8h]
  __int64 v161; // [rsp+98h] [rbp-E8h]
  __int64 v162; // [rsp+A8h] [rbp-D8h] BYREF
  _BYTE *v163; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE *v164; // [rsp+B8h] [rbp-C8h]
  _BYTE *v165; // [rsp+C0h] [rbp-C0h]
  __int64 v166[2]; // [rsp+D0h] [rbp-B0h] BYREF
  char v167; // [rsp+E0h] [rbp-A0h]
  char v168; // [rsp+E1h] [rbp-9Fh]
  __int64 v169; // [rsp+F0h] [rbp-90h] BYREF
  int v170; // [rsp+F8h] [rbp-88h] BYREF
  int *v171; // [rsp+100h] [rbp-80h]
  int *v172; // [rsp+108h] [rbp-78h]
  int *v173; // [rsp+110h] [rbp-70h]
  __int64 v174; // [rsp+118h] [rbp-68h]
  unsigned __int8 *v175; // [rsp+120h] [rbp-60h] BYREF
  __int64 v176; // [rsp+128h] [rbp-58h]
  _BYTE v177[80]; // [rsp+130h] [rbp-50h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  v170 = 0;
  v171 = 0;
  v172 = &v170;
  v173 = &v170;
  v174 = 0;
  v152 = v2;
  v151 = a2 + 72;
  if ( v2 != a2 + 72 )
  {
    v3 = a1 + 152;
    while ( 1 )
    {
      if ( !v152 )
LABEL_206:
        BUG();
      if ( *(_QWORD *)(v152 + 24) != v152 + 16 )
        break;
LABEL_27:
      v152 = *(_QWORD *)(v152 + 8);
      if ( v151 == v152 )
        goto LABEL_28;
    }
    v4 = *(_QWORD *)(v152 + 24);
    while ( 1 )
    {
      if ( !v4 )
        BUG();
      if ( *(_BYTE *)(v4 - 8) != 54 )
        goto LABEL_26;
      v5 = &v170;
      v6 = sub_1911FD0(v3, *(_QWORD *)(v4 - 48));
      v7 = **(_QWORD **)(v4 - 48);
      v8 = v171;
      v166[0] = v7;
      if ( !v171 )
        goto LABEL_15;
      do
      {
        while ( 1 )
        {
          v9 = *((_QWORD *)v8 + 2);
          v10 = *((_QWORD *)v8 + 3);
          if ( *((_QWORD *)v8 + 4) >= v7 )
            break;
          v8 = (int *)*((_QWORD *)v8 + 3);
          if ( !v10 )
            goto LABEL_13;
        }
        v5 = v8;
        v8 = (int *)*((_QWORD *)v8 + 2);
      }
      while ( v9 );
LABEL_13:
      if ( v5 == &v170 || *((_QWORD *)v5 + 4) > v7 )
      {
LABEL_15:
        v175 = (unsigned __int8 *)v166;
        v5 = (int *)sub_190F830(&v169, v5, (unsigned __int64 **)&v175);
      }
      v11 = (int *)*((_QWORD *)v5 + 7);
      v12 = v5 + 12;
      if ( !v11 )
        break;
      while ( 1 )
      {
        v13 = v11[8];
        v14 = (int *)*((_QWORD *)v11 + 3);
        if ( v6 < v13 )
          v14 = (int *)*((_QWORD *)v11 + 2);
        if ( !v14 )
          break;
        v11 = v14;
      }
      if ( v6 < v13 )
      {
        if ( v11 == *((int **)v5 + 8) )
        {
LABEL_24:
          v15 = 1;
          if ( v12 != v11 )
            goto LABEL_47;
        }
        else
        {
LABEL_44:
          if ( v6 <= *(_DWORD *)(sub_220EF80(v11) + 32) )
            goto LABEL_26;
          v12 = v5 + 12;
          if ( !v11 )
            goto LABEL_26;
          v15 = 1;
          if ( v5 + 12 != v11 )
LABEL_47:
            v15 = v6 < v11[8];
        }
LABEL_25:
        v153 = v15;
        v157 = v12;
        v16 = sub_22077B0(40);
        *(_DWORD *)(v16 + 32) = v6;
        sub_220F040(v153, v16, v11, v157);
        ++*((_QWORD *)v5 + 10);
        goto LABEL_26;
      }
      if ( v6 > v13 )
        goto LABEL_24;
LABEL_26:
      v4 = *(_QWORD *)(v4 + 8);
      if ( v152 + 16 == v4 )
        goto LABEL_27;
    }
    v11 = v5 + 12;
    if ( v12 != *((int **)v5 + 8) )
      goto LABEL_44;
    v15 = 1;
    goto LABEL_25;
  }
LABEL_28:
  v17 = 0;
  v163 = 0;
  v164 = 0;
  v165 = 0;
  v158 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v159 = *(_QWORD *)(a2 + 80);
  if ( v151 == v159 )
    goto LABEL_50;
  do
  {
    if ( !v159 )
      goto LABEL_206;
    if ( v159 + 16 == *(_QWORD *)(v159 + 24) )
      goto LABEL_49;
    v18 = *(_QWORD *)(v159 + 24);
    v19 = v17;
    v20 = v159 + 16;
    do
    {
      if ( !v18 )
        BUG();
      if ( *(_BYTE *)(v18 - 8) == 55 )
      {
        v162 = v18 - 24;
        if ( (++v19 >= dword_4FAEB20 || dword_4FAEB20 == -1)
          && (v19 <= dword_4FAEA40 || dword_4FAEA40 == -1)
          && (*(_BYTE *)(v18 - 6) & 1) == 0 )
        {
          v21 = *(_QWORD *)(v18 - 72);
          if ( *(_BYTE *)(sub_14AD280(*(_QWORD *)(v18 - 48), v158, 6u) + 16) != 78 && *(_BYTE *)(v21 + 16) == 87 )
          {
            v27 = v162;
            v28 = v171;
            v29 = &v170;
            v30 = **(_QWORD **)(v162 - 24);
            v31 = v171;
            if ( !v171 )
              goto LABEL_79;
            do
            {
              while ( 1 )
              {
                v32 = *((_QWORD *)v31 + 2);
                v33 = *((_QWORD *)v31 + 3);
                if ( *((_QWORD *)v31 + 4) >= v30 )
                  break;
                v31 = (int *)*((_QWORD *)v31 + 3);
                if ( !v33 )
                  goto LABEL_63;
              }
              v29 = v31;
              v31 = (int *)*((_QWORD *)v31 + 2);
            }
            while ( v32 );
LABEL_63:
            if ( v29 == &v170 || *((_QWORD *)v29 + 4) > v30 )
              goto LABEL_79;
            v166[0] = **(_QWORD **)(v162 - 24);
            v34 = &v170;
            do
            {
              while ( 1 )
              {
                v35 = *((_QWORD *)v28 + 2);
                v36 = *((_QWORD *)v28 + 3);
                if ( *((_QWORD *)v28 + 4) >= v30 )
                  break;
                v28 = (int *)*((_QWORD *)v28 + 3);
                if ( !v36 )
                  goto LABEL_69;
              }
              v34 = v28;
              v28 = (int *)*((_QWORD *)v28 + 2);
            }
            while ( v35 );
LABEL_69:
            if ( v34 == &v170 || *((_QWORD *)v34 + 4) > v30 )
            {
              v175 = (unsigned __int8 *)v166;
              v37 = sub_190F830(&v169, v34, (unsigned __int64 **)&v175);
              v27 = v162;
              v34 = (int *)v37;
            }
            v38 = sub_1911FD0(a1 + 152, *(_QWORD *)(v27 - 24));
            v39 = *((_QWORD *)v34 + 7);
            if ( !v39 )
              goto LABEL_79;
            v40 = v34 + 12;
            do
            {
              while ( 1 )
              {
                v41 = *(_QWORD *)(v39 + 16);
                v42 = *(_QWORD *)(v39 + 24);
                if ( v38 <= *(_DWORD *)(v39 + 32) )
                  break;
                v39 = *(_QWORD *)(v39 + 24);
                if ( !v42 )
                  goto LABEL_77;
              }
              v40 = (int *)v39;
              v39 = *(_QWORD *)(v39 + 16);
            }
            while ( v41 );
LABEL_77:
            if ( v40 == v34 + 12 || v38 < v40[8] )
            {
LABEL_79:
              v43 = v164;
              if ( v164 == v165 )
              {
                sub_190D490((__int64)&v163, v164, &v162);
                v154 = v162;
              }
              else
              {
                v154 = v162;
                if ( v164 )
                {
                  *(_QWORD *)v164 = v162;
                  v43 = v164;
                }
                v164 = v43 + 8;
              }
              v44 = *(_QWORD ****)(v154 - 48);
              if ( *((_BYTE *)v44 + 16) != 87 )
              {
LABEL_84:
                v45 = *((_BYTE *)v44 + 16);
                if ( v45 == 9 )
                  goto LABEL_40;
                if ( v45 != 7 )
                  goto LABEL_92;
                v146 = (__int64)*v44;
                if ( !*((_DWORD *)*v44 + 3) )
                  goto LABEL_92;
                v150 = v19;
                v46 = 0;
                v148 = v18;
                v47 = 0;
                v48 = *((_DWORD *)*v44 + 3);
                do
                {
                  if ( *(_BYTE *)(sub_15A0A60((__int64)v44, v47) + 16) == 9 )
                    v46 = 1;
                  ++v47;
                }
                while ( v48 != v47 );
                v49 = v46;
                v18 = v148;
                v19 = v150;
                if ( !v49 )
                {
LABEL_92:
                  v160 = *(_QWORD *)(v154 - 24);
                  v50 = sub_1648A60(64, 2u);
                  v51 = (__int64)v50;
                  if ( v50 )
                    sub_15F9660((__int64)v50, (__int64)v44, v160, v154);
                  v52 = (unsigned __int64 **)(v51 + 48);
                  v53 = *(unsigned __int64 **)(v154 + 48);
                  v175 = (unsigned __int8 *)v53;
                  if ( v53 )
                  {
                    sub_1623A60((__int64)&v175, (__int64)v53, 2);
                    if ( v52 == (unsigned __int64 **)&v175 )
                    {
                      if ( v175 )
                        sub_161E7C0(v51 + 48, (__int64)v175);
                      goto LABEL_98;
                    }
                    v88 = *(_QWORD *)(v51 + 48);
                    if ( !v88 )
                    {
LABEL_134:
                      v89 = v175;
                      *(_QWORD *)(v51 + 48) = v175;
                      if ( v89 )
                        sub_1623210((__int64)&v175, v89, v51 + 48);
                      goto LABEL_98;
                    }
                  }
                  else if ( v52 == (unsigned __int64 **)&v175 || (v88 = *(_QWORD *)(v51 + 48)) == 0 )
                  {
LABEL_98:
                    sub_15F9450(v51, 1 << (*(unsigned __int16 *)(v154 + 18) >> 1) >> 1);
                    goto LABEL_40;
                  }
                  sub_161E7C0(v51 + 48, v88);
                  goto LABEL_134;
                }
                if ( !*(_DWORD *)(v146 + 12) )
                  goto LABEL_40;
                v135 = *(unsigned int *)(v146 + 12);
                v161 = 0;
                v133 = v20;
                v54 = v154;
                while ( 1 )
                {
                  v175 = v177;
                  v176 = 0x400000000LL;
                  v55 = sub_1643350(**v44);
                  v58 = sub_159C470(v55, 0, 0);
                  v59 = (unsigned int)v176;
                  if ( (unsigned int)v176 >= HIDWORD(v176) )
                  {
                    sub_16CD150((__int64)&v175, v177, 0, 8, v56, v57);
                    v59 = (unsigned int)v176;
                  }
                  *(_QWORD *)&v175[8 * v59] = v58;
                  LODWORD(v176) = v176 + 1;
                  if ( *(_BYTE *)(sub_15A0A60((__int64)v44, v161) + 16) == 9 )
                  {
                    v87 = (unsigned __int64 *)v175;
                    if ( v175 == v177 )
                      goto LABEL_130;
LABEL_129:
                    _libc_free((unsigned __int64)v87);
                    goto LABEL_130;
                  }
                  v60 = sub_1643350(**v44);
                  v63 = sub_159C470(v60, v161, 0);
                  v64 = (unsigned int)v176;
                  if ( (unsigned int)v176 >= HIDWORD(v176) )
                  {
                    sub_16CD150((__int64)&v175, v177, 0, 8, v61, v62);
                    v64 = (unsigned int)v176;
                  }
                  *(_QWORD *)&v175[8 * v64] = v63;
                  v168 = 1;
                  v65 = (__int64 *)v175;
                  v66 = (unsigned int)(v176 + 1);
                  v166[0] = (__int64)"splitStore";
                  v67 = v176 + 2;
                  LODWORD(v176) = v176 + 1;
                  v167 = 3;
                  v155 = *(_QWORD *)(v54 - 24);
                  v68 = *(_QWORD *)v155;
                  if ( *(_BYTE *)(*(_QWORD *)v155 + 8LL) == 16 )
                    v68 = **(_QWORD **)(v68 + 16);
                  v140 = v67;
                  v144 = *(_QWORD *)(v68 + 24);
                  v69 = sub_1648A60(72, v67);
                  if ( v69 )
                  {
                    v70 = *(_QWORD *)v155;
                    if ( *(_BYTE *)(*(_QWORD *)v155 + 8LL) == 16 )
                      v70 = **(_QWORD **)(v70 + 16);
                    v137 = &v69[-3 * v140];
                    v138 = v140;
                    v141 = *(_DWORD *)(v70 + 8) >> 8;
                    v71 = (__int64 *)sub_15F9F50(v144, (__int64)v65, v66);
                    v72 = (__int64 *)sub_1646BA0(v71, v141);
                    v73 = v138;
                    v74 = (__int64)v137;
                    v75 = v72;
                    if ( *(_BYTE *)(*(_QWORD *)v155 + 8LL) == 16 )
                    {
                      v94 = sub_16463B0(v72, *(_QWORD *)(*(_QWORD *)v155 + 32LL));
                      v73 = v138;
                      v74 = (__int64)v137;
                      v75 = v94;
                    }
                    else
                    {
                      v76 = &v65[v66];
                      if ( v65 != v76 )
                      {
                        v77 = v65;
                        while ( 1 )
                        {
                          v78 = *(_QWORD *)*v77;
                          if ( *(_BYTE *)(v78 + 8) == 16 )
                            break;
                          if ( v76 == ++v77 )
                            goto LABEL_117;
                        }
                        v79 = sub_16463B0(v75, *(_QWORD *)(v78 + 32));
                        v74 = (__int64)v137;
                        v73 = v138;
                        v75 = v79;
                      }
                    }
LABEL_117:
                    sub_15F1EA0((__int64)v69, (__int64)v75, 32, v74, v73, v54);
                    v69[7] = v144;
                    v69[8] = sub_15F9F50(v144, (__int64)v65, v66);
                    sub_15F9CE0((__int64)v69, v155, v65, v66, (__int64)v166);
                  }
                  v80 = *(_QWORD *)(v54 + 48);
                  v81 = v69 + 6;
                  v166[0] = v80;
                  if ( !v80 )
                    break;
                  sub_1623A60((__int64)v166, v80, 2);
                  if ( v81 == v166 )
                  {
                    if ( v166[0] )
                      sub_161E7C0((__int64)v166, v166[0]);
                    goto LABEL_122;
                  }
                  v92 = v69[6];
                  if ( v92 )
                    goto LABEL_145;
LABEL_146:
                  v93 = (unsigned __int8 *)v166[0];
                  v69[6] = v166[0];
                  if ( v93 )
                    sub_1623210((__int64)v166, v93, (__int64)(v69 + 6));
LABEL_122:
                  v82 = sub_1909170(v54, (__int64)v69, v158);
                  v156 = sub_15A0A60((__int64)v44, v161);
                  v83 = sub_1648A60(64, 2u);
                  v84 = (__int64)v83;
                  if ( v83 )
                    sub_15F9650((__int64)v83, v156, (__int64)v69, 0, 0);
                  v85 = (__int64 *)(v84 + 48);
                  sub_15F9450(v84, v82);
                  v86 = *(_QWORD *)(v54 + 48);
                  v166[0] = v86;
                  if ( !v86 )
                  {
                    if ( v85 == v166 )
                      goto LABEL_128;
                    v90 = *(_QWORD *)(v84 + 48);
                    if ( !v90 )
                      goto LABEL_128;
LABEL_140:
                    sub_161E7C0(v84 + 48, v90);
                    goto LABEL_141;
                  }
                  sub_1623A60((__int64)v166, v86, 2);
                  if ( v85 == v166 )
                  {
                    if ( v166[0] )
                      sub_161E7C0((__int64)v166, v166[0]);
                    goto LABEL_128;
                  }
                  v90 = *(_QWORD *)(v84 + 48);
                  if ( v90 )
                    goto LABEL_140;
LABEL_141:
                  v91 = (unsigned __int8 *)v166[0];
                  *(_QWORD *)(v84 + 48) = v166[0];
                  if ( v91 )
                    sub_1623210((__int64)v166, v91, v84 + 48);
LABEL_128:
                  sub_15F2180(v84, v54);
                  v87 = (unsigned __int64 *)v175;
                  if ( v175 != v177 )
                    goto LABEL_129;
LABEL_130:
                  if ( v135 == ++v161 )
                  {
                    v19 = v150;
                    v18 = v148;
                    v20 = v133;
                    goto LABEL_40;
                  }
                }
                if ( v81 == v166 )
                  goto LABEL_122;
                v92 = v69[6];
                if ( !v92 )
                  goto LABEL_122;
LABEL_145:
                sub_161E7C0((__int64)(v69 + 6), v92);
                goto LABEL_146;
              }
              v139 = v19;
              v136 = v20;
              while ( 1 )
              {
                v95 = (unsigned int *)v44[7];
                v96 = *((unsigned int *)v44 + 16);
                v175 = v177;
                v176 = 0x400000000LL;
                v97 = sub_1643350(**v44);
                v100 = sub_159C470(v97, 0, 0);
                v101 = (unsigned int)v176;
                if ( (unsigned int)v176 >= HIDWORD(v176) )
                {
                  sub_16CD150((__int64)&v175, v177, 0, 8, v98, v99);
                  v101 = (unsigned int)v176;
                }
                v102 = &v95[v96];
                *(_QWORD *)&v175[8 * v101] = v100;
                v103 = v176 + 1;
                for ( LODWORD(v176) = v176 + 1; v102 != v95; LODWORD(v176) = v176 + 1 )
                {
                  v104 = *v95;
                  v105 = sub_1643350(**v44);
                  v108 = sub_159C470(v105, v104, 0);
                  v109 = (unsigned int)v176;
                  if ( (unsigned int)v176 >= HIDWORD(v176) )
                  {
                    sub_16CD150((__int64)&v175, v177, 0, 8, v106, v107);
                    v109 = (unsigned int)v176;
                  }
                  ++v95;
                  *(_QWORD *)&v175[8 * v109] = v108;
                  v103 = v176 + 1;
                }
                v168 = 1;
                v110 = v103;
                v166[0] = (__int64)"splitStore";
                v167 = 3;
                v111 = (__int64 *)v175;
                v112 = *(_QWORD *)(v154 - 24);
                v147 = v103 + 1;
                v113 = *(_QWORD *)v112;
                if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) == 16 )
                  v113 = **(_QWORD **)(v113 + 16);
                v145 = *(_QWORD *)(v113 + 24);
                v114 = sub_1648A60(72, v147);
                v115 = v114;
                if ( v114 )
                {
                  v142 = (__int64)&v114[-3 * v147];
                  v116 = *(_QWORD *)v112;
                  if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) == 16 )
                    v116 = **(_QWORD **)(v116 + 16);
                  v134 = *(_DWORD *)(v116 + 8) >> 8;
                  v117 = (__int64 *)sub_15F9F50(v145, (__int64)v111, v110);
                  v118 = (__int64 *)sub_1646BA0(v117, v134);
                  if ( *(_BYTE *)(*(_QWORD *)v112 + 8LL) == 16 )
                  {
                    v118 = sub_16463B0(v118, *(_QWORD *)(*(_QWORD *)v112 + 32LL));
                  }
                  else
                  {
                    v119 = &v111[v110];
                    if ( v111 != v119 )
                    {
                      v120 = v111;
                      while ( 1 )
                      {
                        v121 = *(_QWORD *)*v120;
                        if ( *(_BYTE *)(v121 + 8) == 16 )
                          break;
                        if ( v119 == ++v120 )
                          goto LABEL_175;
                      }
                      v118 = sub_16463B0(v118, *(_QWORD *)(v121 + 32));
                    }
                  }
LABEL_175:
                  sub_15F1EA0((__int64)v115, (__int64)v118, 32, v142, v147, v154);
                  v115[7] = v145;
                  v115[8] = sub_15F9F50(v145, (__int64)v111, v110);
                  sub_15F9CE0((__int64)v115, v112, v111, v110, (__int64)v166);
                }
                v122 = v115 + 6;
                sub_15FA2E0((__int64)v115, 1);
                v123 = *(_QWORD *)(v154 + 48);
                v166[0] = v123;
                if ( !v123 )
                  break;
                sub_1623A60((__int64)v166, v123, 2);
                if ( v122 == v166 )
                {
                  if ( v166[0] )
                    sub_161E7C0((__int64)v166, v166[0]);
                  goto LABEL_185;
                }
                v124 = v115[6];
                if ( v124 )
                  goto LABEL_182;
LABEL_183:
                v125 = (unsigned __int8 *)v166[0];
                v115[6] = v166[0];
                if ( v125 )
                  sub_1623210((__int64)v166, v125, (__int64)(v115 + 6));
LABEL_185:
                v126 = sub_1909170(v154, (__int64)v115, v158);
                v149 = (__int64)*(v44 - 3);
                v127 = sub_1648A60(64, 2u);
                v128 = (__int64)v127;
                if ( v127 )
                  sub_15F9650((__int64)v127, v149, (__int64)v115, 0, 0);
                v129 = (__int64 *)(v128 + 48);
                sub_15F9450(v128, v126);
                v130 = *(_QWORD *)(v154 + 48);
                v166[0] = v130;
                if ( v130 )
                {
                  sub_1623A60((__int64)v166, v130, 2);
                  if ( v129 != v166 )
                  {
                    v131 = *(_QWORD *)(v128 + 48);
                    if ( v131 )
LABEL_197:
                      sub_161E7C0(v128 + 48, v131);
                    v132 = (unsigned __int8 *)v166[0];
                    *(_QWORD *)(v128 + 48) = v166[0];
                    if ( v132 )
                      sub_1623210((__int64)v166, v132, v128 + 48);
                    goto LABEL_191;
                  }
                  if ( v166[0] )
                    sub_161E7C0((__int64)v166, v166[0]);
                }
                else if ( v129 != v166 )
                {
                  v131 = *(_QWORD *)(v128 + 48);
                  if ( v131 )
                    goto LABEL_197;
                }
LABEL_191:
                v44 = (_QWORD ***)*(v44 - 6);
                sub_15F2180(v128, v154);
                if ( v175 != v177 )
                  _libc_free((unsigned __int64)v175);
                if ( *((_BYTE *)v44 + 16) != 87 )
                {
                  v19 = v139;
                  v20 = v136;
                  goto LABEL_84;
                }
              }
              if ( v122 == v166 )
                goto LABEL_185;
              v124 = v115[6];
              if ( !v124 )
                goto LABEL_185;
LABEL_182:
              sub_161E7C0((__int64)(v115 + 6), v124);
              goto LABEL_183;
            }
          }
        }
      }
LABEL_40:
      v18 = *(_QWORD *)(v18 + 8);
    }
    while ( v20 != v18 );
    v17 = v19;
LABEL_49:
    v159 = *(_QWORD *)(v159 + 8);
  }
  while ( v151 != v159 );
LABEL_50:
  v22 = v163;
  v23 = v164;
  v24 = (_QWORD **)v163;
  if ( v164 != v163 )
  {
    do
    {
      v25 = *v24++;
      sub_15F20C0(v25);
    }
    while ( v23 != (_BYTE *)v24 );
    v22 = v163;
  }
  if ( v22 )
    j_j___libc_free_0(v22, v165 - v22);
  return sub_1909D50(v171);
}
