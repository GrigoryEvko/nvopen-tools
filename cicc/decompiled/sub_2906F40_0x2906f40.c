// Function: sub_2906F40
// Address: 0x2906f40
//
char *__fastcall sub_2906F40(__int64 a1, char *a2, __int64 *a3)
{
  __int64 v4; // rax
  char v5; // dl
  char *v6; // r12
  unsigned __int8 v7; // al
  int v8; // ecx
  __int64 *v9; // r15
  char *v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // eax
  char *v13; // r15
  char *v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // r13
  char ***v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 *v20; // rax
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 **v25; // r13
  int v26; // eax
  __int64 *v27; // rbx
  __int64 v28; // r8
  __int64 v29; // rsi
  int v30; // eax
  char ***v31; // r12
  unsigned int v32; // edi
  char ***v33; // rsi
  char **v34; // r11
  __int64 v35; // rdx
  __int64 v36; // rbx
  __int64 v37; // r14
  __int64 v38; // r15
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // rbx
  __int64 v44; // rax
  unsigned __int64 v45; // r12
  __int64 v46; // rbx
  char ***v47; // rbx
  char *v48; // rsi
  __int64 *v49; // rax
  int v50; // eax
  char **v51; // rax
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned __int64 v54; // rsi
  unsigned int v55; // edi
  int v56; // esi
  int v57; // ebx
  char ***v58; // rbx
  char ***v59; // r13
  char *v61; // rbx
  unsigned __int8 *v62; // r13
  __int64 v63; // rax
  char v64; // al
  char *v65; // r8
  __int64 v66; // r9
  __int64 *v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rbx
  unsigned __int64 v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdi
  __int64 v73; // r13
  unsigned int v74; // edx
  __int64 v75; // rsi
  __int64 v76; // rax
  __int64 *v77; // rbx
  unsigned __int8 *v78; // r12
  unsigned __int8 v79; // al
  __int64 v80; // rax
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  __int64 **v86; // rbx
  __int64 **v87; // r12
  __int64 *v88; // rax
  __int64 v89; // rsi
  char **v90; // r13
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  int v95; // eax
  unsigned int v96; // eax
  __int64 v97; // r13
  char *v98; // r8
  int v99; // eax
  int v100; // ecx
  unsigned int v101; // eax
  __int64 v102; // r9
  bool v103; // zf
  __int64 *v104; // rax
  int v105; // esi
  int v106; // edx
  __int64 v107; // rdx
  __int64 v108; // rdx
  __int64 v109; // rsi
  __int64 v110; // rdx
  unsigned __int64 v111; // rax
  int v112; // edx
  __int64 v113; // rax
  bool v114; // cf
  __int64 v115; // rdx
  __int64 v116; // rcx
  int v117; // r8d
  __int64 *v118; // rdi
  unsigned int v119; // edx
  __int64 *v120; // rax
  __int64 v121; // r9
  __int64 *v122; // rax
  int v123; // edi
  __int64 v124; // rax
  __int64 v125; // rax
  __int64 v126; // rdi
  __int64 v127; // rsi
  __int64 v128; // rax
  int v129; // edx
  __int64 v130; // [rsp+8h] [rbp-268h]
  __int64 **v131; // [rsp+8h] [rbp-268h]
  __int64 v132; // [rsp+10h] [rbp-260h]
  __int64 v133; // [rsp+28h] [rbp-248h]
  __int64 *v134; // [rsp+30h] [rbp-240h]
  char ***v135; // [rsp+30h] [rbp-240h]
  __int64 **v136; // [rsp+30h] [rbp-240h]
  __int64 v137; // [rsp+30h] [rbp-240h]
  char *v138; // [rsp+38h] [rbp-238h]
  char *v139; // [rsp+38h] [rbp-238h]
  char ***v140; // [rsp+40h] [rbp-230h]
  char v141; // [rsp+48h] [rbp-228h]
  unsigned __int64 v142; // [rsp+48h] [rbp-228h]
  __int64 **v143; // [rsp+48h] [rbp-228h]
  __int64 **v144; // [rsp+58h] [rbp-218h]
  char *v147; // [rsp+78h] [rbp-1F8h] BYREF
  __int64 *v148; // [rsp+80h] [rbp-1F0h] BYREF
  __int64 v149; // [rsp+88h] [rbp-1E8h] BYREF
  __int64 *v150; // [rsp+90h] [rbp-1E0h] BYREF
  __int64 *v151; // [rsp+98h] [rbp-1D8h] BYREF
  char *v152; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 *v153; // [rsp+A8h] [rbp-1C8h]
  __int64 *v154; // [rsp+B0h] [rbp-1C0h] BYREF
  __int64 v155; // [rsp+C0h] [rbp-1B0h] BYREF
  __int64 v156; // [rsp+C8h] [rbp-1A8h]
  __int64 v157; // [rsp+D0h] [rbp-1A0h]
  unsigned int v158; // [rsp+D8h] [rbp-198h]
  char ***v159; // [rsp+E0h] [rbp-190h]
  __int64 v160; // [rsp+E8h] [rbp-188h]
  char **v161; // [rsp+F0h] [rbp-180h] BYREF
  char *v162; // [rsp+F8h] [rbp-178h]
  __int64 *v163; // [rsp+100h] [rbp-170h]
  __int64 *v164; // [rsp+108h] [rbp-168h]
  _QWORD v165[2]; // [rsp+110h] [rbp-160h] BYREF
  __int64 v166; // [rsp+120h] [rbp-150h]
  char **v167; // [rsp+130h] [rbp-140h] BYREF
  char ***v168; // [rsp+138h] [rbp-138h]
  char *v169; // [rsp+140h] [rbp-130h]
  void *v170; // [rsp+148h] [rbp-128h]
  unsigned __int64 v171; // [rsp+150h] [rbp-120h] BYREF
  __int64 v172; // [rsp+158h] [rbp-118h]
  unsigned __int8 *v173; // [rsp+160h] [rbp-110h]
  char *v174; // [rsp+170h] [rbp-100h] BYREF
  __int64 v175; // [rsp+178h] [rbp-F8h] BYREF
  _QWORD v176[3]; // [rsp+180h] [rbp-F0h] BYREF
  __int64 v177[3]; // [rsp+198h] [rbp-D8h] BYREF
  __int64 *v178; // [rsp+1B0h] [rbp-C0h] BYREF
  unsigned __int64 v179; // [rsp+1B8h] [rbp-B8h] BYREF
  __int64 v180; // [rsp+1C0h] [rbp-B0h] BYREF
  char *v181; // [rsp+1C8h] [rbp-A8h]
  int v182; // [rsp+1D0h] [rbp-A0h]
  unsigned __int64 v183; // [rsp+1D8h] [rbp-98h] BYREF
  __int64 v184; // [rsp+1E0h] [rbp-90h]
  char **v185; // [rsp+1E8h] [rbp-88h]

  v147 = (char *)sub_2906530(a1, (__int64)a2, (__int64)a3);
  sub_29011C0(&v178, a3, (__int64)v147);
  if ( v180 == a3[1] + 16LL * *((unsigned int *)a3 + 6) )
    v4 = 16LL * *((unsigned int *)a3 + 10);
  else
    v4 = 16LL * *(unsigned int *)(v180 + 8);
  v5 = *(_BYTE *)(a3[4] + v4 + 8);
  v6 = v147;
  if ( v5 )
  {
    v7 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v147 + 1) + 8LL) - 17 <= 1 )
    {
      v8 = v7;
      if ( v7 == 17 )
        return v6;
      goto LABEL_6;
    }
    v8 = v7;
    v5 = 0;
    if ( v7 != 17 )
    {
LABEL_6:
      if ( v5 == (v8 == 18) )
        return v6;
    }
  }
  v9 = (__int64 *)&v167;
  v159 = &v161;
  v179 = 0x1000000000LL;
  v155 = 0;
  v156 = 0;
  v157 = 0;
  v158 = 0;
  v160 = 0;
  v178 = &v180;
  sub_94F890((__int64)&v178, (__int64)v147);
  v10 = v147;
  v167 = 0;
  v168 = 0;
  v169 = v147;
  if ( v147 != 0 && v147 + 4096 != 0 && v147 != (char *)-8192LL )
  {
    sub_BD73F0((__int64)&v167);
    v10 = v147;
  }
  v174 = v10;
  LODWORD(v170) = 0;
  v171 = 0;
  v172 = 0;
  v173 = 0;
  sub_28FF950((__int64)&v175, (__int64)&v167);
  sub_2906B20((__int64)&v155, (__int64 *)&v174, (__int64)&v175);
  sub_D68D70(v177);
  sub_D68D70(&v175);
  sub_D68D70(&v171);
  sub_D68D70(&v167);
  v12 = v179;
  if ( (_DWORD)v179 )
  {
    do
    {
      v13 = (char *)v178[v12 - 1];
      LODWORD(v179) = v12 - 1;
      v176[0] = 0;
      v14 = (char *)sub_22077B0(0x20u);
      if ( v14 )
      {
        *(_QWORD *)v14 = a2;
        *((_QWORD *)v14 + 1) = a3;
        *((_QWORD *)v14 + 2) = &v155;
        *((_QWORD *)v14 + 3) = &v178;
      }
      v174 = v14;
      v176[1] = sub_2906CF0;
      v176[0] = sub_28FECF0;
      sub_2901070(v13, (__int64 *)&v174, v15);
      if ( v176[0] )
        ((void (__fastcall *)(char **, char **, __int64))v176[0])(&v174, &v174, 3);
      v12 = v179;
    }
    while ( (_DWORD)v179 );
    v9 = (__int64 *)&v167;
  }
  if ( v178 != &v180 )
    _libc_free((unsigned __int64)v178);
  HIDWORD(v175) = 6;
  v174 = (char *)v176;
  do
  {
    LODWORD(v175) = 0;
    v16 = (unsigned __int64)(unsigned int)v160 << 6;
    v140 = (char ***)((char *)v159 + v16);
    if ( v159 == (char ***)((char *)v159 + v16) )
      break;
    v17 = v159;
    do
    {
      v20 = (__int64 *)*v17;
      v179 = 0;
      v180 = 0;
      v178 = v20;
      v181 = (char *)v17[3];
      if ( v181 + 4096 != 0 && v181 != 0 && v181 != (char *)-8192LL )
        sub_BD6050(&v179, (unsigned __int64)v17[1] & 0xFFFFFFFFFFFFFFF8LL);
      v21 = *((_DWORD *)v17 + 8);
      v183 = 0;
      v184 = 0;
      v182 = v21;
      v185 = v17[7];
      LOBYTE(v11) = v185 != 0;
      if ( v185 != 0 && v185 + 512 != 0 && v185 != (char **)-8192LL )
        sub_BD6050(&v183, (unsigned __int64)v17[5] & 0xFFFFFFFFFFFFFFF8LL);
      v161 = &v152;
      LOBYTE(v151) = 1;
      v162 = a2;
      v152 = (char *)v178;
      v163 = a3;
      v167 = (char **)&v151;
      v164 = &v155;
      v168 = &v161;
      v170 = sub_29066B0;
      v169 = (char *)sub_28FEB90;
      sub_2901070((char *)v178, v9, v11);
      if ( v169 )
        ((void (__fastcall *)(__int64 *, __int64 *, __int64))v169)(v9, v9, 3);
      if ( (_BYTE)v151 )
      {
        v23 = (unsigned int)v175;
        v18 = HIDWORD(v175);
        v19 = (__int64)v152;
        v24 = (unsigned int)v175 + 1LL;
        if ( v24 > HIDWORD(v175) )
        {
          v139 = v152;
          sub_C8D5F0((__int64)&v174, v176, v24, 8u, v22, (__int64)v152);
          v23 = (unsigned int)v175;
          v19 = (__int64)v139;
        }
        v11 = (__int64)v174;
        *(_QWORD *)&v174[8 * v23] = v19;
        LODWORD(v175) = v175 + 1;
      }
      if ( v185 + 512 != 0 && v185 != 0 && v185 != (char **)-8192LL )
        sub_BD60C0(&v183);
      LOBYTE(v18) = v181 + 4096 != 0;
      if ( ((v181 != 0) & (unsigned __int8)v18) != 0 && v181 != (char *)-8192LL )
        sub_BD60C0(&v179);
      v17 += 8;
    }
    while ( v140 != v17 );
    v11 = (unsigned int)v175;
    v25 = (__int64 **)v174;
    v26 = v175;
    v138 = &v174[8 * (unsigned int)v175];
    if ( v174 != v138 )
    {
      v134 = v9;
      do
      {
        v27 = *v25;
        v28 = v156;
        v178 = *v25;
        if ( v158 )
        {
          v19 = v158 - 1;
          v18 = (unsigned int)v19 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v11 = v156 + 16 * v18;
          v29 = *(_QWORD *)v11;
          if ( v27 == *(__int64 **)v11 )
          {
LABEL_45:
            if ( v11 != v156 + 16LL * v158 )
            {
              v18 = (__int64)v159;
              v30 = v160;
              v31 = &v159[8 * (unsigned __int64)*(unsigned int *)(v11 + 8)];
              v11 = (__int64)&v159[8 * (unsigned __int64)(unsigned int)v160];
              if ( (char ***)v11 != v31 )
              {
                v32 = v19 & (((unsigned int)*v31 >> 9) ^ ((unsigned int)*v31 >> 4));
                v33 = (char ***)(v156 + 16LL * v32);
                v34 = *v33;
                if ( *v33 == *v31 )
                {
LABEL_48:
                  *v33 = (char **)-8192LL;
                  LODWORD(v157) = v157 - 1;
                  v18 = (__int64)v159;
                  ++HIDWORD(v157);
                  v30 = v160;
                  v11 = (__int64)&v159[8 * (unsigned __int64)(unsigned int)v160];
                }
                else
                {
                  v56 = 1;
                  while ( v34 != (char **)-4096LL )
                  {
                    v57 = v56 + 1;
                    v32 = v19 & (v32 + v56);
                    v33 = (char ***)(v156 + 16LL * v32);
                    v34 = *v33;
                    if ( *v31 == *v33 )
                      goto LABEL_48;
                    v56 = v57;
                  }
                }
                v35 = v11 - (_QWORD)(v31 + 8);
                v36 = v35 >> 6;
                if ( v35 > 0 )
                {
                  v37 = (__int64)(v31 + 5);
                  do
                  {
                    v38 = *(_QWORD *)(v37 + 48);
                    *(_QWORD *)(v37 - 40) = *(_QWORD *)(v37 + 24);
                    v39 = *(_QWORD *)(v37 - 16);
                    if ( v38 != v39 )
                    {
                      if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
                        sub_BD60C0((_QWORD *)(v37 - 32));
                      *(_QWORD *)(v37 - 16) = v38;
                      if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
                        sub_BD73F0(v37 - 32);
                    }
                    v40 = *(_QWORD *)(v37 + 80);
                    *(_DWORD *)(v37 - 8) = *(_DWORD *)(v37 + 56);
                    v41 = *(_QWORD *)(v37 + 16);
                    if ( v40 != v41 )
                    {
                      if ( v41 != 0 && v41 != -4096 && v41 != -8192 )
                        sub_BD60C0((_QWORD *)v37);
                      *(_QWORD *)(v37 + 16) = v40;
                      if ( v40 != 0 && v40 != -4096 && v40 != -8192 )
                        sub_BD73F0(v37);
                    }
                    v37 += 64;
                    --v36;
                  }
                  while ( v36 );
                  v30 = v160;
                  v18 = (__int64)v159;
                }
                v42 = (unsigned int)(v30 - 1);
                LODWORD(v160) = v42;
                v43 = (_QWORD *)(v18 + (v42 << 6));
                v44 = v43[7];
                if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
                  sub_BD60C0(v43 + 5);
                sub_D68D70(v43 + 1);
                v11 = (__int64)v159;
                if ( v31 != &v159[8 * (unsigned __int64)(unsigned int)v160] )
                {
                  v11 = (unsigned int)v157;
                  v45 = ((char *)v31 - (char *)v159) >> 6;
                  if ( (_DWORD)v157 )
                  {
                    v11 = v156;
                    v18 = v156 + 16LL * v158;
                    if ( v156 != v18 )
                    {
                      while ( 1 )
                      {
                        v53 = v11;
                        if ( *(_QWORD *)v11 != -4096 && *(_QWORD *)v11 != -8192 )
                          break;
                        v11 += 16;
                        if ( v18 == v11 )
                          goto LABEL_72;
                      }
                      while ( v18 != v53 )
                      {
                        v54 = *(unsigned int *)(v53 + 8);
                        v11 = v54;
                        if ( v45 < v54 )
                        {
                          v11 = (unsigned int)(v54 - 1);
                          *(_DWORD *)(v53 + 8) = v11;
                        }
                        v53 += 16;
                        if ( v53 == v18 )
                          break;
                        while ( 1 )
                        {
                          v11 = *(_QWORD *)v53;
                          if ( *(_QWORD *)v53 != -8192 && v11 != -4096 )
                            break;
                          v53 += 16;
                          if ( v18 == v53 )
                            goto LABEL_72;
                        }
                      }
                    }
                  }
                }
LABEL_72:
                v27 = v178;
              }
            }
          }
          else
          {
            v11 = 1;
            while ( v29 != -4096 )
            {
              v55 = v11 + 1;
              v18 = (unsigned int)v19 & ((_DWORD)v11 + (_DWORD)v18);
              v11 = v156 + 16LL * (unsigned int)v18;
              v29 = *(_QWORD *)v11;
              if ( v27 == *(__int64 **)v11 )
                goto LABEL_45;
              v11 = v55;
            }
          }
        }
        ++v25;
        *(_QWORD *)sub_1152A40((__int64)a2, (__int64 *)&v178, v11, v18, v28, v19) = v27;
      }
      while ( v138 != (char *)v25 );
      v9 = v134;
      v26 = v175;
    }
  }
  while ( v26 );
  v46 = v156 + 16LL * v158;
  sub_29011C0(&v178, &v155, (__int64)v147);
  if ( v46 == v180 )
  {
    v6 = v147;
  }
  else
  {
    v148 = &v155;
    do
    {
      v47 = v159;
      v135 = &v159[8 * (unsigned __int64)(unsigned int)v160];
      if ( v135 == v159 )
        goto LABEL_235;
      v141 = 0;
      do
      {
        v49 = (__int64 *)*v47;
        v179 = 0;
        v180 = 0;
        v178 = v49;
        v181 = (char *)v47[3];
        if ( v181 != 0 && v181 + 4096 != 0 && v181 != (char *)-8192LL )
          sub_BD6050(&v179, (unsigned __int64)v47[1] & 0xFFFFFFFFFFFFFFF8LL);
        v50 = *((_DWORD *)v47 + 8);
        v183 = 0;
        v184 = 0;
        v182 = v50;
        v185 = v47[7];
        if ( v185 != 0 && v185 + 512 != 0 && v185 != (char **)-8192LL )
          sub_BD6050(&v183, (unsigned __int64)v47[5] & 0xFFFFFFFFFFFFFFF8LL);
        v161 = 0;
        v162 = 0;
        v152 = (char *)v178;
        v163 = v178;
        if ( v178 != 0 && v178 + 512 != 0 && v178 != (__int64 *)-8192LL )
          sub_BD73F0((__int64)&v161);
        LODWORD(v164) = 0;
        v165[0] = 0;
        v165[1] = 0;
        v166 = 0;
        v169 = 0;
        v51 = (char **)sub_22077B0(0x20u);
        if ( v51 )
        {
          v51[3] = (char *)&v161;
          *v51 = a2;
          v51[2] = (char *)&v148;
          v51[1] = (char *)a3;
        }
        v167 = v51;
        v170 = sub_29067B0;
        v169 = (char *)sub_28FEC80;
        sub_2901070(v152, v9, v52);
        if ( v169 )
          ((void (__fastcall *)(__int64 *, __int64 *, __int64))v169)(v9, v9, 3);
        if ( v166 )
        {
          v48 = v152;
          if ( (unsigned __int8)(*v152 - 90) <= 2u
            || (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v166 + 8) + 8LL) - 17 <= 1 != (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v152 + 1) + 8LL)
                                                                                          - 17 <= 1 )
          {
            v167 = 0;
            v168 = 0;
            v169 = v152;
            if ( v152 != (char *)-8192LL && v152 != (char *)-4096LL )
            {
              sub_BD73F0((__int64)v9);
              v48 = v169;
            }
            LODWORD(v170) = 2;
            v171 = 0;
            v172 = 0;
            v173 = 0;
            sub_FC7530(&v161, (__int64)v48);
            LODWORD(v164) = (_DWORD)v170;
            sub_FC7530(v165, (__int64)v173);
            sub_D68D70(&v171);
            sub_D68D70(v9);
          }
        }
        v167 = 0;
        v168 = 0;
        v169 = v181;
        if ( v181 + 4096 != 0 && v181 != 0 && v181 != (char *)-8192LL )
          sub_BD6050((unsigned __int64 *)v9, v179 & 0xFFFFFFFFFFFFFFF8LL);
        v171 = 0;
        v172 = 0;
        LODWORD(v170) = v182;
        v173 = (unsigned __int8 *)v185;
        if ( v185 + 512 != 0 && v185 != 0 && v185 != (char **)-8192LL )
          sub_BD6050(&v171, v183 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v169 != (char *)v163 || v173 != (unsigned __int8 *)v166 || (_DWORD)v170 != (_DWORD)v164 )
        {
          v142 = sub_2904570((__int64)&v155, (__int64 *)&v152);
          sub_FC7530((_QWORD *)v142, (__int64)v163);
          *(_DWORD *)(v142 + 24) = (_DWORD)v164;
          sub_FC7530((_QWORD *)(v142 + 32), v166);
          v141 = 1;
        }
        v47 += 8;
        sub_D68D70(&v171);
        sub_D68D70(v9);
        sub_D68D70(v165);
        sub_D68D70(&v161);
        sub_D68D70(&v183);
        sub_D68D70(&v179);
      }
      while ( v135 != v47 );
    }
    while ( v141 );
    v136 = (__int64 **)&v159[8 * (unsigned __int64)(unsigned int)v160];
    if ( v136 == (__int64 **)v159 )
    {
LABEL_235:
      v152 = a2;
      v153 = a3;
      v154 = &v155;
    }
    else
    {
      v143 = (__int64 **)v159;
      do
      {
        v178 = *v143;
        sub_28FF950((__int64)&v179, (__int64)(v143 + 1));
        v61 = (char *)v178;
        sub_28FF950((__int64)&v161, (__int64)&v179);
        if ( (_DWORD)v164 == 2 )
        {
          v62 = (unsigned __int8 *)sub_B47F80(v61);
          v63 = v130;
          LOWORD(v63) = 0;
          v130 = v63;
          sub_B44220(v62, (__int64)(v61 + 24), v63);
          v64 = *v61;
          v65 = "base_phi";
          v66 = 8;
          if ( *v61 != 84 )
          {
            v65 = "base_select";
            v66 = 11;
            if ( v64 != 86 )
            {
              v65 = "base_ee";
              if ( v64 != 90 )
              {
                v65 = "base_ie";
                if ( v64 != 91 )
                  v65 = "base_sv";
              }
              v66 = 7;
            }
          }
          sub_28FF1A0((__int64)&v152, (__int64)v61, ".base", (void *)5, v65, (_BYTE *)v66);
          LOWORD(v171) = 260;
          v167 = &v152;
          sub_BD6B50(v62, (const char **)v9);
          if ( v152 != (char *)&v154 )
            j_j___libc_free_0((unsigned __int64)v152);
          v67 = (__int64 *)sub_BD5C60((__int64)v61);
          v68 = sub_B9C770(v67, 0, 0, 0, 1);
          sub_B9A090((__int64)v62, "is_base_value", 0xDu, v68);
          v169 = v61;
          v167 = 0;
          v168 = 0;
          if ( v61 != (char *)-4096LL && v61 != (char *)-8192LL )
            sub_BD73F0((__int64)v9);
          LODWORD(v170) = 2;
          v171 = 0;
          v173 = v62;
          v172 = 0;
          if ( v62 != 0 && v62 + 4096 != 0 && v62 != (unsigned __int8 *)-8192LL )
            sub_BD73F0((__int64)&v171);
          v152 = v61;
          v69 = sub_2904570((__int64)&v155, (__int64 *)&v152);
          sub_FC7530((_QWORD *)v69, (__int64)v169);
          *(_DWORD *)(v69 + 24) = (_DWORD)v170;
          sub_FC7530((_QWORD *)(v69 + 32), (__int64)v173);
          sub_D68D70(&v171);
          sub_D68D70(v9);
          v167 = (char **)v62;
          *(_BYTE *)sub_2905EE0((__int64)a3, v9) = 1;
          sub_D68D70(v165);
        }
        else
        {
          sub_D68D70(v165);
        }
        sub_D68D70(&v161);
        sub_D68D70(&v183);
        sub_D68D70(&v179);
        v143 += 8;
      }
      while ( v136 != v143 );
      v152 = a2;
      v70 = (unsigned __int64)(unsigned int)v160 << 6;
      v153 = a3;
      v131 = (__int64 **)((char *)v159 + v70);
      v154 = &v155;
      if ( v159 != (char ***)((char *)v159 + v70) )
      {
        v144 = (__int64 **)v159;
        while ( 1 )
        {
          v178 = *v144;
          sub_28FF950((__int64)&v179, (__int64)(v144 + 1));
          v77 = v178;
          sub_28FF950((__int64)v9, (__int64)&v179);
          if ( (_DWORD)v170 != 2 )
            goto LABEL_173;
          v78 = v173;
          v79 = *v173;
          v73 = (__int64)(v173 - 64);
          if ( *v173 <= 0x1Cu )
            goto LABEL_170;
          if ( v79 == 84 )
          {
            v95 = *((_DWORD *)v77 + 1);
            v161 = 0;
            v162 = 0;
            v163 = 0;
            LODWORD(v164) = 0;
            v96 = v95 & 0x7FFFFFF;
            if ( !v96 )
            {
              v126 = 0;
              v127 = 0;
LABEL_213:
              sub_C7D6A0(v126, v127, 8);
              goto LABEL_173;
            }
            v97 = 0;
            v98 = 0;
            v133 = 8LL * v96;
            v99 = 0;
            while ( 2 )
            {
              v108 = *(v77 - 1);
              v132 = 4 * v97;
              v109 = *(_QWORD *)(v108 + 4 * v97);
              v110 = *(_QWORD *)(32LL * *((unsigned int *)v77 + 18) + v108 + v97);
              v149 = v110;
              if ( v99 )
              {
                v100 = v99 - 1;
                v101 = (v99 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
                v102 = *(_QWORD *)&v98[16 * v101];
                if ( v110 != v102 )
                {
                  v123 = 1;
                  while ( v102 != -4096 )
                  {
                    v101 = v100 & (v123 + v101);
                    v102 = *(_QWORD *)&v98[16 * v101];
                    if ( v110 == v102 )
                      goto LABEL_187;
                    ++v123;
                  }
                  goto LABEL_196;
                }
                goto LABEL_187;
              }
LABEL_196:
              v111 = *(_QWORD *)(v110 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v111 == v110 + 48 )
              {
                v115 = 0;
              }
              else
              {
                if ( !v111 )
                  BUG();
                v112 = *(unsigned __int8 *)(v111 - 24);
                v113 = v111 - 24;
                v114 = (unsigned int)(v112 - 30) < 0xB;
                v115 = 0;
                if ( v114 )
                  v115 = v113;
              }
              v137 = sub_29069F0((__int64 *)&v152, v109, v115);
              if ( (_DWORD)v164 )
              {
                v116 = v149;
                v117 = 1;
                v118 = 0;
                v119 = ((_DWORD)v164 - 1) & (((unsigned int)v149 >> 9) ^ ((unsigned int)v149 >> 4));
                v120 = (__int64 *)&v162[16 * v119];
                v121 = *v120;
                if ( v149 == *v120 )
                {
LABEL_202:
                  v122 = v120 + 1;
                  goto LABEL_203;
                }
                while ( v121 != -4096 )
                {
                  if ( v121 == -8192 && !v118 )
                    v118 = v120;
                  v119 = ((_DWORD)v164 - 1) & (v117 + v119);
                  v120 = (__int64 *)&v162[16 * v119];
                  v121 = *v120;
                  if ( v149 == *v120 )
                    goto LABEL_202;
                  ++v117;
                }
                if ( v118 )
                  v120 = v118;
                v161 = (char **)((char *)v161 + 1);
                v129 = (_DWORD)v163 + 1;
                v151 = v120;
                if ( 4 * ((int)v163 + 1) < (unsigned int)(3 * (_DWORD)v164) )
                {
                  if ( (int)v164 - HIDWORD(v163) - v129 > (unsigned int)v164 >> 3 )
                  {
LABEL_226:
                    LODWORD(v163) = v129;
                    if ( *v120 != -4096 )
                      --HIDWORD(v163);
                    *v120 = v116;
                    v122 = v120 + 1;
                    *v122 = 0;
LABEL_203:
                    *v122 = v137;
LABEL_187:
                    v103 = (unsigned __int8)sub_116E230((__int64)&v161, &v149, &v150) == 0;
                    v104 = v150;
                    if ( v103 )
                    {
                      v105 = (int)v164;
                      v151 = v150;
                      v161 = (char **)((char *)v161 + 1);
                      v106 = (_DWORD)v163 + 1;
                      if ( 4 * ((int)v163 + 1) >= (unsigned int)(3 * (_DWORD)v164) )
                      {
                        v105 = 2 * (_DWORD)v164;
                      }
                      else if ( (int)v164 - HIDWORD(v163) - v106 > (unsigned int)v164 >> 3 )
                      {
                        goto LABEL_190;
                      }
                      sub_116E750((__int64)&v161, v105);
                      sub_116E230((__int64)&v161, &v149, &v151);
                      v106 = (_DWORD)v163 + 1;
                      v104 = v151;
LABEL_190:
                      LODWORD(v163) = v106;
                      if ( *v104 != -4096 )
                        --HIDWORD(v163);
                      v107 = v149;
                      v104[1] = 0;
                      *v104 = v107;
                    }
                    v97 += 8;
                    sub_AC2B30(*((_QWORD *)v78 - 1) + v132, v104[1]);
                    if ( v133 == v97 )
                    {
                      v126 = (__int64)v162;
                      v127 = 16LL * (unsigned int)v164;
                      goto LABEL_213;
                    }
                    v98 = v162;
                    v99 = (int)v164;
                    continue;
                  }
                  sub_116E750((__int64)&v161, (int)v164);
LABEL_232:
                  sub_116E230((__int64)&v161, &v149, &v151);
                  v116 = v149;
                  v129 = (_DWORD)v163 + 1;
                  v120 = v151;
                  goto LABEL_226;
                }
              }
              else
              {
                v161 = (char **)((char *)v161 + 1);
                v151 = 0;
              }
              break;
            }
            sub_116E750((__int64)&v161, 2 * (_DWORD)v164);
            goto LABEL_232;
          }
          if ( v79 == 86 )
          {
            v124 = sub_29069F0((__int64 *)&v152, *(v77 - 8), (__int64)v173);
            sub_AC2B30((__int64)(v78 - 64), v124);
            v125 = sub_29069F0((__int64 *)&v152, *(v77 - 4), (__int64)v78);
            sub_AC2B30((__int64)(v78 - 32), v125);
            goto LABEL_173;
          }
          v73 = (__int64)(v173 - 64);
          if ( v79 != 90 )
            break;
          v75 = *(v77 - 8);
LABEL_172:
          v76 = sub_29069F0((__int64 *)&v152, v75, (__int64)v78);
          sub_AC2B30(v73, v76);
LABEL_173:
          sub_D68D70(&v171);
          sub_D68D70(v9);
          sub_D68D70(&v183);
          sub_D68D70(&v179);
          v144 += 8;
          if ( v131 == v144 )
            goto LABEL_181;
        }
        if ( v79 == 91 )
        {
          v80 = sub_29069F0((__int64 *)&v152, *(v77 - 12), (__int64)v173);
          sub_AC2B30((__int64)(v78 - 96), v80);
          v81 = sub_29069F0((__int64 *)&v152, *(v77 - 8), (__int64)v78);
          sub_AC2B30((__int64)(v78 - 64), v81);
          goto LABEL_173;
        }
LABEL_170:
        v71 = sub_29069F0((__int64 *)&v152, *(v77 - 8), (__int64)v173);
        v72 = v73;
        v73 = (__int64)(v78 - 32);
        sub_AC2B30(v72, v71);
        v74 = *((_DWORD *)v77 + 20);
        if ( *(_DWORD *)(*(_QWORD *)(*(v77 - 8) + 8) + 32LL) == v74
          && (unsigned __int8)sub_B4EE20((int *)v77[9], v74, v74) )
        {
          v128 = sub_ACADE0(*(__int64 ***)(*(v77 - 4) + 8));
          sub_AC2B30((__int64)(v78 - 32), v128);
          goto LABEL_173;
        }
        v75 = *(v77 - 4);
        goto LABEL_172;
      }
    }
LABEL_181:
    sub_B43CC0((__int64)v147);
    v86 = (__int64 **)v159;
    v87 = (__int64 **)&v159[8 * (unsigned __int64)(unsigned int)v160];
    if ( v87 != (__int64 **)v159 )
    {
      do
      {
        v88 = *v86;
        v89 = (__int64)(v86 + 1);
        v86 += 8;
        v178 = v88;
        sub_28FF950((__int64)&v179, v89);
        v90 = v185;
        v167 = (char **)v178;
        *(_QWORD *)sub_1152A40((__int64)a2, v9, v91, v92, v93, v94) = v90;
        sub_D68D70(&v183);
        sub_D68D70(&v179);
      }
      while ( v87 != v86 );
    }
    v6 = *(char **)sub_1152A40((__int64)a2, (__int64 *)&v147, v82, v83, v84, v85);
  }
  if ( v174 != (char *)v176 )
    _libc_free((unsigned __int64)v174);
  v58 = v159;
  v59 = &v159[8 * (unsigned __int64)(unsigned int)v160];
  if ( v159 != v59 )
  {
    do
    {
      v59 -= 8;
      sub_D68D70(v59 + 5);
      sub_D68D70(v59 + 1);
    }
    while ( v58 != v59 );
    v59 = v159;
  }
  if ( v59 != &v161 )
    _libc_free((unsigned __int64)v59);
  sub_C7D6A0(v156, 16LL * v158, 8);
  return v6;
}
