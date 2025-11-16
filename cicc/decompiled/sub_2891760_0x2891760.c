// Function: sub_2891760
// Address: 0x2891760
//
__int64 __fastcall sub_2891760(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // r15
  int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char *i; // r13
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _BYTE *v25; // r15
  __int64 v26; // r13
  char *v27; // rbx
  __int64 v28; // rax
  int v29; // eax
  __int64 v30; // rdx
  __int64 *v31; // r14
  unsigned __int64 v32; // rsi
  int v33; // eax
  unsigned __int64 *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned __int64 *v38; // rsi
  __int64 v39; // r12
  __int64 v40; // rax
  int v41; // eax
  unsigned __int8 **v42; // rsi
  unsigned __int64 *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 *v48; // r13
  __int64 v49; // rax
  __int64 *v50; // r15
  __int64 v51; // rbx
  char *v52; // rdx
  int v53; // eax
  char **v54; // r12
  char *v55; // rax
  char *v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rax
  __int64 v59; // r15
  unsigned __int8 v60; // al
  __int64 v61; // rdx
  __int64 v62; // r15
  __int64 v63; // r9
  _QWORD *v64; // rax
  __int64 v65; // r9
  _QWORD *v66; // r11
  unsigned __int16 v67; // r8
  __int64 v68; // rsi
  unsigned __int64 *v69; // r15
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rax
  char *v74; // rbx
  __int64 v75; // rax
  char *v76; // r14
  __int64 v77; // rbx
  _QWORD *v78; // r12
  __int64 v79; // rax
  unsigned __int8 *v81; // rdi
  bool v82; // al
  __int64 v83; // rdx
  _BYTE *v84; // rax
  unsigned int v85; // r15d
  bool v86; // al
  __int64 v87; // rsi
  unsigned __int8 *v88; // rsi
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  __int64 v92; // r9
  unsigned __int64 v93; // rbx
  _QWORD *v94; // r12
  void (__fastcall *v95)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v96; // rax
  _BYTE *v97; // rax
  unsigned int v98; // r15d
  unsigned int v99; // r15d
  bool v100; // r12
  __int64 v101; // rax
  unsigned int v102; // r12d
  bool v103; // dl
  unsigned int v104; // ecx
  bool v105; // r15
  __int64 v106; // rax
  unsigned int v107; // ecx
  unsigned int v108; // r15d
  int v109; // eax
  __int64 v110; // [rsp+0h] [rbp-ED0h]
  __int64 v111; // [rsp+8h] [rbp-EC8h]
  _QWORD *v112; // [rsp+8h] [rbp-EC8h]
  _QWORD *v113; // [rsp+8h] [rbp-EC8h]
  unsigned __int16 v114; // [rsp+10h] [rbp-EC0h]
  _QWORD *v115; // [rsp+10h] [rbp-EC0h]
  __int64 v116; // [rsp+10h] [rbp-EC0h]
  __int64 v117; // [rsp+10h] [rbp-EC0h]
  __int64 v118; // [rsp+10h] [rbp-EC0h]
  __int64 v119; // [rsp+10h] [rbp-EC0h]
  __int64 v120; // [rsp+10h] [rbp-EC0h]
  unsigned int v121; // [rsp+10h] [rbp-EC0h]
  __int64 v123; // [rsp+30h] [rbp-EA0h]
  int v124; // [rsp+30h] [rbp-EA0h]
  unsigned __int8 *v125; // [rsp+30h] [rbp-EA0h]
  bool v127; // [rsp+40h] [rbp-E90h]
  int v128; // [rsp+40h] [rbp-E90h]
  int v129; // [rsp+40h] [rbp-E90h]
  __int64 v130; // [rsp+40h] [rbp-E90h]
  __int64 v131; // [rsp+40h] [rbp-E90h]
  unsigned __int8 *v132; // [rsp+40h] [rbp-E90h]
  int v133; // [rsp+40h] [rbp-E90h]
  __int64 v134; // [rsp+48h] [rbp-E88h]
  __int64 v135; // [rsp+60h] [rbp-E70h]
  char v136; // [rsp+7Eh] [rbp-E52h]
  bool v137; // [rsp+7Fh] [rbp-E51h]
  unsigned __int64 *v138; // [rsp+90h] [rbp-E40h]
  unsigned __int64 v139; // [rsp+98h] [rbp-E38h]
  __int64 v140; // [rsp+98h] [rbp-E38h]
  _BYTE *v141; // [rsp+A0h] [rbp-E30h] BYREF
  __int64 v142; // [rsp+A8h] [rbp-E28h]
  _BYTE v143[64]; // [rsp+B0h] [rbp-E20h] BYREF
  __int64 v144; // [rsp+F0h] [rbp-DE0h] BYREF
  __int64 v145; // [rsp+F8h] [rbp-DD8h]
  _BYTE v146[192]; // [rsp+100h] [rbp-DD0h] BYREF
  unsigned __int64 v147[54]; // [rsp+1C0h] [rbp-D10h] BYREF
  __int64 v148; // [rsp+370h] [rbp-B60h] BYREF
  __int64 *v149; // [rsp+378h] [rbp-B58h]
  int v150; // [rsp+380h] [rbp-B50h]
  int v151; // [rsp+384h] [rbp-B4Ch]
  int v152; // [rsp+388h] [rbp-B48h]
  char v153; // [rsp+38Ch] [rbp-B44h]
  __int64 v154; // [rsp+390h] [rbp-B40h] BYREF
  __int64 *v155; // [rsp+3D0h] [rbp-B00h]
  __int64 v156; // [rsp+3D8h] [rbp-AF8h]
  __int64 v157; // [rsp+3E0h] [rbp-AF0h] BYREF
  int v158; // [rsp+3E8h] [rbp-AE8h]
  __int64 v159; // [rsp+3F0h] [rbp-AE0h]
  int v160; // [rsp+3F8h] [rbp-AD8h]
  __int64 v161; // [rsp+400h] [rbp-AD0h]
  char v162[8]; // [rsp+520h] [rbp-9B0h] BYREF
  unsigned __int64 v163; // [rsp+528h] [rbp-9A8h]
  char v164; // [rsp+53Ch] [rbp-994h]
  char *v165; // [rsp+580h] [rbp-950h]
  char v166; // [rsp+590h] [rbp-940h] BYREF
  __int64 v167; // [rsp+6D0h] [rbp-800h] BYREF
  __int64 v168; // [rsp+6D8h] [rbp-7F8h]
  char v169; // [rsp+6ECh] [rbp-7E4h]
  char *v170; // [rsp+730h] [rbp-7A0h]
  char v171; // [rsp+740h] [rbp-790h] BYREF
  __int64 v172; // [rsp+880h] [rbp-650h] BYREF
  unsigned __int64 v173; // [rsp+888h] [rbp-648h]
  __int64 v174; // [rsp+890h] [rbp-640h]
  __int64 v175; // [rsp+898h] [rbp-638h]
  __int64 *v176; // [rsp+8A0h] [rbp-630h]
  __int64 v177; // [rsp+8A8h] [rbp-628h]
  _BYTE v178[48]; // [rsp+8B0h] [rbp-620h] BYREF
  char *v179; // [rsp+8E0h] [rbp-5F0h]
  char v180; // [rsp+8F0h] [rbp-5E0h] BYREF
  __int64 v181; // [rsp+A30h] [rbp-4A0h] BYREF
  unsigned __int64 v182; // [rsp+A38h] [rbp-498h]
  char *v183; // [rsp+A40h] [rbp-490h] BYREF
  char v184; // [rsp+A4Ch] [rbp-484h]
  char *v185; // [rsp+A90h] [rbp-440h]
  char v186; // [rsp+AA0h] [rbp-430h] BYREF
  unsigned __int64 v187[94]; // [rsp+BE0h] [rbp-2F0h] BYREF

  memset(v187, 0, 0x2C0u);
  if ( a3 )
  {
    v187[68] = a3;
    v187[0] = (unsigned __int64)&v187[2];
    v187[1] = 0x1000000000LL;
    v187[72] = (unsigned __int64)&v187[75];
    v187[66] = 0;
    v187[67] = 0;
    v187[69] = 0;
    LOBYTE(v187[70]) = 1;
    v187[71] = 0;
    v187[73] = 8;
    LODWORD(v187[74]) = 0;
    BYTE4(v187[74]) = 1;
    LOWORD(v187[83]) = 0;
    memset(&v187[84], 0, 24);
    LOBYTE(v187[87]) = 1;
  }
  v3 = sub_B2BEC0(a1);
  v4 = *(_QWORD *)(a1 + 80);
  v134 = v3;
  v144 = (__int64)v146;
  v141 = v143;
  memset(v147, 0, sizeof(v147));
  v156 = 0x800000000LL;
  v147[1] = (unsigned __int64)&v147[4];
  if ( v4 )
    v4 -= 24;
  v147[12] = (unsigned __int64)&v147[14];
  HIDWORD(v147[13]) = 8;
  v149 = &v154;
  v155 = &v157;
  v5 = *(_QWORD *)(v4 + 48);
  v145 = 0x800000000LL;
  v142 = 0x800000000LL;
  v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
  LODWORD(v147[2]) = 8;
  BYTE4(v147[3]) = 1;
  v150 = 8;
  v152 = 0;
  v153 = 1;
  v151 = 1;
  v154 = v4;
  v148 = 1;
  if ( v6 == v4 + 48 )
    goto LABEL_182;
  if ( !v6 )
    BUG();
  v7 = v6 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v6 - 24) - 30 > 0xA )
  {
LABEL_182:
    v8 = 0;
    v9 = 0;
    v7 = 0;
  }
  else
  {
    v8 = sub_B46E30(v7);
    v9 = v7;
  }
  v161 = v4;
  v157 = v9;
  v158 = v8;
  v159 = v7;
  v160 = 0;
  LODWORD(v156) = 1;
  sub_CE27D0((__int64)&v148);
  sub_CE3710((__int64)&v172, (__int64)v147, v10, v11, v12, v13);
  i = v162;
  sub_CE35F0((__int64)&v181, (__int64)&v172);
  sub_CE3710((__int64)v162, (__int64)&v148, v15, v16, v17, v18);
  sub_CE35F0((__int64)&v167, (__int64)v162);
  sub_CE37E0((__int64)&v167, (__int64)&v181, (__int64)&v141, v19, v20, v21);
  if ( v170 != &v171 )
    _libc_free((unsigned __int64)v170);
  if ( !v169 )
    _libc_free(v168);
  if ( v165 != &v166 )
    _libc_free((unsigned __int64)v165);
  if ( !v164 )
    _libc_free(v163);
  if ( v185 != &v186 )
    _libc_free((unsigned __int64)v185);
  if ( !v184 )
    _libc_free(v182);
  if ( v179 != &v180 )
    _libc_free((unsigned __int64)v179);
  if ( !BYTE4(v175) )
    _libc_free(v173);
  if ( v155 != &v157 )
    _libc_free((unsigned __int64)v155);
  if ( !v153 )
    _libc_free((unsigned __int64)v149);
  if ( (unsigned __int64 *)v147[12] != &v147[14] )
    _libc_free(v147[12]);
  if ( !BYTE4(v147[3]) )
    _libc_free(v147[1]);
  v25 = &v141[8 * (unsigned int)v142];
  if ( v25 != v141 )
  {
    v139 = (unsigned __int64)v141;
    do
    {
      v26 = *((_QWORD *)v25 - 1);
      v27 = *(char **)(v26 + 56);
      for ( i = (char *)(v26 + 48); i != v27; v27 = (char *)*((_QWORD *)v27 + 1) )
      {
        while ( 1 )
        {
          if ( !v27 )
            BUG();
          if ( *(v27 - 24) == 85 )
          {
            v28 = *((_QWORD *)v27 - 7);
            if ( v28 )
            {
              if ( !*(_BYTE *)v28 && *(_QWORD *)(v28 + 24) == *((_QWORD *)v27 + 7) && (*(_BYTE *)(v28 + 33) & 0x20) != 0 )
              {
                v29 = *(_DWORD *)(v28 + 36);
                if ( v29 == 206 || v29 == 282 )
                {
                  v181 = 6;
                  v182 = 0;
                  v183 = v27 - 24;
                  if ( v27 != (char *)-4072LL && v27 != (char *)-8168LL )
                    sub_BD73F0((__int64)&v181);
                  v30 = (unsigned int)v145;
                  v31 = &v181;
                  v22 = v144;
                  v32 = (unsigned int)v145 + 1LL;
                  v33 = v145;
                  if ( v32 > HIDWORD(v145) )
                  {
                    if ( v144 > (unsigned __int64)&v181
                      || (unsigned __int64)&v181 >= v144 + 24 * (unsigned __int64)(unsigned int)v145 )
                    {
                      v31 = &v181;
                      sub_F39130((__int64)&v144, v32, (unsigned int)v145, v144, v23, v24);
                      v30 = (unsigned int)v145;
                      v22 = v144;
                      v33 = v145;
                    }
                    else
                    {
                      v76 = (char *)&v181 - v144;
                      sub_F39130((__int64)&v144, v32, (unsigned int)v145, v144, v23, v24);
                      v22 = v144;
                      v30 = (unsigned int)v145;
                      v31 = (__int64 *)&v76[v144];
                      v33 = v145;
                    }
                  }
                  v34 = (unsigned __int64 *)(v22 + 24 * v30);
                  if ( v34 )
                  {
                    *v34 = 6;
                    v35 = v31[2];
                    v34[1] = 0;
                    v34[2] = v35;
                    if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
                      sub_BD6050(v34, *v31 & 0xFFFFFFFFFFFFFFF8LL);
                    v33 = v145;
                  }
                  LODWORD(v145) = v33 + 1;
                  LOBYTE(v22) = v183 + 4096 != 0;
                  if ( ((v183 != 0) & (unsigned __int8)v22) != 0 && v183 != (char *)-8192LL )
                    break;
                }
              }
            }
          }
          v27 = (char *)*((_QWORD *)v27 + 1);
          if ( i == v27 )
            goto LABEL_58;
        }
        sub_BD60C0(&v181);
      }
LABEL_58:
      v25 -= 8;
    }
    while ( (_BYTE *)v139 != v25 );
  }
  v36 = (unsigned int)v145;
  v37 = 3LL * (unsigned int)v145;
  v38 = (unsigned __int64 *)(v144 + 24LL * (unsigned int)v145);
  v138 = v38;
  if ( v38 != (unsigned __int64 *)v144 )
  {
    v136 = 0;
    v140 = v144;
    while ( 1 )
    {
      v39 = *(_QWORD *)(v140 + 16);
      if ( v39 )
      {
        if ( *(_BYTE *)v39 == 85 )
        {
          v40 = *(_QWORD *)(v39 - 32);
          if ( v40 )
          {
            if ( !*(_BYTE *)v40 )
            {
              v38 = *(unsigned __int64 **)(v39 + 80);
              if ( *(unsigned __int64 **)(v40 + 24) == v38 && (*(_BYTE *)(v40 + 33) & 0x20) != 0 )
              {
                v41 = *(_DWORD *)(v40 + 36);
                if ( v41 == 206 )
                {
                  v81 = *(unsigned __int8 **)(v39 - 32LL * (*(_DWORD *)(v39 + 4) & 0x7FFFFFF));
                  if ( *v81 <= 0x15u && (unsigned __int8)sub_AC2F40(v81, (__int64)v38, v37, v22) )
                    v42 = (unsigned __int8 **)sub_AD6400(*(_QWORD *)(v39 + 8));
                  else
                    v42 = (unsigned __int8 **)sub_AD6450(*(_QWORD *)(v39 + 8));
                  goto LABEL_71;
                }
                if ( v41 == 282 )
                {
                  v42 = (unsigned __int8 **)sub_D64C80(*(_QWORD *)(v140 + 16), v134, a2, 1);
LABEL_71:
                  v43 = v187;
                  v177 = 0x800000000LL;
                  if ( !LOBYTE(v187[87]) )
                    v43 = 0;
                  v172 = 0;
                  v135 = (__int64)v43;
                  v173 = 0;
                  v174 = 0;
                  v175 = 0;
                  v176 = (__int64 *)v178;
                  sub_1021A90((unsigned __int8 *)v39, v42, 0, 0, 0, (__int64)&v172);
                  v48 = v176;
                  v182 = 0x800000000LL;
                  v49 = (unsigned int)v177;
                  v181 = (__int64)&v183;
                  v50 = &v176[v49];
                  v51 = (v49 * 8) >> 3;
                  if ( (unsigned __int64)v49 > 8 )
                  {
                    sub_D6B130((__int64)&v181, (v49 * 8) >> 3, v44, v45, v46, v47);
                    v53 = v182;
                    v52 = (char *)v181;
                    v54 = (char **)(v181 + 24LL * (unsigned int)v182);
                  }
                  else
                  {
                    v52 = (char *)&v183;
                    v53 = 0;
                    v54 = &v183;
                  }
                  if ( v48 != v50 )
                  {
                    do
                    {
                      if ( v54 )
                      {
                        v55 = (char *)*v48;
                        *v54 = (char *)4;
                        v54[1] = 0;
                        v54[2] = v55;
                        if ( v55 != 0 && v55 + 4096 != 0 && v55 != (char *)-8192LL )
                          sub_BD73F0((__int64)v54);
                      }
                      ++v48;
                      v54 += 3;
                    }
                    while ( v50 != v48 );
                    v52 = (char *)v181;
                    v53 = v182;
                  }
                  LODWORD(v182) = v53 + v51;
                  i = &v52[24 * (unsigned int)(v53 + v51)];
                  if ( v52 != i )
                  {
                    v137 = 0;
                    v56 = v52;
                    while ( 1 )
                    {
                      v57 = *((_QWORD *)v56 + 2);
                      if ( !v57 )
                        goto LABEL_84;
                      if ( *(_BYTE *)v57 != 31 )
                        goto LABEL_84;
                      v58 = *(_DWORD *)(v57 + 4) & 0x7FFFFFF;
                      if ( (_DWORD)v58 == 1 )
                        goto LABEL_84;
                      v59 = *(_QWORD *)(v57 - 32 * v58);
                      v60 = *(_BYTE *)v59;
                      if ( *(_BYTE *)v59 > 0x15u )
                        goto LABEL_148;
                      if ( !sub_AC30F0(v59) )
                      {
                        if ( *(_BYTE *)v59 == 17 )
                        {
                          if ( *(_DWORD *)(v59 + 32) <= 0x40u )
                          {
                            v82 = *(_QWORD *)(v59 + 24) == 0;
                          }
                          else
                          {
                            v128 = *(_DWORD *)(v59 + 32);
                            v82 = v128 == (unsigned int)sub_C444A0(v59 + 24);
                          }
                        }
                        else
                        {
                          v131 = *(_QWORD *)(v59 + 8);
                          if ( (unsigned int)*(unsigned __int8 *)(v131 + 8) - 17 > 1 )
                            goto LABEL_144;
                          v97 = sub_AD7630(v59, 0, v61);
                          if ( !v97 || *v97 != 17 )
                          {
                            if ( *(_BYTE *)(v131 + 8) == 17 )
                            {
                              v133 = *(_DWORD *)(v131 + 32);
                              if ( v133 )
                              {
                                v125 = (unsigned __int8 *)v59;
                                v104 = 0;
                                v105 = 0;
                                while ( 1 )
                                {
                                  v121 = v104;
                                  v106 = sub_AD69F0(v125, v104);
                                  if ( !v106 )
                                    break;
                                  v107 = v121;
                                  if ( *(_BYTE *)v106 != 13 )
                                  {
                                    if ( *(_BYTE *)v106 != 17 )
                                      break;
                                    v108 = *(_DWORD *)(v106 + 32);
                                    if ( v108 <= 0x40 )
                                    {
                                      v105 = *(_QWORD *)(v106 + 24) == 0;
                                    }
                                    else
                                    {
                                      v109 = sub_C444A0(v106 + 24);
                                      v107 = v121;
                                      v105 = v108 == v109;
                                    }
                                    if ( !v105 )
                                      break;
                                  }
                                  v104 = v107 + 1;
                                  if ( v133 == v104 )
                                  {
                                    if ( v105 )
                                      goto LABEL_90;
                                    break;
                                  }
                                }
                              }
                            }
LABEL_144:
                            v59 = *(_QWORD *)(v57 - 32LL * (*(_DWORD *)(v57 + 4) & 0x7FFFFFF));
                            v60 = *(_BYTE *)v59;
                            if ( *(_BYTE *)v59 == 17 )
                            {
                              if ( *(_DWORD *)(v59 + 32) > 0x40u )
                              {
                                v129 = *(_DWORD *)(v59 + 32);
                                if ( (unsigned int)sub_C444A0(v59 + 24) != v129 - 1 )
                                  goto LABEL_84;
                                goto LABEL_147;
                              }
                              v86 = *(_QWORD *)(v59 + 24) == 1;
LABEL_180:
                              if ( !v86 )
                                goto LABEL_84;
                            }
                            else
                            {
LABEL_148:
                              v130 = *(_QWORD *)(v59 + 8);
                              v83 = (unsigned int)*(unsigned __int8 *)(v130 + 8) - 17;
                              if ( (unsigned int)v83 > 1 || v60 > 0x15u )
                                goto LABEL_84;
                              v84 = sub_AD7630(v59, 0, v83);
                              if ( !v84 || *v84 != 17 )
                              {
                                if ( *(_BYTE *)(v130 + 8) == 17 )
                                {
                                  v124 = *(_DWORD *)(v130 + 32);
                                  if ( v124 )
                                  {
                                    v132 = (unsigned __int8 *)v59;
                                    v120 = v57;
                                    v99 = 0;
                                    v100 = 0;
                                    while ( 1 )
                                    {
                                      v101 = sub_AD69F0(v132, v99);
                                      if ( !v101 )
                                        break;
                                      if ( *(_BYTE *)v101 != 13 )
                                      {
                                        if ( *(_BYTE *)v101 != 17 )
                                          break;
                                        v102 = *(_DWORD *)(v101 + 32);
                                        v100 = v102 <= 0x40
                                             ? *(_QWORD *)(v101 + 24) == 1
                                             : v102 - 1 == (unsigned int)sub_C444A0(v101 + 24);
                                        if ( !v100 )
                                          break;
                                      }
                                      if ( v124 == ++v99 )
                                      {
                                        v103 = v100;
                                        v57 = v120;
                                        if ( v103 )
                                          goto LABEL_147;
                                        goto LABEL_84;
                                      }
                                    }
                                  }
                                }
                                goto LABEL_84;
                              }
                              v85 = *((_DWORD *)v84 + 8);
                              if ( v85 > 0x40 )
                              {
                                v86 = v85 - 1 == (unsigned int)sub_C444A0((__int64)(v84 + 24));
                                goto LABEL_180;
                              }
                              if ( *((_QWORD *)v84 + 3) != 1 )
                                goto LABEL_84;
                            }
LABEL_147:
                            v62 = *(_QWORD *)(v57 - 32);
                            v63 = *(_QWORD *)(v57 - 64);
                            goto LABEL_91;
                          }
                          v98 = *((_DWORD *)v97 + 8);
                          if ( v98 <= 0x40 )
                            v82 = *((_QWORD *)v97 + 3) == 0;
                          else
                            v82 = v98 == (unsigned int)sub_C444A0((__int64)(v97 + 24));
                        }
                        if ( !v82 )
                          goto LABEL_144;
                      }
LABEL_90:
                      v62 = *(_QWORD *)(v57 - 64);
                      v63 = *(_QWORD *)(v57 - 32);
LABEL_91:
                      v127 = v62 != v63 && v62 != 0;
                      if ( v127 )
                      {
                        v111 = v63;
                        v123 = *(_QWORD *)(v57 + 40);
                        sub_AA5980(v63, v123, 0);
                        sub_B43C20((__int64)&v167, v123);
                        v110 = v167;
                        v114 = v168;
                        v64 = sub_BD2C40(72, 1u);
                        v65 = v111;
                        v66 = v64;
                        if ( v64 )
                        {
                          v67 = v114;
                          v115 = v64;
                          sub_B4C8F0((__int64)v64, v62, 1u, v110, v67);
                          v65 = v111;
                          v66 = v115;
                        }
                        v68 = *(_QWORD *)(v57 + 48);
                        v69 = v66 + 6;
                        v167 = v68;
                        if ( !v68 )
                        {
                          if ( v69 == (unsigned __int64 *)&v167 )
                            goto LABEL_98;
                          v87 = v66[6];
                          if ( !v87 )
                            goto LABEL_98;
LABEL_156:
                          v113 = v66;
                          v118 = v65;
                          sub_B91220((__int64)v69, v87);
                          v66 = v113;
                          v65 = v118;
                          goto LABEL_157;
                        }
                        v116 = v65;
                        v112 = v66;
                        sub_B96E90((__int64)&v167, v68, 1);
                        v65 = v116;
                        if ( v69 == (unsigned __int64 *)&v167 )
                        {
                          if ( v167 )
                          {
                            sub_B91220((__int64)&v167, v167);
                            v65 = v116;
                          }
                          goto LABEL_98;
                        }
                        v66 = v112;
                        v87 = v112[6];
                        if ( v87 )
                          goto LABEL_156;
LABEL_157:
                        v88 = (unsigned __int8 *)v167;
                        v66[6] = v167;
                        if ( v88 )
                        {
                          v119 = v65;
                          sub_B976B0((__int64)&v167, v88, (__int64)v69);
                          v65 = v119;
                        }
LABEL_98:
                        v117 = v65;
                        sub_B43D60((_QWORD *)v57);
                        v72 = v117;
                        if ( v135 )
                        {
                          v167 = v123;
                          v168 = v117 | 4;
                          sub_FFB3D0(v135, (unsigned __int64 *)&v167, 1, v70, v71, v117);
                          v72 = v117;
                        }
                        v73 = *(_QWORD *)(v72 + 16);
                        if ( v73 )
                        {
                          while ( (unsigned __int8)(**(_BYTE **)(v73 + 24) - 30) > 0xAu )
                          {
                            v73 = *(_QWORD *)(v73 + 8);
                            if ( !v73 )
                              goto LABEL_103;
                          }
                          goto LABEL_84;
                        }
LABEL_103:
                        v56 += 24;
                        v137 = v127;
                        if ( i == v56 )
                        {
LABEL_104:
                          v74 = (char *)v181;
                          v136 |= v137;
                          i = (char *)(v181 + 24LL * (unsigned int)v182);
                          if ( (char *)v181 != i )
                          {
                            do
                            {
                              v75 = *((_QWORD *)i - 1);
                              i -= 24;
                              if ( v75 != 0 && v75 != -4096 && v75 != -8192 )
                                sub_BD60C0(i);
                            }
                            while ( v74 != i );
                            i = (char *)v181;
                          }
                          break;
                        }
                      }
                      else
                      {
LABEL_84:
                        v56 += 24;
                        if ( i == v56 )
                          goto LABEL_104;
                      }
                    }
                  }
                  if ( i != (char *)&v183 )
                    _libc_free((unsigned __int64)i);
                  if ( v176 != (__int64 *)v178 )
                    _libc_free((unsigned __int64)v176);
                  v38 = (unsigned __int64 *)(8LL * (unsigned int)v175);
                  sub_C7D6A0(v173, (__int64)v38, 8);
                }
              }
            }
          }
        }
      }
      v140 += 24;
      if ( (unsigned __int64 *)v140 == v138 )
      {
        if ( v136 )
        {
          v38 = 0;
          if ( LOBYTE(v187[87]) )
            v38 = v187;
          sub_F62E00(a1, (__int64)v38, 0, v22, v23, v24);
        }
        v36 = (unsigned int)v145;
        break;
      }
    }
  }
  LOBYTE(i) = (_DWORD)v36 != 0;
  if ( v141 != v143 )
  {
    _libc_free((unsigned __int64)v141);
    v36 = (unsigned int)v145;
  }
  v77 = v144;
  v78 = (_QWORD *)(v144 + 24 * v36);
  if ( (_QWORD *)v144 != v78 )
  {
    do
    {
      v79 = *(v78 - 1);
      v78 -= 3;
      LOBYTE(v22) = v79 != -4096;
      LOBYTE(v37) = v79 != 0;
      if ( ((v79 != 0) & (unsigned __int8)v22) != 0 && v79 != -8192 )
        sub_BD60C0(v78);
    }
    while ( (_QWORD *)v77 != v78 );
    v78 = (_QWORD *)v144;
  }
  if ( v78 != (_QWORD *)v146 )
    _libc_free((unsigned __int64)v78);
  if ( LOBYTE(v187[87]) )
  {
    LOBYTE(v187[87]) = 0;
    sub_FFCE90((__int64)v187, (__int64)v38, v37, v22, v23, v24);
    sub_FFD870((__int64)v187, (__int64)v38, v89, v90, v91, v92);
    sub_FFBC40((__int64)v187, (__int64)v38);
    v93 = v187[85];
    v94 = (_QWORD *)v187[84];
    if ( v187[85] != v187[84] )
    {
      do
      {
        v95 = (void (__fastcall *)(_QWORD, _QWORD, _QWORD))v94[7];
        *v94 = &unk_49E5048;
        if ( v95 )
          v95(v94 + 5, v94 + 5, 3);
        *v94 = &unk_49DB368;
        v96 = v94[3];
        if ( v96 != 0 && v96 != -4096 && v96 != -8192 )
          sub_BD60C0(v94 + 1);
        v94 += 9;
      }
      while ( (_QWORD *)v93 != v94 );
      v94 = (_QWORD *)v187[84];
    }
    if ( v94 )
      j_j___libc_free_0((unsigned __int64)v94);
    if ( !BYTE4(v187[74]) )
      _libc_free(v187[72]);
    if ( (unsigned __int64 *)v187[0] != &v187[2] )
      _libc_free(v187[0]);
  }
  return (unsigned int)i;
}
