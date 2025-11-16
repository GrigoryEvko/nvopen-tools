// Function: sub_28D4060
// Address: 0x28d4060
//
_QWORD *__fastcall sub_28D4060(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v3; // r14
  __int64 *v4; // r12
  __int64 v5; // rbx
  __int64 *v6; // rax
  __int64 v7; // r13
  int v8; // eax
  __int64 *v9; // rax
  char v10; // al
  __int64 v11; // rdx
  unsigned int v12; // esi
  __int64 v13; // r13
  unsigned int v14; // edi
  int v15; // r14d
  __int64 **v16; // r10
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  __int64 *v19; // r8
  _QWORD *result; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned int v24; // eax
  __int64 v25; // r10
  unsigned int v26; // edx
  int v27; // eax
  _BYTE *v28; // rax
  _BYTE *v29; // r15
  __int64 *v30; // rsi
  __int64 v31; // r8
  __int64 v32; // rdi
  __int64 *v33; // rdx
  unsigned int v34; // r15d
  __int64 *v35; // rax
  __int64 v36; // rcx
  __int64 *v37; // rax
  int v38; // r15d
  __int64 v39; // rcx
  int v40; // r14d
  __int64 v41; // rax
  int v42; // r11d
  unsigned int v43; // r14d
  __int64 v44; // r15
  _DWORD *v45; // rsi
  __int64 v46; // r13
  int j; // r12d
  __int64 v48; // r10
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // eax
  int v52; // eax
  char v53; // al
  __int64 v54; // rax
  __int64 *v55; // rdx
  __int64 v56; // rdx
  __int64 *v57; // rcx
  __int64 v58; // rsi
  int v59; // ecx
  __int64 v60; // rdi
  int v61; // ecx
  unsigned int v62; // edx
  __int64 *v63; // rax
  __int64 v64; // r8
  unsigned int v65; // ecx
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 *v68; // rax
  __int64 v69; // rsi
  int v70; // ecx
  unsigned int v71; // edx
  __int64 *v72; // rax
  __int64 v73; // r9
  unsigned int v74; // ecx
  __int64 v75; // rdx
  __int64 v76; // rax
  int v77; // ecx
  __int64 v78; // rdi
  __int64 v79; // r8
  __int64 v80; // rax
  __int64 v81; // rsi
  unsigned int v82; // edx
  __int64 v83; // r14
  __int64 *v84; // rcx
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r10
  __int64 *v88; // rax
  int v89; // ecx
  __int64 v90; // rsi
  __int64 v91; // rdi
  int v92; // ecx
  unsigned int v93; // edx
  __int64 *v94; // rax
  __int64 v95; // r9
  unsigned int v96; // ecx
  __int64 v97; // rdx
  __int64 v98; // rax
  __int64 v99; // rsi
  __int64 v100; // rax
  __int64 v101; // rsi
  unsigned int v102; // edx
  __int64 v103; // r14
  __int64 *v104; // rcx
  __int64 v105; // rax
  __int64 *v106; // rdx
  __int64 v107; // rdx
  __int64 *v108; // rax
  int v109; // ecx
  __int64 v110; // rsi
  __int64 v111; // rdi
  int v112; // ecx
  unsigned int v113; // edx
  __int64 *v114; // rax
  __int64 v115; // r8
  unsigned int v116; // ecx
  __int64 v117; // rdx
  __int64 v118; // rax
  unsigned int v119; // ecx
  __int64 **v120; // rax
  __int64 *v121; // r10
  __int64 *v122; // r13
  int v123; // eax
  int v124; // r10d
  unsigned int v125; // esi
  int v126; // eax
  __int64 *v127; // r8
  int v128; // eax
  int v129; // eax
  __int64 v130; // rax
  int v131; // eax
  int v132; // eax
  int v133; // eax
  unsigned __int64 v134; // rdi
  int v135; // r14d
  int v136; // eax
  int v137; // r12d
  unsigned int v138; // r15d
  int i; // ebx
  __int64 **v140; // r14
  char v141; // al
  __int64 v142; // rsi
  int v143; // eax
  __int64 v144; // r11
  int v145; // eax
  int v146; // eax
  __int64 v147; // r14
  __int64 v148; // rax
  int v149; // eax
  __int64 v150; // rdi
  int v151; // eax
  int v152; // eax
  int v153; // r8d
  int v154; // r9d
  int v155; // r9d
  int k; // r10d
  unsigned int v157; // eax
  int v158; // r14d
  int v159; // ebx
  int v160; // r12d
  unsigned int m; // r15d
  __int64 *v162; // r14
  __int64 v163; // rsi
  int v164; // r10d
  int v165; // eax
  int v166; // eax
  int v167; // r8d
  unsigned int v168; // r9d
  int v169; // eax
  unsigned int v170; // r15d
  unsigned int v171; // r15d
  int v172; // [rsp+14h] [rbp-CCh]
  int v173; // [rsp+14h] [rbp-CCh]
  __int64 v174; // [rsp+18h] [rbp-C8h]
  int v175; // [rsp+18h] [rbp-C8h]
  __int64 *v176; // [rsp+18h] [rbp-C8h]
  __int64 *v177; // [rsp+20h] [rbp-C0h]
  __int64 v178; // [rsp+20h] [rbp-C0h]
  __int64 *v179; // [rsp+28h] [rbp-B8h]
  __int64 v180; // [rsp+28h] [rbp-B8h]
  __int64 v181; // [rsp+30h] [rbp-B0h]
  __int64 v182; // [rsp+30h] [rbp-B0h]
  __int64 *v183; // [rsp+30h] [rbp-B0h]
  __int64 *v184; // [rsp+30h] [rbp-B0h]
  __int64 v185; // [rsp+30h] [rbp-B0h]
  __int64 *v186; // [rsp+30h] [rbp-B0h]
  __int64 v188; // [rsp+48h] [rbp-98h]
  __int64 v189; // [rsp+48h] [rbp-98h]
  __int64 *v190; // [rsp+58h] [rbp-88h] BYREF
  __int64 *v191; // [rsp+60h] [rbp-80h] BYREF
  __int64 *v192; // [rsp+68h] [rbp-78h]
  __int64 v193; // [rsp+70h] [rbp-70h]
  __int64 v194; // [rsp+78h] [rbp-68h]
  __int64 *v195; // [rsp+80h] [rbp-60h] BYREF
  __int64 v196; // [rsp+88h] [rbp-58h]
  __int64 v197; // [rsp+90h] [rbp-50h]
  __int64 v198; // [rsp+98h] [rbp-48h]

  v3 = a1 + 1432;
  v4 = (__int64 *)a2;
  v5 = a1;
  v195 = (__int64 *)a2;
  v6 = sub_28C7580(a1 + 1432, (__int64 *)&v195);
  v7 = (__int64)v6;
  if ( v6 )
    v7 = v6[1];
  v8 = a3[2];
  if ( v8 == 2 )
  {
    v195 = (__int64 *)*((_QWORD *)a3 + 3);
    v9 = sub_28C7580(v3, (__int64 *)&v195);
    if ( !v9 )
      goto LABEL_16;
    v188 = v9[1];
  }
  else
  {
    if ( v8 != 3 )
      goto LABEL_16;
    v188 = *(_QWORD *)(a1 + 1392);
  }
  if ( v188 )
    goto LABEL_7;
LABEL_16:
  v196 = 0;
  v195 = (__int64 *)a3;
  if ( (unsigned __int8)sub_28C8F20(a1 + 2016, (__int64 *)&v195, &v190) )
  {
    v188 = v190[1];
    goto LABEL_7;
  }
  v125 = *(_DWORD *)(a1 + 2040);
  v126 = *(_DWORD *)(a1 + 2032);
  v127 = v190;
  ++*(_QWORD *)(a1 + 2016);
  v128 = v126 + 1;
  v191 = v127;
  if ( 4 * v128 >= 3 * v125 )
  {
    v125 *= 2;
    goto LABEL_211;
  }
  if ( v125 - *(_DWORD *)(a1 + 2036) - v128 <= v125 >> 3 )
  {
LABEL_211:
    sub_28C99C0(a1 + 2016, v125);
    sub_28C8F20(a1 + 2016, (__int64 *)&v195, &v191);
    v127 = v191;
    v128 = *(_DWORD *)(a1 + 2032) + 1;
  }
  *(_DWORD *)(a1 + 2032) = v128;
  if ( *v127 != -8 )
    --*(_DWORD *)(a1 + 2036);
  v183 = v127;
  *v127 = (__int64)v195;
  v127[1] = v196;
  v188 = sub_28CC470(a1, 0, (__int64)a3);
  v183[1] = v188;
  v129 = a3[2];
  if ( v129 == 1 )
  {
    v130 = *((_QWORD *)a3 + 3);
    *(_DWORD *)(v188 + 16) = 0;
    *(_QWORD *)(v188 + 8) = v130;
  }
  else if ( v129 == 12 )
  {
    v142 = *((_QWORD *)a3 + 7);
    v143 = sub_28C8CE0(a1, v142);
    *(_QWORD *)(v188 + 8) = v142;
    *(_DWORD *)(v188 + 16) = v143;
    *(_QWORD *)(v188 + 40) = *(_QWORD *)(v144 + 64);
  }
  else
  {
    v152 = sub_28C8CE0(a1, (__int64)v4);
    *(_QWORD *)(v188 + 8) = v4;
    *(_DWORD *)(v188 + 16) = v152;
  }
LABEL_7:
  v10 = sub_25DDDB0(a1 + 2056, (__int64)v4);
  if ( v188 == v7 && !v10 )
    goto LABEL_9;
  if ( v188 == v7 )
    goto LABEL_75;
  if ( v4 == *(__int64 **)(v7 + 24) )
  {
    *(_QWORD *)(v7 + 24) = 0;
    *(_DWORD *)(v7 + 32) = -1;
  }
  sub_25DDDB0(v7 + 64, (__int64)v4);
  sub_2411830((__int64)&v195, v188 + 64, v4, v21, v22, v23);
  if ( v4 != *(__int64 **)(v188 + 8) )
  {
    v24 = sub_28C8CE0(a1, (__int64)v4);
    v26 = *(_DWORD *)(v188 + 16);
    if ( v24 < v26 )
    {
      *(_QWORD *)(v188 + 24) = v25;
      *(_DWORD *)(v188 + 32) = v26;
      *(_QWORD *)(v188 + 8) = v4;
      *(_DWORD *)(v188 + 16) = v24;
      sub_28CADF0(a1, v188);
    }
    else if ( v24 < *(_DWORD *)(v188 + 32) )
    {
      *(_QWORD *)(v188 + 24) = v4;
      *(_DWORD *)(v188 + 32) = v24;
    }
  }
  if ( *(_BYTE *)v4 == 62 )
  {
    --*(_DWORD *)(v7 + 176);
    v27 = *(_DWORD *)(v188 + 176);
    if ( !v27 && !*(_QWORD *)(v188 + 40) && a3[2] == 12 )
    {
      *(_QWORD *)(v188 + 40) = *((_QWORD *)a3 + 8);
      sub_28CADF0(a1, v188);
      v165 = sub_28C8CE0(a1, (__int64)v4);
      *(_QWORD *)(v188 + 8) = v4;
      *(_DWORD *)(v188 + 16) = v165;
      v27 = *(_DWORD *)(v188 + 176);
    }
    *(_DWORD *)(v188 + 176) = v27 + 1;
  }
  v28 = (_BYTE *)sub_28C8480(a1, (__int64)v4);
  v29 = v28;
  if ( v28 && *v28 == 27 )
  {
    if ( !*(_QWORD *)(v188 + 48) )
    {
      *(_QWORD *)(v188 + 48) = v28;
      sub_28CABC0(a1, v188);
    }
    sub_28CC2D0(a1, v29, v188);
    if ( v29 == *(_BYTE **)(v7 + 48) )
    {
      if ( !*(_DWORD *)(v7 + 176) && *(_DWORD *)(v7 + 148) == *(_DWORD *)(v7 + 152) )
      {
        *(_QWORD *)(v7 + 48) = 0;
      }
      else
      {
        *(_QWORD *)(v7 + 48) = sub_28CBF00(a1, v7);
        sub_28CABC0(a1, v7);
      }
    }
  }
  v30 = (__int64 *)*(unsigned int *)(a1 + 1456);
  v191 = v4;
  if ( !(_DWORD)v30 )
  {
    ++*(_QWORD *)(a1 + 1432);
    v195 = 0;
    goto LABEL_191;
  }
  v31 = 1;
  v32 = *(_QWORD *)(a1 + 1440);
  v33 = 0;
  v34 = ((_DWORD)v30 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v35 = (__int64 *)(v32 + 16LL * v34);
  v36 = *v35;
  if ( v4 != (__int64 *)*v35 )
  {
    while ( v36 != -4096 )
    {
      if ( v36 == -8192 && !v33 )
        v33 = v35;
      v168 = v31 + 1;
      v31 = ((_DWORD)v30 - 1) & (v34 + (unsigned int)v31);
      v34 = v31;
      v35 = (__int64 *)(v32 + 16LL * (unsigned int)v31);
      v36 = *v35;
      if ( v4 == (__int64 *)*v35 )
        goto LABEL_36;
      v31 = v168;
    }
    if ( !v33 )
      v33 = v35;
    v169 = *(_DWORD *)(v5 + 1448);
    ++*(_QWORD *)(v5 + 1432);
    v151 = v169 + 1;
    v195 = v33;
    if ( 4 * v151 < (unsigned int)(3 * (_DWORD)v30) )
    {
      v150 = (__int64)v4;
      if ( (int)v30 - *(_DWORD *)(v5 + 1452) - v151 > (unsigned int)v30 >> 3 )
        goto LABEL_193;
      goto LABEL_192;
    }
LABEL_191:
    LODWORD(v30) = 2 * (_DWORD)v30;
LABEL_192:
    sub_28C9810(v3, (int)v30);
    v30 = (__int64 *)&v191;
    sub_28C74C0(v3, (__int64 *)&v191, &v195);
    v150 = (__int64)v191;
    v33 = v195;
    v151 = *(_DWORD *)(v5 + 1448) + 1;
LABEL_193:
    *(_DWORD *)(v5 + 1448) = v151;
    if ( *v33 != -4096 )
      --*(_DWORD *)(v5 + 1452);
    *v33 = v150;
    v37 = v33 + 1;
    v33[1] = 0;
    goto LABEL_37;
  }
LABEL_36:
  v37 = v35 + 1;
LABEL_37:
  *v37 = v188;
  if ( *(_DWORD *)(v7 + 84) == *(_DWORD *)(v7 + 88) && v7 != *(_QWORD *)(v5 + 1392) )
  {
    v33 = *(__int64 **)(v7 + 56);
    if ( !v33 )
      goto LABEL_39;
    v135 = *(_DWORD *)(v5 + 2040);
    v178 = *(_QWORD *)(v5 + 2024);
    if ( !v135 )
      goto LABEL_39;
    v184 = *(__int64 **)(v7 + 56);
    v136 = sub_28CB200(v33);
    v179 = v4;
    v33 = v184;
    v137 = v135 - 1;
    v185 = v5;
    v138 = (v135 - 1) & v136;
    for ( i = 1; ; ++i )
    {
      v140 = (__int64 **)(v178 + 16LL * v138);
      v30 = *v140;
      if ( *v140 == (__int64 *)-8LL )
      {
LABEL_246:
        v5 = v185;
        v4 = v179;
        goto LABEL_39;
      }
      if ( v30 != (__int64 *)0x7FFFFFFF0LL )
      {
        v176 = v33;
        v141 = (*(__int64 (__fastcall **)(__int64 *))(*v33 + 24))(v33);
        v33 = v176;
        if ( v141 )
        {
          v5 = v185;
          v4 = v179;
          if ( v140 != (__int64 **)(*(_QWORD *)(v185 + 2024) + 16LL * *(unsigned int *)(v185 + 2040)) )
          {
            *v140 = (__int64 *)0x7FFFFFFF0LL;
            --*(_DWORD *)(v185 + 2032);
            ++*(_DWORD *)(v185 + 2036);
          }
          goto LABEL_39;
        }
        v30 = *v140;
      }
      if ( v30 == (__int64 *)-8LL )
        goto LABEL_246;
      v170 = i + v138;
      v138 = v137 & v170;
    }
  }
  if ( v4 == *(__int64 **)(v7 + 8) )
  {
    if ( !*(_DWORD *)(v7 + 176) && *(_QWORD *)(v7 + 40) )
      *(_QWORD *)(v7 + 40) = 0;
    v147 = sub_28CB5E0(v5, v7);
    v148 = sub_28CB5E0(v5, v7);
    v149 = sub_28C8CE0(v5, v148);
    *(_QWORD *)(v7 + 8) = v147;
    v30 = (__int64 *)v7;
    *(_DWORD *)(v7 + 16) = v149;
    *(_QWORD *)(v7 + 24) = 0;
    *(_DWORD *)(v7 + 32) = -1;
    sub_28CADF0(v5, v7);
  }
LABEL_39:
  v38 = *(_DWORD *)(v5 + 1752);
  if ( !v38 )
    goto LABEL_75;
  v39 = *(_QWORD *)(v5 + 1736);
  v40 = *((_QWORD *)a3 + 2);
  if ( !v40 )
  {
    v181 = *(_QWORD *)(v5 + 1736);
    v41 = (*(__int64 (__fastcall **)(_DWORD *, __int64 *, __int64 *, __int64, __int64))(*(_QWORD *)a3 + 32LL))(
            a3,
            v30,
            v33,
            v39,
            v31);
    v39 = v181;
    v40 = v41;
    *((_QWORD *)a3 + 2) = v41;
  }
  v42 = v38 - 1;
  v43 = (v38 - 1) & v40;
  v44 = v39 + 56LL * v43;
  v45 = *(_DWORD **)v44;
  if ( a3 != *(_DWORD **)v44 )
  {
    v182 = v7;
    v46 = v39;
    v177 = v4;
    for ( j = 1; ; ++j )
    {
      if ( v45 != (_DWORD *)0x7FFFFFFF0LL && v45 + 2 != 0 && a3 != (_DWORD *)0x7FFFFFFF0LL && a3 + 2 != 0 )
      {
        v48 = *((_QWORD *)v45 + 2);
        if ( !(_DWORD)v48 )
        {
          v172 = v42;
          v49 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)v45 + 32LL))(v45);
          v42 = v172;
          v48 = v49;
          *((_QWORD *)v45 + 2) = v49;
        }
        v50 = *((_QWORD *)a3 + 2);
        if ( !(_DWORD)v50 )
        {
          v173 = v42;
          v174 = v48;
          v50 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a3 + 32LL))(a3);
          v42 = v173;
          v48 = v174;
          *((_QWORD *)a3 + 2) = v50;
        }
        if ( v50 == v48 )
        {
          v51 = a3[3];
          if ( v51 == v45[3] )
          {
            if ( v51 > 0xFFFFFFFD )
              break;
            v52 = a3[2];
            if ( (unsigned int)(v52 - 11) <= 1 || v52 == v45[2] )
            {
              v175 = v42;
              v53 = (*(__int64 (__fastcall **)(_DWORD *))(*(_QWORD *)a3 + 16LL))(a3);
              v42 = v175;
              if ( v53 )
                break;
            }
          }
        }
      }
      if ( *(_QWORD *)v44 == -8 )
      {
        v7 = v182;
        v4 = v177;
        goto LABEL_75;
      }
      v43 = v42 & (j + v43);
      v44 = v46 + 56LL * v43;
      v45 = *(_DWORD **)v44;
      if ( a3 == *(_DWORD **)v44 )
        break;
    }
    v7 = v182;
    v4 = v177;
  }
  if ( v44 == *(_QWORD *)(v5 + 1736) + 56LL * *(unsigned int *)(v5 + 1752) )
    goto LABEL_75;
  v54 = *(_QWORD *)(v44 + 16);
  if ( *(_BYTE *)(v44 + 36) )
    v55 = (__int64 *)(v54 + 8LL * *(unsigned int *)(v44 + 28));
  else
    v55 = (__int64 *)(v54 + 8LL * *(unsigned int *)(v44 + 24));
  v191 = *(__int64 **)(v44 + 16);
  v192 = v55;
  sub_254BBF0((__int64)&v191);
  v193 = v44 + 8;
  v194 = *(_QWORD *)(v44 + 8);
  if ( *(_BYTE *)(v44 + 36) )
    v56 = *(unsigned int *)(v44 + 28);
  else
    v56 = *(unsigned int *)(v44 + 24);
  v195 = (__int64 *)(*(_QWORD *)(v44 + 16) + 8 * v56);
  v196 = (__int64)v195;
  sub_254BBF0((__int64)&v195);
  v197 = v44 + 8;
  v57 = v191;
  v198 = *(_QWORD *)(v44 + 8);
  if ( v191 != v195 )
  {
    while ( 1 )
    {
      v58 = *v57;
      v59 = *(_DWORD *)(v5 + 2440);
      v60 = *(_QWORD *)(v5 + 2424);
      if ( !v59 )
        goto LABEL_149;
      v61 = v59 - 1;
      v62 = v61 & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v63 = (__int64 *)(v60 + 16LL * v62);
      v64 = *v63;
      if ( *v63 != v58 )
        break;
LABEL_67:
      v65 = *((_DWORD *)v63 + 2);
      v66 = 1LL << v65;
      v67 = 8LL * (v65 >> 6);
LABEL_68:
      *(_QWORD *)(*(_QWORD *)(v5 + 2280) + v67) |= v66;
      v57 = v192;
      v68 = v191 + 1;
      v191 = v68;
      if ( v68 != v192 )
      {
        while ( (unsigned __int64)(*v68 + 2) <= 1 )
        {
          v191 = ++v68;
          if ( v68 == v192 )
            goto LABEL_71;
        }
        v57 = v191;
      }
LABEL_71:
      if ( v195 == v57 )
        goto LABEL_72;
    }
    v133 = 1;
    while ( v64 != -4096 )
    {
      v155 = v133 + 1;
      v62 = v61 & (v133 + v62);
      v63 = (__int64 *)(v60 + 16LL * v62);
      v64 = *v63;
      if ( v58 == *v63 )
        goto LABEL_67;
      v133 = v155;
    }
LABEL_149:
    v66 = 1;
    v67 = 0;
    goto LABEL_68;
  }
LABEL_72:
  if ( !*(_BYTE *)(v44 + 36) )
    _libc_free(*(_QWORD *)(v44 + 16));
  *(_QWORD *)v44 = 0x7FFFFFFF0LL;
  --*(_DWORD *)(v5 + 1744);
  ++*(_DWORD *)(v5 + 1748);
LABEL_75:
  v69 = v4[2];
  if ( v69 )
  {
    while ( 1 )
    {
      v77 = *(_DWORD *)(v5 + 2440);
      v78 = *(_QWORD *)(v69 + 24);
      v79 = *(_QWORD *)(v5 + 2424);
      if ( !v77 )
        goto LABEL_81;
      v70 = v77 - 1;
      v71 = v70 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
      v72 = (__int64 *)(v79 + 16LL * v71);
      v73 = *v72;
      if ( v78 != *v72 )
        break;
LABEL_78:
      v74 = *((_DWORD *)v72 + 2);
      v75 = 1LL << v74;
      v76 = 8LL * (v74 >> 6);
LABEL_79:
      *(_QWORD *)(*(_QWORD *)(v5 + 2280) + v76) |= v75;
      v69 = *(_QWORD *)(v69 + 8);
      if ( !v69 )
        goto LABEL_82;
    }
    v123 = 1;
    while ( v73 != -4096 )
    {
      v124 = v123 + 1;
      v71 = v70 & (v123 + v71);
      v72 = (__int64 *)(v79 + 16LL * v71);
      v73 = *v72;
      if ( v78 == *v72 )
        goto LABEL_78;
      v123 = v124;
    }
LABEL_81:
    v75 = 1;
    v76 = 0;
    goto LABEL_79;
  }
LABEL_82:
  v80 = *(unsigned int *)(v5 + 1720);
  v81 = *(_QWORD *)(v5 + 1704);
  if ( !(_DWORD)v80 )
    goto LABEL_98;
  v82 = (v80 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v83 = v81 + 56LL * v82;
  v84 = *(__int64 **)v83;
  if ( v4 != *(__int64 **)v83 )
  {
    for ( k = 1; ; ++k )
    {
      if ( v84 == (__int64 *)-4096LL )
        goto LABEL_98;
      v82 = (v80 - 1) & (k + v82);
      v83 = v81 + 56LL * v82;
      v84 = *(__int64 **)v83;
      if ( v4 == *(__int64 **)v83 )
        break;
    }
  }
  if ( v83 == v81 + 56 * v80 )
    goto LABEL_98;
  v85 = *(_QWORD *)(v83 + 16);
  if ( *(_BYTE *)(v83 + 36) )
    v86 = v85 + 8LL * *(unsigned int *)(v83 + 28);
  else
    v86 = v85 + 8LL * *(unsigned int *)(v83 + 24);
  v195 = *(__int64 **)(v83 + 16);
  v196 = v86;
  sub_254BBF0((__int64)&v195);
  v197 = v83 + 8;
  v198 = *(_QWORD *)(v83 + 8);
  if ( *(_BYTE *)(v83 + 36) )
  {
    v87 = *(_QWORD *)(v83 + 16) + 8LL * *(unsigned int *)(v83 + 28);
    v88 = v195;
    if ( (__int64 *)v87 != v195 )
      goto LABEL_89;
    goto LABEL_97;
  }
  v134 = *(_QWORD *)(v83 + 16);
  v87 = v134 + 8LL * *(unsigned int *)(v83 + 24);
  v88 = v195;
  if ( (__int64 *)v87 == v195 )
    goto LABEL_151;
  do
  {
LABEL_89:
    v89 = *(_DWORD *)(v5 + 2440);
    v90 = *v88;
    v91 = *(_QWORD *)(v5 + 2424);
    if ( v89 )
    {
      v92 = v89 - 1;
      v93 = v92 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
      v94 = (__int64 *)(v91 + 16LL * v93);
      v95 = *v94;
      if ( v90 == *v94 )
      {
LABEL_91:
        v96 = *((_DWORD *)v94 + 2);
        v97 = 1LL << v96;
        v98 = 8LL * (v96 >> 6);
        goto LABEL_92;
      }
      v131 = 1;
      while ( v95 != -4096 )
      {
        v153 = v131 + 1;
        v93 = v92 & (v131 + v93);
        v94 = (__int64 *)(v91 + 16LL * v93);
        v95 = *v94;
        if ( v90 == *v94 )
          goto LABEL_91;
        v131 = v153;
      }
    }
    v97 = 1;
    v98 = 0;
LABEL_92:
    *(_QWORD *)(*(_QWORD *)(v5 + 2280) + v98) |= v97;
    v88 = v195 + 1;
    v195 = v88;
    if ( (__int64 *)v196 != v88 )
    {
      while ( (unsigned __int64)(*v88 + 2) <= 1 )
      {
        v195 = ++v88;
        if ( (__int64 *)v196 == v88 )
          goto LABEL_95;
      }
      v88 = v195;
    }
LABEL_95:
    ;
  }
  while ( (__int64 *)v87 != v88 );
  if ( *(_BYTE *)(v83 + 36) )
    goto LABEL_97;
  v134 = *(_QWORD *)(v83 + 16);
LABEL_151:
  _libc_free(v134);
LABEL_97:
  *(_QWORD *)v83 = -8192;
  --*(_DWORD *)(v5 + 1712);
  ++*(_DWORD *)(v5 + 1716);
LABEL_98:
  v99 = sub_28C8480(v5, (__int64)v4);
  if ( v99 )
    sub_28CA760(v5, v99);
  if ( (unsigned __int8)(*(_BYTE *)v4 - 82) <= 1u )
  {
    v100 = *(unsigned int *)(v5 + 1880);
    v101 = *(_QWORD *)(v5 + 1864);
    if ( (_DWORD)v100 )
    {
      v102 = (v100 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v103 = v101 + 56LL * v102;
      v104 = *(__int64 **)v103;
      if ( v4 == *(__int64 **)v103 )
      {
LABEL_103:
        if ( v103 != v101 + 56 * v100 )
        {
          v105 = *(_QWORD *)(v103 + 16);
          if ( *(_BYTE *)(v103 + 36) )
            v106 = (__int64 *)(v105 + 8LL * *(unsigned int *)(v103 + 28));
          else
            v106 = (__int64 *)(v105 + 8LL * *(unsigned int *)(v103 + 24));
          v191 = *(__int64 **)(v103 + 16);
          v192 = v106;
          sub_254BBF0((__int64)&v191);
          v193 = v103 + 8;
          v194 = *(_QWORD *)(v103 + 8);
          if ( *(_BYTE *)(v103 + 36) )
            v107 = *(unsigned int *)(v103 + 28);
          else
            v107 = *(unsigned int *)(v103 + 24);
          v195 = (__int64 *)(*(_QWORD *)(v103 + 16) + 8 * v107);
          v196 = (__int64)v195;
          sub_254BBF0((__int64)&v195);
          v197 = v103 + 8;
          v198 = *(_QWORD *)(v103 + 8);
          v108 = v191;
          if ( v191 != v195 )
          {
            while ( 1 )
            {
              v109 = *(_DWORD *)(v5 + 2440);
              v110 = *v108;
              v111 = *(_QWORD *)(v5 + 2424);
              if ( !v109 )
                goto LABEL_145;
              v112 = v109 - 1;
              v113 = v112 & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
              v114 = (__int64 *)(v111 + 16LL * v113);
              v115 = *v114;
              if ( v110 != *v114 )
                break;
LABEL_111:
              v116 = *((_DWORD *)v114 + 2);
              v117 = 1LL << v116;
              v118 = 8LL * (v116 >> 6);
LABEL_112:
              *(_QWORD *)(*(_QWORD *)(v5 + 2280) + v118) |= v117;
              v108 = v191 + 1;
              v191 = v108;
              if ( v192 != v108 )
              {
                while ( (unsigned __int64)(*v108 + 2) <= 1 )
                {
                  v191 = ++v108;
                  if ( v192 == v108 )
                    goto LABEL_115;
                }
                v108 = v191;
              }
LABEL_115:
              if ( v195 == v108 )
                goto LABEL_116;
            }
            v132 = 1;
            while ( v115 != -4096 )
            {
              v154 = v132 + 1;
              v113 = v112 & (v132 + v113);
              v114 = (__int64 *)(v111 + 16LL * v113);
              v115 = *v114;
              if ( v110 == *v114 )
                goto LABEL_111;
              v132 = v154;
            }
LABEL_145:
            v117 = 1;
            v118 = 0;
            goto LABEL_112;
          }
LABEL_116:
          if ( !*(_BYTE *)(v103 + 36) )
            _libc_free(*(_QWORD *)(v103 + 16));
          *(_QWORD *)v103 = -8192;
          --*(_DWORD *)(v5 + 1872);
          ++*(_DWORD *)(v5 + 1876);
        }
      }
      else
      {
        v164 = 1;
        while ( v104 != (__int64 *)-4096LL )
        {
          v102 = (v100 - 1) & (v164 + v102);
          v103 = v101 + 56LL * v102;
          v104 = *(__int64 **)v103;
          if ( v4 == *(__int64 **)v103 )
            goto LABEL_103;
          ++v164;
        }
      }
    }
  }
  v11 = *(_QWORD *)(v5 + 1472);
  v12 = *(_DWORD *)(v5 + 1488);
  if ( v188 == v7 || *(_BYTE *)v4 != 62 )
    goto LABEL_10;
  if ( !v12 )
  {
    v191 = v4;
    v13 = v5 + 1464;
    goto LABEL_182;
  }
  v14 = v12 - 1;
  v119 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v120 = (__int64 **)(v11 + 16LL * v119);
  v121 = *v120;
  if ( v4 == *v120 )
  {
LABEL_123:
    v122 = v120[1];
    if ( v122 && *((_DWORD *)v122 + 2) == 12 )
    {
      v157 = a3[3];
      if ( v157 == *((_DWORD *)v122 + 3) )
      {
        if ( v157 > 0xFFFFFFFD )
          goto LABEL_125;
        if ( (unsigned int)(a3[2] - 11) <= 1
          && (*(unsigned __int8 (__fastcall **)(_DWORD *, __int64 *))(*(_QWORD *)a3 + 16LL))(a3, v122) )
        {
          goto LABEL_9;
        }
      }
      v158 = *(_DWORD *)(v5 + 2040);
      v180 = *(_QWORD *)(v5 + 2024);
      if ( !v158 )
        goto LABEL_9;
      v189 = v5;
      v186 = v4;
      v159 = 1;
      v160 = v158 - 1;
      for ( m = (v158 - 1) & sub_28CB200(v122); ; m = v160 & v171 )
      {
        v162 = (__int64 *)(v180 + 16LL * m);
        v163 = *v162;
        if ( *v162 != 0x7FFFFFFF0LL )
        {
          if ( v163 == -8 )
            goto LABEL_255;
          if ( (*(unsigned __int8 (__fastcall **)(__int64 *, __int64, __int64))(*v122 + 24))(v122, v163, 0x7FFFFFFF0LL) )
          {
            v5 = v189;
            v4 = v186;
            if ( v162 != (__int64 *)(*(_QWORD *)(v189 + 2024) + 16LL * *(unsigned int *)(v189 + 2040)) )
            {
              *v162 = 0x7FFFFFFF0LL;
              --*(_DWORD *)(v189 + 2032);
              ++*(_DWORD *)(v189 + 2036);
            }
LABEL_9:
            v11 = *(_QWORD *)(v5 + 1472);
            v12 = *(_DWORD *)(v5 + 1488);
LABEL_10:
            v191 = v4;
            v13 = v5 + 1464;
            if ( v12 )
            {
              v14 = v12 - 1;
              goto LABEL_12;
            }
LABEL_182:
            ++*(_QWORD *)(v5 + 1464);
            v12 = 0;
            v195 = 0;
LABEL_183:
            v12 *= 2;
LABEL_184:
            sub_28D3EB0(v13, v12);
            sub_28CBD10(v13, (__int64 *)&v191, &v195);
            v4 = v191;
            v16 = (__int64 **)v195;
            v146 = *(_DWORD *)(v5 + 1480) + 1;
LABEL_178:
            *(_DWORD *)(v5 + 1480) = v146;
            if ( *v16 != (__int64 *)-4096LL )
              --*(_DWORD *)(v5 + 1484);
            *v16 = v4;
            result = v16 + 1;
            v16[1] = 0;
            goto LABEL_14;
          }
          v163 = *v162;
        }
        if ( v163 == -8 )
        {
LABEL_255:
          v5 = v189;
          v4 = v186;
          goto LABEL_9;
        }
        v171 = v159 + m;
        ++v159;
      }
    }
  }
  else
  {
    v166 = 1;
    while ( v121 != (__int64 *)-4096LL )
    {
      v167 = v166 + 1;
      v119 = v14 & (v166 + v119);
      v120 = (__int64 **)(v11 + 16LL * v119);
      v121 = *v120;
      if ( v4 == *v120 )
        goto LABEL_123;
      v166 = v167;
    }
  }
LABEL_125:
  v191 = v4;
  v13 = v5 + 1464;
LABEL_12:
  v15 = 1;
  v16 = 0;
  v17 = v14 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v18 = (_QWORD *)(v11 + 16LL * v17);
  v19 = (__int64 *)*v18;
  if ( v4 != (__int64 *)*v18 )
  {
    while ( v19 != (__int64 *)-4096LL )
    {
      if ( !v16 && v19 == (__int64 *)-8192LL )
        v16 = (__int64 **)v18;
      v17 = v14 & (v15 + v17);
      v18 = (_QWORD *)(v11 + 16LL * v17);
      v19 = (__int64 *)*v18;
      if ( v4 == (__int64 *)*v18 )
        goto LABEL_13;
      ++v15;
    }
    if ( !v16 )
      v16 = (__int64 **)v18;
    v145 = *(_DWORD *)(v5 + 1480);
    ++*(_QWORD *)(v5 + 1464);
    v146 = v145 + 1;
    v195 = (__int64 *)v16;
    if ( 4 * v146 >= 3 * v12 )
      goto LABEL_183;
    if ( v12 - (v146 + *(_DWORD *)(v5 + 1484)) <= v12 >> 3 )
      goto LABEL_184;
    goto LABEL_178;
  }
LABEL_13:
  result = v18 + 1;
LABEL_14:
  *result = a3;
  return result;
}
