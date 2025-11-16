// Function: sub_1AE8380
// Address: 0x1ae8380
//
_QWORD *__fastcall sub_1AE8380(unsigned __int64 a1, unsigned __int8 a2, _QWORD *a3, int a4)
{
  int v5; // r9d
  _QWORD *v7; // r12
  __int64 v8; // rbx
  _QWORD *v9; // rsi
  _QWORD *v10; // r8
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // r15
  __int64 i; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int8 v18; // r11
  int v19; // r9d
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // rax
  bool v23; // al
  _BOOL8 v24; // rdi
  _QWORD *v25; // r14
  __int64 v26; // rdi
  unsigned __int64 v28; // rdi
  unsigned int v29; // r10d
  int v30; // eax
  __int64 v31; // r12
  size_t v32; // r12
  __int64 v33; // rax
  __int64 v34; // rcx
  __int64 v35; // r8
  int v36; // r9d
  __int64 v37; // rdx
  bool v38; // zf
  __int64 v39; // r10
  __int64 v40; // rax
  size_t v41; // rax
  unsigned int v42; // ebx
  __int64 v43; // r13
  char *v44; // rax
  _BYTE *v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rax
  char v49; // r12
  __int64 v50; // rax
  unsigned int v51; // eax
  unsigned int v52; // r10d
  unsigned __int8 v53; // r11
  _QWORD *v54; // rbx
  __int64 v55; // r12
  char v56; // al
  unsigned int v57; // r10d
  unsigned __int8 v58; // r11
  __int64 v59; // rax
  __int64 v60; // rcx
  int v61; // r8d
  int v62; // r9d
  unsigned int v63; // r10d
  __int64 v64; // rax
  _QWORD *v65; // rax
  _QWORD *v66; // rcx
  __int64 v67; // rax
  unsigned int v68; // r12d
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rbx
  __int64 v72; // rax
  char *k; // rdx
  __int64 v74; // rcx
  int v75; // r9d
  __int64 v76; // r8
  _QWORD *v77; // rbx
  __int64 v78; // rax
  unsigned int v79; // r10d
  __int64 v80; // r13
  char *v81; // rax
  __int64 v82; // rdi
  __int64 v83; // rdx
  int v84; // eax
  char v85; // cl
  char v86; // si
  _QWORD *v87; // rdi
  __int64 v88; // rax
  __int64 v89; // rcx
  int v90; // r8d
  int v91; // r9d
  unsigned int v92; // r10d
  __int64 v93; // r13
  __int64 v94; // rax
  __int64 v95; // rdx
  char *j; // rax
  __int64 v97; // rdi
  __int64 v98; // rdx
  __int64 v99; // rax
  _QWORD *v100; // rax
  _QWORD *v101; // rax
  char *v102; // rsi
  unsigned __int64 v103; // rdi
  __int64 v104; // rdx
  __int64 v105; // rdi
  __int64 v106; // rbx
  __int64 v107; // r11
  __int64 v108; // r11
  unsigned __int64 v109; // rax
  unsigned int v110; // edx
  int v111; // eax
  __int64 v112; // rdi
  char v113; // al
  _QWORD *v114; // [rsp+0h] [rbp-B0h]
  char v115; // [rsp+0h] [rbp-B0h]
  int v116; // [rsp+8h] [rbp-A8h]
  unsigned int v117; // [rsp+8h] [rbp-A8h]
  int v118; // [rsp+Ch] [rbp-A4h]
  int v119; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v120; // [rsp+Ch] [rbp-A4h]
  unsigned int v121; // [rsp+Ch] [rbp-A4h]
  unsigned __int8 v122; // [rsp+Ch] [rbp-A4h]
  unsigned int v123; // [rsp+Ch] [rbp-A4h]
  unsigned int v124; // [rsp+Ch] [rbp-A4h]
  unsigned int v125; // [rsp+Ch] [rbp-A4h]
  int v126; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v127; // [rsp+10h] [rbp-A0h]
  int v128; // [rsp+10h] [rbp-A0h]
  int v129; // [rsp+10h] [rbp-A0h]
  int v130; // [rsp+10h] [rbp-A0h]
  int v131; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v132; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v133; // [rsp+10h] [rbp-A0h]
  unsigned int v134; // [rsp+10h] [rbp-A0h]
  char v135; // [rsp+10h] [rbp-A0h]
  int v136; // [rsp+10h] [rbp-A0h]
  int v137; // [rsp+10h] [rbp-A0h]
  unsigned int v138; // [rsp+10h] [rbp-A0h]
  int v139; // [rsp+10h] [rbp-A0h]
  _QWORD *v140; // [rsp+10h] [rbp-A0h]
  __int64 v141; // [rsp+10h] [rbp-A0h]
  unsigned __int8 v142; // [rsp+18h] [rbp-98h]
  unsigned __int64 *v143; // [rsp+18h] [rbp-98h]
  unsigned __int8 v144; // [rsp+18h] [rbp-98h]
  unsigned __int8 v145; // [rsp+18h] [rbp-98h]
  unsigned __int8 v146; // [rsp+18h] [rbp-98h]
  unsigned __int8 v147; // [rsp+18h] [rbp-98h]
  unsigned int v148; // [rsp+18h] [rbp-98h]
  unsigned int v149; // [rsp+18h] [rbp-98h]
  unsigned __int64 *v150; // [rsp+18h] [rbp-98h]
  unsigned int v151; // [rsp+18h] [rbp-98h]
  const void **v152; // [rsp+18h] [rbp-98h]
  unsigned __int8 v153; // [rsp+18h] [rbp-98h]
  unsigned __int8 v154; // [rsp+18h] [rbp-98h]
  __int64 v155; // [rsp+18h] [rbp-98h]
  __int64 v156; // [rsp+18h] [rbp-98h]
  __int64 v157; // [rsp+18h] [rbp-98h]
  unsigned int v158; // [rsp+18h] [rbp-98h]
  unsigned int v159; // [rsp+18h] [rbp-98h]
  unsigned int v160; // [rsp+18h] [rbp-98h]
  unsigned int v161; // [rsp+18h] [rbp-98h]
  unsigned __int8 v162; // [rsp+18h] [rbp-98h]
  unsigned int v163; // [rsp+18h] [rbp-98h]
  unsigned int v164; // [rsp+18h] [rbp-98h]
  unsigned int v165; // [rsp+18h] [rbp-98h]
  __int64 v166; // [rsp+18h] [rbp-98h]
  unsigned int v167; // [rsp+18h] [rbp-98h]
  unsigned __int64 v168; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v169; // [rsp+28h] [rbp-88h]
  _QWORD *v170; // [rsp+30h] [rbp-80h] BYREF
  unsigned int v171; // [rsp+38h] [rbp-78h]
  __int64 v172; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v173; // [rsp+48h] [rbp-68h] BYREF
  __int64 v174; // [rsp+50h] [rbp-60h]
  _BYTE v175[32]; // [rsp+58h] [rbp-58h] BYREF
  char v176; // [rsp+78h] [rbp-38h]

  v5 = a4;
  v7 = a3 + 1;
  v8 = a1;
  v9 = (_QWORD *)a3[2];
  if ( !v9 )
  {
    v176 = 0;
    goto LABEL_97;
  }
  v10 = a3 + 1;
  v11 = (_QWORD *)a3[2];
  do
  {
    while ( 1 )
    {
      v12 = v11[2];
      v13 = v11[3];
      if ( v11[4] >= a1 )
        break;
      v11 = (_QWORD *)v11[3];
      if ( !v13 )
        goto LABEL_6;
    }
    v10 = v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v12 );
LABEL_6:
  if ( v7 != v10 && v10[4] <= a1 )
    return v10 + 5;
  v176 = 0;
  v14 = v7;
  do
  {
    while ( 1 )
    {
      i = v9[2];
      v16 = v9[3];
      if ( v9[4] >= a1 )
        break;
      v9 = (_QWORD *)v9[3];
      if ( !v16 )
        goto LABEL_12;
    }
    v14 = v9;
    v9 = (_QWORD *)v9[2];
  }
  while ( i );
LABEL_12:
  if ( v7 == v14 )
  {
LABEL_97:
    v136 = v5;
    v153 = a2;
    v64 = sub_22077B0(104);
    v38 = a3[5] == 0;
    v18 = v153;
    *(_QWORD *)(v64 + 32) = a1;
    v19 = v136;
    v20 = v64;
    v14 = (_QWORD *)v64;
    *(_BYTE *)(v64 + 96) = 0;
    if ( v38 || (v21 = a3[4], *(_QWORD *)(v21 + 32) >= a1) )
    {
      v65 = sub_1AE7600((__int64)a3, (unsigned __int64 *)(v64 + 32));
      v18 = v153;
      v19 = v136;
      v66 = v65;
      v67 = v21;
      goto LABEL_100;
    }
    goto LABEL_102;
  }
  if ( v14[4] <= a1 )
  {
    v25 = v14 + 5;
    goto LABEL_32;
  }
  v126 = v5;
  v142 = a2;
  v17 = sub_22077B0(104);
  v18 = v142;
  v19 = v126;
  *(_QWORD *)(v17 + 32) = a1;
  v20 = v17;
  *(_BYTE *)(v17 + 96) = 0;
  if ( v14[4] > a1 )
  {
    v21 = a3[3];
    v150 = (unsigned __int64 *)(v17 + 32);
    if ( v14 != (_QWORD *)v21 )
    {
      v119 = v126;
      v132 = v18;
      v48 = sub_220EF80(v14);
      v18 = v132;
      v19 = v119;
      v21 = v48;
      if ( *(_QWORD *)(v48 + 32) >= a1 )
      {
        v14 = (_QWORD *)v20;
        v100 = sub_1AE7600((__int64)a3, v150);
        v18 = v132;
        v19 = v119;
        v66 = v100;
        v67 = v21;
        goto LABEL_100;
      }
      if ( !*(_QWORD *)(v48 + 24) )
        goto LABEL_102;
      v21 = (__int64)v14;
    }
    v23 = 1;
LABEL_21:
    v14 = (_QWORD *)v20;
    goto LABEL_22;
  }
  v143 = (unsigned __int64 *)(v17 + 32);
  if ( v14[4] >= a1 )
  {
LABEL_104:
    v137 = v19;
    v154 = v18;
    j_j___libc_free_0(v20, 104);
    v5 = v137;
    a2 = v154;
    goto LABEL_26;
  }
  v21 = a3[4];
  if ( v14 == (_QWORD *)v21 )
  {
LABEL_102:
    v23 = 0;
    goto LABEL_21;
  }
  v118 = v126;
  v127 = v18;
  v22 = sub_220EEE0(v14);
  v18 = v127;
  v19 = v118;
  v21 = v22;
  if ( *(_QWORD *)(v22 + 32) > a1 )
  {
    if ( !v14[3] )
      v21 = (__int64)v14;
    v23 = v14[3] != 0;
    goto LABEL_21;
  }
  v14 = (_QWORD *)v20;
  v101 = sub_1AE7600((__int64)a3, v143);
  v19 = v118;
  v18 = v127;
  v66 = v101;
  v67 = v21;
LABEL_100:
  if ( !v67 )
  {
    v20 = (__int64)v14;
    v14 = v66;
    goto LABEL_104;
  }
  v23 = v66 != 0;
LABEL_22:
  v24 = v7 == (_QWORD *)v21 || v23 || a1 < *(_QWORD *)(v21 + 32);
  v128 = v19;
  v144 = v18;
  sub_220F040(v24, v14, v21, v7);
  ++a3[5];
  a2 = v144;
  v5 = v128;
LABEL_26:
  v25 = v14 + 5;
  if ( v176 )
  {
    v26 = (__int64)(v14 + 6);
    if ( *((_BYTE *)v14 + 96) )
    {
      v14[5] = v172;
      v129 = v5;
      v145 = a2;
      sub_1AE7900(v26, (char **)&v173, i, v12, (int)v10, v5);
      a2 = v145;
      v5 = v129;
    }
    else
    {
      v14[5] = v172;
      v14[6] = v14 + 8;
      v14[7] = 0x2000000000LL;
      LODWORD(v10) = v174;
      if ( (_DWORD)v174 )
      {
        v139 = v5;
        v162 = a2;
        sub_1AE7900(v26, (char **)&v173, i, v12, v174, v5);
        v5 = v139;
        a2 = v162;
      }
      *((_BYTE *)v14 + 96) = 1;
    }
    goto LABEL_36;
  }
LABEL_32:
  if ( !*((_BYTE *)v14 + 96) )
    goto LABEL_39;
  v28 = v14[6];
  if ( (_QWORD *)v28 != v14 + 8 )
  {
    v130 = v5;
    v146 = a2;
    _libc_free(v28);
    v5 = v130;
    a2 = v146;
  }
  *((_BYTE *)v14 + 96) = 0;
LABEL_36:
  if ( v176 && (_BYTE *)v173 != v175 )
  {
    v131 = v5;
    v147 = a2;
    _libc_free(v173);
    v5 = v131;
    a2 = v147;
  }
LABEL_39:
  v29 = *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8;
  if ( v5 == 48 )
    return v25;
  v30 = *(unsigned __int8 *)(v8 + 16);
  if ( (unsigned __int8)v30 <= 0x17u )
    goto LABEL_56;
  if ( (_BYTE)v30 != 51 )
  {
    i = (unsigned int)(v30 - 47);
    if ( (unsigned __int8)(v30 - 47) <= 1u )
    {
      if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
        v10 = *(_QWORD **)(v8 - 8);
      else
        v10 = (_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
      v31 = v10[3];
      if ( *(_BYTE *)(v31 + 16) == 13 )
      {
        if ( *(_DWORD *)(v31 + 32) > 0x40u )
        {
          v122 = a2;
          v140 = v10;
          v163 = *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8;
          v116 = *(_DWORD *)(v31 + 32);
          if ( v116 - (unsigned int)sub_16A57B0(v31 + 24) > 0x40 )
            return v25;
          v29 = v163;
          v10 = v140;
          a2 = v122;
          v32 = **(_QWORD **)(v31 + 24);
          if ( v32 > 0xFFFFFFFF )
            return v25;
        }
        else
        {
          v32 = *(_QWORD *)(v31 + 24);
          if ( v32 > 0xFFFFFFFF )
            return v25;
        }
        if ( v29 >= (unsigned int)v32 )
        {
          v33 = sub_1AE8380(*v10, a2, a3);
          if ( *(_BYTE *)(v33 + 56) )
          {
            v37 = *(_QWORD *)v33;
            v38 = *((_BYTE *)v14 + 96) == 0;
            v14[5] = *(_QWORD *)v33;
            if ( v38 )
            {
              v14[6] = v14 + 8;
              v14[7] = 0x2000000000LL;
              if ( *(_DWORD *)(v33 + 16) )
                sub_1AE7820((__int64)(v14 + 6), v33 + 8, (__int64)(v14 + 8), v34, v35, v36);
              *((_BYTE *)v14 + 96) = 1;
            }
            else
            {
              sub_1AE7820((__int64)(v14 + 6), v33 + 8, v37, v34, v35, v36);
            }
            v39 = v14[6];
            v40 = *((unsigned int *)v14 + 14);
            if ( *(_BYTE *)(v8 + 16) == 47 )
            {
              v102 = (char *)v14[6];
              *((_DWORD *)v14 + 14) = v40 - v32;
              LOBYTE(v172) = -1;
              sub_1AE7A90((__int64)(v14 + 6), v102, v32, (unsigned __int8 *)&v172, v35, v36);
            }
            else
            {
              v41 = v40 - v32;
              v42 = v41;
              if ( v41 )
              {
                v42 = (unsigned int)memmove((void *)v14[6], (const void *)(v39 + v32), v41) + v41 - v14[6];
                v39 = v14[6];
              }
              *((_DWORD *)v14 + 14) = v42;
              LOBYTE(v172) = -1;
              sub_1AE7A90((__int64)(v14 + 6), (char *)(v39 + v42), v32, (unsigned __int8 *)&v172, v35, v36);
            }
          }
        }
        return v25;
      }
LABEL_56:
      v43 = v29;
      v172 = v8;
      v173 = (unsigned __int64)v175;
      v174 = 0x2000000000LL;
      if ( v29 )
      {
        v44 = v175;
        v45 = v175;
        if ( v29 > 0x20uLL )
        {
          v164 = v29;
          sub_16CD150((__int64)&v173, v175, v29, 1, (int)v10, v5);
          v45 = (_BYTE *)v173;
          v29 = v164;
          v44 = (char *)(v173 + (unsigned int)v174);
        }
        for ( i = (__int64)&v45[v43]; (char *)i != v44; ++v44 )
        {
          if ( v44 )
            *v44 = 0;
        }
        LODWORD(v174) = v29;
        v8 = v172;
      }
      v46 = (__int64)(v14 + 6);
      if ( *((_BYTE *)v14 + 96) )
      {
        v14[5] = v8;
        v149 = v29;
        sub_1AE7900(v46, (char **)&v173, i, v12, (int)v10, v5);
        v29 = v149;
      }
      else
      {
        v14[5] = v8;
        v14[6] = v14 + 8;
        v14[7] = 0x2000000000LL;
        if ( (_DWORD)v174 )
        {
          v158 = v29;
          sub_1AE7900(v46, (char **)&v173, i, v12, (int)v10, v5);
          v29 = v158;
        }
        *((_BYTE *)v14 + 96) = 1;
      }
      if ( (_BYTE *)v173 != v175 )
      {
        v148 = v29;
        _libc_free(v173);
        v29 = v148;
      }
      if ( v29 )
      {
        v47 = 0;
        do
        {
          *(_BYTE *)(v14[6] + v47) = v47;
          ++v47;
        }
        while ( v29 > (unsigned int)v47 );
      }
      return v25;
    }
    if ( (_BYTE)v30 != 50 )
    {
      if ( (_BYTE)v30 == 61 )
      {
        if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
          v87 = *(_QWORD **)(v8 - 8);
        else
          v87 = (_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
        v159 = *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8;
        v88 = sub_1AE8380(*v87, a2, a3);
        v92 = v159;
        v93 = v88;
        if ( *(_BYTE *)(v88 + 56) )
        {
          v94 = *(_QWORD *)v88;
          v95 = v159;
          v173 = (unsigned __int64)v175;
          v172 = v94;
          v174 = 0x2000000000LL;
          if ( v159 )
          {
            if ( v159 > 0x20uLL )
            {
              sub_16CD150((__int64)&v173, v175, v159, 1, v90, v91);
              v92 = v159;
              v95 = v159;
            }
            v95 += v173;
            for ( j = (char *)(v173 + (unsigned int)v174); (char *)v95 != j; ++j )
            {
              if ( j )
                *j = 0;
            }
            LODWORD(v174) = v92;
            v94 = v172;
          }
          v97 = (__int64)(v14 + 6);
          if ( *((_BYTE *)v14 + 96) )
          {
            v14[5] = v94;
            v160 = v92;
            sub_1AE7900(v97, (char **)&v173, v95, v89, v90, v91);
            v92 = v160;
          }
          else
          {
            v14[5] = v94;
            v14[6] = v14 + 8;
            v14[7] = 0x2000000000LL;
            if ( (_DWORD)v174 )
            {
              v165 = v92;
              sub_1AE7900(v97, (char **)&v173, (unsigned int)v174, v89, v90, v91);
              v92 = v165;
            }
            *((_BYTE *)v14 + 96) = 1;
          }
          if ( (_BYTE *)v173 != v175 )
          {
            v161 = v92;
            _libc_free(v173);
            v92 = v161;
          }
          v98 = 0;
          for ( LODWORD(v99) = *(_DWORD *)(**(_QWORD **)(v8 - 24) + 8LL) >> 8; (unsigned int)v99 > (unsigned int)v98; ++v98 )
            *(_BYTE *)(v14[6] + v98) = *(_BYTE *)(*(_QWORD *)(v93 + 8) + v98);
          if ( v92 > (unsigned int)v99 )
          {
            v99 = (unsigned int)v99;
            do
              *(_BYTE *)(v14[6] + v99++) = -1;
            while ( v92 > (unsigned int)v99 );
          }
        }
        return v25;
      }
      goto LABEL_56;
    }
    v49 = *(_BYTE *)(v8 + 23) & 0x40;
    if ( v49 )
    {
      v50 = *(_QWORD *)(v8 - 8);
    }
    else
    {
      i = 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
      v50 = v8 - i;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v50 + 24) + 16LL) != 13 )
      goto LABEL_56;
    v133 = a2;
    v151 = *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8;
    v51 = sub_1643030(*(_QWORD *)v8);
    v52 = v151;
    v53 = v133;
    v169 = v51;
    if ( v51 > 0x40 )
    {
      sub_16A4EF0((__int64)&v168, 1, 0);
      v53 = v133;
      v52 = v151;
      v49 = *(_BYTE *)(v8 + 23) & 0x40;
    }
    else
    {
      v168 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v51) & 1;
    }
    if ( v49 )
      v54 = *(_QWORD **)(v8 - 8);
    else
      v54 = (_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
    v55 = v54[3];
    v120 = v53;
    v134 = v52;
    v152 = (const void **)(v55 + 24);
    if ( *(_DWORD *)(v55 + 32) > 0x40u )
    {
      v56 = sub_16A5940((__int64)v152);
      v58 = v120;
      v57 = v134;
    }
    else
    {
      v56 = sub_39FAC40(*(_QWORD *)(v55 + 24));
      v57 = v134;
      v58 = v120;
    }
    if ( !v58 && (v56 & 7) != 0 )
      goto LABEL_93;
    v121 = v57;
    v59 = sub_1AE8380(*v54, v58, a3);
    v63 = v121;
    v135 = *(_BYTE *)(v59 + 56);
    if ( !v135 )
      goto LABEL_93;
    v104 = *(_QWORD *)v59;
    v38 = *((_BYTE *)v14 + 96) == 0;
    v105 = (__int64)(v14 + 6);
    v14[5] = *(_QWORD *)v59;
    if ( v38 )
    {
      v14[6] = v14 + 8;
      v14[7] = 0x2000000000LL;
      if ( *(_DWORD *)(v59 + 16) )
      {
        sub_1AE7820(v105, v59 + 8, (__int64)(v14 + 8), v60, v61, v62);
        v63 = v121;
      }
      *((_BYTE *)v14 + 96) = 1;
    }
    else
    {
      sub_1AE7820(v105, v59 + 8, v104, v60, v61, v62);
      v63 = v121;
    }
    if ( !v63 )
    {
LABEL_93:
      if ( v169 > 0x40 && v168 )
        j_j___libc_free_0_0(v168);
      return v25;
    }
    v106 = 0;
    while ( 1 )
    {
      v171 = *(_DWORD *)(v55 + 32);
      if ( v171 <= 0x40 )
        break;
      v123 = v63;
      sub_16A4FD0((__int64)&v170, v152);
      v63 = v123;
      if ( v171 <= 0x40 )
      {
        v107 = (__int64)v170;
        goto LABEL_188;
      }
      sub_16A8890((__int64 *)&v170, (__int64 *)&v168);
      v110 = v171;
      v108 = (__int64)v170;
      v171 = 0;
      v63 = v123;
      LODWORD(v173) = v110;
      v172 = (__int64)v170;
      v117 = v110;
      if ( v110 <= 0x40 )
        goto LABEL_189;
      v114 = v170;
      v111 = sub_16A57B0((__int64)&v172);
      v63 = v123;
      if ( v117 - v111 > 0x40 || *v114 )
      {
        if ( !v114 )
          goto LABEL_190;
        j_j___libc_free_0_0(v114);
        v63 = v123;
        if ( v171 <= 0x40 )
          goto LABEL_190;
        v112 = (__int64)v170;
        v113 = 0;
        if ( !v170 )
          goto LABEL_190;
      }
      else
      {
        j_j___libc_free_0_0(v114);
        v63 = v123;
        if ( v171 <= 0x40 || (v112 = (__int64)v170) == 0 )
        {
LABEL_203:
          *(_BYTE *)(v14[6] + v106) = -1;
          goto LABEL_190;
        }
        v113 = v135;
      }
      v115 = v113;
      v124 = v63;
      j_j___libc_free_0_0(v112);
      v63 = v124;
      if ( v115 )
        goto LABEL_203;
LABEL_190:
      if ( v169 > 0x40 )
      {
        v125 = v63;
        sub_16A7DC0((__int64 *)&v168, 1u);
        v63 = v125;
      }
      else
      {
        v109 = 0;
        if ( v169 != 1 )
          v109 = (2 * v168) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v169);
        v168 = v109;
      }
      if ( v63 <= (unsigned int)++v106 )
        goto LABEL_93;
    }
    v107 = *(_QWORD *)(v55 + 24);
LABEL_188:
    v108 = v168 & v107;
    v171 = 0;
    v170 = (_QWORD *)v108;
LABEL_189:
    if ( !v108 )
      goto LABEL_203;
    goto LABEL_190;
  }
  v68 = a2;
  if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
    v69 = *(_QWORD **)(v8 - 8);
  else
    v69 = (_QWORD *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  v138 = *(_DWORD *)(*(_QWORD *)v8 + 8LL) >> 8;
  v70 = sub_1AE8380(*v69, a2, a3);
  if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
    v71 = *(_QWORD *)(v8 - 8);
  else
    v71 = v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
  v155 = v70;
  v72 = sub_1AE8380(*(_QWORD *)(v71 + 24), v68, a3);
  v76 = v155;
  v77 = (_QWORD *)v72;
  if ( *(_BYTE *)(v155 + 56) )
  {
    if ( *(_BYTE *)(v72 + 56) )
    {
      v78 = *(_QWORD *)v155;
      if ( *(_QWORD *)v155 )
      {
        v79 = v138;
        if ( v78 == *v77 )
        {
          v80 = v138;
          v172 = *(_QWORD *)v155;
          v173 = (unsigned __int64)v175;
          v174 = 0x2000000000LL;
          if ( v138 )
          {
            if ( v138 > 0x20uLL )
            {
              v141 = v155;
              v167 = v79;
              sub_16CD150((__int64)&v173, v175, v79, 1, v76, v75);
              v76 = v141;
              v79 = v167;
            }
            v81 = (char *)(v173 + (unsigned int)v174);
            for ( k = (char *)(v80 + v173); k != v81; ++v81 )
            {
              if ( v81 )
                *v81 = 0;
            }
            LODWORD(v174) = v79;
            v78 = v172;
          }
          v82 = (__int64)(v14 + 6);
          if ( *((_BYTE *)v14 + 96) )
          {
            v14[5] = v78;
            v156 = v76;
            sub_1AE7900(v82, (char **)&v173, (__int64)k, v74, v76, v75);
            v76 = v156;
          }
          else
          {
            v14[5] = v78;
            v14[6] = v14 + 8;
            v14[7] = 0x2000000000LL;
            if ( (_DWORD)v174 )
            {
              v166 = v76;
              sub_1AE7900(v82, (char **)&v173, (__int64)k, v74, v76, v75);
              v76 = v166;
            }
            *((_BYTE *)v14 + 96) = 1;
          }
          if ( (_BYTE *)v173 != v175 )
          {
            v157 = v76;
            _libc_free(v173);
            v76 = v157;
          }
          if ( *(_DWORD *)(v76 + 16) )
          {
            v83 = 0;
            v84 = 0;
            do
            {
              v85 = *(_BYTE *)(*(_QWORD *)(v76 + 8) + v83);
              v86 = *(_BYTE *)(v77[1] + v83);
              if ( v85 == -1 )
              {
                *(_BYTE *)(v14[6] + v83) = v86;
              }
              else
              {
                if ( v85 != v86 && v86 != -1 )
                {
                  if ( *((_BYTE *)v14 + 96) )
                  {
                    v103 = v14[6];
                    if ( (_QWORD *)v103 != v14 + 8 )
                      _libc_free(v103);
                    *((_BYTE *)v14 + 96) = 0;
                  }
                  return v25;
                }
                *(_BYTE *)(v14[6] + v83) = v85;
              }
              v83 = (unsigned int)(v84 + 1);
              v84 = v83;
            }
            while ( *(_DWORD *)(v76 + 16) > (unsigned int)v83 );
          }
        }
      }
    }
  }
  return v25;
}
