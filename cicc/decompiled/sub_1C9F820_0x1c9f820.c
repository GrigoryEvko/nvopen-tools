// Function: sub_1C9F820
// Address: 0x1c9f820
//
__int64 __fastcall sub_1C9F820(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        int *a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  _QWORD *v16; // r13
  _QWORD *v19; // rdx
  __int64 v20; // r9
  _QWORD *v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rdx
  unsigned int v30; // r10d
  char v31; // al
  __int64 v32; // rdx
  double v33; // xmm4_8
  double v34; // xmm5_8
  _QWORD *v35; // r9
  _QWORD *v36; // r8
  unsigned __int64 v37; // r11
  unsigned __int8 v38; // al
  char v39; // al
  char v40; // al
  unsigned __int64 v41; // r11
  _QWORD *v42; // r14
  _QWORD *v43; // rdx
  _QWORD *v44; // r15
  _QWORD *v45; // rdi
  _QWORD *v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rax
  int v49; // eax
  int v51; // r10d
  __int64 v52; // r12
  __int64 v53; // r11
  unsigned __int8 v54; // al
  __int64 v55; // rcx
  unsigned __int64 *v56; // r10
  __int64 v57; // rsi
  _QWORD *v58; // rax
  _QWORD *v59; // rdx
  _QWORD *v60; // r8
  _QWORD *v61; // r9
  _BOOL4 v62; // r11d
  __int64 v63; // rax
  unsigned int v64; // eax
  _DWORD *v65; // rax
  _DWORD *v66; // r8
  int v67; // eax
  _QWORD *v68; // r12
  char v69; // al
  __int64 v70; // rax
  _QWORD *v71; // rax
  _QWORD *v72; // rdx
  _BOOL8 v73; // rdi
  char v74; // al
  char v75; // al
  int v76; // r11d
  int v77; // eax
  __int64 v78; // rax
  _QWORD *v79; // rdi
  unsigned __int64 v80; // rdx
  _QWORD *v81; // rax
  _QWORD *v82; // rcx
  int v83; // r10d
  int v84; // edx
  __int64 v85; // r12
  unsigned int v86; // eax
  _DWORD *v87; // rax
  int *v88; // rax
  int v89; // edx
  unsigned __int64 v90; // rsi
  int v91; // eax
  unsigned __int64 v92; // rsi
  __int64 *v93; // r12
  int v94; // r10d
  _QWORD *v95; // r8
  int *v96; // r12
  int *v97; // r13
  char v98; // bl
  int *v99; // rax
  _BYTE *v100; // rdx
  char v101; // al
  __int64 *v102; // r12
  int v103; // r15d
  _QWORD *v104; // r14
  _QWORD *v105; // rax
  _QWORD *v106; // rdx
  char v107; // al
  _QWORD *v108; // r8
  __int64 v109; // r12
  int v110; // eax
  char v111; // al
  int v112; // r10d
  unsigned __int64 v113; // r12
  _QWORD *v114; // rdx
  unsigned __int64 v115; // rcx
  _QWORD *i; // rax
  _QWORD *v117; // rdi
  _QWORD *v118; // r9
  unsigned __int64 *v119; // rax
  _QWORD *v120; // rax
  const __m128i *v121; // rax
  __int64 v122; // rax
  __int64 v123; // r8
  __int64 v124; // r12
  __int64 v125; // r11
  _QWORD *v126; // rcx
  int *v127; // r12
  _QWORD *v128; // rbx
  __int64 *v129; // rdx
  __int64 v130; // rax
  int v131; // eax
  __int64 v132; // rax
  int v133; // eax
  char v134; // r11
  int v135; // eax
  _QWORD *v136; // [rsp+0h] [rbp-E0h]
  _QWORD *v137; // [rsp+0h] [rbp-E0h]
  _QWORD *v138; // [rsp+0h] [rbp-E0h]
  __int64 v139; // [rsp+0h] [rbp-E0h]
  __int64 v140; // [rsp+8h] [rbp-D8h]
  __int64 v141; // [rsp+8h] [rbp-D8h]
  _QWORD *v142; // [rsp+10h] [rbp-D0h]
  int v143; // [rsp+18h] [rbp-C8h]
  _QWORD *v144; // [rsp+18h] [rbp-C8h]
  _QWORD *v145; // [rsp+18h] [rbp-C8h]
  char v146; // [rsp+20h] [rbp-C0h]
  _QWORD *v147; // [rsp+20h] [rbp-C0h]
  _BOOL4 v148; // [rsp+20h] [rbp-C0h]
  _QWORD *v149; // [rsp+20h] [rbp-C0h]
  _QWORD *v150; // [rsp+20h] [rbp-C0h]
  _QWORD *v152; // [rsp+28h] [rbp-B8h]
  _QWORD *v153; // [rsp+28h] [rbp-B8h]
  _QWORD *v154; // [rsp+28h] [rbp-B8h]
  _QWORD *v155; // [rsp+28h] [rbp-B8h]
  _QWORD *v156; // [rsp+28h] [rbp-B8h]
  _QWORD *v157; // [rsp+28h] [rbp-B8h]
  _QWORD *v158; // [rsp+28h] [rbp-B8h]
  char v159; // [rsp+28h] [rbp-B8h]
  _QWORD *v160; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v161; // [rsp+30h] [rbp-B0h]
  unsigned int v162; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v163; // [rsp+30h] [rbp-B0h]
  _QWORD *v164; // [rsp+30h] [rbp-B0h]
  _QWORD *v165; // [rsp+30h] [rbp-B0h]
  unsigned int v166; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v167; // [rsp+30h] [rbp-B0h]
  unsigned int v168; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v169[2]; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v170; // [rsp+48h] [rbp-98h] BYREF
  __int64 v171; // [rsp+50h] [rbp-90h] BYREF
  int v172; // [rsp+58h] [rbp-88h] BYREF
  __int64 v173; // [rsp+60h] [rbp-80h]
  int *v174; // [rsp+68h] [rbp-78h]
  int *v175; // [rsp+70h] [rbp-70h]
  __int64 v176; // [rsp+78h] [rbp-68h]
  unsigned __int64 *v177; // [rsp+80h] [rbp-60h] BYREF
  __int64 v178; // [rsp+88h] [rbp-58h] BYREF
  __int64 v179; // [rsp+90h] [rbp-50h] BYREF
  __int64 *v180; // [rsp+98h] [rbp-48h]
  __int64 *v181; // [rsp+A0h] [rbp-40h]
  __int64 v182; // [rsp+A8h] [rbp-38h]

  v16 = a4;
  v169[0] = a3;
  v19 = (_QWORD *)a4[2];
  if ( v19 )
  {
    v20 = (__int64)(a4 + 1);
    v21 = (_QWORD *)a4[2];
    v22 = a4 + 1;
    do
    {
      while ( 1 )
      {
        v23 = v21[2];
        v24 = v21[3];
        if ( v21[4] >= a3 )
          break;
        v21 = (_QWORD *)v21[3];
        if ( !v24 )
          goto LABEL_6;
      }
      v22 = v21;
      v21 = (_QWORD *)v21[2];
    }
    while ( v23 );
LABEL_6:
    if ( (_QWORD *)v20 != v22 )
    {
      v25 = v20;
      if ( v22[4] <= a3 )
      {
        while ( 1 )
        {
          v47 = v19[2];
          v48 = v19[3];
          if ( v19[4] < a3 )
          {
            v19 = (_QWORD *)v19[3];
            if ( !v48 )
              goto LABEL_31;
          }
          else
          {
            v25 = (__int64)v19;
            v19 = (_QWORD *)v19[2];
            if ( !v47 )
            {
LABEL_31:
              if ( v20 == v25 || *(_QWORD *)(v25 + 32) > a3 )
              {
                v177 = v169;
                v25 = sub_1C9E550(v16, (_QWORD *)v25, &v177);
              }
              v49 = *(_DWORD *)(v25 + 40);
              *a6 = v49;
              return v49 != 0;
            }
          }
        }
      }
    }
  }
  v26 = (_QWORD *)a5[2];
  if ( v26 )
  {
    v27 = a5 + 1;
    do
    {
      while ( 1 )
      {
        v28 = v26[2];
        v29 = v26[3];
        if ( v26[4] >= a3 )
          break;
        v26 = (_QWORD *)v26[3];
        if ( !v29 )
          goto LABEL_13;
      }
      v27 = v26;
      v26 = (_QWORD *)v26[2];
    }
    while ( v28 );
LABEL_13:
    if ( v27 != a5 + 1 )
    {
      v30 = 2;
      if ( v27[4] <= a3 )
        return v30;
    }
  }
  v160 = a5 + 1;
  v31 = sub_1C9F660(a1, a2, a3);
  v35 = v160;
  v36 = a5;
  v146 = v31;
  if ( !v31 )
    goto LABEL_140;
  v37 = v169[0];
  v38 = *(_BYTE *)(v169[0] + 16);
  if ( v38 <= 0x17u )
  {
    if ( v38 == 17 )
    {
      if ( unk_4FBE1ED
        && (v39 = sub_1C2F070(a2), v37 = v169[0], v39)
        && (v74 = sub_15E0450(v169[0]), v37 = v169[0], !v74) )
      {
        *a6 = 1;
      }
      else
      {
        v161 = v37;
        v40 = sub_15E0450(v37);
        v41 = v161;
        if ( !v40 || (v69 = sub_1C2F070(a2), v41 = v161, v69) )
        {
          v42 = *(_QWORD **)(a1 + 8);
          if ( !v42 )
            goto LABEL_75;
          v43 = (_QWORD *)v42[2];
          v44 = v42 + 1;
          if ( !v43 )
            goto LABEL_75;
          v45 = v42 + 1;
          v46 = (_QWORD *)v42[2];
          do
          {
            if ( v46[4] < v41 )
            {
              v46 = (_QWORD *)v46[3];
            }
            else
            {
              v45 = v46;
              v46 = (_QWORD *)v46[2];
            }
          }
          while ( v46 );
          if ( v44 != v45 && v45[4] <= v41 )
          {
            v68 = v42 + 1;
            do
            {
              if ( v43[4] < v41 )
              {
                v43 = (_QWORD *)v43[3];
              }
              else
              {
                v68 = v43;
                v43 = (_QWORD *)v43[2];
              }
            }
            while ( v43 );
            if ( v68 == v44 || v68[4] > v41 )
            {
              v163 = v41;
              v156 = v68;
              v70 = sub_22077B0(48);
              *(_DWORD *)(v70 + 40) = 0;
              v68 = (_QWORD *)v70;
              *(_QWORD *)(v70 + 32) = v163;
              v71 = sub_1C704D0(v42, v156, (unsigned __int64 *)(v70 + 32));
              if ( v72 )
              {
                v73 = v71 || v44 == v72 || v72[4] > v163;
                sub_220F040(v73, v68, v72, v42 + 1);
                ++v42[5];
              }
              else
              {
                v164 = v71;
                j_j___libc_free_0(v68, 48);
                v68 = v164;
              }
            }
            *a6 = *((_DWORD *)v68 + 10);
          }
          else
          {
LABEL_75:
            *a6 = 0;
          }
        }
        else
        {
          *a6 = 5;
        }
      }
      goto LABEL_76;
    }
    if ( v38 == 3 )
    {
      *a6 = *(_DWORD *)(*(_QWORD *)v169[0] + 8LL) >> 8;
LABEL_76:
      v66 = (_DWORD *)sub_1C9E600(v16, v169);
      v67 = *a6;
      *v66 = *a6;
      return v67 != 0;
    }
    if ( (v38 & 0xFD) != 0x4D )
    {
      if ( v38 == 5 )
      {
        v91 = sub_1C95850(v169[0], a2);
        v92 = v169[0];
        *a6 = v91;
        sub_1C9E680(v16, v92, v91, a5);
        return *a6 != 0;
      }
      goto LABEL_140;
    }
LABEL_44:
    v51 = *(_DWORD *)(*(_QWORD *)v169[0] + 8LL) >> 8;
    *a6 = v51;
    if ( v51 )
    {
      sub_1C9E680(v16, v37, v51, a5);
      return 1;
    }
    else
    {
      v174 = &v172;
      v175 = &v172;
      v172 = 0;
      v173 = 0;
      v176 = 0;
      LODWORD(v178) = 0;
      v179 = 0;
      v180 = &v178;
      v181 = &v178;
      v182 = 0;
      sub_1C99AC0(v37, (__int64)&v171, (__int64)&v177, (__int64)v16);
      v93 = v180;
      v94 = 0;
      v95 = a5;
      if ( v180 != &v178 )
      {
        do
        {
          v170 = v93[4];
          sub_1C99680((__int64)a5, (unsigned __int64 *)&v170);
          v93 = (__int64 *)sub_220EF30(v93);
        }
        while ( v93 != &v178 );
        v94 = 0;
        v95 = a5;
      }
      v96 = v174;
      if ( v174 == &v172 )
        goto LABEL_164;
      v158 = v16;
      v97 = a6;
      v98 = 0;
      do
      {
        v100 = (_BYTE *)*((_QWORD *)v96 + 4);
        v101 = v100[16];
        if ( v101 != 9 && v101 != 15 )
        {
          LODWORD(v170) = *(_DWORD *)(*(_QWORD *)v100 + 8LL) >> 8;
          if ( (_DWORD)v170 )
            goto LABEL_161;
          v138 = v95;
          v133 = sub_1C9F820(a1, a2, v100, v158, v95, &v170);
          v95 = v138;
          if ( !v133 )
          {
LABEL_163:
            a6 = v97;
            v94 = 0;
            v16 = v158;
            goto LABEL_164;
          }
          if ( v133 != 2 )
          {
LABEL_161:
            if ( v98 )
            {
              if ( *v97 != (_DWORD)v170 )
                goto LABEL_163;
            }
            else
            {
              *v97 = v170;
            }
            v98 = v146;
          }
        }
        v136 = v95;
        v99 = (int *)sub_220EF30(v96);
        v95 = v136;
        v96 = v99;
      }
      while ( v99 != &v172 );
      v134 = v98;
      v94 = 0;
      a6 = v97;
      v16 = v158;
      if ( !v134 )
LABEL_164:
        *a6 = 0;
      else
        v94 = 1;
      v102 = v180;
      v103 = v94;
      v104 = v95;
      if ( v180 != &v178 )
      {
        do
        {
          sub_1C9E680(v16, v102[4], *a6, v104);
          v102 = (__int64 *)sub_220EF30(v102);
        }
        while ( v102 != &v178 );
        v94 = v103;
      }
      v166 = v94;
      sub_1C97470(v179);
      sub_1C97470(v173);
      return v166;
    }
  }
  switch ( v38 )
  {
    case 'F':
      *a6 = 1;
      goto LABEL_74;
    case 'V':
      v179 = v169[0];
      v177 = (unsigned __int64 *)&v179;
      v178 = 0x400000001LL;
      v52 = *(_QWORD *)(v169[0] - 24);
      v53 = 1;
      while ( 1 )
      {
        v54 = *(_BYTE *)(v52 + 16);
        if ( v54 != 86 )
          break;
        if ( (unsigned int)v53 >= HIDWORD(v178) )
        {
          v145 = v36;
          v149 = v35;
          sub_16CD150((__int64)&v177, &v179, 0, 8, (int)v36, (int)v35);
          v53 = (unsigned int)v178;
          v36 = v145;
          v35 = v149;
        }
        v177[v53] = v52;
        v53 = (unsigned int)(v178 + 1);
        LODWORD(v178) = v178 + 1;
        v52 = *(_QWORD *)(v52 - 24);
        if ( !v52 )
          BUG();
      }
      v55 = (unsigned int)v53;
      if ( (_DWORD)v53 )
      {
        v56 = v177;
        while ( 1 )
        {
          v57 = v56[v55 - 1];
          if ( v54 <= 0x17u )
            break;
          if ( v54 == 54 )
          {
            if ( *(_QWORD *)v52 == **(_QWORD **)(v57 - 24) )
            {
              LODWORD(v178) = v53 - 1;
              if ( (_DWORD)v53 != 1 )
              {
LABEL_72:
                sub_1C9E680(v16, v169[0], 0, v36);
                v30 = 0;
LABEL_62:
                if ( v177 != (unsigned __int64 *)&v179 )
                {
                  v162 = v30;
                  _libc_free((unsigned __int64)v177);
                  return v162;
                }
                return v30;
              }
LABEL_54:
              v152 = v36;
              v147 = v35;
              v58 = sub_1819210((__int64)v36, v169);
              v60 = v152;
              if ( v59 )
              {
                v61 = v147;
                v62 = v58 || v147 == v59 || v169[0] < v59[4];
                v142 = v152;
                v144 = v59;
                v148 = v62;
                v153 = v61;
                v63 = sub_22077B0(40);
                *(_QWORD *)(v63 + 32) = v169[0];
                sub_220F040(v148, v63, v144, v153);
                v60 = v142;
                ++v142[5];
              }
              v154 = v60;
              v64 = sub_1C9F820(a1, a2, v52, v16, v60, a6);
              v30 = v64;
              if ( v64 )
              {
                if ( v64 == 1 )
                {
                  sub_1C9E680(v16, v169[0], *a6, v154);
                  v30 = 1;
                }
              }
              else
              {
                sub_1C9E680(v16, v169[0], 0, v154);
                v30 = 0;
              }
              goto LABEL_62;
            }
LABEL_53:
            if ( (_DWORD)v178 )
              goto LABEL_72;
            goto LABEL_54;
          }
          if ( v54 != 87 || *(_QWORD *)v52 != **(_QWORD **)(v57 - 24) )
            goto LABEL_71;
          v157 = v36;
          v75 = sub_1C957E0(v52, v57, v32, v55, (unsigned int)v36);
          v36 = v157;
          if ( v75 )
          {
            LODWORD(v53) = v76 - 1;
            LODWORD(v178) = v53;
            v52 = *(_QWORD *)(v52 - 24);
          }
          else
          {
            v52 = *(_QWORD *)(v52 - 48);
            LODWORD(v53) = v178;
          }
          v55 = (unsigned int)v53;
          if ( !(_DWORD)v53 )
            goto LABEL_87;
          v54 = *(_BYTE *)(v52 + 16);
        }
        if ( v54 == 17 )
        {
          if ( **(_QWORD **)(v57 - 24) == *(_QWORD *)v52 )
          {
            LODWORD(v178) = 0;
            goto LABEL_54;
          }
          goto LABEL_53;
        }
LABEL_71:
        if ( (_DWORD)v178 )
          goto LABEL_72;
      }
LABEL_87:
      v150 = v36;
      v155 = v35;
      sub_164D160(v169[0], v52, a7, a8, a9, a10, v33, v34, a13, a14);
      v36 = v150;
      v35 = v155;
      goto LABEL_54;
    case '5':
      if ( !*(_BYTE *)(a1 + 1) )
        goto LABEL_140;
      v77 = *(_DWORD *)(*(_QWORD *)v169[0] + 8LL) >> 8;
      if ( !v77 )
        v77 = 5;
LABEL_115:
      *a6 = v77;
      goto LABEL_74;
    case 'H':
      v77 = *(_DWORD *)(*(_QWORD *)v169[0] + 8LL) >> 8;
      if ( !v77 )
      {
        *a6 = *(_DWORD *)(**(_QWORD **)(v169[0] - 24) + 8LL) >> 8;
        goto LABEL_74;
      }
      goto LABEL_115;
    case 'N':
      v78 = *(_QWORD *)(v169[0] - 24);
      if ( *(_BYTE *)(v78 + 16) || (*(_BYTE *)(v78 + 33) & 0x20) == 0 )
      {
        v79 = *(_QWORD **)(a1 + 16);
        if ( !v79 )
          goto LABEL_141;
        v80 = *(_QWORD *)((v169[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
        if ( *(_BYTE *)(v80 + 16) )
          goto LABEL_141;
        v177 = *(unsigned __int64 **)((v169[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
        v81 = (_QWORD *)v79[2];
        if ( !v81 )
          goto LABEL_141;
        v82 = v79 + 1;
        do
        {
          if ( v81[4] < v80 )
          {
            v81 = (_QWORD *)v81[3];
          }
          else
          {
            v82 = v81;
            v81 = (_QWORD *)v81[2];
          }
        }
        while ( v81 );
        if ( v82 == v79 + 1 || v82[4] > v80 )
        {
LABEL_141:
          sub_1C9E680(v16, v169[0], 0, a5);
          return 0;
        }
        v165 = a5;
        v88 = (int *)sub_1C9E790(v79, (unsigned __int64 *)&v177);
LABEL_145:
        v89 = *v88;
        v90 = v169[0];
        *a6 = *v88;
        sub_1C9E680(v16, v90, v89, v165);
        return 1;
      }
      v83 = *(_DWORD *)(*(_QWORD *)v169[0] + 8LL) >> 8;
      if ( !v83 )
      {
        if ( *(_DWORD *)(v78 + 36) != 3660 )
          goto LABEL_140;
        v84 = *a6;
        if ( *a6 )
        {
LABEL_139:
          sub_1C9E680(v16, v37, v84, a5);
          return 1;
        }
        v85 = *(_QWORD *)(v169[0] - 24LL * (*(_DWORD *)(v169[0] + 20) & 0xFFFFFFF));
LABEL_133:
        sub_1C99680((__int64)a5, v169);
        v86 = sub_1C9F820(a1, a2, v85, v16, a5, a6);
        v30 = v86;
        if ( v86 )
        {
          if ( v86 == 1 )
          {
            sub_1C9E680(v16, v169[0], *a6, a5);
            return 1;
          }
        }
        else
        {
          sub_1C9E680(v16, v169[0], 0, a5);
          return 0;
        }
        return v30;
      }
      *a6 = v83;
LABEL_74:
      v65 = (_DWORD *)sub_1C9E600(v16, v169);
      v30 = 1;
      *v65 = *a6;
      return v30;
    case 'G':
      v85 = *(_QWORD *)(v169[0] - 24);
      goto LABEL_138;
    case '8':
      v85 = *(_QWORD *)(v169[0] - 24LL * (*(_DWORD *)(v169[0] + 20) & 0xFFFFFFF));
LABEL_138:
      v84 = *(_DWORD *)(*(_QWORD *)v85 + 8LL) >> 8;
      *a6 = v84;
      if ( v84 )
        goto LABEL_139;
      goto LABEL_133;
  }
  if ( (v38 & 0xFD) == 0x4D )
    goto LABEL_44;
  if ( v38 != 54 )
  {
LABEL_140:
    v87 = (_DWORD *)sub_1C9E600(v16, v169);
    v30 = 0;
    *v87 = 0;
    return v30;
  }
  *a6 = 0;
  v171 = v37;
  v167 = v37;
  if ( (unsigned int)sub_1776BC0(v37) == 4 )
  {
    *a6 = 1;
    sub_1C9E680(v16, v167, 1, a5);
    return 1;
  }
  v105 = *(_QWORD **)(a1 + 512);
  if ( v105 )
  {
    v106 = (_QWORD *)(a1 + 504);
    do
    {
      if ( v105[4] < v167 )
      {
        v105 = (_QWORD *)v105[3];
      }
      else
      {
        v106 = v105;
        v105 = (_QWORD *)v105[2];
      }
    }
    while ( v105 );
    if ( v106 != (_QWORD *)(a1 + 504) && v106[4] <= v167 )
    {
      v165 = a5;
      v88 = (int *)sub_1C9D4C0((_QWORD *)(a1 + 496), (unsigned __int64 *)&v171);
      goto LABEL_145;
    }
  }
  v107 = sub_1C2F070(a2);
  v108 = a5;
  if ( v107 && unk_4FBE1ED )
  {
    v109 = sub_1649C60(*(_QWORD *)(v171 - 24));
    v110 = sub_1776BC0(v171);
    v108 = a5;
    if ( v110 == 101 )
    {
      *a6 = 1;
      if ( *(_BYTE *)(v109 + 16) != 17 )
        goto LABEL_183;
LABEL_181:
      v111 = sub_15E0450(v109);
      v108 = a5;
      if ( v111 )
      {
        *a6 = 1;
LABEL_183:
        v112 = 1;
LABEL_184:
        v113 = v169[0];
        goto LABEL_185;
      }
      goto LABEL_189;
    }
    if ( *(_BYTE *)(v109 + 16) == 17 )
      goto LABEL_181;
  }
LABEL_189:
  v112 = *a6;
  if ( *a6 )
    goto LABEL_183;
  v114 = (_QWORD *)(a1 + 312);
  v115 = *(_QWORD *)(v171 - 24);
  for ( i = *(_QWORD **)(a1 + 320); i; i = v117 )
  {
    v117 = (_QWORD *)i[2];
    v118 = (_QWORD *)i[3];
    if ( i[4] < v115 )
    {
      i = v114;
      v117 = v118;
    }
    v114 = i;
  }
  if ( (_QWORD *)(a1 + 312) == v114 )
    goto LABEL_184;
  if ( v115 < v114[4] )
    goto LABEL_184;
  v177 = *(unsigned __int64 **)(v171 - 24);
  v140 = (__int64)v108;
  v119 = sub_1C9AB60((_QWORD *)(a1 + 304), (unsigned __int64 *)&v177);
  v120 = sub_1C98E50(a1 + 256, v119);
  v112 = 0;
  v108 = (_QWORD *)v140;
  if ( (_QWORD *)(a1 + 264) == v120 )
    goto LABEL_184;
  v177 = *(unsigned __int64 **)(v171 - 24);
  v121 = (const __m128i *)sub_1C9AB60((_QWORD *)(a1 + 304), (unsigned __int64 *)&v177);
  v122 = sub_1C9D010((_QWORD *)(a1 + 256), v121);
  v123 = v140;
  v124 = v122;
  if ( *(_BYTE *)(*(_QWORD *)v171 + 8LL) == 15 )
  {
    sub_1C99680(v140, v169);
    v123 = v140;
  }
  v125 = *(_QWORD *)(v124 + 24);
  v159 = 0;
  v141 = v124 + 8;
  v126 = (_QWORD *)v169[0];
  v127 = a6;
  v128 = (_QWORD *)v123;
  while ( v125 != v141 )
  {
    v129 = *(__int64 **)(v125 + 32);
    v130 = *v129;
    if ( *(_BYTE *)(*v129 + 8) != 15 || v130 != *v126 )
    {
LABEL_222:
      v112 = 0;
      v108 = v128;
      a6 = v127;
      v113 = (unsigned __int64)v126;
LABEL_223:
      *a6 = 0;
      goto LABEL_185;
    }
    v131 = *(_DWORD *)(v130 + 8) >> 8;
    if ( v131 )
    {
      LODWORD(v177) = v131;
LABEL_206:
      if ( v159 )
      {
        if ( v143 != (_DWORD)v177 )
          goto LABEL_222;
      }
      else
      {
        v143 = (int)v177;
        v159 = v146;
      }
      goto LABEL_208;
    }
    v139 = v125;
    v135 = sub_1C9F820(a1, a2, v129, v16, v128, &v177);
    v125 = v139;
    if ( !v135 )
    {
      v108 = v128;
      v112 = 0;
      a6 = v127;
      v113 = v169[0];
      goto LABEL_223;
    }
    v126 = (_QWORD *)v169[0];
    if ( v135 != 2 )
      goto LABEL_206;
LABEL_208:
    v137 = v126;
    v132 = sub_220EF30(v125);
    v126 = v137;
    v125 = v132;
  }
  v108 = v128;
  v112 = 0;
  a6 = v127;
  v113 = (unsigned __int64)v126;
  if ( !v159 )
    goto LABEL_223;
  v112 = 1;
  *a6 = v143;
LABEL_185:
  v168 = v112;
  sub_1C9E680(v16, v113, *a6, v108);
  return v168;
}
