// Function: sub_294BC60
// Address: 0x294bc60
//
void __fastcall sub_294BC60(_BYTE *a1, char a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rax
  const char *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // r9
  const char *v14; // r13
  unsigned int *v15; // rax
  int v16; // ecx
  unsigned int *v17; // rdx
  unsigned int v18; // r13d
  int v19; // ebx
  _BYTE *v20; // rax
  __int64 v21; // rax
  unsigned int v22; // ebx
  __int64 v23; // r12
  unsigned __int8 *v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int8 *v27; // r12
  __int64 (__fastcall *v28)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v29; // r14
  __int64 (__fastcall *v30)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // r14
  unsigned __int64 v35; // rsi
  int v36; // eax
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned __int8 *v39; // r12
  __int64 (__fastcall *v40)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v41; // r13
  __int16 v42; // ax
  _QWORD *v43; // rax
  _BYTE *v44; // r12
  unsigned int *v45; // r13
  unsigned int *v46; // rbx
  __int64 v47; // rdx
  unsigned int v48; // esi
  __int64 v49; // rax
  unsigned __int8 *v50; // r9
  __int64 (__fastcall *v51)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v52; // rax
  __int64 v53; // rbx
  __int64 v54; // r12
  __int64 v55; // rdi
  const char *v56; // rsi
  __int64 v57; // r9
  const char *v58; // r13
  unsigned int *v59; // rax
  int v60; // ecx
  unsigned int *v61; // rdx
  __int64 v62; // r13
  int v63; // eax
  int v64; // eax
  unsigned int v65; // edx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rdx
  int v69; // eax
  int v70; // eax
  unsigned int v71; // edx
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // rdx
  _QWORD *v75; // rax
  unsigned int *v76; // r12
  unsigned int *v77; // rbx
  __int64 v78; // rdx
  unsigned int v79; // esi
  _QWORD *v80; // rax
  __int64 v81; // r9
  unsigned int *v82; // r13
  unsigned int *v83; // r12
  __int64 v84; // rdx
  unsigned int v85; // esi
  unsigned int *v86; // r13
  unsigned int *v87; // rbx
  __int64 v88; // rdx
  unsigned int v89; // esi
  _QWORD **v90; // rdx
  int v91; // ecx
  __int64 *v92; // rax
  __int64 v93; // rsi
  unsigned int *v94; // r12
  __int64 v95; // rax
  unsigned int *v96; // r13
  __int64 v97; // rdx
  unsigned int v98; // esi
  __int64 v99; // rax
  char v100; // al
  __int16 v101; // cx
  __int64 v102; // rax
  __int64 v103; // rax
  unsigned __int8 *v104; // r12
  __int64 (__fastcall *v105)(__int64, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v106; // rax
  unsigned int *v107; // r12
  __int64 v108; // rax
  unsigned int *v109; // r13
  __int64 v110; // rdx
  unsigned int v111; // esi
  __int64 v112; // r13
  __int64 v113; // rax
  __int64 v114; // rax
  unsigned __int8 *v115; // r12
  __int64 (__fastcall *v116)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v117; // r14
  char v118; // r15
  _QWORD *v119; // rax
  _BYTE *v120; // r12
  unsigned int *v121; // r14
  unsigned int *v122; // rbx
  __int64 v123; // rdx
  unsigned int v124; // esi
  __int64 v125; // rax
  unsigned __int8 *v126; // rbx
  __int64 (__fastcall *v127)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v128; // rdi
  __int64 v129; // r14
  _QWORD *v130; // rax
  unsigned int *v131; // r12
  unsigned int *v132; // rbx
  __int64 v133; // rdx
  unsigned int v134; // esi
  _QWORD *v135; // rax
  __int64 v136; // r9
  unsigned int *v137; // r12
  unsigned int *v138; // rbx
  __int64 v139; // rdx
  unsigned int v140; // esi
  __int64 v141; // rax
  unsigned __int64 v142; // r8
  unsigned __int64 v143; // rax
  unsigned __int64 v144; // rsi
  __int64 v145; // [rsp+8h] [rbp-258h]
  __int64 v148; // [rsp+38h] [rbp-228h]
  __int64 v151; // [rsp+68h] [rbp-1F8h]
  __int64 v152; // [rsp+70h] [rbp-1F0h]
  __int64 v153; // [rsp+78h] [rbp-1E8h]
  __int64 v154; // [rsp+80h] [rbp-1E0h]
  __int64 v155; // [rsp+88h] [rbp-1D8h]
  __int64 v156; // [rsp+90h] [rbp-1D0h]
  char v157; // [rsp+9Ah] [rbp-1C6h]
  char v158; // [rsp+9Bh] [rbp-1C5h]
  unsigned int v159; // [rsp+9Ch] [rbp-1C4h]
  _BYTE *v160; // [rsp+A8h] [rbp-1B8h]
  unsigned __int8 *v161; // [rsp+B8h] [rbp-1A8h]
  unsigned __int8 *v162; // [rsp+C0h] [rbp-1A0h]
  unsigned __int8 *v163; // [rsp+C0h] [rbp-1A0h]
  unsigned __int64 v164; // [rsp+C0h] [rbp-1A0h]
  __int16 v165; // [rsp+D0h] [rbp-190h]
  __int64 v166; // [rsp+D8h] [rbp-188h]
  __int64 v167; // [rsp+E0h] [rbp-180h]
  __int64 v168; // [rsp+108h] [rbp-158h]
  char v169[32]; // [rsp+110h] [rbp-150h] BYREF
  __int16 v170; // [rsp+130h] [rbp-130h]
  _QWORD v171[2]; // [rsp+140h] [rbp-120h] BYREF
  int v172; // [rsp+150h] [rbp-110h]
  __int16 v173; // [rsp+160h] [rbp-100h]
  char *v174; // [rsp+170h] [rbp-F0h] BYREF
  unsigned int v175; // [rsp+178h] [rbp-E8h]
  __int16 v176; // [rsp+190h] [rbp-D0h]
  unsigned int *v177; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v178; // [rsp+1A8h] [rbp-B8h]
  _BYTE v179[32]; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v180; // [rsp+1D0h] [rbp-90h]
  __int64 v181; // [rsp+1D8h] [rbp-88h]
  __int64 v182; // [rsp+1E0h] [rbp-80h]
  __int64 *v183; // [rsp+1E8h] [rbp-78h]
  void **v184; // [rsp+1F0h] [rbp-70h]
  void **v185; // [rsp+1F8h] [rbp-68h]
  __int64 v186; // [rsp+200h] [rbp-60h]
  int v187; // [rsp+208h] [rbp-58h]
  __int16 v188; // [rsp+20Ch] [rbp-54h]
  char v189; // [rsp+20Eh] [rbp-52h]
  __int64 v190; // [rsp+210h] [rbp-50h]
  __int64 v191; // [rsp+218h] [rbp-48h]
  void *v192; // [rsp+220h] [rbp-40h] BYREF
  void *v193; // [rsp+228h] [rbp-38h] BYREF

  v7 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v160 = *(_BYTE **)(a3 - 32 * v7);
  v8 = *(_QWORD *)(a3 + 32 * (1 - v7));
  v151 = *(_QWORD *)(a3 + 32 * (2 - v7));
  v166 = *(_QWORD *)(a3 + 32 * (3 - v7));
  v152 = *(_QWORD *)(a3 + 8);
  v153 = *(_QWORD *)(v152 + 24);
  v189 = 7;
  v183 = (__int64 *)sub_BD5C60(a3);
  v184 = &v192;
  v185 = &v193;
  v177 = (unsigned int *)v179;
  v192 = &unk_49DA100;
  v178 = 0x200000000LL;
  v193 = &unk_49DA0B0;
  v9 = *(_QWORD *)(a3 + 40);
  v186 = 0;
  v156 = v9;
  v187 = 0;
  v188 = 512;
  v190 = 0;
  v191 = 0;
  v180 = 0;
  v181 = 0;
  LOWORD(v182) = 0;
  sub_D5F1F0((__int64)&v177, a3);
  v10 = *(_QWORD *)(v8 + 24);
  if ( *(_DWORD *)(v8 + 32) > 0x40u )
    v10 = *(_QWORD *)v10;
  v158 = 0;
  if ( v10 )
  {
    _BitScanReverse64(&v10, v10);
    v158 = 1;
    v157 = 63 - (v10 ^ 0x3F);
  }
  v11 = *(const char **)(a3 + 48);
  v174 = (char *)v11;
  if ( v11 && (sub_B96E90((__int64)&v174, (__int64)v11, 1), (v14 = v174) != 0) )
  {
    v15 = v177;
    v16 = v178;
    v17 = &v177[4 * (unsigned int)v178];
    if ( v177 != v17 )
    {
      while ( *v15 )
      {
        v15 += 4;
        if ( v17 == v15 )
          goto LABEL_147;
      }
      *((_QWORD *)v15 + 1) = v174;
      goto LABEL_12;
    }
LABEL_147:
    if ( (unsigned int)v178 >= (unsigned __int64)HIDWORD(v178) )
    {
      v144 = (unsigned int)v178 + 1LL;
      if ( HIDWORD(v178) < v144 )
      {
        sub_C8D5F0((__int64)&v177, v179, v144, 0x10u, v12, v13);
        v17 = &v177[4 * (unsigned int)v178];
      }
      *(_QWORD *)v17 = 0;
      *((_QWORD *)v17 + 1) = v14;
      v14 = v174;
      LODWORD(v178) = v178 + 1;
    }
    else
    {
      if ( v17 )
      {
        *v17 = 0;
        *((_QWORD *)v17 + 1) = v14;
        v16 = v178;
        v14 = v174;
      }
      LODWORD(v178) = v16 + 1;
    }
  }
  else
  {
    sub_93FB40((__int64)&v177, 0);
    v14 = v174;
  }
  if ( v14 )
LABEL_12:
    sub_B91220((__int64)&v174, (__int64)v14);
  v159 = *(_DWORD *)(v152 + 32);
  if ( *(_BYTE *)v151 > 0x15u )
  {
LABEL_19:
    if ( v159 == 1 || a2 == 1 )
    {
      v161 = 0;
    }
    else
    {
      v21 = sub_BCD140(v183, v159);
      v174 = "scalar_mask";
      v176 = 259;
      v161 = (unsigned __int8 *)sub_A83570(&v177, v151, v21, (__int64)&v174);
    }
    if ( !v159 )
    {
      v62 = v166;
LABEL_141:
      sub_BD84D0(a3, v62);
      sub_B43D60((_QWORD *)a3);
      *a5 = 1;
      goto LABEL_142;
    }
    v167 = 0;
    while ( 1 )
    {
      if ( v161 )
      {
        v22 = v159 - 1 - v167;
        if ( !*a1 )
          v22 = v167;
        v175 = v159;
        v23 = 1LL << v22;
        if ( v159 <= 0x40 )
        {
          v174 = 0;
          goto LABEL_29;
        }
        sub_C43690((__int64)&v174, 0, 0);
        if ( v175 <= 0x40 )
LABEL_29:
          v174 = (char *)(v23 | (unsigned __int64)v174);
        else
          *(_QWORD *)&v174[8 * (v22 >> 6)] |= v23;
        v24 = (unsigned __int8 *)sub_ACCFD0(v183, (__int64)&v174);
        if ( v175 > 0x40 && v174 )
          j_j___libc_free_0_0((unsigned __int64)v174);
        v173 = 257;
        v25 = sub_BCD140(v183, v159);
        v26 = sub_ACD640(v25, 0, 0);
        v170 = 257;
        v27 = (unsigned __int8 *)v26;
        v28 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v184 + 2);
        if ( v28 != sub_9202E0 )
        {
          v29 = v28((__int64)v184, 28u, v161, v24);
          goto LABEL_38;
        }
        if ( *v161 > 0x15u || *v24 > 0x15u )
          goto LABEL_108;
        v29 = (unsigned __int8)sub_AC47B0(28) ? sub_AD5570(28, (__int64)v161, v24, 0, 0) : sub_AABE40(0x1Cu, v161, v24);
LABEL_38:
        if ( !v29 )
        {
LABEL_108:
          v176 = 257;
          v29 = sub_B504D0(28, (__int64)v161, (__int64)v24, (__int64)&v174, 0, 0);
          (*((void (__fastcall **)(void **, __int64, char *, __int64, __int64))*v185 + 2))(v185, v29, v169, v181, v182);
          v86 = v177;
          v87 = &v177[4 * (unsigned int)v178];
          if ( v177 != v87 )
          {
            do
            {
              v88 = *((_QWORD *)v86 + 1);
              v89 = *v86;
              v86 += 4;
              sub_B99FD0(v29, v89, v88);
            }
            while ( v87 != v86 );
          }
        }
        v30 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v184 + 7);
        if ( v30 == sub_928890 )
        {
          if ( *(_BYTE *)v29 <= 0x15u && *v27 <= 0x15u )
          {
            v31 = sub_AAB310(0x21u, (unsigned __int8 *)v29, v27);
LABEL_43:
            if ( v31 )
              goto LABEL_44;
          }
          v176 = 257;
          v31 = (__int64)sub_BD2C40(72, unk_3F10FD0);
          if ( v31 )
          {
            v90 = *(_QWORD ***)(v29 + 8);
            v91 = *((unsigned __int8 *)v90 + 8);
            if ( (unsigned int)(v91 - 17) > 1 )
            {
              v93 = sub_BCB2A0(*v90);
            }
            else
            {
              BYTE4(v168) = (_BYTE)v91 == 18;
              LODWORD(v168) = *((_DWORD *)v90 + 8);
              v92 = (__int64 *)sub_BCB2A0(*v90);
              v93 = sub_BCE1B0(v92, v168);
            }
            sub_B523C0(v31, v93, 53, 33, v29, (__int64)v27, (__int64)&v174, 0, 0, 0);
          }
          (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v185 + 2))(
            v185,
            v31,
            v171,
            v181,
            v182);
          v94 = v177;
          v95 = 4LL * (unsigned int)v178;
          v96 = &v177[v95];
          if ( v177 != &v177[v95] )
          {
            do
            {
              v97 = *((_QWORD *)v94 + 1);
              v98 = *v94;
              v94 += 4;
              sub_B99FD0(v31, v98, v97);
            }
            while ( v96 != v94 );
          }
          goto LABEL_44;
        }
        v31 = v30((__int64)v184, 33u, (_BYTE *)v29, v27);
        goto LABEL_43;
      }
      v173 = 2307;
      v171[0] = "Mask";
      v172 = v167;
      v103 = sub_BCB2E0(v183);
      v104 = (unsigned __int8 *)sub_ACD640(v103, v167, 0);
      v105 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v184 + 12);
      if ( v105 != sub_948070 )
        break;
      if ( *(_BYTE *)v151 <= 0x15u && *v104 <= 0x15u )
      {
        v31 = sub_AD5840(v151, v104, 0);
        goto LABEL_132;
      }
LABEL_133:
      v176 = 257;
      v106 = sub_BD2C40(72, 2u);
      v31 = (__int64)v106;
      if ( v106 )
        sub_B4DE80((__int64)v106, v151, (__int64)v104, (__int64)&v174, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v185 + 2))(v185, v31, v171, v181, v182);
      v107 = v177;
      v108 = 4LL * (unsigned int)v178;
      v109 = &v177[v108];
      if ( v177 != &v177[v108] )
      {
        do
        {
          v110 = *((_QWORD *)v107 + 1);
          v111 = *v107;
          v107 += 4;
          sub_B99FD0(v31, v111, v110);
        }
        while ( v109 != v107 );
      }
LABEL_44:
      v32 = v154;
      LOWORD(v32) = 0;
      v154 = v32;
      v33 = sub_F38250(v31, (__int64 *)(a3 + 24), v32, 0, 0, a4, 0, 0);
      v34 = *(_QWORD *)(v33 + 40);
      v155 = v33;
      v174 = "cond.load";
      v176 = 259;
      sub_BD6B50((unsigned __int8 *)v34, (const char **)&v174);
      v35 = *(_QWORD *)(v34 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v35 == v34 + 48 )
      {
        v37 = 0;
      }
      else
      {
        if ( !v35 )
          BUG();
        v36 = *(unsigned __int8 *)(v35 - 24);
        v37 = v35 - 24;
        if ( (unsigned int)(v36 - 30) >= 0xB )
          v37 = 0;
      }
      sub_D5F1F0((__int64)&v177, v37);
      v171[0] = "Ptr";
      v172 = v167;
      v173 = 2307;
      v38 = sub_BCB2E0(v183);
      v39 = (unsigned __int8 *)sub_ACD640(v38, v167, 0);
      v40 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v184 + 12);
      if ( v40 != sub_948070 )
      {
        v41 = v40((__int64)v184, v160, v39);
LABEL_52:
        if ( v41 )
          goto LABEL_53;
        goto LABEL_96;
      }
      if ( *v160 <= 0x15u && *v39 <= 0x15u )
      {
        v41 = sub_AD5840((__int64)v160, v39, 0);
        goto LABEL_52;
      }
LABEL_96:
      v176 = 257;
      v75 = sub_BD2C40(72, 2u);
      v41 = (__int64)v75;
      if ( v75 )
        sub_B4DE80((__int64)v75, (__int64)v160, (__int64)v39, (__int64)&v174, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v185 + 2))(v185, v41, v171, v181, v182);
      v76 = v177;
      v77 = &v177[4 * (unsigned int)v178];
      if ( v177 != v77 )
      {
        do
        {
          v78 = *((_QWORD *)v76 + 1);
          v79 = *v76;
          v76 += 4;
          sub_B99FD0(v41, v79, v78);
        }
        while ( v77 != v76 );
      }
LABEL_53:
      v171[0] = "Load";
      v172 = v167;
      v173 = 2307;
      HIBYTE(v42) = HIBYTE(v165);
      LOBYTE(v42) = v157;
      v165 = v42;
      if ( !v158 )
      {
        v99 = sub_AA4E30(v180);
        v100 = sub_AE5020(v99, v153);
        HIBYTE(v101) = HIBYTE(v165);
        LOBYTE(v101) = v100;
        v165 = v101;
      }
      v176 = 257;
      v43 = sub_BD2C40(80, 1u);
      v44 = v43;
      if ( v43 )
        sub_B4D190((__int64)v43, v153, v41, (__int64)&v174, 0, v165, 0, 0);
      (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v185 + 2))(v185, v44, v171, v181, v182);
      v45 = v177;
      v46 = &v177[4 * (unsigned int)v178];
      if ( v177 != v46 )
      {
        do
        {
          v47 = *((_QWORD *)v45 + 1);
          v48 = *v45;
          v45 += 4;
          sub_B99FD0((__int64)v44, v48, v47);
        }
        while ( v46 != v45 );
      }
      v171[0] = "Res";
      v173 = 2307;
      v172 = v167;
      v49 = sub_BCB2E0(v183);
      v50 = (unsigned __int8 *)sub_ACD640(v49, v167, 0);
      v51 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))*((_QWORD *)*v184 + 13);
      if ( v51 != sub_948040 )
      {
        v163 = v50;
        v102 = v51((__int64)v184, (_BYTE *)v166, v44, v50);
        v50 = v163;
        v53 = v102;
LABEL_64:
        if ( v53 )
          goto LABEL_65;
        goto LABEL_103;
      }
      if ( *(_BYTE *)v166 <= 0x15u && *v44 <= 0x15u && *v50 <= 0x15u )
      {
        v162 = v50;
        v52 = sub_AD5A90(v166, v44, v50, 0);
        v50 = v162;
        v53 = v52;
        goto LABEL_64;
      }
LABEL_103:
      v148 = (__int64)v50;
      v176 = 257;
      v80 = sub_BD2C40(72, 3u);
      v81 = v148;
      v53 = (__int64)v80;
      if ( v80 )
        sub_B4DFA0((__int64)v80, v166, (__int64)v44, v148, (__int64)&v174, v148, 0, 0);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64, __int64))*v185 + 2))(
        v185,
        v53,
        v171,
        v181,
        v182,
        v81);
      v82 = v177;
      v83 = &v177[4 * (unsigned int)v178];
      if ( v177 != v83 )
      {
        do
        {
          v84 = *((_QWORD *)v82 + 1);
          v85 = *v82;
          v82 += 4;
          sub_B99FD0(v53, v85, v84);
        }
        while ( v83 != v82 );
      }
LABEL_65:
      v54 = sub_B46EC0(v155, 0);
      v176 = 259;
      v174 = "else";
      sub_BD6B50((unsigned __int8 *)v54, (const char **)&v174);
      v55 = *(_QWORD *)(v54 + 56);
      v180 = v54;
      LOWORD(v182) = 1;
      v181 = v55;
      if ( v55 == v54 + 48 )
        goto LABEL_76;
      if ( v55 )
        v55 -= 24;
      v56 = *(const char **)sub_B46C60(v55);
      v174 = (char *)v56;
      if ( v56 && (sub_B96E90((__int64)&v174, (__int64)v56, 1), (v58 = v174) != 0) )
      {
        v59 = v177;
        v60 = v178;
        v61 = &v177[4 * (unsigned int)v178];
        if ( v177 != v61 )
        {
          while ( *v59 )
          {
            v59 += 4;
            if ( v61 == v59 )
              goto LABEL_121;
          }
          *((_QWORD *)v59 + 1) = v174;
LABEL_75:
          sub_B91220((__int64)&v174, (__int64)v58);
          goto LABEL_76;
        }
LABEL_121:
        if ( (unsigned int)v178 >= (unsigned __int64)HIDWORD(v178) )
        {
          v142 = (unsigned int)v178 + 1LL;
          v143 = v145 & 0xFFFFFFFF00000000LL;
          v145 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v178) < v142 )
          {
            v164 = v143;
            sub_C8D5F0((__int64)&v177, v179, v142, 0x10u, v142, v57);
            v143 = v164;
            v61 = &v177[4 * (unsigned int)v178];
          }
          *(_QWORD *)v61 = v143;
          *((_QWORD *)v61 + 1) = v58;
          v58 = v174;
          LODWORD(v178) = v178 + 1;
        }
        else
        {
          if ( v61 )
          {
            *v61 = 0;
            *((_QWORD *)v61 + 1) = v58;
            v60 = v178;
            v58 = v174;
          }
          LODWORD(v178) = v60 + 1;
        }
      }
      else
      {
        sub_93FB40((__int64)&v177, 0);
        v58 = v174;
      }
      if ( v58 )
        goto LABEL_75;
LABEL_76:
      v174 = "res.phi.else";
      v176 = 259;
      v62 = sub_D5C860((__int64 *)&v177, v152, 2, (__int64)&v174);
      v63 = *(_DWORD *)(v62 + 4) & 0x7FFFFFF;
      if ( v63 == *(_DWORD *)(v62 + 72) )
      {
        sub_B48D90(v62);
        v63 = *(_DWORD *)(v62 + 4) & 0x7FFFFFF;
      }
      v64 = (v63 + 1) & 0x7FFFFFF;
      v65 = v64 | *(_DWORD *)(v62 + 4) & 0xF8000000;
      v66 = *(_QWORD *)(v62 - 8) + 32LL * (unsigned int)(v64 - 1);
      *(_DWORD *)(v62 + 4) = v65;
      if ( *(_QWORD *)v66 )
      {
        v67 = *(_QWORD *)(v66 + 8);
        **(_QWORD **)(v66 + 16) = v67;
        if ( v67 )
          *(_QWORD *)(v67 + 16) = *(_QWORD *)(v66 + 16);
      }
      *(_QWORD *)v66 = v53;
      if ( v53 )
      {
        v68 = *(_QWORD *)(v53 + 16);
        *(_QWORD *)(v66 + 8) = v68;
        if ( v68 )
          *(_QWORD *)(v68 + 16) = v66 + 8;
        *(_QWORD *)(v66 + 16) = v53 + 16;
        *(_QWORD *)(v53 + 16) = v66;
      }
      *(_QWORD *)(*(_QWORD *)(v62 - 8)
                + 32LL * *(unsigned int *)(v62 + 72)
                + 8LL * ((*(_DWORD *)(v62 + 4) & 0x7FFFFFFu) - 1)) = v34;
      v69 = *(_DWORD *)(v62 + 4) & 0x7FFFFFF;
      if ( v69 == *(_DWORD *)(v62 + 72) )
      {
        sub_B48D90(v62);
        v69 = *(_DWORD *)(v62 + 4) & 0x7FFFFFF;
      }
      v70 = (v69 + 1) & 0x7FFFFFF;
      v71 = v70 | *(_DWORD *)(v62 + 4) & 0xF8000000;
      v72 = *(_QWORD *)(v62 - 8) + 32LL * (unsigned int)(v70 - 1);
      *(_DWORD *)(v62 + 4) = v71;
      if ( *(_QWORD *)v72 )
      {
        v73 = *(_QWORD *)(v72 + 8);
        **(_QWORD **)(v72 + 16) = v73;
        if ( v73 )
          *(_QWORD *)(v73 + 16) = *(_QWORD *)(v72 + 16);
      }
      *(_QWORD *)v72 = v166;
      if ( v166 )
      {
        v74 = *(_QWORD *)(v166 + 16);
        *(_QWORD *)(v72 + 8) = v74;
        if ( v74 )
          *(_QWORD *)(v74 + 16) = v72 + 8;
        *(_QWORD *)(v72 + 16) = v166 + 16;
        *(_QWORD *)(v166 + 16) = v72;
      }
      ++v167;
      *(_QWORD *)(*(_QWORD *)(v62 - 8)
                + 32LL * *(unsigned int *)(v62 + 72)
                + 8LL * ((*(_DWORD *)(v62 + 4) & 0x7FFFFFFu) - 1)) = v156;
      if ( v159 == v167 )
        goto LABEL_141;
      v166 = v62;
      v156 = v54;
    }
    v31 = v105((__int64)v184, (_BYTE *)v151, v104);
LABEL_132:
    if ( v31 )
      goto LABEL_44;
    goto LABEL_133;
  }
  v18 = 0;
  v19 = *(_DWORD *)(*(_QWORD *)(v151 + 8) + 32LL);
  if ( v19 )
  {
    do
    {
      v20 = (_BYTE *)sub_AD69F0((unsigned __int8 *)v151, v18);
      if ( !v20 || *v20 != 17 )
        goto LABEL_19;
    }
    while ( v19 != ++v18 );
  }
  if ( v159 )
  {
    v112 = 0;
    while ( 1 )
    {
      v113 = sub_AD69F0((unsigned __int8 *)v151, (unsigned int)v112);
      if ( !sub_AC30F0(v113) )
        break;
LABEL_178:
      if ( v159 == ++v112 )
        goto LABEL_179;
    }
    v172 = v112;
    v173 = 2307;
    v171[0] = "Ptr";
    v114 = sub_BCB2E0(v183);
    v115 = (unsigned __int8 *)sub_ACD640(v114, v112, 0);
    v116 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v184 + 12);
    if ( v116 == sub_948070 )
    {
      if ( *v160 > 0x15u || *v115 > 0x15u )
      {
LABEL_180:
        v176 = 257;
        v130 = sub_BD2C40(72, 2u);
        v117 = (__int64)v130;
        if ( v130 )
          sub_B4DE80((__int64)v130, (__int64)v160, (__int64)v115, (__int64)&v174, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v185 + 2))(v185, v117, v171, v181, v182);
        v131 = v177;
        v132 = &v177[4 * (unsigned int)v178];
        if ( v177 != v132 )
        {
          do
          {
            v133 = *((_QWORD *)v131 + 1);
            v134 = *v131;
            v131 += 4;
            sub_B99FD0(v117, v134, v133);
          }
          while ( v132 != v131 );
        }
LABEL_163:
        v118 = v157;
        v171[0] = "Load";
        v173 = 2307;
        v172 = v112;
        if ( !v158 )
        {
          v141 = sub_AA4E30(v180);
          v118 = sub_AE5020(v141, v153);
        }
        v176 = 257;
        v119 = sub_BD2C40(80, 1u);
        v120 = v119;
        if ( v119 )
          sub_B4D190((__int64)v119, v153, v117, (__int64)&v174, 0, v118, 0, 0);
        (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v185 + 2))(v185, v120, v171, v181, v182);
        v121 = v177;
        v122 = &v177[4 * (unsigned int)v178];
        if ( v177 != v122 )
        {
          do
          {
            v123 = *((_QWORD *)v121 + 1);
            v124 = *v121;
            v121 += 4;
            sub_B99FD0((__int64)v120, v124, v123);
          }
          while ( v122 != v121 );
        }
        v171[0] = "Res";
        v172 = v112;
        v173 = 2307;
        v125 = sub_BCB2E0(v183);
        v126 = (unsigned __int8 *)sub_ACD640(v125, v112, 0);
        v127 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))*((_QWORD *)*v184 + 13);
        if ( v127 == sub_948040 )
        {
          v128 = 0;
          if ( *(_BYTE *)v166 <= 0x15u )
            v128 = v166;
          if ( *v120 > 0x15u || *v126 > 0x15u || !v128 )
            goto LABEL_185;
          v129 = sub_AD5A90(v128, v120, v126, 0);
        }
        else
        {
          v129 = v127((__int64)v184, (_BYTE *)v166, v120, v126);
        }
        if ( v129 )
        {
LABEL_177:
          v166 = v129;
          goto LABEL_178;
        }
LABEL_185:
        v176 = 257;
        v135 = sub_BD2C40(72, 3u);
        v136 = 0;
        v129 = (__int64)v135;
        if ( v135 )
          sub_B4DFA0((__int64)v135, v166, (__int64)v120, (__int64)v126, (__int64)&v174, 0, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64, __int64))*v185 + 2))(
          v185,
          v129,
          v171,
          v181,
          v182,
          v136);
        v137 = v177;
        v138 = &v177[4 * (unsigned int)v178];
        if ( v177 != v138 )
        {
          do
          {
            v139 = *((_QWORD *)v137 + 1);
            v140 = *v137;
            v137 += 4;
            sub_B99FD0(v129, v140, v139);
          }
          while ( v138 != v137 );
        }
        goto LABEL_177;
      }
      v117 = sub_AD5840((__int64)v160, v115, 0);
    }
    else
    {
      v117 = v116((__int64)v184, v160, v115);
    }
    if ( v117 )
      goto LABEL_163;
    goto LABEL_180;
  }
LABEL_179:
  sub_BD84D0(a3, v166);
  sub_B43D60((_QWORD *)a3);
LABEL_142:
  nullsub_61();
  v192 = &unk_49DA100;
  nullsub_63();
  if ( v177 != (unsigned int *)v179 )
    _libc_free((unsigned __int64)v177);
}
