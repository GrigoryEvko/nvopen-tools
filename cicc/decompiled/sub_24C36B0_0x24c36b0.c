// Function: sub_24C36B0
// Address: 0x24c36b0
//
void __fastcall sub_24C36B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, char a6)
{
  __int16 v8; // dx
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int *v17; // r13
  unsigned int *v18; // rax
  unsigned int v19; // ecx
  unsigned int *v20; // rdx
  __int64 v21; // rdi
  _BYTE *v22; // rax
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // rax
  char v29; // r15
  __int64 v30; // r9
  _QWORD *v31; // r13
  __int64 v32; // rsi
  unsigned int *v33; // r15
  unsigned int *v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // r14
  _BYTE *v38; // rax
  __int64 v39; // rax
  unsigned __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rax
  char v43; // r14
  _QWORD *v44; // rax
  __int64 v45; // r9
  __int64 v46; // r15
  unsigned int *v47; // r14
  unsigned int *v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rdi
  _BYTE *v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // r14
  __int64 v57; // rax
  char v58; // r15
  unsigned __int8 *v59; // r13
  unsigned int *v60; // r15
  unsigned int *v61; // r14
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // rdi
  unsigned __int8 *v65; // r14
  __int64 (__fastcall *v66)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v67; // r10
  __int64 v68; // rax
  char v69; // r14
  _QWORD *v70; // rax
  __int64 v71; // r9
  __int64 v72; // r15
  unsigned int *v73; // r14
  unsigned int *v74; // r12
  __int64 v75; // rdx
  __int64 v76; // r13
  __int64 v77; // r15
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 (__fastcall *v80)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v81; // rax
  _BYTE **v82; // rcx
  __int64 v83; // r14
  bool v84; // zf
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rsi
  __int64 v87; // rdx
  __int64 v88; // r12
  __int64 v89; // rdx
  __int64 v90; // r12
  __int64 v91; // rdi
  __int64 v92; // r12
  unsigned __int64 v93; // rax
  __int64 **v94; // rcx
  unsigned __int64 v95; // rax
  __int64 v96; // r14
  __int64 v97; // r15
  __int64 v98; // rax
  char v99; // al
  _QWORD *v100; // rax
  __int64 v101; // r9
  _BYTE *v102; // r13
  unsigned int *v103; // r15
  unsigned int *i; // r14
  __int64 v105; // rdx
  unsigned int v106; // esi
  __int64 v107; // r14
  __int64 v108; // rax
  unsigned __int64 v109; // rax
  __int64 v110; // r12
  __int64 v111; // rax
  char v112; // bl
  _QWORD *v113; // rax
  __int64 v114; // r9
  __int64 v115; // r14
  __int64 v116; // rsi
  unsigned int *v117; // rbx
  unsigned int *v118; // r12
  __int64 v119; // rdx
  unsigned int *v120; // r14
  unsigned int *v121; // r15
  __int64 v122; // rbx
  __int64 v123; // rdx
  unsigned int v124; // esi
  __int64 v125; // r11
  unsigned int *v126; // r15
  unsigned int *v127; // r13
  __int64 v128; // rdx
  unsigned int v129; // esi
  __int64 v130; // rax
  __int64 v131; // rcx
  int v132; // esi
  __int64 v133; // rax
  __int64 *v134; // rdi
  _QWORD *v135; // rax
  unsigned __int64 v136; // rsi
  __int64 v137; // rdx
  __int64 v138; // r12
  __int64 *v139; // rax
  __int64 v140; // rax
  int v141; // edx
  int v142; // edx
  char v143; // dl
  int v144; // eax
  __int64 v145; // [rsp-10h] [rbp-200h]
  __int64 v146; // [rsp-10h] [rbp-200h]
  unsigned __int64 v147; // [rsp-10h] [rbp-200h]
  __int64 v148; // [rsp+8h] [rbp-1E8h]
  char v150; // [rsp+17h] [rbp-1D9h]
  __int64 v152; // [rsp+18h] [rbp-1D8h]
  char v153; // [rsp+18h] [rbp-1D8h]
  __int64 v156; // [rsp+30h] [rbp-1C0h]
  __int64 v157; // [rsp+38h] [rbp-1B8h]
  __int64 v158; // [rsp+38h] [rbp-1B8h]
  __int64 v159; // [rsp+38h] [rbp-1B8h]
  __int64 v160; // [rsp+48h] [rbp-1A8h]
  __int64 v161; // [rsp+48h] [rbp-1A8h]
  unsigned int *v162; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v163; // [rsp+58h] [rbp-198h] BYREF
  _BYTE *v164; // [rsp+60h] [rbp-190h] BYREF
  __int64 v165; // [rsp+68h] [rbp-188h]
  _BYTE *v166; // [rsp+70h] [rbp-180h] BYREF
  __int64 v167; // [rsp+78h] [rbp-178h]
  __int16 v168; // [rsp+90h] [rbp-160h]
  unsigned int *v169; // [rsp+A0h] [rbp-150h] BYREF
  unsigned int v170; // [rsp+A8h] [rbp-148h]
  unsigned int v171; // [rsp+ACh] [rbp-144h]
  _BYTE v172[32]; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v173; // [rsp+D0h] [rbp-120h]
  __int64 v174; // [rsp+D8h] [rbp-118h]
  __int64 v175; // [rsp+E0h] [rbp-110h]
  __int64 *v176; // [rsp+E8h] [rbp-108h]
  __int64 v177; // [rsp+F0h] [rbp-100h]
  __int64 v178; // [rsp+F8h] [rbp-F8h]
  void *v179; // [rsp+120h] [rbp-D0h]
  unsigned int *v180; // [rsp+130h] [rbp-C0h] BYREF
  unsigned int v181; // [rsp+138h] [rbp-B8h]
  unsigned __int64 v182; // [rsp+140h] [rbp-B0h] BYREF
  unsigned int v183; // [rsp+148h] [rbp-A8h]
  __int16 v184; // [rsp+150h] [rbp-A0h]
  __int64 v185; // [rsp+160h] [rbp-90h]
  __int64 v186; // [rsp+168h] [rbp-88h]
  __int64 v187; // [rsp+170h] [rbp-80h]
  __int64 v188; // [rsp+188h] [rbp-68h]
  void *v189; // [rsp+1B0h] [rbp-40h]

  v160 = sub_AA5190(a3);
  if ( v160 )
  {
    LOBYTE(v9) = v8;
    v10 = HIBYTE(v8);
  }
  else
  {
    v10 = 0;
    LOBYTE(v9) = 0;
  }
  v162 = 0;
  v9 = (unsigned __int8)v9;
  BYTE1(v9) = v10;
  v11 = v9;
  v12 = *(_QWORD *)(a2 + 80);
  if ( v12 )
  {
    v150 = 0;
    if ( a3 == v12 - 24 )
    {
      v130 = sub_B92180(a2);
      v131 = v130;
      if ( v130 )
      {
        v132 = *(_DWORD *)(v130 + 20);
        v133 = *(_QWORD *)(v130 + 8);
        v134 = (__int64 *)(v133 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v133 & 4) != 0 )
          v134 = (__int64 *)*v134;
        v135 = sub_B01860(v134, v132, 0, v131, 0, 0, 0, 1);
        sub_B10CB0(&v180, (__int64)v135);
        v162 = v180;
        if ( v180 )
          sub_B976B0((__int64)&v180, (unsigned __int8 *)v180, (__int64)&v162);
      }
      v150 = 1;
      v160 = sub_29F3B00(a3, v160, v11);
    }
  }
  else
  {
    v150 = 0;
  }
  v13 = v160 - 24;
  if ( !v160 )
    v13 = 0;
  v161 = v13;
  sub_23E3770((__int64)&v169, v13);
  v14 = (unsigned __int64)v162;
  if ( v162 )
  {
    v180 = v162;
    sub_B96E90((__int64)&v180, (__int64)v162, 1);
    v17 = v180;
    if ( v180 )
    {
      v14 = v170;
      v18 = v169;
      v19 = v170;
      v20 = &v169[4 * v170];
      if ( v169 != v20 )
      {
        while ( 1 )
        {
          v16 = *v18;
          if ( !(_DWORD)v16 )
            break;
          v18 += 4;
          if ( v20 == v18 )
            goto LABEL_63;
        }
        *((_QWORD *)v18 + 1) = v180;
        goto LABEL_14;
      }
LABEL_63:
      if ( v170 >= (unsigned __int64)v171 )
      {
        v14 = v170 + 1LL;
        if ( v171 < v14 )
        {
          v14 = (unsigned __int64)v172;
          sub_C8D5F0((__int64)&v169, v172, v170 + 1LL, 0x10u, v15, v16);
          v20 = &v169[4 * v170];
        }
        *(_QWORD *)v20 = 0;
        *((_QWORD *)v20 + 1) = v17;
        v17 = v180;
        ++v170;
      }
      else
      {
        if ( v20 )
        {
          *v20 = 0;
          *((_QWORD *)v20 + 1) = v17;
          v19 = v170;
          v17 = v180;
        }
        v170 = v19 + 1;
      }
    }
    else
    {
      v14 = 0;
      sub_93FB40((__int64)&v169, 0);
      v17 = v180;
    }
    if ( v17 )
    {
LABEL_14:
      v14 = (unsigned __int64)v17;
      sub_B91220((__int64)&v180, (__int64)v17);
    }
  }
  if ( *(_BYTE *)(a1 + 1026) )
  {
    v89 = *(_QWORD *)(a1 + 64);
    v184 = 257;
    v90 = sub_921880(&v169, *(_QWORD *)(a1 + 56), v89, 0, 0, (__int64)&v180, 0);
    v14 = sub_BD5C60(v90);
    *(_QWORD *)(v90 + 72) = sub_A7A090((__int64 *)(v90 + 72), (__int64 *)v14, -1, 32);
  }
  if ( *(_BYTE *)(a1 + 1027) )
  {
    v76 = *(_QWORD *)(a1 + 624);
    v168 = 257;
    v77 = *(_QWORD *)(v76 + 24);
    v78 = sub_BCB2E0(v176);
    v164 = (_BYTE *)sub_ACD640(v78, 0, 0);
    v79 = sub_BCB2E0(v176);
    v165 = sub_ACD640(v79, a4, 0);
    v80 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v177 + 64LL);
    if ( v80 == sub_920540 )
    {
      if ( sub_BCEA30(v77) )
        goto LABEL_83;
      if ( *(_BYTE *)v76 > 0x15u )
        goto LABEL_83;
      v81 = sub_24C2610(&v164, (__int64)&v166);
      if ( v82 != v81 )
        goto LABEL_83;
      LOBYTE(v184) = 0;
      v83 = sub_AD9FD0(v77, (unsigned __int8 *)v76, (__int64 *)&v164, 2, 3u, (__int64)&v180, 0);
      if ( (_BYTE)v184 )
      {
        LOBYTE(v184) = 0;
        if ( v183 > 0x40 && v182 )
          j_j___libc_free_0_0(v182);
        if ( v181 > 0x40 && v180 )
          j_j___libc_free_0_0((unsigned __int64)v180);
      }
    }
    else
    {
      v83 = v80(v177, v77, (_BYTE *)v76, &v164, 2, 3);
    }
    if ( v83 )
    {
LABEL_59:
      v84 = *(_BYTE *)(a1 + 1036) == 0;
      v164 = (_BYTE *)v83;
      if ( v84 )
      {
        v136 = *(_QWORD *)(a1 + 72);
        v137 = *(_QWORD *)(a1 + 80);
        v184 = 257;
        v138 = sub_921880(&v169, v136, v137, (int)&v164, 1, (__int64)&v180, 0);
        v139 = (__int64 *)sub_BD5C60(v138);
        *(_QWORD *)(v138 + 72) = sub_A7A090((__int64 *)(v138 + 72), v139, -1, 32);
        v14 = v147;
      }
      else
      {
        v85 = sub_24C3400((_QWORD *)a1, a2, a5, v161);
        sub_23D0AB0((__int64)&v180, v85, 0, 0, 0);
        v86 = *(_QWORD *)(a1 + 72);
        v87 = *(_QWORD *)(a1 + 80);
        v168 = 257;
        v88 = sub_921880(&v180, v86, v87, (int)&v164, 1, (__int64)&v166, 0);
        v14 = sub_BD5C60(v88);
        *(_QWORD *)(v88 + 72) = sub_A7A090((__int64 *)(v88 + 72), (__int64 *)v14, -1, 32);
        nullsub_61();
        v189 = &unk_49DA100;
        nullsub_63();
        if ( v180 != (unsigned int *)&v182 )
          _libc_free((unsigned __int64)v180);
      }
      goto LABEL_18;
    }
LABEL_83:
    v184 = 257;
    v83 = (__int64)sub_BD2C40(88, 3u);
    if ( !v83 )
      goto LABEL_86;
    v125 = *(_QWORD *)(v76 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v125 + 8) - 17 <= 1 )
    {
LABEL_85:
      sub_B44260(v83, v125, 34, 3u, 0, 0);
      *(_QWORD *)(v83 + 72) = v77;
      *(_QWORD *)(v83 + 80) = sub_B4DC50(v77, (__int64)&v164, 2);
      sub_B4D9A0(v83, v76, (__int64 *)&v164, 2, (__int64)&v180);
LABEL_86:
      sub_B4DDE0(v83, 3);
      (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v178 + 16LL))(
        v178,
        v83,
        &v166,
        v174,
        v175);
      v126 = v169;
      v127 = &v169[4 * v170];
      if ( v169 != v127 )
      {
        do
        {
          v128 = *((_QWORD *)v126 + 1);
          v129 = *v126;
          v126 += 4;
          sub_B99FD0(v83, v129, v128);
        }
        while ( v127 != v126 );
      }
      goto LABEL_59;
    }
    v140 = *((_QWORD *)v164 + 1);
    v141 = *(unsigned __int8 *)(v140 + 8);
    if ( v141 != 17 )
    {
      if ( v141 == 18 )
      {
LABEL_105:
        v143 = 1;
LABEL_106:
        v144 = *(_DWORD *)(v140 + 32);
        BYTE4(v163) = v143;
        LODWORD(v163) = v144;
        v125 = sub_BCE1B0((__int64 *)v125, v163);
        goto LABEL_85;
      }
      v140 = *(_QWORD *)(v165 + 8);
      v142 = *(unsigned __int8 *)(v140 + 8);
      if ( v142 != 17 )
      {
        if ( v142 != 18 )
          goto LABEL_85;
        goto LABEL_105;
      }
    }
    v143 = 0;
    goto LABEL_106;
  }
LABEL_18:
  if ( !*(_BYTE *)(a1 + 1028) )
    goto LABEL_19;
  v50 = *(_QWORD *)(a1 + 464);
  v184 = 257;
  v51 = (_BYTE *)sub_AD64C0(v50, 0, 0);
  v52 = *(_QWORD *)(a1 + 464);
  v166 = v51;
  v53 = sub_AD64C0(v52, a4, 0);
  v54 = *(_QWORD *)(a1 + 632);
  v167 = v53;
  v55 = sub_921130(&v169, *(_QWORD *)(v54 + 24), v54, &v166, 2, (__int64)&v180, 0);
  v56 = *(_QWORD *)(a1 + 496);
  v168 = 257;
  v148 = v55;
  v57 = sub_AA4E30(v173);
  v58 = sub_AE5020(v57, v56);
  v184 = 257;
  v59 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A14);
  if ( v59 )
    sub_B4D190((__int64)v59, v56, v148, (__int64)&v180, 0, v58, 0, 0);
  (*(void (__fastcall **)(__int64, unsigned __int8 *, _BYTE **, __int64, __int64))(*(_QWORD *)v178 + 16LL))(
    v178,
    v59,
    &v166,
    v174,
    v175);
  v60 = v169;
  v61 = &v169[4 * v170];
  if ( v169 != v61 )
  {
    do
    {
      v62 = *((_QWORD *)v60 + 1);
      v63 = *v60;
      v60 += 4;
      sub_B99FD0((__int64)v59, v63, v62);
    }
    while ( v61 != v60 );
  }
  v64 = *(_QWORD *)(a1 + 496);
  v168 = 257;
  v65 = (unsigned __int8 *)sub_AD64C0(v64, 1, 0);
  v66 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v177 + 32LL);
  if ( v66 == sub_9201A0 )
  {
    if ( *v59 > 0x15u || *v65 > 0x15u )
    {
LABEL_79:
      v184 = 257;
      v158 = sub_B504D0(13, (__int64)v59, (__int64)v65, (__int64)&v180, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v178 + 16LL))(
        v178,
        v158,
        &v166,
        v174,
        v175);
      v67 = v158;
      v120 = &v169[4 * v170];
      if ( v169 != v120 )
      {
        v159 = a1;
        v121 = v169;
        v122 = v67;
        do
        {
          v123 = *((_QWORD *)v121 + 1);
          v124 = *v121;
          v121 += 4;
          sub_B99FD0(v122, v124, v123);
        }
        while ( v120 != v121 );
        v67 = v122;
        a1 = v159;
      }
      goto LABEL_48;
    }
    if ( (unsigned __int8)sub_AC47B0(13) )
      v67 = sub_AD5570(13, (__int64)v59, v65, 0, 0);
    else
      v67 = sub_AABE40(0xDu, v59, v65);
  }
  else
  {
    v67 = v66(v177, 13u, v59, v65, 0, 0);
  }
  if ( !v67 )
    goto LABEL_79;
LABEL_48:
  v157 = v67;
  v68 = sub_AA4E30(v173);
  v69 = sub_AE5020(v68, *(_QWORD *)(v157 + 8));
  v184 = 257;
  v70 = sub_BD2C40(80, unk_3F10A10);
  v72 = (__int64)v70;
  if ( v70 )
    sub_B4D3C0((__int64)v70, v157, v148, 0, v69, v71, 0, 0);
  v14 = v72;
  (*(void (__fastcall **)(__int64, __int64, unsigned int **, __int64, __int64))(*(_QWORD *)v178 + 16LL))(
    v178,
    v72,
    &v180,
    v174,
    v175);
  v73 = v169;
  v74 = &v169[4 * v170];
  if ( v169 != v74 )
  {
    do
    {
      v75 = *((_QWORD *)v73 + 1);
      v14 = *v73;
      v73 += 4;
      sub_B99FD0(v72, v14, v75);
    }
    while ( v74 != v73 );
  }
  sub_B9D8E0((__int64)v59, v14);
  sub_B9D8E0(v72, v14);
LABEL_19:
  if ( *(_BYTE *)(a1 + 1029) )
  {
    v21 = *(_QWORD *)(a1 + 464);
    v184 = 257;
    v22 = (_BYTE *)sub_AD64C0(v21, 0, 0);
    v23 = *(_QWORD *)(a1 + 464);
    v166 = v22;
    v24 = sub_AD64C0(v23, a4, 0);
    v25 = *(_QWORD *)(a1 + 640);
    v167 = v24;
    v26 = sub_921130(&v169, *(_QWORD *)(v25 + 24), v25, &v166, 2, (__int64)&v180, 0);
    v27 = *(_QWORD *)(a1 + 504);
    v168 = 257;
    v152 = v26;
    v28 = sub_AA4E30(v173);
    v29 = sub_AE5020(v28, v27);
    v184 = 257;
    v31 = sub_BD2C40(80, unk_3F10A14);
    if ( v31 )
    {
      sub_B4D190((__int64)v31, v27, v152, (__int64)&v180, 0, v29, 0, 0);
      v30 = v145;
    }
    v32 = (__int64)v31;
    (*(void (__fastcall **)(__int64, _QWORD *, _BYTE **, __int64, __int64, __int64))(*(_QWORD *)v178 + 16LL))(
      v178,
      v31,
      &v166,
      v174,
      v175,
      v30);
    v33 = v169;
    v34 = &v169[4 * v170];
    if ( v169 != v34 )
    {
      do
      {
        v35 = *((_QWORD *)v33 + 1);
        v32 = *v33;
        v33 += 4;
        sub_B99FD0((__int64)v31, v32, v35);
      }
      while ( v34 != v33 );
    }
    v166 = v176;
    v36 = sub_B8C340(&v166);
    v184 = 257;
    v37 = v36;
    v38 = (_BYTE *)sub_AD6530(v31[1], v32);
    v39 = sub_92B530(&v169, 0x20u, (__int64)v31, v38, (__int64)&v180);
    v40 = sub_F38250(v39, (__int64 *)(v161 + 24), 0, 0, v37, 0, 0, 0);
    sub_23D0AB0((__int64)&v180, v40, 0, 0, 0);
    v41 = sub_AD6400(*(_QWORD *)(a1 + 504));
    v42 = sub_AA4E30(v185);
    v43 = sub_AE5020(v42, *(_QWORD *)(v41 + 8));
    v168 = 257;
    v44 = sub_BD2C40(80, unk_3F10A10);
    v46 = (__int64)v44;
    if ( v44 )
      sub_B4D3C0((__int64)v44, v41, v152, 0, v43, v45, 0, 0);
    v14 = v46;
    (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v188 + 16LL))(
      v188,
      v46,
      &v166,
      v186,
      v187);
    v47 = v180;
    v48 = &v180[4 * v181];
    if ( v180 != v48 )
    {
      do
      {
        v49 = *((_QWORD *)v47 + 1);
        v14 = *v47;
        v47 += 4;
        sub_B99FD0(v46, v14, v49);
      }
      while ( v48 != v47 );
    }
    sub_B9D8E0((__int64)v31, v14);
    sub_B9D8E0(v46, v14);
    nullsub_61();
    v189 = &unk_49DA100;
    nullsub_63();
    if ( v180 != (unsigned int *)&v182 )
      _libc_free((unsigned __int64)v180);
  }
  if ( *(_BYTE *)(a1 + 1032) && a6 != 1 && v150 )
  {
    v91 = *(_QWORD *)(a1 + 480);
    HIDWORD(v166) = 0;
    v184 = 257;
    v92 = *(_QWORD *)(a2 + 40);
    v164 = (_BYTE *)sub_AD6530(v91, v14);
    v163 = sub_BCE3C0(v176, *(_DWORD *)(v92 + 316));
    v93 = sub_B33D10((__int64)&v169, 0xB2u, (__int64)&v163, 1, (int)&v164, 1, (__int64)v166, (__int64)&v180);
    v94 = *(__int64 ***)(a1 + 464);
    v184 = 257;
    v95 = sub_24C3260((__int64 *)&v169, 0x2Fu, v93, v94, (__int64)&v180, 0, (int)v166, 0);
    v168 = 257;
    v96 = *(_QWORD *)(a1 + 464);
    v156 = v95;
    v97 = *(_QWORD *)(a1 + 440);
    v98 = sub_AA4E30(v173);
    v99 = sub_AE5020(v98, v96);
    v184 = 257;
    v153 = v99;
    v100 = sub_BD2C40(80, unk_3F10A14);
    v102 = v100;
    if ( v100 )
    {
      sub_B4D190((__int64)v100, v96, v97, (__int64)&v180, 0, v153, 0, 0);
      v101 = v146;
    }
    (*(void (__fastcall **)(__int64, _BYTE *, _BYTE **, __int64, __int64, __int64))(*(_QWORD *)v178 + 16LL))(
      v178,
      v102,
      &v166,
      v174,
      v175,
      v101);
    v103 = &v169[4 * v170];
    for ( i = v169; v103 != i; i += 4 )
    {
      v105 = *((_QWORD *)i + 1);
      v106 = *i;
      sub_B99FD0((__int64)v102, v106, v105);
    }
    v184 = 257;
    v107 = sub_92B530(&v169, 0x24u, v156, v102, (__int64)&v180);
    v180 = (unsigned int *)v176;
    v108 = sub_B8C340(&v180);
    v109 = sub_F38250(v107, (__int64 *)(v161 + 24), 0, 0, v108, 0, 0, 0);
    sub_23D0AB0((__int64)&v180, v109, 0, 0, 0);
    v110 = *(_QWORD *)(a1 + 440);
    v111 = sub_AA4E30(v185);
    v112 = sub_AE5020(v111, *(_QWORD *)(v156 + 8));
    v168 = 257;
    v113 = sub_BD2C40(80, unk_3F10A10);
    v115 = (__int64)v113;
    if ( v113 )
      sub_B4D3C0((__int64)v113, v156, v110, 0, v112, v114, 0, 0);
    v116 = v115;
    (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v188 + 16LL))(
      v188,
      v115,
      &v166,
      v186,
      v187);
    v117 = v180;
    v118 = &v180[4 * v181];
    if ( v180 != v118 )
    {
      do
      {
        v119 = *((_QWORD *)v117 + 1);
        v116 = *v117;
        v117 += 4;
        sub_B99FD0(v115, v116, v119);
      }
      while ( v118 != v117 );
    }
    sub_B9D8E0((__int64)v102, v116);
    sub_B9D8E0(v115, v116);
    nullsub_61();
    v189 = &unk_49DA100;
    nullsub_63();
    if ( v180 != (unsigned int *)&v182 )
      _libc_free((unsigned __int64)v180);
  }
  nullsub_61();
  v179 = &unk_49DA100;
  nullsub_63();
  if ( v169 != (unsigned int *)v172 )
    _libc_free((unsigned __int64)v169);
  if ( v162 )
    sub_B91220((__int64)&v162, (__int64)v162);
}
