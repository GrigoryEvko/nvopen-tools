// Function: sub_23CF870
// Address: 0x23cf870
//
void __fastcall sub_23CF870(__int64 *a1, __int64 a2, __int64 a3, unsigned __int64 a4, char a5)
{
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 v7; // rax
  char *v8; // rsi
  __int64 v9; // r8
  __int64 v10; // r9
  char *v11; // r13
  unsigned int *v12; // rax
  int v13; // ecx
  unsigned int *v14; // rdx
  __int64 v15; // r15
  const char *v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rbx
  __int64 v24; // rdx
  unsigned __int64 v25; // r8
  __int64 v26; // r13
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // r12
  unsigned int *v35; // r13
  unsigned int *v36; // rbx
  __int64 v37; // rdx
  unsigned int v38; // esi
  unsigned __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 v43; // r14
  __int64 v44; // rax
  char v45; // si
  _QWORD *v46; // rax
  unsigned __int64 v47; // r13
  unsigned int *v48; // r14
  unsigned int *v49; // rbx
  __int64 v50; // rdx
  unsigned int v51; // esi
  __int64 (__fastcall *v52)(__int64, unsigned int, _BYTE *, __int64); // rax
  unsigned __int8 *v53; // r14
  __int64 v54; // rax
  unsigned __int8 *v55; // r13
  __int64 (__fastcall *v56)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v57; // rbx
  __int64 v58; // r14
  _BYTE *v59; // rax
  __int64 v60; // rax
  __int64 v61; // r12
  _QWORD *v62; // rax
  __int64 v63; // r9
  __int64 v64; // r13
  unsigned int *v65; // r14
  unsigned int *v66; // r12
  __int64 v67; // rdx
  unsigned int v68; // esi
  __int64 v69; // r12
  int v70; // eax
  int v71; // eax
  unsigned int v72; // edx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // r8
  unsigned __int64 v77; // rsi
  _QWORD *v78; // r9
  unsigned __int64 v79; // r13
  __int64 v80; // rax
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rcx
  __int64 v83; // rdx
  unsigned __int64 v84; // r12
  __int64 v85; // rax
  unsigned __int64 v86; // r8
  char *v87; // rdx
  __int64 v88; // r8
  unsigned __int64 v89; // rbx
  __int64 v90; // rax
  char *v91; // rdx
  unsigned __int64 v92; // rsi
  unsigned int *v93; // r14
  unsigned int *v94; // r13
  __int64 v95; // rdx
  unsigned int v96; // esi
  _QWORD *v97; // rax
  __int64 v98; // r13
  unsigned int *v99; // r14
  unsigned int *v100; // r12
  __int64 v101; // rdx
  unsigned int v102; // esi
  unsigned __int8 *v103; // rax
  unsigned int *v104; // r13
  unsigned int *v105; // rbx
  __int64 v106; // rdx
  unsigned int v107; // esi
  unsigned int *v108; // r14
  unsigned int *v109; // r13
  __int64 v110; // rdx
  unsigned int v111; // esi
  unsigned __int64 v112; // rbx
  __int64 *v113; // rdx
  __int64 v114; // rbx
  __int64 v115; // rax
  char *v116; // rax
  __int64 v117; // rdi
  unsigned __int64 v118; // rsi
  __int64 v122; // [rsp+70h] [rbp-270h]
  __int64 v123; // [rsp+78h] [rbp-268h]
  __int64 v124; // [rsp+80h] [rbp-260h]
  __int64 v125; // [rsp+90h] [rbp-250h]
  __int64 v127; // [rsp+98h] [rbp-248h]
  __int64 **v128; // [rsp+A0h] [rbp-240h]
  unsigned __int64 v129; // [rsp+A0h] [rbp-240h]
  __int64 v130; // [rsp+A0h] [rbp-240h]
  __int64 v131; // [rsp+A8h] [rbp-238h]
  unsigned __int64 v132; // [rsp+A8h] [rbp-238h]
  _BYTE v134[32]; // [rsp+C0h] [rbp-220h] BYREF
  __int16 v135; // [rsp+E0h] [rbp-200h]
  _BYTE v136[32]; // [rsp+F0h] [rbp-1F0h] BYREF
  __int16 v137; // [rsp+110h] [rbp-1D0h]
  _QWORD v138[4]; // [rsp+120h] [rbp-1C0h] BYREF
  __int16 v139; // [rsp+140h] [rbp-1A0h]
  _BYTE *v140; // [rsp+150h] [rbp-190h] BYREF
  __int64 i; // [rsp+158h] [rbp-188h]
  _BYTE v142[48]; // [rsp+160h] [rbp-180h] BYREF
  unsigned int *v143; // [rsp+190h] [rbp-150h] BYREF
  __int64 v144; // [rsp+198h] [rbp-148h]
  _BYTE v145[32]; // [rsp+1A0h] [rbp-140h] BYREF
  __int64 v146; // [rsp+1C0h] [rbp-120h]
  __int64 v147; // [rsp+1C8h] [rbp-118h]
  __int64 v148; // [rsp+1D0h] [rbp-110h]
  _QWORD *v149; // [rsp+1D8h] [rbp-108h]
  void **v150; // [rsp+1E0h] [rbp-100h]
  void **v151; // [rsp+1E8h] [rbp-F8h]
  __int64 v152; // [rsp+1F0h] [rbp-F0h]
  int v153; // [rsp+1F8h] [rbp-E8h]
  __int16 v154; // [rsp+1FCh] [rbp-E4h]
  char v155; // [rsp+1FEh] [rbp-E2h]
  __int64 v156; // [rsp+200h] [rbp-E0h]
  __int64 v157; // [rsp+208h] [rbp-D8h]
  void *v158; // [rsp+210h] [rbp-D0h] BYREF
  void *v159; // [rsp+218h] [rbp-C8h] BYREF
  char *v160; // [rsp+220h] [rbp-C0h] BYREF
  __int64 v161; // [rsp+228h] [rbp-B8h]
  _QWORD v162[2]; // [rsp+230h] [rbp-B0h] BYREF
  __int16 v163; // [rsp+240h] [rbp-A0h]

  v5 = (_QWORD *)sub_BD5C60(*a1);
  v154 = 512;
  v6 = (__int64)v5;
  v152 = 0;
  v143 = (unsigned int *)v145;
  v144 = 0x200000000LL;
  v150 = &v158;
  v151 = &v159;
  LOWORD(v148) = 0;
  v149 = v5;
  v153 = 0;
  v158 = &unk_49DA100;
  v155 = 7;
  v156 = 0;
  v159 = &unk_49DA0B0;
  v7 = *a1;
  v157 = 0;
  v146 = 0;
  v147 = 0;
  v8 = *(char **)(v7 + 48);
  v160 = v8;
  if ( !v8 || (sub_B96E90((__int64)&v160, (__int64)v8, 1), (v11 = v160) == 0) )
  {
    sub_93FB40((__int64)&v143, 0);
    v11 = v160;
    goto LABEL_120;
  }
  v12 = v143;
  v13 = v144;
  v14 = &v143[4 * (unsigned int)v144];
  if ( v143 == v14 )
  {
LABEL_116:
    if ( (unsigned int)v144 >= (unsigned __int64)HIDWORD(v144) )
    {
      v118 = (unsigned int)v144 + 1LL;
      if ( HIDWORD(v144) < v118 )
      {
        sub_C8D5F0((__int64)&v143, v145, v118, 0x10u, v9, v10);
        v14 = &v143[4 * (unsigned int)v144];
      }
      *(_QWORD *)v14 = 0;
      *((_QWORD *)v14 + 1) = v11;
      v11 = v160;
      LODWORD(v144) = v144 + 1;
    }
    else
    {
      if ( v14 )
      {
        *v14 = 0;
        *((_QWORD *)v14 + 1) = v11;
        v13 = v144;
        v11 = v160;
      }
      LODWORD(v144) = v13 + 1;
    }
LABEL_120:
    if ( !v11 )
      goto LABEL_9;
    goto LABEL_8;
  }
  while ( *v12 )
  {
    v12 += 4;
    if ( v14 == v12 )
      goto LABEL_116;
  }
  *((_QWORD *)v12 + 1) = v160;
LABEL_8:
  sub_B91220((__int64)&v160, (__int64)v11);
LABEL_9:
  v15 = *(_QWORD *)(*a1 + 40);
  v124 = v15;
  v16 = sub_BD5D20(v15);
  v17 = a1[2];
  v160 = (char *)v16;
  v162[0] = ".tail";
  v18 = (__int64 *)(*a1 + 24);
  v161 = v19;
  v163 = 773;
  v138[0] = 0;
  v122 = sub_F36990(v15, v18, 0, v17, 0, 0, (void **)&v160, 0);
  v140 = v142;
  for ( i = 0x600000000LL; a4 > v138[0]; ++v138[0] )
  {
    v20 = *(_QWORD *)(v15 + 72);
    v160 = "sub_";
    v163 = 2819;
    v162[0] = v138;
    v21 = sub_22077B0(0x50u);
    v23 = v21;
    if ( v21 )
      sub_AA4D50(v21, v6, (__int64)&v160, v20, v122);
    v24 = (unsigned int)i;
    v25 = (unsigned int)i + 1LL;
    if ( v25 > HIDWORD(i) )
    {
      sub_C8D5F0((__int64)&v140, v142, (unsigned int)i + 1LL, 8u, v25, v22);
      v24 = (unsigned int)i;
    }
    *(_QWORD *)&v140[8 * v24] = v23;
    LODWORD(i) = i + 1;
  }
  v26 = *(_QWORD *)(v15 + 72);
  v160 = "ne";
  v163 = 259;
  v123 = sub_22077B0(0x50u);
  if ( v123 )
    sub_AA4D50(v123, v6, (__int64)&v160, v26, v122);
  v27 = *(_QWORD *)(v15 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v27 == v15 + 48 )
    goto LABEL_127;
  if ( !v27 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v27 - 24) - 30 > 0xA )
LABEL_127:
    BUG();
  v28 = *(_QWORD *)v140;
  if ( *(_QWORD *)(v27 - 56) )
  {
    v29 = *(_QWORD *)(v27 - 48);
    **(_QWORD **)(v27 - 40) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = *(_QWORD *)(v27 - 40);
  }
  *(_QWORD *)(v27 - 56) = v28;
  if ( v28 )
  {
    v30 = *(_QWORD *)(v28 + 16);
    *(_QWORD *)(v27 - 48) = v30;
    if ( v30 )
      *(_QWORD *)(v30 + 16) = v27 - 48;
    *(_QWORD *)(v27 - 40) = v28 + 16;
    *(_QWORD *)(v28 + 16) = v27 - 56;
  }
  v163 = 257;
  v146 = v123;
  v147 = v123 + 48;
  LOWORD(v148) = 0;
  v31 = sub_D5C860((__int64 *)&v143, *(_QWORD *)(*a1 + 8), a4, (__int64)&v160);
  v163 = 257;
  v32 = v31;
  v33 = sub_BD2C40(72, 1u);
  v34 = (__int64)v33;
  if ( v33 )
    sub_B4C8F0((__int64)v33, v122, 1u, 0, 0);
  (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v151 + 2))(v151, v34, &v160, v147, v148);
  v35 = v143;
  v36 = &v143[4 * (unsigned int)v144];
  if ( v143 != v36 )
  {
    do
    {
      v37 = *((_QWORD *)v35 + 1);
      v38 = *v35;
      v35 += 4;
      sub_B99FD0(v34, v38, v37);
    }
    while ( v36 != v35 );
  }
  if ( a4 )
  {
    v39 = 0;
    while ( 1 )
    {
      v125 = 8 * v39;
      v146 = *(_QWORD *)&v140[8 * v39];
      v147 = v146 + 48;
      LOWORD(v148) = 0;
      v139 = 257;
      v128 = *(__int64 ***)(*a1 + 8);
      v137 = 257;
      v135 = 257;
      v40 = sub_BCB2E0(v149);
      v160 = (char *)sub_ACD640(v40, v39, 0);
      v41 = sub_BCB2B0(v149);
      v42 = sub_921130(&v143, v41, a2, &v160, 1, (__int64)v134, 3u);
      v43 = sub_BCB2B0(v149);
      v44 = sub_AA4E30(v146);
      v45 = sub_AE5020(v44, v43);
      v163 = 257;
      v46 = sub_BD2C40(80, unk_3F10A14);
      v47 = (unsigned __int64)v46;
      if ( v46 )
        sub_B4D190((__int64)v46, v43, v42, (__int64)&v160, 0, v45, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int64, _BYTE *, __int64, __int64))*v151 + 2))(
        v151,
        v47,
        v136,
        v147,
        v148);
      v48 = v143;
      v49 = &v143[4 * (unsigned int)v144];
      if ( v143 != v49 )
      {
        do
        {
          v50 = *((_QWORD *)v48 + 1);
          v51 = *v48;
          v48 += 4;
          sub_B99FD0(v47, v51, v50);
        }
        while ( v49 != v48 );
      }
      if ( v128 == *(__int64 ***)(v47 + 8) )
      {
        v53 = (unsigned __int8 *)v47;
        goto LABEL_43;
      }
      v52 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v150 + 15);
      if ( v52 != sub_920130 )
        break;
      if ( *(_BYTE *)v47 <= 0x15u )
      {
        if ( (unsigned __int8)sub_AC4810(0x27u) )
          v53 = (unsigned __int8 *)sub_ADAB70(39, v47, v128, 0);
        else
          v53 = (unsigned __int8 *)sub_AA93C0(0x27u, v47, (__int64)v128);
LABEL_42:
        if ( v53 )
          goto LABEL_43;
      }
      v163 = 257;
      v103 = (unsigned __int8 *)sub_BD2C40(72, unk_3F10A14);
      v53 = v103;
      if ( v103 )
        sub_B515B0((__int64)v103, v47, (__int64)v128, (__int64)&v160, 0, 0);
      (*((void (__fastcall **)(void **, unsigned __int8 *, _QWORD *, __int64, __int64))*v151 + 2))(
        v151,
        v53,
        v138,
        v147,
        v148);
      v104 = v143;
      v105 = &v143[4 * (unsigned int)v144];
      if ( v143 != v105 )
      {
        do
        {
          v106 = *((_QWORD *)v104 + 1);
          v107 = *v104;
          v104 += 4;
          sub_B99FD0((__int64)v53, v107, v106);
        }
        while ( v105 != v104 );
      }
LABEL_43:
      v54 = sub_AD64C0(*(_QWORD *)(*a1 + 8), *(unsigned __int8 *)(a3 + v39), 0);
      v139 = 257;
      v55 = (unsigned __int8 *)v54;
      v56 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v150 + 4);
      if ( a5 )
      {
        if ( v56 == sub_9201A0 )
        {
          if ( *v55 > 0x15u || *v53 > 0x15u )
          {
LABEL_96:
            v163 = 257;
            v57 = sub_B504D0(15, (__int64)v55, (__int64)v53, (__int64)&v160, 0, 0);
            (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v151 + 2))(
              v151,
              v57,
              v138,
              v147,
              v148);
            v108 = v143;
            v109 = &v143[4 * (unsigned int)v144];
            if ( v143 != v109 )
            {
              do
              {
                v110 = *((_QWORD *)v108 + 1);
                v111 = *v108;
                v108 += 4;
                sub_B99FD0(v57, v111, v110);
              }
              while ( v109 != v108 );
            }
            goto LABEL_50;
          }
          if ( (unsigned __int8)sub_AC47B0(15) )
            v57 = sub_AD5570(15, (__int64)v55, v53, 0, 0);
          else
            v57 = sub_AABE40(0xFu, v55, v53);
        }
        else
        {
          v57 = v56((__int64)v150, 15u, v55, v53, 0, 0);
        }
        if ( !v57 )
          goto LABEL_96;
      }
      else
      {
        if ( v56 == sub_9201A0 )
        {
          if ( *v53 > 0x15u || *v55 > 0x15u )
          {
LABEL_82:
            v163 = 257;
            v57 = sub_B504D0(15, (__int64)v53, (__int64)v55, (__int64)&v160, 0, 0);
            (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v151 + 2))(
              v151,
              v57,
              v138,
              v147,
              v148);
            v93 = v143;
            v94 = &v143[4 * (unsigned int)v144];
            if ( v143 != v94 )
            {
              do
              {
                v95 = *((_QWORD *)v93 + 1);
                v96 = *v93;
                v93 += 4;
                sub_B99FD0(v57, v96, v95);
              }
              while ( v94 != v93 );
            }
            goto LABEL_50;
          }
          if ( (unsigned __int8)sub_AC47B0(15) )
            v57 = sub_AD5570(15, (__int64)v53, v55, 0, 0);
          else
            v57 = sub_AABE40(0xFu, v53, v55);
        }
        else
        {
          v57 = v56((__int64)v150, 15u, v53, v55, 0, 0);
        }
        if ( !v57 )
          goto LABEL_82;
      }
LABEL_50:
      v129 = v39 + 1;
      if ( a4 - 1 <= v39 )
      {
        v163 = 257;
        v97 = sub_BD2C40(72, 1u);
        v98 = (__int64)v97;
        if ( v97 )
          sub_B4C8F0((__int64)v97, v123, 1u, 0, 0);
        (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v151 + 2))(v151, v98, &v160, v147, v148);
        v99 = v143;
        v100 = &v143[4 * (unsigned int)v144];
        if ( v143 != v100 )
        {
          do
          {
            v101 = *((_QWORD *)v99 + 1);
            v102 = *v99;
            v99 += 4;
            sub_B99FD0(v98, v102, v101);
          }
          while ( v100 != v99 );
        }
      }
      else
      {
        v58 = *(_QWORD *)&v140[v125 + 8];
        v139 = 257;
        v59 = (_BYTE *)sub_AD64C0(*(_QWORD *)(*a1 + 8), 0, 0);
        v60 = sub_92B530(&v143, 0x21u, v57, v59, (__int64)v138);
        v163 = 257;
        v61 = v60;
        v62 = sub_BD2C40(72, 3u);
        v64 = (__int64)v62;
        if ( v62 )
          sub_B4C9A0((__int64)v62, v123, v58, v61, 3u, v63, 0, 0);
        (*((void (__fastcall **)(void **, __int64, char **, __int64, __int64))*v151 + 2))(v151, v64, &v160, v147, v148);
        v65 = v143;
        v66 = &v143[4 * (unsigned int)v144];
        if ( v143 != v66 )
        {
          do
          {
            v67 = *((_QWORD *)v65 + 1);
            v68 = *v65;
            v65 += 4;
            sub_B99FD0(v64, v68, v67);
          }
          while ( v66 != v65 );
        }
      }
      v69 = *(_QWORD *)&v140[v125];
      v70 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
      if ( v70 == *(_DWORD *)(v32 + 72) )
      {
        sub_B48D90(v32);
        v70 = *(_DWORD *)(v32 + 4) & 0x7FFFFFF;
      }
      v71 = (v70 + 1) & 0x7FFFFFF;
      v72 = v71 | *(_DWORD *)(v32 + 4) & 0xF8000000;
      v73 = *(_QWORD *)(v32 - 8) + 32LL * (unsigned int)(v71 - 1);
      *(_DWORD *)(v32 + 4) = v72;
      if ( *(_QWORD *)v73 )
      {
        v74 = *(_QWORD *)(v73 + 8);
        **(_QWORD **)(v73 + 16) = v74;
        if ( v74 )
          *(_QWORD *)(v74 + 16) = *(_QWORD *)(v73 + 16);
      }
      *(_QWORD *)v73 = v57;
      if ( v57 )
      {
        v75 = *(_QWORD *)(v57 + 16);
        *(_QWORD *)(v73 + 8) = v75;
        if ( v75 )
          *(_QWORD *)(v75 + 16) = v73 + 8;
        *(_QWORD *)(v73 + 16) = v57 + 16;
        *(_QWORD *)(v57 + 16) = v73;
      }
      *(_QWORD *)(*(_QWORD *)(v32 - 8)
                + 32LL * *(unsigned int *)(v32 + 72)
                + 8LL * ((*(_DWORD *)(v32 + 4) & 0x7FFFFFFu) - 1)) = v69;
      v39 = v129;
      if ( a4 == v129 )
        goto LABEL_65;
    }
    v53 = (unsigned __int8 *)v52((__int64)v150, 39u, (_BYTE *)v47, (__int64)v128);
    goto LABEL_42;
  }
LABEL_65:
  sub_BD84D0(*a1, v32);
  sub_B43D60((_QWORD *)*a1);
  if ( !a1[2] )
    goto LABEL_107;
  v77 = (unsigned __int64)v140;
  v78 = v162;
  v161 = 0x800000000LL;
  v79 = a4;
  v160 = (char *)v162;
  v80 = *(_QWORD *)v140;
  v162[0] = v124;
  LODWORD(v161) = 1;
  v162[1] = v80 & 0xFFFFFFFFFFFFFFFBLL;
  if ( !a4 )
  {
    v83 = 1;
    v112 = v122 & 0xFFFFFFFFFFFFFFFBLL;
    goto LABEL_103;
  }
  v81 = 0;
  v82 = 8;
  v83 = 1;
  v84 = a4 - 1;
  while ( 1 )
  {
    v88 = 8 * v81;
    v89 = v81 + 1;
    if ( v84 <= v81 )
    {
      v85 = *(_QWORD *)(v77 + 8 * v81);
      v86 = v83 + 1;
      if ( v83 + 1 <= v82 )
        goto LABEL_69;
LABEL_75:
      v131 = v85;
      sub_C8D5F0((__int64)&v160, v162, v86, 0x10u, v86, (__int64)v78);
      v83 = (unsigned int)v161;
      v85 = v131;
      goto LABEL_69;
    }
    v90 = *(_QWORD *)(v77 + 8 * v81);
    v78 = (_QWORD *)(*(_QWORD *)(v77 + v88 + 8) & 0xFFFFFFFFFFFFFFFBLL);
    if ( v83 + 1 > v82 )
    {
      v127 = v90;
      v130 = v88;
      v132 = *(_QWORD *)(v77 + v88 + 8) & 0xFFFFFFFFFFFFFFFBLL;
      sub_C8D5F0((__int64)&v160, v162, v83 + 1, 0x10u, v88, (__int64)v78);
      v83 = (unsigned int)v161;
      v90 = v127;
      v88 = v130;
      v78 = (_QWORD *)v132;
    }
    v91 = &v160[16 * v83];
    *(_QWORD *)v91 = v90;
    v92 = (unsigned __int64)v140;
    *((_QWORD *)v91 + 1) = v78;
    LODWORD(v161) = v161 + 1;
    v83 = (unsigned int)v161;
    v85 = *(_QWORD *)(v92 + v88);
    v86 = (unsigned int)v161 + 1LL;
    if ( v86 > HIDWORD(v161) )
      goto LABEL_75;
LABEL_69:
    v87 = &v160[16 * v83];
    *(_QWORD *)v87 = v85;
    *((_QWORD *)v87 + 1) = v123 & 0xFFFFFFFFFFFFFFFBLL;
    v83 = (unsigned int)(v161 + 1);
    LODWORD(v161) = v161 + 1;
    if ( v79 == v89 )
      break;
    v77 = (unsigned __int64)v140;
    v82 = HIDWORD(v161);
    v81 = v89;
  }
  v76 = v83 + 1;
  v112 = v122 & 0xFFFFFFFFFFFFFFFBLL;
  if ( v83 + 1 > (unsigned __int64)HIDWORD(v161) )
  {
    sub_C8D5F0((__int64)&v160, v162, v83 + 1, 0x10u, v76, (__int64)v162);
    v83 = (unsigned int)v161;
  }
LABEL_103:
  v113 = (__int64 *)&v160[16 * v83];
  v113[1] = v112;
  v114 = v112 | 4;
  *v113 = v123;
  LODWORD(v161) = v161 + 1;
  v115 = (unsigned int)v161;
  if ( (unsigned __int64)(unsigned int)v161 + 1 > HIDWORD(v161) )
  {
    sub_C8D5F0((__int64)&v160, v162, (unsigned int)v161 + 1LL, 0x10u, v76, (__int64)v162);
    v115 = (unsigned int)v161;
  }
  v116 = &v160[16 * v115];
  *((_QWORD *)v116 + 1) = v114;
  *(_QWORD *)v116 = v124;
  v117 = a1[2];
  LODWORD(v161) = v161 + 1;
  sub_FFB3D0(v117, (unsigned __int64 *)v160, (unsigned int)v161, (__int64)a1, v76, (__int64)v162);
  if ( v160 != (char *)v162 )
    _libc_free((unsigned __int64)v160);
LABEL_107:
  if ( v140 != v142 )
    _libc_free((unsigned __int64)v140);
  nullsub_61();
  v158 = &unk_49DA100;
  nullsub_63();
  if ( v143 != (unsigned int *)v145 )
    _libc_free((unsigned __int64)v143);
}
