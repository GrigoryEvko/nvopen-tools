// Function: sub_241A060
// Address: 0x241a060
//
void __fastcall sub_241A060(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  char v11; // bl
  bool v12; // al
  __int64 *v13; // rdi
  __int64 v14; // rsi
  unsigned int *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // r8
  unsigned int *v18; // r15
  unsigned __int64 v19; // rax
  __int64 v20; // r9
  unsigned __int64 v21; // rax
  char v22; // dl
  __int64 v23; // r9
  unsigned __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // r13
  _QWORD *v28; // rax
  __int64 v29; // r9
  __int64 v30; // r12
  unsigned int *v31; // r13
  unsigned int *v32; // rbx
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // rsi
  __int64 v36; // rdi
  __int64 **v37; // rax
  unsigned int v38; // ebx
  unsigned int *v40; // rax
  __int64 v41; // rbx
  __int64 *v42; // rsi
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned __int8 v45; // bl
  __int64 v46; // r15
  __int64 v47; // rsi
  __int64 v48; // r9
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r12
  _BYTE *v52; // rbx
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int8 *v55; // r15
  __int64 (__fastcall *v56)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v57; // rax
  __int64 v58; // r9
  __int64 v59; // r14
  unsigned int *v60; // r15
  unsigned int *v61; // rbx
  __int64 v62; // rdx
  unsigned int v63; // esi
  __int64 v64; // rax
  __int64 v65; // r15
  _QWORD *v66; // rax
  __int64 v67; // r9
  __int64 v68; // r12
  unsigned int *v69; // r15
  unsigned int *v70; // rbx
  __int64 v71; // rdx
  unsigned int v72; // esi
  _QWORD *v73; // rdx
  __int64 v74; // rax
  unsigned int v75; // ecx
  unsigned int **v76; // r15
  unsigned int *v77; // rdi
  __int64 v78; // rsi
  bool v79; // al
  unsigned int v80; // edx
  _QWORD *v81; // rcx
  unsigned int v82; // ebx
  unsigned int **v83; // rax
  unsigned int *v84; // r8
  __int64 v85; // rax
  __int64 v86; // rdi
  _BYTE *v87; // rax
  __int64 v88; // rax
  unsigned __int64 v89; // rax
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  __int64 v95; // rdx
  __int64 v96; // rcx
  __int64 v97; // r8
  __int64 v98; // r9
  _QWORD *v99; // r12
  _QWORD *v100; // r15
  void (__fastcall *v101)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v102; // rax
  unsigned int v103; // r12d
  int v104; // eax
  int v105; // edi
  __int64 v106; // [rsp-10h] [rbp-5A0h]
  __int64 v107; // [rsp-8h] [rbp-598h]
  __int64 v108; // [rsp+10h] [rbp-580h]
  __int64 v109; // [rsp+18h] [rbp-578h]
  unsigned int *v110; // [rsp+20h] [rbp-570h]
  unsigned __int64 v111; // [rsp+28h] [rbp-568h]
  __int64 v112; // [rsp+30h] [rbp-560h]
  __int64 v113; // [rsp+38h] [rbp-558h]
  unsigned __int64 v114; // [rsp+40h] [rbp-550h]
  unsigned __int64 v116; // [rsp+50h] [rbp-540h]
  __int64 **v117; // [rsp+50h] [rbp-540h]
  __int64 v119; // [rsp+70h] [rbp-520h]
  __int64 v120; // [rsp+78h] [rbp-518h]
  unsigned int v121; // [rsp+80h] [rbp-510h]
  char v122; // [rsp+86h] [rbp-50Ah]
  char v123; // [rsp+87h] [rbp-509h]
  _QWORD **v124; // [rsp+88h] [rbp-508h]
  unsigned __int16 v125; // [rsp+90h] [rbp-500h]
  __int64 v126; // [rsp+90h] [rbp-500h]
  __int64 v127; // [rsp+90h] [rbp-500h]
  __int64 v128; // [rsp+A8h] [rbp-4E8h] BYREF
  unsigned __int64 v129; // [rsp+B0h] [rbp-4E0h] BYREF
  _BYTE *v130; // [rsp+B8h] [rbp-4D8h]
  __int64 v131; // [rsp+C0h] [rbp-4D0h]
  __int64 v132[4]; // [rsp+D0h] [rbp-4C0h] BYREF
  unsigned int *v133; // [rsp+F0h] [rbp-4A0h] BYREF
  int v134; // [rsp+F8h] [rbp-498h]
  char v135; // [rsp+100h] [rbp-490h] BYREF
  __int64 v136; // [rsp+128h] [rbp-468h]
  __int64 v137; // [rsp+130h] [rbp-460h]
  __int64 v138; // [rsp+140h] [rbp-450h]
  __int64 v139; // [rsp+148h] [rbp-448h]
  void *v140; // [rsp+170h] [rbp-420h]
  unsigned int *v141[2]; // [rsp+180h] [rbp-410h] BYREF
  char v142; // [rsp+190h] [rbp-400h] BYREF
  __int64 v143; // [rsp+1B8h] [rbp-3D8h]
  void *v144; // [rsp+200h] [rbp-390h]
  unsigned int *v145[2]; // [rsp+210h] [rbp-380h] BYREF
  _QWORD v146[2]; // [rsp+220h] [rbp-370h] BYREF
  __int16 v147; // [rsp+230h] [rbp-360h]
  void *v148; // [rsp+290h] [rbp-300h]
  const char *v149; // [rsp+2A0h] [rbp-2F0h] BYREF
  __int64 v150; // [rsp+2A8h] [rbp-2E8h]
  _BYTE v151[16]; // [rsp+2B0h] [rbp-2E0h] BYREF
  __int16 v152; // [rsp+2C0h] [rbp-2D0h]
  __int64 v153; // [rsp+4B0h] [rbp-E0h]
  __int64 v154; // [rsp+4B8h] [rbp-D8h]
  _QWORD **v155; // [rsp+4C0h] [rbp-D0h]
  __int64 v156; // [rsp+4C8h] [rbp-C8h]
  char v157; // [rsp+4D0h] [rbp-C0h]
  __int64 v158; // [rsp+4D8h] [rbp-B8h]
  char *v159; // [rsp+4E0h] [rbp-B0h]
  __int64 v160; // [rsp+4E8h] [rbp-A8h]
  int v161; // [rsp+4F0h] [rbp-A0h]
  char v162; // [rsp+4F4h] [rbp-9Ch]
  char v163; // [rsp+4F8h] [rbp-98h] BYREF
  __int16 v164; // [rsp+538h] [rbp-58h]
  _QWORD *v165; // [rsp+540h] [rbp-50h]
  _QWORD *v166; // [rsp+548h] [rbp-48h]
  __int64 v167; // [rsp+550h] [rbp-40h]

  v3 = sub_B43CC0(a2);
  v4 = *(_QWORD *)(a2 - 64);
  v5 = *(_QWORD *)(v4 + 8);
  v6 = sub_9208B0(v3, v5);
  v150 = v7;
  v149 = (const char *)((unsigned __int64)(v6 + 7) >> 3);
  v114 = sub_CA1930(&v149);
  if ( !v114 )
    return;
  if ( !sub_B46500((unsigned __int8 *)a2) )
  {
    v11 = sub_240D530(a2, v5, v8, v9, v10);
    if ( !v11 )
      goto LABEL_5;
    goto LABEL_4;
  }
  switch ( (*(_WORD *)(a2 + 2) >> 7) & 7 )
  {
    case 0:
      v43 = 0;
      break;
    case 1:
    case 2:
    case 5:
      v43 = 640;
      break;
    case 3:
      BUG();
    case 4:
    case 6:
      v43 = 768;
      break;
    case 7:
      v43 = 896;
      break;
  }
  *(_WORD *)(a2 + 2) = v43 | *(_WORD *)(a2 + 2) & 0xFC7F;
  v11 = sub_240D530(a2, v5, v43, a2, v10);
  if ( v11 )
LABEL_4:
    v11 = !sub_B46500((unsigned __int8 *)a2);
LABEL_5:
  v129 = 0;
  v130 = 0;
  v131 = 0;
  memset(v132, 0, 24);
  v12 = sub_B46500((unsigned __int8 *)a2);
  v13 = (__int64 *)*a1;
  if ( !v12 )
  {
    v14 = sub_24159D0((__int64)v13, v4);
    goto LABEL_7;
  }
  v35 = *(_QWORD *)(v4 + 8);
  v36 = *v13;
  if ( (unsigned __int8)(*(_BYTE *)(v35 + 8) - 15) > 1u )
  {
    v14 = *(_QWORD *)(v36 + 72);
LABEL_7:
    v128 = v14;
    if ( !v11 )
      goto LABEL_8;
    goto LABEL_37;
  }
  v37 = (__int64 **)sub_240F000(v36, v35);
  v14 = sub_AC9350(v37);
  v128 = v14;
  if ( !v11 )
  {
LABEL_8:
    if ( (_BYTE)qword_4FE37A8 )
    {
      v145[0] = (unsigned int *)sub_24159D0(*a1, *(_QWORD *)(a2 - 32));
      v15 = v145[0];
LABEL_10:
      v16 = *a1;
      v113 = a2 + 24;
      v120 = sub_2416F70(*a1, v128, (unsigned __int64)v15, a2 + 24, 0);
      goto LABEL_11;
    }
    goto LABEL_47;
  }
LABEL_37:
  sub_9281F0((__int64)&v129, v130, &v128);
  v149 = (const char *)sub_2414930((__int64 *)*a1, (_BYTE *)v4);
  sub_240DEA0((__int64)v132, &v149);
  if ( (_BYTE)qword_4FE37A8 )
  {
    v145[0] = (unsigned int *)sub_24159D0(*a1, *(_QWORD *)(a2 - 32));
    sub_24141E0((__int64)&v129, (char **)v145);
    v149 = (const char *)sub_2414930((__int64 *)*a1, *(_BYTE **)(a2 - 32));
    sub_240DEA0((__int64)v132, &v149);
    v15 = v145[0];
    goto LABEL_10;
  }
  v14 = v128;
LABEL_47:
  v16 = *a1;
  v113 = a2 + 24;
  v120 = sub_2415280(*a1, v14, a2 + 24, 0);
LABEL_11:
  v109 = 0;
  if ( v11 )
  {
    v16 = *a1;
    v109 = sub_2415600(*a1, &v129, v132, v113, 0, 0);
    v11 = v109 != 0;
  }
  v18 = *(unsigned int **)(a2 - 32);
  v124 = (_QWORD **)*a1;
  v110 = v18;
  v125 = *(_WORD *)(a2 + 2);
  _BitScanReverse64(&v19, 1LL << (v125 >> 1));
  v121 = 63 - (v19 ^ 0x3F);
  v122 = v11 & sub_240D530(v16, a2, v121, v125 >> 1, v17);
  if ( *(_BYTE *)v18 == 60 )
  {
    v73 = v124[31];
    v74 = *((unsigned int *)v124 + 66);
    if ( (_DWORD)v74 )
    {
      v75 = (v74 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v76 = (unsigned int **)&v73[2 * v75];
      v77 = *v76;
      if ( v110 == *v76 )
      {
LABEL_87:
        if ( v76 != &v73[2 * v74] )
        {
          sub_2412230((__int64)&v149, *(_QWORD *)(a2 + 40), v113, 0, 0, v20, 0, 0);
          v78 = v120;
          sub_240E460((__int64 *)&v149, v120, (__int64)v76[1], 0, 0);
          if ( v122 )
          {
            v78 = v120;
            v79 = *(_BYTE *)v120 == 14;
            if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v120 + 8) + 8LL) - 15) > 1u )
            {
              if ( *(_BYTE *)v120 != 17 )
              {
LABEL_91:
                v80 = *((_DWORD *)v124 + 74);
                v81 = v124[35];
                if ( v80 )
                {
                  v82 = (v80 - 1) & (((unsigned int)v110 >> 9) ^ ((unsigned int)v110 >> 4));
                  v83 = (unsigned int **)&v81[2 * v82];
                  v84 = *v83;
                  if ( *v83 == v110 )
                  {
LABEL_93:
                    v78 = v109;
                    sub_240E460((__int64 *)&v149, v109, (__int64)v83[1], 0, 0);
                    goto LABEL_94;
                  }
                  v104 = 1;
                  while ( v84 != (unsigned int *)-4096LL )
                  {
                    v105 = v104 + 1;
                    v82 = (v80 - 1) & (v104 + v82);
                    v83 = (unsigned int **)&v81[2 * v82];
                    v84 = *v83;
                    if ( v110 == *v83 )
                      goto LABEL_93;
                    v104 = v105;
                  }
                }
                v83 = (unsigned int **)&v81[2 * v80];
                goto LABEL_93;
              }
              v103 = *(_DWORD *)(v120 + 32);
              if ( v103 <= 0x40 )
                v79 = *(_QWORD *)(v120 + 24) == 0;
              else
                v79 = v103 == (unsigned int)sub_C444A0(v120 + 24);
            }
            if ( !v79 )
              goto LABEL_91;
          }
LABEL_94:
          sub_F94A20(&v149, v78);
          goto LABEL_29;
        }
      }
      else
      {
        v20 = 1;
        while ( v77 != (unsigned int *)-4096LL )
        {
          v75 = (v74 - 1) & (v20 + v75);
          v76 = (unsigned int **)&v73[2 * v75];
          v77 = *v76;
          if ( v110 == *v76 )
            goto LABEL_87;
          v20 = (unsigned int)(v20 + 1);
        }
      }
    }
  }
  v123 = 0;
  if ( (_BYTE)qword_4FE3A68 )
  {
    _BitScanReverse64(&v21, 1LL << v121);
    v123 = 63 - (v21 ^ 0x3F);
  }
  v22 = *(_BYTE *)v120;
  if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)(v120 + 8) + 8LL) - 15) > 1u )
  {
    if ( v22 != 17 )
      goto LABEL_18;
    v38 = *(_DWORD *)(v120 + 32);
    if ( !(v38 <= 0x40 ? *(_QWORD *)(v120 + 24) == 0 : v38 == (unsigned int)sub_C444A0(v120 + 24)) )
      goto LABEL_18;
LABEL_44:
    sub_2413070((__int64 *)v124, (unsigned __int64)v110, v114, v123, v113, 0);
    if ( !(_BYTE)qword_4FE3408 )
      goto LABEL_30;
    goto LABEL_45;
  }
  if ( v22 == 14 )
    goto LABEL_44;
LABEL_18:
  sub_2412230((__int64)&v133, *(_QWORD *)(a2 + 40), v113, 0, 0, v20, 0, 0);
  v119 = sub_2412430(*v124, (unsigned __int64)v110, v121, v113, 0, v23);
  v24 = v114;
  v108 = v25;
  if ( v114 <= 7 )
  {
    v26 = 0;
LABEL_20:
    v126 = v26;
    v116 = v26 + v24;
    do
    {
      v152 = 257;
      v27 = sub_94B060(&v133, (*v124)[6], v119, v126, (__int64)&v149);
      v152 = 257;
      v28 = sub_BD2C40(80, unk_3F10A10);
      v30 = (__int64)v28;
      if ( v28 )
        sub_B4D3C0((__int64)v28, v120, v27, 0, v123, v29, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v139 + 16LL))(
        v139,
        v30,
        &v149,
        v136,
        v137);
      v31 = v133;
      v32 = &v133[4 * v134];
      if ( v133 != v32 )
      {
        do
        {
          v33 = *((_QWORD *)v31 + 1);
          v34 = *v31;
          v31 += 4;
          sub_B99FD0(v30, v34, v33);
        }
        while ( v32 != v31 );
      }
      ++v126;
    }
    while ( v126 != v116 );
    goto LABEL_26;
  }
  v51 = 0;
  v117 = (__int64 **)sub_BCDA70((__int64 *)(*v124)[6], 8);
  v52 = (_BYTE *)sub_ACADE0(v117);
  do
  {
    while ( 1 )
    {
      v147 = 257;
      v54 = sub_BCB2D0((_QWORD *)(*v124)[1]);
      v55 = (unsigned __int8 *)sub_ACD640(v54, v51, 0);
      v56 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v138 + 104LL);
      if ( v56 == sub_948040 )
        break;
      v53 = v56(v138, v52, (_BYTE *)v120, v55);
LABEL_66:
      if ( !v53 )
        goto LABEL_70;
      ++v51;
      v52 = (_BYTE *)v53;
      if ( v51 == 8 )
        goto LABEL_75;
    }
    if ( *v52 <= 0x15u && *(_BYTE *)v120 <= 0x15u && *v55 <= 0x15u )
    {
      v53 = sub_AD5A90((__int64)v52, (_BYTE *)v120, v55, 0);
      goto LABEL_66;
    }
LABEL_70:
    v152 = 257;
    v57 = sub_BD2C40(72, 3u);
    v58 = 0;
    v59 = (__int64)v57;
    if ( v57 )
      sub_B4DFA0((__int64)v57, (__int64)v52, v120, (__int64)v55, (__int64)&v149, 0, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, unsigned int **, __int64, __int64, __int64))(*(_QWORD *)v139 + 16LL))(
      v139,
      v59,
      v145,
      v136,
      v137,
      v58);
    v60 = v133;
    v61 = &v133[4 * v134];
    if ( v133 != v61 )
    {
      do
      {
        v62 = *((_QWORD *)v60 + 1);
        v63 = *v60;
        v60 += 4;
        sub_B99FD0(v59, v63, v62);
      }
      while ( v61 != v60 );
    }
    ++v51;
    v52 = (_BYTE *)v59;
  }
  while ( v51 != 8 );
LABEL_75:
  v112 = (__int64)v52;
  v127 = 0;
  v111 = (v114 - 8) >> 3;
  while ( 1 )
  {
    v152 = 257;
    v64 = sub_94B060(&v133, (__int64)v117, v119, v127, (__int64)&v149);
    v152 = 257;
    v65 = v64;
    v66 = sub_BD2C40(80, unk_3F10A10);
    v68 = (__int64)v66;
    if ( v66 )
    {
      sub_B4D3C0((__int64)v66, v112, v65, 0, v123, v67, 0, 0);
      v67 = v106;
    }
    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64, __int64))(*(_QWORD *)v139 + 16LL))(
      v139,
      v68,
      &v149,
      v136,
      v137,
      v67);
    v69 = v133;
    v70 = &v133[4 * v134];
    if ( v133 != v70 )
    {
      do
      {
        v71 = *((_QWORD *)v69 + 1);
        v72 = *v69;
        v69 += 4;
        sub_B99FD0(v68, v72, v71);
      }
      while ( v70 != v69 );
    }
    if ( v127 == v111 )
      break;
    ++v127;
  }
  v26 = 8 * (v111 + 1);
  v24 = v114 & 7;
  if ( (v114 & 7) != 0 )
    goto LABEL_20;
LABEL_26:
  if ( v122 )
  {
    _BitScanReverse64(&v44, 1LL << v121);
    v45 = 63 - (v44 ^ 0x3F);
    if ( (unsigned __int8)byte_4FE3AA8 >= v45 )
      v45 = byte_4FE3AA8;
    v46 = sub_2415280((__int64)v124, v120, v113, 0);
    v47 = *(_QWORD *)(a2 + 40);
    sub_2412230((__int64)v141, v47, v113, 0, 0, v48, 0, 0);
    if ( *(_BYTE *)v46 > 0x15u )
    {
      if ( (int)qword_4FE3088 < 0 || (int)qword_4FE3088 > *((_DWORD *)v124 + 120) )
      {
        v149 = "_dfscmp";
        v152 = 259;
        v86 = *(_QWORD *)(v46 + 8);
        if ( *(_DWORD *)(v86 + 8) >> 8 != 1 )
        {
          v87 = (_BYTE *)sub_AD64C0(v86, 0, 0);
          v46 = sub_92B530(v141, 0x21u, v46, v87, (__int64)&v149);
        }
        v164 = 0;
        v149 = v151;
        v150 = 0x1000000000LL;
        v155 = v124 + 2;
        v153 = 0;
        v154 = 0;
        v156 = 0;
        v157 = 1;
        v158 = 0;
        v159 = &v163;
        v160 = 8;
        v161 = 0;
        v162 = 1;
        v165 = 0;
        v166 = 0;
        v167 = 0;
        v88 = v143;
        if ( v143 )
          v88 = v143 - 24;
        v89 = sub_F38250(v46, (__int64 *)(v88 + 24), 0, 0, (*v124)[98], (__int64)&v149, 0, 0);
        sub_23D0AB0((__int64)v145, v89, 0, 0, 0);
        v90 = sub_2411150((__int64)v124, v109, v145);
        sub_2410910((__int64 *)v124, (__int64)v145, v90, v108, v114, v45);
        ++*((_DWORD *)v124 + 120);
        nullsub_61();
        v148 = &unk_49DA100;
        nullsub_63();
        if ( (_QWORD *)v145[0] != v146 )
          _libc_free((unsigned __int64)v145[0]);
        sub_FFCE90((__int64)&v149, (__int64)v145, v91, v92, v93, v94);
        sub_FFD870((__int64)&v149, (__int64)v145, v95, v96, v97, v98);
        sub_FFBC40((__int64)&v149, (__int64)v145);
        v99 = v166;
        v100 = v165;
        if ( v166 != v165 )
        {
          do
          {
            v101 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v100[7];
            *v100 = &unk_49E5048;
            if ( v101 )
              v101(v100 + 5, v100 + 5, 3);
            *v100 = &unk_49DB368;
            v102 = v100[3];
            if ( v102 != 0 && v102 != -4096 && v102 != -8192 )
              sub_BD60C0(v100 + 1);
            v100 += 9;
          }
          while ( v99 != v100 );
          v100 = v165;
        }
        if ( v100 )
          j_j___libc_free_0((unsigned __int64)v100);
        if ( !v162 )
          _libc_free((unsigned __int64)v159);
        if ( v149 != v151 )
          _libc_free((unsigned __int64)v149);
      }
      else
      {
        v145[0] = (unsigned int *)v46;
        v152 = 257;
        v145[1] = v110;
        v146[0] = sub_ACD640((*v124)[8], v114, 0);
        v146[1] = v109;
        sub_921880(v141, (*v124)[75], (*v124)[76], (int)v145, 4, (__int64)&v149, 0);
      }
    }
    else if ( !sub_AD7890(v46, v47, v107, v49, v50) )
    {
      v85 = sub_2411150((__int64)v124, v109, v141);
      sub_2410910((__int64 *)v124, (__int64)v141, v85, v108, v114, v45);
    }
    nullsub_61();
    v144 = &unk_49DA100;
    nullsub_63();
    if ( (char *)v141[0] != &v142 )
      _libc_free((unsigned __int64)v141[0]);
  }
  nullsub_61();
  v140 = &unk_49DA100;
  nullsub_63();
  if ( v133 != (unsigned int *)&v135 )
    _libc_free((unsigned __int64)v133);
LABEL_29:
  if ( (_BYTE)qword_4FE3408 )
  {
LABEL_45:
    sub_23D0AB0((__int64)&v149, a2, 0, 0, 0);
    v40 = *(unsigned int **)(a2 - 32);
    v147 = 257;
    v141[1] = v40;
    v141[0] = (unsigned int *)v120;
    v41 = sub_921880(
            (unsigned int **)&v149,
            *(_QWORD *)(*(_QWORD *)*a1 + 408LL),
            *(_QWORD *)(*(_QWORD *)*a1 + 416LL),
            (int)v141,
            2,
            (__int64)v145,
            0);
    v42 = (__int64 *)sub_BD5C60(v41);
    *(_QWORD *)(v41 + 72) = sub_A7A090((__int64 *)(v41 + 72), v42, 1, 79);
    sub_F94A20(&v149, (__int64)v42);
  }
LABEL_30:
  if ( v132[0] )
    j_j___libc_free_0(v132[0]);
  if ( v129 )
    j_j___libc_free_0(v129);
}
