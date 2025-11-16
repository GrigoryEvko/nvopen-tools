// Function: sub_29D6A20
// Address: 0x29d6a20
//
__int64 __fastcall sub_29D6A20(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  _BYTE *v6; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // r12
  _QWORD *v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned int v16; // r12d
  __int64 i; // rsi
  unsigned __int8 v19; // dl
  __int16 *v20; // rax
  __int16 *v21; // r10
  __int64 *v22; // r12
  void **v23; // r10
  __int64 v24; // r14
  __int64 v25; // r13
  unsigned __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rax
  __int16 *v32; // rax
  __int16 *v33; // rcx
  __int16 *v34; // rax
  __int16 *v35; // rcx
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rax
  bool v39; // zf
  __int64 v40; // rbx
  __int16 *v41; // rax
  __int16 *v42; // rdx
  unsigned __int64 v43; // rax
  __int64 *v44; // rax
  unsigned __int64 v45; // rax
  int v46; // edx
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rbx
  __int64 v50; // rcx
  unsigned __int64 v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // r8
  __int64 *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 *v59; // rax
  __int64 v60; // rbx
  _QWORD *v61; // rdi
  unsigned __int64 v62; // rax
  int v63; // edx
  unsigned __int64 v64; // rax
  unsigned __int8 *v65; // r11
  __int64 (__fastcall *v66)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v67; // rax
  __int64 v68; // rbx
  __int64 v69; // rax
  __int64 v70; // rcx
  __int64 v71; // rax
  unsigned __int64 v72; // rbx
  _BYTE *v73; // r14
  __int64 v74; // r12
  unsigned __int64 v75; // rdi
  _QWORD *v76; // rdi
  __int64 v77; // rdx
  __int64 v78; // rcx
  unsigned __int64 v79; // rax
  _BYTE *v80; // r15
  _QWORD *v81; // rax
  __int64 v82; // r12
  __int64 v83; // rcx
  int v84; // edx
  unsigned __int8 v85; // al
  unsigned __int8 v86; // al
  int v87; // r12d
  unsigned int *v88; // r12
  __int64 v89; // r14
  unsigned int *v90; // r13
  __int64 v91; // rdx
  unsigned int v92; // esi
  __int64 v93; // rax
  __int64 v94; // rax
  __int64 v95; // rbx
  bool v96; // cl
  __int64 v97; // rax
  unsigned __int64 v98; // rax
  __int64 v99; // rdx
  int v100; // edi
  __int64 v101; // rbx
  _QWORD *v102; // rdi
  __int64 *v103; // [rsp+0h] [rbp-260h]
  __int64 v104; // [rsp+18h] [rbp-248h]
  __int64 v105; // [rsp+20h] [rbp-240h]
  int v106; // [rsp+28h] [rbp-238h]
  __int16 v107; // [rsp+28h] [rbp-238h]
  __int64 v108; // [rsp+30h] [rbp-230h]
  __int64 v109; // [rsp+30h] [rbp-230h]
  __int64 v110; // [rsp+38h] [rbp-228h]
  __int64 v111; // [rsp+38h] [rbp-228h]
  char v112; // [rsp+40h] [rbp-220h]
  __int64 v113; // [rsp+40h] [rbp-220h]
  unsigned __int8 *v114; // [rsp+40h] [rbp-220h]
  unsigned __int8 *v115; // [rsp+40h] [rbp-220h]
  unsigned int v116; // [rsp+48h] [rbp-218h]
  void **v117; // [rsp+48h] [rbp-218h]
  void **v118; // [rsp+48h] [rbp-218h]
  void **v119; // [rsp+48h] [rbp-218h]
  unsigned __int64 v120; // [rsp+48h] [rbp-218h]
  unsigned __int64 v121; // [rsp+50h] [rbp-210h]
  __int64 *v122; // [rsp+50h] [rbp-210h]
  __int64 v123; // [rsp+50h] [rbp-210h]
  __int64 v124; // [rsp+50h] [rbp-210h]
  __int64 v125; // [rsp+50h] [rbp-210h]
  __int64 v126; // [rsp+50h] [rbp-210h]
  _QWORD *v127; // [rsp+58h] [rbp-208h]
  __int64 v128; // [rsp+58h] [rbp-208h]
  __int64 v129; // [rsp+58h] [rbp-208h]
  __int64 v130; // [rsp+58h] [rbp-208h]
  bool v131; // [rsp+58h] [rbp-208h]
  __int64 v133; // [rsp+60h] [rbp-200h]
  __int64 v134; // [rsp+78h] [rbp-1E8h] BYREF
  __int64 v135; // [rsp+80h] [rbp-1E0h] BYREF
  _QWORD *v136; // [rsp+88h] [rbp-1D8h] BYREF
  _QWORD *v137; // [rsp+90h] [rbp-1D0h] BYREF
  __int64 v138; // [rsp+98h] [rbp-1C8h]
  __int16 v139; // [rsp+B0h] [rbp-1B0h]
  unsigned int **v140; // [rsp+C0h] [rbp-1A0h] BYREF
  _QWORD v141[3]; // [rsp+C8h] [rbp-198h] BYREF
  __int64 v142; // [rsp+E0h] [rbp-180h]
  __int16 v143; // [rsp+E8h] [rbp-178h]
  __int64 v144; // [rsp+F0h] [rbp-170h] BYREF
  unsigned int *v145; // [rsp+100h] [rbp-160h] BYREF
  __int64 v146; // [rsp+108h] [rbp-158h]
  _BYTE v147[32]; // [rsp+110h] [rbp-150h] BYREF
  __int64 v148; // [rsp+130h] [rbp-130h]
  __int64 v149; // [rsp+138h] [rbp-128h]
  __int64 v150; // [rsp+140h] [rbp-120h]
  __int64 v151; // [rsp+148h] [rbp-118h]
  void **v152; // [rsp+150h] [rbp-110h]
  void **v153; // [rsp+158h] [rbp-108h]
  __int64 v154; // [rsp+160h] [rbp-100h]
  int v155; // [rsp+168h] [rbp-F8h]
  __int16 v156; // [rsp+16Ch] [rbp-F4h]
  char v157; // [rsp+16Eh] [rbp-F2h]
  __int64 v158; // [rsp+170h] [rbp-F0h]
  __int64 v159; // [rsp+178h] [rbp-E8h]
  void *v160; // [rsp+180h] [rbp-E0h] BYREF
  void *v161; // [rsp+188h] [rbp-D8h] BYREF
  __int64 v162; // [rsp+190h] [rbp-D0h] BYREF
  __int16 *v163; // [rsp+198h] [rbp-C8h]
  __int64 v164; // [rsp+1A0h] [rbp-C0h]
  int v165; // [rsp+1A8h] [rbp-B8h]
  unsigned __int8 v166; // [rsp+1ACh] [rbp-B4h]
  __int16 v167; // [rsp+1B0h] [rbp-B0h] BYREF

  v2 = a2;
  v151 = sub_AA48A0(a2);
  v145 = (unsigned int *)v147;
  v160 = &unk_49DA100;
  v146 = 0x200000000LL;
  LOWORD(v150) = 0;
  v161 = &unk_49DA0B0;
  v149 = a2 + 48;
  v5 = *(_QWORD *)(a2 + 56);
  v152 = &v160;
  v153 = &v161;
  v154 = 0;
  v155 = 0;
  v156 = 512;
  v157 = 7;
  v158 = 0;
  v159 = 0;
  v148 = a2;
  if ( !v5 )
    BUG();
  if ( *(_BYTE *)(v5 - 24) == 84 )
    goto LABEL_3;
  for ( i = *(_QWORD *)(a2 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
      break;
  }
  v162 = 0;
  v163 = &v167;
  v164 = 16;
  v165 = 0;
  v166 = 1;
  sub_F58DD0((__int64)&v162, (__int64 *)i, 0, 0, v3, v4);
  v19 = v166;
  v20 = v163;
  if ( v166 )
    v21 = &v163[4 * HIDWORD(v164)];
  else
    v21 = &v163[4 * (unsigned int)v164];
  if ( v163 == v21 )
    goto LABEL_38;
  while ( 1 )
  {
    v22 = (__int64 *)v20;
    if ( *(_QWORD *)v20 < 0xFFFFFFFFFFFFFFFELL )
      break;
    v20 += 4;
    if ( v21 == v20 )
      goto LABEL_38;
  }
  if ( v20 == v21 )
  {
LABEL_38:
    if ( v19 )
      goto LABEL_67;
    _libc_free((unsigned __int64)v163);
    v5 = *(_QWORD *)(v2 + 56);
LABEL_68:
    if ( !v5 )
      BUG();
LABEL_3:
    if ( *(_BYTE *)(v5 - 24) == 84 )
      goto LABEL_27;
    v6 = sub_F35A80(v2, &v134, &v135);
    if ( !v6 )
      goto LABEL_27;
    v7 = *((_QWORD *)v6 - 12);
    if ( *(_BYTE *)v7 <= 0x1Cu )
      goto LABEL_27;
    v8 = *(_QWORD *)(v7 + 40);
    if ( (*(_WORD *)(v8 + 2) & 0x7FFF) != 0 )
      goto LABEL_27;
    v9 = sub_F35A80(*(_QWORD *)(v7 + 40), (__int64 *)&v136, (__int64 *)&v137);
    if ( !v9 )
      goto LABEL_27;
    v10 = *((_QWORD *)v9 - 12);
    if ( *(_BYTE *)v10 <= 0x1Cu )
      goto LABEL_27;
    v11 = *(_QWORD **)(v10 + 40);
    v127 = v11;
    if ( (_QWORD *)v8 == v11 )
      goto LABEL_27;
    if ( v11 == v137 )
    {
      v57 = v135;
      v58 = v134;
      if ( v8 == v135 )
      {
        v112 = 0;
        v57 = v134;
      }
      else
      {
        if ( v8 != v134 )
          goto LABEL_27;
        v134 = v135;
        v135 = v58;
        v112 = 1;
      }
      v116 = 29;
      if ( !sub_29D66F0(a1, (__int64)v136, v57, v8) )
        goto LABEL_27;
    }
    else
    {
      if ( v11 != v136 )
        goto LABEL_27;
      v12 = v134;
      if ( v8 == v134 )
      {
        v112 = 0;
        v12 = v135;
      }
      else
      {
        if ( v8 != v135 )
          goto LABEL_27;
        v134 = v8;
        v135 = v12;
        v112 = 1;
      }
      if ( !sub_29D66F0(a1, (__int64)v137, v12, v8) )
        goto LABEL_27;
      v116 = 28;
    }
    v13 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v8 + 48 == v13 )
    {
      v15 = *(_QWORD *)(v8 + 56);
    }
    else
    {
      if ( !v13 )
        BUG();
      v14 = *(_QWORD *)(v8 + 56);
      v15 = v14;
      if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 <= 0xA )
      {
        v121 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v13 == v14 )
          goto LABEL_139;
        while ( 1 )
        {
          if ( !v14 )
            BUG();
LABEL_25:
          if ( *(_BYTE *)(v14 - 24) == 84
            || (unsigned __int8)sub_B46970((unsigned __int8 *)(v14 - 24))
            || !sub_991A70((unsigned __int8 *)(v14 - 24), 0, 0, 0, 0, 1u, 0) )
          {
            break;
          }
          v14 = *(_QWORD *)(v14 + 8);
          if ( v14 == v121 )
            goto LABEL_139;
        }
LABEL_27:
        v16 = 0;
        goto LABEL_28;
      }
    }
    v121 = 0;
    v14 = v15;
    if ( v15 )
      goto LABEL_25;
LABEL_139:
    v60 = 0;
    v133 = v127[6];
    v61 = (_QWORD *)((v133 & 0xFFFFFFFFFFFFFFF8LL) - 24);
    if ( (v133 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      v61 = 0;
    sub_B43D60(v61);
    sub_AA80F0((__int64)v127, v127 + 6, 0, v8, *(__int64 **)(v8 + 56), 1, (__int64 *)(v8 + 48), 0);
    v62 = v127[6] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v127 + 6 == (_QWORD *)v62 )
    {
      v123 = 0;
    }
    else
    {
      if ( !v62 )
        BUG();
      v63 = *(unsigned __int8 *)(v62 - 24);
      v64 = v62 - 24;
      if ( (unsigned int)(v63 - 30) < 0xB )
        v60 = v64;
      v123 = v60;
    }
    v110 = v148;
    v109 = v149;
    v107 = v150;
    sub_D5F1F0((__int64)&v145, v123);
    if ( v112 )
      sub_F35C60(v123, (__int64 *)&v145);
    LOWORD(v142) = 257;
    v65 = *(unsigned __int8 **)(v123 - 96);
    v66 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *))*((_QWORD *)*v152 + 2);
    if ( v66 == sub_9202E0 )
    {
      if ( *(_BYTE *)v10 > 0x15u || *v65 > 0x15u )
        goto LABEL_179;
      v114 = *(unsigned __int8 **)(v123 - 96);
      if ( (unsigned __int8)sub_AC47B0(v116) )
        v67 = sub_AD5570(v116, v10, v114, 0, 0);
      else
        v67 = sub_AABE40(v116, (unsigned __int8 *)v10, v114);
      v65 = v114;
      v68 = v67;
    }
    else
    {
      v115 = *(unsigned __int8 **)(v123 - 96);
      v93 = v66((__int64)v152, v116, (_BYTE *)v10, v65);
      v65 = v115;
      v68 = v93;
    }
    if ( v68 )
    {
LABEL_155:
      sub_BD2ED0(v123, *(_QWORD *)(v123 - 96), v68);
      sub_A88F30((__int64)&v145, v110, v109, v107);
      if ( v127 != v136 )
      {
        sub_AA5200((__int64)v136);
        sub_AA5450(v136);
      }
      if ( v127 != v137 )
      {
        sub_AA5200((__int64)v137);
        sub_AA5450(v137);
      }
      v16 = 1;
      sub_AA5200(v8);
      sub_AA5450((_QWORD *)v8);
      goto LABEL_28;
    }
LABEL_179:
    v167 = 257;
    v68 = sub_B504D0(v116, v10, (__int64)v65, (__int64)&v162, 0, 0);
    if ( *(_BYTE *)v68 > 0x1Cu )
    {
      switch ( *(_BYTE *)v68 )
      {
        case ')':
        case '+':
        case '-':
        case '/':
        case '2':
        case '5':
        case 'J':
        case 'K':
        case 'S':
          goto LABEL_195;
        case 'T':
        case 'U':
        case 'V':
          v82 = *(_QWORD *)(v68 + 8);
          v83 = v82;
          v84 = *(unsigned __int8 *)(v82 + 8);
          if ( (unsigned int)(v84 - 17) <= 1 )
            v83 = **(_QWORD **)(v82 + 16);
          v85 = *(_BYTE *)(v83 + 8);
          if ( v85 <= 3u || v85 == 5 || (v85 & 0xFD) == 4 )
            goto LABEL_195;
          if ( (_BYTE)v84 == 15 )
          {
            if ( (*(_BYTE *)(v82 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v68 + 8)) )
              break;
            v82 = **(_QWORD **)(v82 + 16);
          }
          else if ( (_BYTE)v84 == 16 )
          {
            do
              v82 = *(_QWORD *)(v82 + 24);
            while ( *(_BYTE *)(v82 + 8) == 16 );
          }
          if ( (unsigned int)*(unsigned __int8 *)(v82 + 8) - 17 <= 1 )
            v82 = **(_QWORD **)(v82 + 16);
          v86 = *(_BYTE *)(v82 + 8);
          if ( v86 <= 3u || v86 == 5 || (v86 & 0xFD) == 4 )
          {
LABEL_195:
            v87 = v155;
            if ( v154 )
              sub_B99FD0(v68, 3u, v154);
            sub_B45150(v68, v87);
          }
          break;
        default:
          break;
      }
    }
    (*((void (__fastcall **)(void **, __int64, unsigned int ***, __int64, __int64))*v153 + 2))(
      v153,
      v68,
      &v140,
      v149,
      v150);
    v88 = v145;
    if ( v145 != &v145[4 * (unsigned int)v146] )
    {
      v89 = v8;
      v90 = &v145[4 * (unsigned int)v146];
      do
      {
        v91 = *((_QWORD *)v88 + 1);
        v92 = *v88;
        v88 += 4;
        sub_B99FD0(v68, v92, v91);
      }
      while ( v90 != v88 );
      v8 = v89;
    }
    goto LABEL_155;
  }
  v108 = 0;
  v106 = -1;
  v113 = 0;
  v128 = 0;
  v122 = (__int64 *)v21;
  v23 = &v161;
  v24 = v2;
  v25 = *(_QWORD *)v20;
  do
  {
    v26 = *(_QWORD *)(v25 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v26 == v25 + 48 )
      goto LABEL_228;
    if ( !v26 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30 > 0xA )
LABEL_228:
      BUG();
    if ( *(_BYTE *)(v26 - 24) != 31 )
      goto LABEL_55;
    v117 = v23;
    v27 = sub_AA54C0(v25);
    v23 = v117;
    v28 = v27;
    if ( (*(_DWORD *)(v26 - 20) & 0x7FFFFFF) == 1 )
    {
      v19 = v166;
      if ( v128 || !v27 )
      {
LABEL_51:
        v2 = v24;
        goto LABEL_38;
      }
      if ( v166 )
      {
        v32 = v163;
        v33 = &v163[4 * HIDWORD(v164)];
        if ( v163 == v33 )
          goto LABEL_51;
        while ( v28 != *(_QWORD *)v32 )
        {
          v32 += 4;
          if ( v33 == v32 )
            goto LABEL_51;
        }
      }
      else
      {
        v31 = sub_C8CA60((__int64)&v162, v27);
        v23 = v117;
        if ( !v31 )
          goto LABEL_55;
      }
      if ( (*(_WORD *)(v25 + 2) & 0x7FFF) != 0 )
        goto LABEL_55;
      v128 = v25;
    }
    else
    {
      v29 = *(_QWORD *)(v26 - 120);
      v19 = v166;
      if ( !v29 )
        goto LABEL_51;
      v30 = *(_QWORD *)(v29 + 16);
      if ( !v30 || *(_QWORD *)(v30 + 8) )
        goto LABEL_51;
      if ( v28 )
      {
        if ( v166 )
        {
          v34 = v163;
          v35 = &v163[4 * HIDWORD(v164)];
          if ( v163 != v35 )
          {
            while ( v28 != *(_QWORD *)v34 )
            {
              v34 += 4;
              if ( v35 == v34 )
                goto LABEL_127;
            }
LABEL_77:
            if ( (*(_WORD *)(v25 + 2) & 0x7FFF) != 0 )
            {
LABEL_55:
              v2 = v24;
              v19 = v166;
              goto LABEL_38;
            }
            v103 = v22;
            v36 = *(_QWORD *)(v25 + 56);
            v118 = v23;
            while ( 1 )
            {
              if ( v26 == v36 )
              {
                v22 = v103;
                v23 = v118;
                v19 = v166;
                goto LABEL_84;
              }
              v37 = *(_QWORD *)(v36 + 8);
              if ( *(_BYTE *)(v36 - 24) == 84 || !sub_991A70((unsigned __int8 *)(v36 - 24), 0, 0, 0, 0, 1u, 0) )
                break;
              v36 = v37;
            }
            v2 = v24;
            v19 = v166;
            goto LABEL_38;
          }
        }
        else
        {
          v56 = sub_C8CA60((__int64)&v162, v28);
          v23 = v117;
          if ( v56 )
            goto LABEL_77;
          v19 = v166;
        }
      }
LABEL_127:
      if ( v113 )
        goto LABEL_51;
      v113 = v25;
LABEL_84:
      v38 = *(_QWORD *)(v26 - 56);
      v39 = v24 == v38;
      if ( v24 == v38 )
        v38 = *(_QWORD *)(v26 - 88);
      v40 = v38;
      if ( v106 == -1 )
      {
        v106 = !v39;
      }
      else if ( !v39 != v106 )
      {
        goto LABEL_51;
      }
      if ( v19 )
      {
        v41 = v163;
        v42 = &v163[4 * HIDWORD(v164)];
        if ( v163 != v42 )
        {
          while ( v40 != *(_QWORD *)v41 )
          {
            v41 += 4;
            if ( v42 == v41 )
              goto LABEL_135;
          }
LABEL_93:
          v43 = *(_QWORD *)(v40 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v43 == v40 + 48 )
            goto LABEL_241;
          if ( !v43 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 > 0xA )
LABEL_241:
            BUG();
          if ( *(_BYTE *)(v43 - 24) == 31 )
          {
            if ( (*(_DWORD *)(v43 - 20) & 0x7FFFFFF) != 1 )
              v25 = v108;
            v108 = v25;
          }
          goto LABEL_101;
        }
      }
      else
      {
        v119 = v23;
        v59 = sub_C8CA60((__int64)&v162, v38);
        v23 = v119;
        if ( v59 )
          goto LABEL_93;
      }
LABEL_135:
      v108 = v25;
    }
LABEL_101:
    v44 = v22 + 1;
    if ( v22 + 1 == v122 )
      break;
    while ( 1 )
    {
      v25 = *v44;
      v22 = v44;
      if ( (unsigned __int64)*v44 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v122 == ++v44 )
        goto LABEL_104;
    }
  }
  while ( v122 != v44 );
LABEL_104:
  v2 = v24;
  if ( v108 == v113 || v108 == 0 || v113 == 0 )
    goto LABEL_122;
  v129 = v108 + 48;
  v45 = *(_QWORD *)(v108 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v45 == v108 + 48 )
  {
    v47 = 0;
  }
  else
  {
    if ( !v45 )
      BUG();
    v46 = *(unsigned __int8 *)(v45 - 24);
    v47 = 0;
    v48 = v45 - 24;
    if ( (unsigned int)(v46 - 30) < 0xB )
      v47 = v48;
  }
  v49 = sub_B46EC0(v47, 0);
  v50 = sub_B46EC0(v47, 1u);
  v51 = *(_QWORD *)(v49 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v51 == v49 + 48 )
    goto LABEL_235;
  if ( !v51 )
    BUG();
  v52 = v51 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v51 - 24) - 30 > 0xA )
LABEL_235:
    BUG();
  v39 = *(_BYTE *)(v51 - 24) == 31;
  v53 = 0;
  if ( v39 )
    v53 = v52;
  v54 = *(_QWORD *)(v50 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v54 == v50 + 48 )
    goto LABEL_239;
  if ( !v54 )
    BUG();
  v55 = v54 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 > 0xA )
LABEL_239:
    BUG();
  if ( *(_BYTE *)(v54 - 24) == 31 )
  {
    if ( v53 )
    {
      if ( (*(_DWORD *)(v53 + 4) & 0x7FFFFFF) == 1 )
      {
        v111 = v54 - 24;
        v120 = *(_QWORD *)(v50 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        v125 = v50;
        v94 = sub_B46EC0(v52, 0);
        v70 = v125;
        v54 = v120;
        v55 = v111;
        if ( v125 == v94 )
          goto LABEL_165;
      }
    }
    if ( (*(_DWORD *)(v54 - 20) & 0x7FFFFFF) != 1 || v49 != sub_B46EC0(v55, 0) )
    {
LABEL_122:
      v19 = v166;
      goto LABEL_38;
    }
    v95 = v108;
    v96 = 0;
    while ( 1 )
    {
      v98 = *(_QWORD *)(v95 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v98 == v129 )
        goto LABEL_236;
      if ( !v98 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v98 - 24) - 30 > 0xA )
LABEL_236:
        BUG();
      v99 = *(_QWORD *)(v98 - 120);
      if ( (unsigned __int8)(*(_BYTE *)v99 - 82) <= 1u )
      {
        v100 = *(_WORD *)(v99 + 2) & 0x3F;
        if ( v100 == 6 || v100 == 33 )
        {
          v126 = v98 - 24;
          *(_WORD *)(v99 + 2) = sub_B52870(v100) | *(_WORD *)(v99 + 2) & 0xFFC0;
          sub_B4CC70(v126);
          v96 = v100 == 6 || v100 == 33;
        }
      }
      v131 = v96;
      v97 = sub_AA54C0(v95);
      v96 = v131;
      v95 = v97;
      if ( v97 == v113 )
        break;
      v129 = v97 + 48;
    }
    v16 = v131;
    if ( !v166 )
      _libc_free((unsigned __int64)v163);
    if ( !v131 )
    {
LABEL_67:
      v5 = *(_QWORD *)(v2 + 56);
      goto LABEL_68;
    }
  }
  else
  {
    if ( !v53 )
      goto LABEL_122;
    if ( (*(_DWORD *)(v53 + 4) & 0x7FFFFFF) != 1 )
      goto LABEL_122;
    v130 = v50;
    v69 = sub_B46EC0(v52, 0);
    v70 = v130;
    if ( v130 != v69 )
      goto LABEL_122;
LABEL_165:
    v71 = *(_QWORD *)(v70 + 56);
    if ( !v71 )
      BUG();
    if ( *(_BYTE *)(v71 - 24) == 84 )
      goto LABEL_122;
    v141[0] = 0;
    v141[1] = 0;
    v72 = sub_986580(v113);
    v140 = &v145;
    v141[2] = v148;
    if ( v148 != 0 && v148 != -4096 && v148 != -8192 )
      sub_BD73F0((__int64)v141);
    v142 = v149;
    v143 = v150;
    sub_B33910(&v144, (__int64 *)&v145);
    v73 = *(_BYTE **)(v72 - 96);
    while ( 1 )
    {
      v74 = *(_QWORD *)(v72 + -32 - 32LL * (1 - v106));
      v75 = *(_QWORD *)(v113 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      v76 = v75 ? (_QWORD *)(v75 - 24) : 0LL;
      sub_B43D60(v76);
      v77 = v104;
      v78 = v105;
      LOWORD(v77) = 0;
      LOWORD(v78) = 1;
      v104 = v77;
      v105 = v78;
      sub_AA80F0(
        v113,
        (unsigned __int64 *)(v113 + 48),
        0,
        v74,
        *(__int64 **)(v74 + 56),
        v78,
        (__int64 *)(v74 + 48),
        v77);
      v79 = sub_986580(v113);
      v80 = *(_BYTE **)(v79 - 96);
      v72 = v79;
      sub_D5F1F0((__int64)&v145, v79);
      v139 = 257;
      v73 = (_BYTE *)(v106 ? sub_A82350(&v145, v73, v80, (__int64)&v137) : sub_A82480(&v145, v73, v80, (__int64)&v137));
      sub_BD2ED0(v72, (__int64)v80, (__int64)v73);
      if ( v74 == v108 )
        break;
      sub_AA5200(v74);
      v124 = sub_AA48A0(v74);
      sub_B43C20((__int64)&v137, v74);
      v81 = sub_BD2C40(72, unk_3F148B8);
      if ( v81 )
        sub_B4C8A0((__int64)v81, v124, (__int64)v137, v138);
    }
    sub_AA5200(v74);
    v101 = sub_AA48A0(v74);
    sub_B43C20((__int64)&v137, v74);
    v102 = sub_BD2C40(72, unk_3F148B8);
    if ( v102 )
      sub_B4C8A0((__int64)v102, v101, (__int64)v137, v138);
    sub_F11320((__int64)&v140);
    v16 = v166;
    if ( !v166 )
    {
      v16 = 1;
      _libc_free((unsigned __int64)v163);
    }
  }
LABEL_28:
  nullsub_61();
  v160 = &unk_49DA100;
  nullsub_63();
  if ( v145 != (unsigned int *)v147 )
    _libc_free((unsigned __int64)v145);
  return v16;
}
