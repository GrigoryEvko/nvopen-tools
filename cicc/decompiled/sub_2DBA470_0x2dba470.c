// Function: sub_2DBA470
// Address: 0x2dba470
//
__int64 __fastcall sub_2DBA470(__int64 a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 i; // r15
  __int64 v7; // rbx
  int v8; // eax
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  int v14; // ecx
  char v15; // al
  __int64 v16; // rcx
  __int64 v17; // rdx
  bool v18; // cl
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // eax
  __int64 v24; // rbx
  __int64 v25; // r12
  _QWORD *v26; // rax
  signed __int64 v27; // rsi
  __int64 v28; // r9
  signed __int64 v29; // r13
  unsigned __int64 v30; // rax
  int v31; // ecx
  unsigned __int64 *v32; // rdx
  unsigned int v33; // ebx
  _BYTE *v34; // r15
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  _BYTE *v41; // r14
  __int64 v42; // rax
  unsigned __int8 *v43; // rbx
  __int64 (__fastcall *v44)(__int64, _BYTE *, unsigned __int8 *); // rax
  unsigned __int8 *v45; // r12
  _BYTE *v46; // rbx
  __int64 v47; // rax
  unsigned __int8 *v48; // r11
  __int64 (__fastcall *v49)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v50; // rax
  unsigned __int8 *v51; // r14
  int v52; // r11d
  __int64 (__fastcall *v53)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  unsigned int v54; // r10d
  __int64 v55; // rax
  __int64 v56; // rbx
  __int64 v57; // rax
  unsigned __int8 *v58; // r12
  __int64 (__fastcall *v59)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v60; // rax
  __int64 v61; // r14
  unsigned __int64 v62; // r15
  unsigned __int64 *v63; // r12
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // r12
  int v67; // edx
  unsigned int v68; // ecx
  int v69; // r12d
  unsigned __int64 *v70; // r12
  unsigned __int64 v71; // r14
  __int64 v72; // rdx
  unsigned int v73; // esi
  unsigned __int8 v74; // al
  __int64 *v75; // rax
  unsigned __int8 *v76; // rax
  unsigned __int64 v77; // r12
  unsigned __int64 *v78; // rbx
  __int64 v79; // rdx
  unsigned int v80; // esi
  unsigned __int8 *v81; // rax
  unsigned __int64 v82; // r14
  unsigned __int64 *v83; // rbx
  __int64 v84; // rdx
  unsigned int v85; // esi
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned __int8 *v88; // rax
  unsigned __int8 *v89; // rcx
  __int64 v90; // rdx
  int v91; // eax
  _BYTE *v92; // rdi
  unsigned __int64 v93; // r8
  unsigned __int64 v94; // rbx
  __int64 v95; // rdx
  unsigned __int64 v96; // rcx
  unsigned __int64 v97; // rsi
  bool v98; // zf
  unsigned __int64 v99; // rax
  unsigned int v100; // r13d
  unsigned __int64 v101; // r9
  int v102; // eax
  int v103; // r13d
  __int64 v104; // rax
  unsigned __int64 v105; // rdx
  __int64 v106; // [rsp+8h] [rbp-1D8h]
  unsigned __int8 v107; // [rsp+27h] [rbp-1B9h]
  unsigned __int8 *v108; // [rsp+40h] [rbp-1A0h]
  unsigned int v109; // [rsp+40h] [rbp-1A0h]
  __int64 v110; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v111; // [rsp+40h] [rbp-1A0h]
  unsigned __int8 *v112; // [rsp+40h] [rbp-1A0h]
  unsigned int v113; // [rsp+40h] [rbp-1A0h]
  __int64 v114; // [rsp+50h] [rbp-190h]
  unsigned __int64 v115; // [rsp+50h] [rbp-190h]
  unsigned __int64 v116; // [rsp+50h] [rbp-190h]
  bool v117; // [rsp+50h] [rbp-190h]
  unsigned int v118; // [rsp+58h] [rbp-188h]
  unsigned __int8 *v119; // [rsp+58h] [rbp-188h]
  _BYTE v120[32]; // [rsp+60h] [rbp-180h] BYREF
  __int16 v121; // [rsp+80h] [rbp-160h]
  unsigned __int64 v122; // [rsp+90h] [rbp-150h] BYREF
  unsigned int v123; // [rsp+98h] [rbp-148h]
  __int16 v124; // [rsp+B0h] [rbp-130h]
  _BYTE *v125; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v126; // [rsp+C8h] [rbp-118h]
  _BYTE v127[32]; // [rsp+D0h] [rbp-110h] BYREF
  _BYTE *v128; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v129; // [rsp+F8h] [rbp-E8h]
  _BYTE v130[32]; // [rsp+100h] [rbp-E0h] BYREF
  unsigned __int64 *v131; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v132; // [rsp+128h] [rbp-B8h]
  _BYTE v133[32]; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v134; // [rsp+150h] [rbp-90h]
  __int64 v135; // [rsp+158h] [rbp-88h]
  __int64 v136; // [rsp+160h] [rbp-80h]
  _QWORD *v137; // [rsp+168h] [rbp-78h]
  void **v138; // [rsp+170h] [rbp-70h]
  void **v139; // [rsp+178h] [rbp-68h]
  __int64 v140; // [rsp+180h] [rbp-60h]
  int v141; // [rsp+188h] [rbp-58h]
  __int16 v142; // [rsp+18Ch] [rbp-54h]
  char v143; // [rsp+18Eh] [rbp-52h]
  __int64 v144; // [rsp+190h] [rbp-50h]
  __int64 v145; // [rsp+198h] [rbp-48h]
  void *v146; // [rsp+1A0h] [rbp-40h] BYREF
  void *v147; // [rsp+1A8h] [rbp-38h] BYREF

  v2 = qword_501D2A8;
  v125 = v127;
  if ( (_DWORD)qword_501D2A8 == 0x800000 )
    v2 = *(_DWORD *)(a2 + 88);
  v126 = 0x400000000LL;
  v128 = v130;
  v129 = 0x400000000LL;
  v107 = 0;
  if ( v2 <= 0x7FFFFF )
  {
    v4 = *(_QWORD *)(a1 + 80);
    v5 = a1 + 72;
    if ( a1 + 72 == v4 )
      goto LABEL_11;
    if ( !v4 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v4 + 32);
      if ( i != v4 + 24 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v5 == v4 )
        goto LABEL_11;
      if ( !v4 )
        BUG();
    }
    if ( v5 == v4 )
      goto LABEL_11;
    v107 = 0;
    v118 = v2;
    v7 = a1 + 72;
    while ( 1 )
    {
      if ( !i )
        BUG();
      v8 = *(unsigned __int8 *)(i - 24);
      if ( (unsigned int)(v8 - 29) > 0x14 )
      {
        if ( (unsigned int)(v8 - 51) > 1 )
          goto LABEL_21;
      }
      else if ( (unsigned int)(v8 - 29) <= 0x12 )
      {
        goto LABEL_21;
      }
      v9 = i - 24;
      if ( (*(_BYTE *)(i - 17) & 0x40) != 0 )
        v10 = *(_QWORD *)(i - 32);
      else
        v10 = v9 - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
      if ( sub_BCEA30(*(_QWORD *)(*(_QWORD *)v10 + 8LL)) )
        goto LABEL_21;
      v13 = *(_QWORD *)(i - 16);
      v14 = *(unsigned __int8 *)(v13 + 8);
      if ( (unsigned int)(v14 - 17) <= 1 )
      {
        v13 = **(_QWORD **)(v13 + 16);
        LOBYTE(v14) = *(_BYTE *)(v13 + 8);
      }
      if ( (_BYTE)v14 != 12 || v118 >= *(_DWORD *)(v13 + 8) >> 8 )
        goto LABEL_21;
      v15 = *(_BYTE *)(i - 17);
      if ( (v15 & 0x40) != 0 )
        v16 = *(_QWORD *)(i - 32);
      else
        v16 = v9 - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
      v17 = *(_QWORD *)(v16 + 32);
      if ( *(_BYTE *)v17 != 17 )
        goto LABEL_42;
      v18 = *(_BYTE *)(i - 24) == 52 || *(_BYTE *)(i - 24) == 49;
      v19 = *(_DWORD *)(v17 + 32);
      v123 = v19;
      if ( v19 <= 0x40 )
        break;
      v117 = v18;
      sub_C43780((__int64)&v122, (const void **)(v17 + 24));
      if ( !v117 )
        goto LABEL_177;
      v19 = v123;
      v95 = 1LL << ((unsigned __int8)v123 - 1);
      if ( v123 <= 0x40 )
        goto LABEL_167;
      v101 = v122;
      if ( (*(_QWORD *)(v122 + 8LL * ((v123 - 1) >> 6)) & v95) != 0 )
      {
        LODWORD(v132) = v123;
        sub_C43780((__int64)&v131, (const void **)&v122);
        v19 = v132;
        if ( (unsigned int)v132 > 0x40 )
        {
          sub_C43D10((__int64)&v131);
          goto LABEL_172;
        }
        v96 = (unsigned __int64)v131;
LABEL_169:
        v97 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
        v98 = v19 == 0;
        v99 = 0;
        if ( !v98 )
          v99 = v97;
        v131 = (unsigned __int64 *)(v99 & ~v96);
LABEL_172:
        sub_C46250((__int64)&v131);
        v100 = v132;
        LODWORD(v132) = 0;
        if ( v123 > 0x40 && v122 )
        {
          v115 = (unsigned __int64)v131;
          j_j___libc_free_0_0(v122);
          v123 = v100;
          v122 = v115;
          if ( (unsigned int)v132 > 0x40 && v131 )
          {
            j_j___libc_free_0_0((unsigned __int64)v131);
LABEL_177:
            v100 = v123;
          }
        }
        else
        {
          v122 = (unsigned __int64)v131;
          v123 = v100;
        }
        if ( v100 <= 0x40 )
        {
LABEL_39:
          if ( v122 && (v122 & (v122 - 1)) == 0 )
            goto LABEL_21;
LABEL_41:
          v15 = *(_BYTE *)(i - 17);
LABEL_42:
          if ( (v15 & 0x40) != 0 )
            v20 = *(_QWORD *)(i - 32);
          else
            v20 = v9 - 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)v20 + 8LL) + 8LL) - 17 > 1 )
          {
            v104 = (unsigned int)v126;
            v105 = (unsigned int)v126 + 1LL;
            if ( v105 > HIDWORD(v126) )
            {
              sub_C8D5F0((__int64)&v125, v127, v105, 8u, v11, v12);
              v104 = (unsigned int)v126;
            }
            v107 = 1;
            *(_QWORD *)&v125[8 * v104] = v9;
            LODWORD(v126) = v126 + 1;
          }
          else
          {
            v21 = (unsigned int)v129;
            v22 = (unsigned int)v129 + 1LL;
            if ( v22 > HIDWORD(v129) )
            {
              sub_C8D5F0((__int64)&v128, v130, v22, 8u, v11, v12);
              v21 = (unsigned int)v129;
            }
            v107 = 1;
            *(_QWORD *)&v128[8 * v21] = v9;
            LODWORD(v129) = v129 + 1;
          }
          goto LABEL_21;
        }
        v101 = v122;
      }
      v116 = v101;
      v102 = sub_C44630((__int64)&v122);
      v12 = v116;
      v103 = v102;
      if ( v116 )
        j_j___libc_free_0_0(v116);
      if ( v103 != 1 )
        goto LABEL_41;
LABEL_21:
      for ( i = *(_QWORD *)(i + 8); i == v4 - 24 + 48; i = *(_QWORD *)(v4 + 32) )
      {
        v4 = *(_QWORD *)(v4 + 8);
        if ( v7 == v4 )
          goto LABEL_49;
        if ( !v4 )
          BUG();
      }
      if ( v7 == v4 )
      {
LABEL_49:
        v23 = v129;
        while ( (_DWORD)v129 )
        {
          v24 = *(_QWORD *)&v128[8 * v23 - 8];
          LODWORD(v129) = v23 - 1;
          v119 = (unsigned __int8 *)v24;
          v25 = *(_QWORD *)(v24 + 8);
          v26 = (_QWORD *)sub_BD5C60(v24);
          v142 = 512;
          v137 = v26;
          v131 = (unsigned __int64 *)v133;
          v138 = &v146;
          v134 = 0;
          v139 = &v147;
          LOWORD(v136) = 0;
          v135 = 0;
          v132 = 0x200000000LL;
          v146 = &unk_49DA100;
          v140 = 0;
          v141 = 0;
          v143 = 7;
          v144 = 0;
          v145 = 0;
          v147 = &unk_49DA0B0;
          v134 = *(_QWORD *)(v24 + 40);
          v135 = v24 + 24;
          v27 = *(_QWORD *)sub_B46C60(v24);
          v122 = v27;
          if ( v27 && (sub_B96E90((__int64)&v122, v27, 1), (v29 = v122) != 0) )
          {
            v30 = (unsigned __int64)v131;
            v31 = v132;
            v32 = &v131[2 * (unsigned int)v132];
            if ( v131 != v32 )
            {
              while ( *(_DWORD *)v30 )
              {
                v30 += 16LL;
                if ( v32 == (unsigned __int64 *)v30 )
                  goto LABEL_156;
              }
              *(_QWORD *)(v30 + 8) = v122;
LABEL_57:
              sub_B91220((__int64)&v122, v29);
              goto LABEL_58;
            }
LABEL_156:
            if ( (unsigned int)v132 >= (unsigned __int64)HIDWORD(v132) )
            {
              v93 = (unsigned int)v132 + 1LL;
              v94 = v106 & 0xFFFFFFFF00000000LL;
              v106 &= 0xFFFFFFFF00000000LL;
              if ( HIDWORD(v132) < v93 )
              {
                sub_C8D5F0((__int64)&v131, v133, v93, 0x10u, v93, v28);
                v32 = &v131[2 * (unsigned int)v132];
              }
              *v32 = v94;
              v32[1] = v29;
              v29 = v122;
              LODWORD(v132) = v132 + 1;
            }
            else
            {
              if ( v32 )
              {
                *(_DWORD *)v32 = 0;
                v32[1] = v29;
                v31 = v132;
                v29 = v122;
              }
              LODWORD(v132) = v31 + 1;
            }
          }
          else
          {
            sub_93FB40((__int64)&v131, 0);
            v29 = v122;
          }
          if ( v29 )
            goto LABEL_57;
LABEL_58:
          v33 = *(_DWORD *)(v25 + 32);
          v34 = (_BYTE *)sub_ACADE0((__int64 **)v25);
          if ( v33 )
          {
            v35 = 0;
            v114 = v33;
            while ( 1 )
            {
              v121 = 257;
              v41 = (_BYTE *)*((_QWORD *)v119 - 8);
              v42 = sub_BCB2E0(v137);
              v43 = (unsigned __int8 *)sub_ACD640(v42, v35, 0);
              v44 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v138 + 12);
              if ( v44 != sub_948070 )
                break;
              if ( *v41 <= 0x15u && *v43 <= 0x15u )
              {
                v45 = (unsigned __int8 *)sub_AD5840((__int64)v41, v43, 0);
                goto LABEL_74;
              }
LABEL_126:
              v124 = 257;
              v81 = (unsigned __int8 *)sub_BD2C40(72, 2u);
              v45 = v81;
              if ( v81 )
                sub_B4DE80((__int64)v81, (__int64)v41, (__int64)v43, (__int64)&v122, 0, 0);
              (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v139 + 2))(
                v139,
                v45,
                v120,
                v135,
                v136);
              v82 = (unsigned __int64)v131;
              v83 = &v131[2 * (unsigned int)v132];
              if ( v131 != v83 )
              {
                do
                {
                  v84 = *(_QWORD *)(v82 + 8);
                  v85 = *(_DWORD *)v82;
                  v82 += 16LL;
                  sub_B99FD0((__int64)v45, v85, v84);
                }
                while ( v83 != (unsigned __int64 *)v82 );
              }
LABEL_75:
              v121 = 257;
              v46 = (_BYTE *)*((_QWORD *)v119 - 4);
              v47 = sub_BCB2E0(v137);
              v48 = (unsigned __int8 *)sub_ACD640(v47, v35, 0);
              v49 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v138 + 12);
              if ( v49 != sub_948070 )
              {
                v112 = v48;
                v86 = v49((__int64)v138, v46, v48);
                v48 = v112;
                v51 = (unsigned __int8 *)v86;
LABEL_79:
                if ( v51 )
                  goto LABEL_80;
                goto LABEL_120;
              }
              if ( *v46 <= 0x15u && *v48 <= 0x15u )
              {
                v108 = v48;
                v50 = sub_AD5840((__int64)v46, v48, 0);
                v48 = v108;
                v51 = (unsigned __int8 *)v50;
                goto LABEL_79;
              }
LABEL_120:
              v110 = (__int64)v48;
              v124 = 257;
              v76 = (unsigned __int8 *)sub_BD2C40(72, 2u);
              v51 = v76;
              if ( v76 )
                sub_B4DE80((__int64)v76, (__int64)v46, v110, (__int64)&v122, 0, 0);
              (*((void (__fastcall **)(void **, unsigned __int8 *, _BYTE *, __int64, __int64))*v139 + 2))(
                v139,
                v51,
                v120,
                v135,
                v136);
              if ( v131 != &v131[2 * (unsigned int)v132] )
              {
                v111 = v45;
                v77 = (unsigned __int64)v131;
                v78 = &v131[2 * (unsigned int)v132];
                do
                {
                  v79 = *(_QWORD *)(v77 + 8);
                  v80 = *(_DWORD *)v77;
                  v77 += 16LL;
                  sub_B99FD0((__int64)v51, v80, v79);
                }
                while ( v78 != (unsigned __int64 *)v77 );
                v45 = v111;
              }
LABEL_80:
              v121 = 257;
              v52 = *v119;
              v53 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v138 + 2);
              v54 = v52 - 29;
              if ( v53 != sub_9202E0 )
              {
                v113 = v52 - 29;
                v87 = v53((__int64)v138, v54, v45, v51);
                v54 = v113;
                v56 = v87;
                goto LABEL_86;
              }
              if ( *v45 <= 0x15u && *v51 <= 0x15u )
              {
                v109 = v52 - 29;
                if ( (unsigned __int8)sub_AC47B0(v54) )
                  v55 = sub_AD5570(v109, (__int64)v45, v51, 0, 0);
                else
                  v55 = sub_AABE40(v109, v45, v51);
                v54 = v109;
                v56 = v55;
LABEL_86:
                if ( v56 )
                  goto LABEL_87;
              }
              v124 = 257;
              v56 = sub_B504D0(v54, (__int64)v45, (__int64)v51, (__int64)&v122, 0, 0);
              if ( *(_BYTE *)v56 > 0x1Cu )
              {
                switch ( *(_BYTE *)v56 )
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
                    goto LABEL_104;
                  case 'T':
                  case 'U':
                  case 'V':
                    v66 = *(_QWORD *)(v56 + 8);
                    v67 = *(unsigned __int8 *)(v66 + 8);
                    v68 = v67 - 17;
                    v74 = *(_BYTE *)(v66 + 8);
                    if ( (unsigned int)(v67 - 17) <= 1 )
                      v74 = *(_BYTE *)(**(_QWORD **)(v66 + 16) + 8LL);
                    if ( v74 <= 3u || v74 == 5 || (v74 & 0xFD) == 4 )
                      goto LABEL_104;
                    if ( (_BYTE)v67 == 15 )
                    {
                      if ( (*(_BYTE *)(v66 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v56 + 8)) )
                        break;
                      v75 = *(__int64 **)(v66 + 16);
                      v66 = *v75;
                      v67 = *(unsigned __int8 *)(*v75 + 8);
                      v68 = v67 - 17;
                    }
                    else if ( (_BYTE)v67 == 16 )
                    {
                      do
                      {
                        v66 = *(_QWORD *)(v66 + 24);
                        LOBYTE(v67) = *(_BYTE *)(v66 + 8);
                      }
                      while ( (_BYTE)v67 == 16 );
                      v68 = (unsigned __int8)v67 - 17;
                    }
                    if ( v68 <= 1 )
                      LOBYTE(v67) = *(_BYTE *)(**(_QWORD **)(v66 + 16) + 8LL);
                    if ( (unsigned __int8)v67 <= 3u || (_BYTE)v67 == 5 || (v67 & 0xFD) == 4 )
                    {
LABEL_104:
                      v69 = v141;
                      if ( v140 )
                        sub_B99FD0(v56, 3u, v140);
                      sub_B45150(v56, v69);
                    }
                    break;
                  default:
                    break;
                }
              }
              (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v139 + 2))(
                v139,
                v56,
                v120,
                v135,
                v136);
              v70 = &v131[2 * (unsigned int)v132];
              if ( v131 != v70 )
              {
                v71 = (unsigned __int64)v131;
                do
                {
                  v72 = *(_QWORD *)(v71 + 8);
                  v73 = *(_DWORD *)v71;
                  v71 += 16LL;
                  sub_B99FD0(v56, v73, v72);
                }
                while ( v70 != (unsigned __int64 *)v71 );
              }
LABEL_87:
              v121 = 257;
              v57 = sub_BCB2E0(v137);
              v58 = (unsigned __int8 *)sub_ACD640(v57, v35, 0);
              v59 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))*((_QWORD *)*v138 + 13);
              if ( v59 == sub_948040 )
              {
                if ( *v34 > 0x15u || *(_BYTE *)v56 > 0x15u || *v58 > 0x15u )
                {
LABEL_89:
                  v124 = 257;
                  v60 = sub_BD2C40(72, 3u);
                  v61 = (__int64)v60;
                  if ( v60 )
                    sub_B4DFA0((__int64)v60, (__int64)v34, v56, (__int64)v58, (__int64)&v122, 0, 0, 0);
                  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v139 + 2))(
                    v139,
                    v61,
                    v120,
                    v135,
                    v136);
                  v62 = (unsigned __int64)v131;
                  v63 = &v131[2 * (unsigned int)v132];
                  if ( v131 != v63 )
                  {
                    do
                    {
                      v64 = *(_QWORD *)(v62 + 8);
                      v65 = *(_DWORD *)v62;
                      v62 += 16LL;
                      sub_B99FD0(v61, v65, v64);
                    }
                    while ( v63 != (unsigned __int64 *)v62 );
                  }
                  v34 = (_BYTE *)v61;
                  goto LABEL_65;
                }
                v36 = sub_AD5A90((__int64)v34, (_BYTE *)v56, v58, 0);
              }
              else
              {
                v36 = v59((__int64)v138, v34, (_BYTE *)v56, v58);
              }
              if ( !v36 )
                goto LABEL_89;
              v34 = (_BYTE *)v36;
LABEL_65:
              if ( (unsigned __int8)(*(_BYTE *)v56 - 42) <= 0x11u )
              {
                sub_B45260((unsigned __int8 *)v56, v56, 1);
                v39 = (unsigned int)v126;
                v40 = (unsigned int)v126 + 1LL;
                if ( v40 > HIDWORD(v126) )
                {
                  sub_C8D5F0((__int64)&v125, v127, v40, 8u, v37, v38);
                  v39 = (unsigned int)v126;
                }
                *(_QWORD *)&v125[8 * v39] = v56;
                LODWORD(v126) = v126 + 1;
              }
              if ( v114 == ++v35 )
                goto LABEL_136;
            }
            v45 = (unsigned __int8 *)v44((__int64)v138, v41, v43);
LABEL_74:
            if ( v45 )
              goto LABEL_75;
            goto LABEL_126;
          }
LABEL_136:
          sub_BD84D0((__int64)v119, (__int64)v34);
          if ( (v119[7] & 0x40) != 0 )
          {
            v88 = (unsigned __int8 *)*((_QWORD *)v119 - 1);
            v89 = &v88[32 * (*((_DWORD *)v119 + 1) & 0x7FFFFFF)];
          }
          else
          {
            v89 = v119;
            v88 = &v119[-32 * (*((_DWORD *)v119 + 1) & 0x7FFFFFF)];
          }
          for ( ; v88 != v89; v88 += 32 )
          {
            if ( *(_QWORD *)v88 )
            {
              v90 = *((_QWORD *)v88 + 1);
              **((_QWORD **)v88 + 2) = v90;
              if ( v90 )
                *(_QWORD *)(v90 + 16) = *((_QWORD *)v88 + 2);
            }
            *(_QWORD *)v88 = 0;
          }
          sub_B43D60(v119);
          nullsub_61();
          v146 = &unk_49DA100;
          nullsub_63();
          if ( v131 != (unsigned __int64 *)v133 )
            _libc_free((unsigned __int64)v131);
          v23 = v129;
        }
        v91 = v126;
        if ( (_DWORD)v126 )
        {
          do
          {
            v92 = *(_BYTE **)&v125[8 * v91 - 8];
            LODWORD(v126) = v91 - 1;
            if ( (unsigned __int8)(*v92 - 48) <= 1u )
              sub_3198EC0();
            else
              sub_3199950();
            v91 = v126;
          }
          while ( (_DWORD)v126 );
        }
        else
        {
LABEL_11:
          v107 = 0;
        }
        if ( v128 != v130 )
          _libc_free((unsigned __int64)v128);
        if ( v125 != v127 )
          _libc_free((unsigned __int64)v125);
        return v107;
      }
    }
    v122 = *(_QWORD *)(v17 + 24);
    if ( !v18 )
      goto LABEL_39;
    v95 = 1LL << ((unsigned __int8)v19 - 1);
LABEL_167:
    v96 = v122;
    if ( (v95 & v122) == 0 )
      goto LABEL_39;
    LODWORD(v132) = v19;
    goto LABEL_169;
  }
  return v107;
}
