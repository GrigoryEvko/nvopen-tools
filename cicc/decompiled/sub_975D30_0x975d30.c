// Function: sub_975D30
// Address: 0x975d30
//
__int64 __fastcall sub_975D30(
        char *a1,
        unsigned __int64 a2,
        __int64 a3,
        _QWORD *a4,
        _QWORD *a5,
        __int64 *a6,
        __int64 a7)
{
  _QWORD *v7; // r13
  unsigned __int8 *v8; // r15
  unsigned __int8 v9; // al
  unsigned __int64 v11; // r10
  unsigned int v12; // r12d
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r8
  unsigned __int64 v19; // r10
  __int64 *v20; // r9
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // rax
  unsigned __int8 *v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rsi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // r14
  char v44; // al
  __int64 *v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdx
  unsigned int v48; // ebx
  _BYTE *v49; // rax
  _BYTE *v50; // r12
  __int64 *v51; // r14
  bool v52; // r12
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned int v55; // r12d
  __int64 v56; // r13
  unsigned int v57; // r13d
  __int64 v58; // r14
  __int64 *v59; // rax
  bool v60; // r12
  __int64 v61; // rdx
  bool v62; // bl
  char v63; // al
  unsigned int v64; // esi
  __int64 *v65; // rdi
  char v66; // al
  int v67; // eax
  unsigned __int16 v68; // ax
  unsigned int v69; // eax
  __int64 v70; // rax
  __int64 v71; // rsi
  __int64 v72; // rsi
  unsigned __int8 v73; // al
  char v74; // al
  __int64 v75; // rcx
  __int64 v76; // rdx
  unsigned __int8 v77; // al
  __int64 v78; // rax
  unsigned __int64 v79; // rax
  unsigned int v80; // edi
  unsigned int v81; // ecx
  __int64 v82; // rax
  unsigned __int8 v83; // al
  char v84; // r12
  __int64 v85; // rcx
  unsigned int v86; // r12d
  __int64 v87; // rdx
  __int64 v88; // rax
  __int64 v89; // rax
  char v90; // r12
  _QWORD *v91; // r12
  __int64 v92; // rcx
  __int64 v93; // rcx
  __int64 v94; // r8
  unsigned __int8 v95; // al
  char v96; // al
  __int64 v97; // rcx
  __int64 v98; // rax
  unsigned int v99; // eax
  _BOOL4 v100; // r12d
  __int64 v101; // rcx
  __int64 v102; // rdx
  __int64 v103; // rax
  unsigned __int16 v104; // ax
  double v105; // xmm0_8
  __int64 *v106; // r12
  __int64 v107; // rax
  _QWORD *v108; // rdi
  unsigned __int64 v109; // r10
  __int64 *v110; // r9
  _BYTE *v111; // r11
  _QWORD *v112; // r12
  size_t v113; // r8
  __int64 v114; // rsi
  char v115; // al
  const void *v116; // rax
  size_t v117; // rdx
  size_t v118; // r14
  const void *v119; // r15
  double v121; // xmm0_8
  double v122; // xmm0_8
  double v123; // xmm0_8
  unsigned int v124; // r14d
  char v125; // al
  __int64 v126; // r8
  __int64 v127; // r9
  int v128; // ebx
  __int64 v129; // rax
  double v130; // xmm0_8
  __int64 v131; // rax
  _QWORD *v132; // r14
  size_t v133; // rdx
  _BYTE *v134; // r14
  size_t v135; // r15
  size_t v136; // rdx
  _BYTE *v137; // r14
  size_t v138; // r15
  __int64 v139; // rdi
  unsigned int v140; // r12d
  unsigned int v141; // esi
  int v142; // eax
  __int64 v143; // rcx
  __int64 v144; // r8
  __int64 v145; // rsi
  __int64 *v146; // [rsp+8h] [rbp-128h]
  unsigned __int64 v147; // [rsp+10h] [rbp-120h]
  __int64 *v148; // [rsp+18h] [rbp-118h]
  unsigned __int64 v149; // [rsp+18h] [rbp-118h]
  __int64 *v150; // [rsp+18h] [rbp-118h]
  size_t v151; // [rsp+18h] [rbp-118h]
  __int64 *v152; // [rsp+18h] [rbp-118h]
  int v154; // [rsp+20h] [rbp-110h]
  unsigned __int64 v155; // [rsp+20h] [rbp-110h]
  char v156; // [rsp+20h] [rbp-110h]
  unsigned __int64 v157; // [rsp+20h] [rbp-110h]
  _BYTE *v158; // [rsp+20h] [rbp-110h]
  unsigned __int64 v159; // [rsp+20h] [rbp-110h]
  unsigned __int64 v160; // [rsp+28h] [rbp-108h]
  _QWORD *v161; // [rsp+28h] [rbp-108h]
  _QWORD *v162; // [rsp+30h] [rbp-100h]
  _QWORD *v163; // [rsp+30h] [rbp-100h]
  __int64 v164; // [rsp+30h] [rbp-100h]
  __int64 v165; // [rsp+30h] [rbp-100h]
  __int64 v166; // [rsp+30h] [rbp-100h]
  __int64 v167; // [rsp+30h] [rbp-100h]
  __int64 v168; // [rsp+30h] [rbp-100h]
  __int64 v169; // [rsp+30h] [rbp-100h]
  __int64 v170; // [rsp+30h] [rbp-100h]
  __int64 v171; // [rsp+30h] [rbp-100h]
  __int64 v172; // [rsp+30h] [rbp-100h]
  __int64 v173; // [rsp+30h] [rbp-100h]
  __int64 v174; // [rsp+30h] [rbp-100h]
  double v175; // [rsp+30h] [rbp-100h]
  __int64 v176; // [rsp+38h] [rbp-F8h]
  _QWORD *v177; // [rsp+38h] [rbp-F8h]
  _QWORD *v178; // [rsp+38h] [rbp-F8h]
  unsigned int v179; // [rsp+38h] [rbp-F8h]
  _QWORD *v180; // [rsp+38h] [rbp-F8h]
  _QWORD *v181; // [rsp+38h] [rbp-F8h]
  __int64 v182; // [rsp+38h] [rbp-F8h]
  _QWORD *v183; // [rsp+38h] [rbp-F8h]
  _QWORD *v184; // [rsp+38h] [rbp-F8h]
  char v185; // [rsp+38h] [rbp-F8h]
  __int64 v186; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v187; // [rsp+48h] [rbp-E8h]
  unsigned __int8 v188; // [rsp+54h] [rbp-DCh]
  _QWORD v189[4]; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v190[4]; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v191; // [rsp+A0h] [rbp-90h] BYREF
  unsigned int v192; // [rsp+A8h] [rbp-88h]
  bool v193; // [rsp+ACh] [rbp-84h]
  __int64 v194; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v195; // [rsp+C8h] [rbp-68h]
  _QWORD dest[2]; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v197; // [rsp+E0h] [rbp-50h]
  __int64 v198; // [rsp+E8h] [rbp-48h]
  __int64 v199; // [rsp+F0h] [rbp-40h]

  v7 = a4;
  v8 = (unsigned __int8 *)*a5;
  if ( (_DWORD)a3 != 206 )
  {
    v9 = *v8;
    v11 = a2;
    v12 = a3;
    LOBYTE(a2) = (_DWORD)a3 == 20;
    v13 = *v8;
    if ( *v8 == 13 && (_DWORD)a3 == 20 )
      return sub_ACADE0(a4);
    v14 = (unsigned int)v9 - 12;
    if ( (unsigned int)v14 > 1 )
    {
      if ( v9 == 20 )
      {
        if ( v12 == 208 || v12 == 346 )
        {
          v177 = a5;
          if ( *(_QWORD *)(a7 + 40) )
          {
            v34 = sub_B491C0(a7, a2);
            if ( v34 )
            {
              v35 = *(_QWORD *)(*v177 + 8LL);
              if ( (unsigned int)*(unsigned __int8 *)(v35 + 8) - 17 <= 1 )
                v35 = **(_QWORD **)(v35 + 16);
              if ( !(unsigned __int8)sub_B2F070(v34, *(_DWORD *)(v35 + 8) >> 8) )
                return *v177;
            }
          }
          return 0;
        }
        goto LABEL_12;
      }
    }
    else
    {
      if ( v12 == 63 || v12 == 66 || v12 - 175 <= 1 || (_BYTE)a2 )
        return sub_AD6530(a4);
      LOBYTE(a2) = v12 == 346 || v12 == 208;
      if ( (_BYTE)a2 )
        return (__int64)v8;
      v14 = v12 - 14;
      if ( (unsigned int)v14 <= 1 )
        return (__int64)v8;
      if ( v9 == 20 )
      {
LABEL_12:
        if ( v12 > 0x184 )
        {
          v14 = v12 - 395;
          if ( (unsigned int)v14 > 6 )
            goto LABEL_55;
        }
        else if ( v12 <= 0x182 )
        {
          return 0;
        }
        v43 = *((_QWORD *)v8 + 1);
        LOBYTE(v13) = *v8;
        if ( *(_BYTE *)(v43 + 8) == 17 )
        {
          if ( v9 == 14 )
          {
            v184 = a5;
            v89 = sub_AD64C0(*(_QWORD *)(v43 + 24), 0, 0);
            a5 = v184;
            v8 = (unsigned __int8 *)v89;
            goto LABEL_112;
          }
          if ( v9 == 13 || (v178 = a5, v44 = sub_AD6C60(v8, a2, v14, a7), a5 = v178, v44) )
          {
            v183 = a5;
            v82 = sub_ACADE0(*(_QWORD *)(v43 + 24));
            a5 = v183;
            v8 = (unsigned __int8 *)v82;
            goto LABEL_112;
          }
          if ( *v8 != 11 && *v8 != 16 || (v45 = 0, v46 = sub_AD69F0(v8, 0), a5 = v178, *(_BYTE *)v46 != 17) )
          {
LABEL_113:
            v8 = (unsigned __int8 *)*a5;
            LOBYTE(v13) = *(_BYTE *)*a5;
            goto LABEL_55;
          }
          v47 = *(unsigned int *)(v46 + 32);
          v192 = v47;
          if ( (unsigned int)v47 > 0x40 )
          {
            v45 = (__int64 *)(v46 + 24);
            sub_C43780(&v191, v46 + 24);
            a5 = v178;
          }
          else
          {
            v191 = *(_QWORD *)(v46 + 24);
          }
          v48 = 1;
          v154 = *(_DWORD *)(v43 + 32);
          if ( v154 != 1 )
          {
            v179 = v12;
            v163 = v7;
            v161 = a5;
            while ( 2 )
            {
              v45 = (__int64 *)v48;
              v49 = (_BYTE *)sub_AD69F0(v8, v48);
              v50 = v49;
              if ( *v49 != 17 )
              {
                v12 = v179;
                v7 = v163;
                v8 = 0;
                a5 = v161;
                goto LABEL_109;
              }
              v51 = (__int64 *)(v49 + 24);
              v47 = v179 - 387;
              switch ( v179 )
              {
                case 0x183u:
                  LODWORD(v195) = v192;
                  if ( v192 > 0x40 )
                    sub_C43780(&v194, &v191);
                  else
                    v194 = v191;
                  v45 = v51;
                  sub_C45EE0(&v194, v51);
                  goto LABEL_117;
                case 0x184u:
                  LODWORD(v195) = v192;
                  if ( v192 <= 0x40 )
                  {
                    v47 = v191;
                    v194 = v191;
LABEL_129:
                    v194 &= *((_QWORD *)v50 + 3);
                    goto LABEL_117;
                  }
                  v45 = &v191;
                  sub_C43780(&v194, &v191);
                  if ( (unsigned int)v195 <= 0x40 )
                    goto LABEL_129;
                  v45 = v51;
                  sub_C43B90(&v194, v51);
LABEL_117:
                  v55 = v195;
                  LODWORD(v195) = 0;
                  v56 = v194;
                  if ( v192 > 0x40 && v191 )
                    j_j___libc_free_0_0(v191);
                  v191 = v56;
                  v192 = v55;
                  if ( (unsigned int)v195 > 0x40 )
                  {
LABEL_121:
                    if ( v194 )
                      j_j___libc_free_0_0(v194);
                  }
LABEL_106:
                  if ( v154 != ++v48 )
                    continue;
                  v12 = v179;
                  v7 = v163;
                  a5 = v161;
                  break;
                case 0x18Bu:
                  v45 = &v191;
                  sub_C472A0(&v194, &v191, v51);
                  if ( v192 > 0x40 && v191 )
                    j_j___libc_free_0_0(v191);
                  v191 = v194;
                  v192 = v195;
                  goto LABEL_106;
                case 0x18Cu:
                  LODWORD(v195) = v192;
                  if ( v192 <= 0x40 )
                  {
                    v47 = v191;
                    v194 = v191;
LABEL_116:
                    v194 |= *((_QWORD *)v50 + 3);
                    goto LABEL_117;
                  }
                  v45 = &v191;
                  sub_C43780(&v194, &v191);
                  if ( (unsigned int)v195 <= 0x40 )
                    goto LABEL_116;
                  v45 = v51;
                  sub_C43BD0(&v194, v51);
                  goto LABEL_117;
                case 0x18Du:
                  v45 = (__int64 *)(v49 + 24);
                  if ( (int)sub_C4C880(&v191, v51) > 0 )
                    v51 = &v191;
                  if ( v192 > 0x40 )
                    goto LABEL_105;
                  goto LABEL_104;
                case 0x18Eu:
                  v45 = (__int64 *)(v49 + 24);
                  if ( (int)sub_C4C880(&v191, v51) < 0 )
                    v51 = &v191;
                  if ( v192 > 0x40 )
                    goto LABEL_105;
                  goto LABEL_104;
                case 0x18Fu:
                  v45 = (__int64 *)(v49 + 24);
                  if ( (int)sub_C49970(&v191, v51) > 0 )
                    v51 = &v191;
                  if ( v192 > 0x40 )
                    goto LABEL_105;
                  goto LABEL_104;
                case 0x190u:
                  v45 = (__int64 *)(v49 + 24);
                  if ( (int)sub_C49970(&v191, v51) < 0 )
                    v51 = &v191;
                  if ( v192 > 0x40 )
                    goto LABEL_105;
LABEL_104:
                  if ( *((_DWORD *)v51 + 2) <= 0x40u )
                  {
                    v47 = *v51;
                    v192 = *((_DWORD *)v51 + 2);
                    v191 = v47;
                  }
                  else
                  {
LABEL_105:
                    v45 = v51;
                    sub_C43990(&v191, v51);
                  }
                  goto LABEL_106;
                case 0x191u:
                  LODWORD(v195) = v192;
                  if ( v192 <= 0x40 )
                  {
                    v47 = v191;
                    v194 = v191;
LABEL_135:
                    v194 ^= *((_QWORD *)v50 + 3);
                    goto LABEL_136;
                  }
                  v45 = &v191;
                  sub_C43780(&v194, &v191);
                  if ( (unsigned int)v195 <= 0x40 )
                    goto LABEL_135;
                  v45 = v51;
                  sub_C43C10(&v194, v51);
LABEL_136:
                  v57 = v195;
                  LODWORD(v195) = 0;
                  v58 = v194;
                  if ( v192 > 0x40 && v191 )
                    j_j___libc_free_0_0(v191);
                  v191 = v58;
                  v192 = v57;
                  if ( (unsigned int)v195 > 0x40 )
                    goto LABEL_121;
                  goto LABEL_106;
                default:
                  goto LABEL_106;
              }
              break;
            }
          }
          v180 = a5;
          v53 = sub_BD5C60(v8, v45, v47);
          v54 = sub_ACCFD0(v53, &v191);
          a5 = v180;
          v8 = (unsigned __int8 *)v54;
LABEL_109:
          if ( v192 > 0x40 && v191 )
          {
            v181 = a5;
            j_j___libc_free_0_0(v191);
            a5 = v181;
          }
LABEL_112:
          if ( v8 )
            return (__int64)v8;
          goto LABEL_113;
        }
LABEL_55:
        if ( (_BYTE)v13 != 16 && (_BYTE)v13 != 11 )
          return 0;
        if ( v12 > 0x3D71 )
        {
          if ( v12 - 15733 > 1 )
            return 0;
          goto LABEL_59;
        }
        if ( v12 <= 0x3D6F )
        {
          if ( v12 > 0x3D4C )
          {
            if ( v12 - 15695 > 1 )
              return 0;
LABEL_59:
            v36 = sub_AD69F0(v8, 0);
            if ( !v36 || *(_BYTE *)v36 != 18 )
              return 0;
            return sub_968DB0((_QWORD *)(v36 + 24), 1, (__int64)v7, 1u);
          }
          if ( v12 <= 0x3D4A )
            return 0;
        }
        v70 = sub_AD69F0(v8, 0);
        if ( !v70 || *(_BYTE *)v70 != 18 )
          return 0;
        return sub_968DB0((_QWORD *)(v70 + 24), 0, (__int64)v7, 1u);
      }
    }
    v160 = v11;
    if ( v9 == 18 )
    {
      v16 = *((_QWORD *)v8 + 3);
      v162 = v8 + 24;
      v17 = sub_C33340(v13, a2, v14, a7, a5);
      v176 = v17;
      if ( v12 != 25 )
      {
        if ( v16 == v17 )
        {
          sub_C3C790(&v186, v162);
          v21 = a7;
          v20 = a6;
          v19 = v160;
        }
        else
        {
          sub_C33EB0(&v186, v162);
          v19 = v160;
          v20 = a6;
          v21 = a7;
        }
        if ( v12 - 14255 <= 1 )
        {
          v59 = &v186;
          if ( v186 == v176 )
            v59 = (__int64 *)v187;
          v8 = 0;
          if ( (*((_BYTE *)v59 + 20) & 7) == 1 )
            goto LABEL_26;
          LODWORD(v195) = *((_DWORD *)v7 + 2) >> 8;
          v60 = v12 != 14255;
          if ( (unsigned int)v195 > 0x40 )
            sub_C43690(&v194, 0, 0);
          else
            v194 = 0;
          BYTE4(v195) = v60;
          LOBYTE(v191) = 0;
          v8 = 0;
          if ( (sub_C41980(&v186, &v194, 0, &v191, v18, v20) & 0xFFFFFFEF) != 0 )
            goto LABEL_97;
        }
        else
        {
          if ( v12 - 175 > 1 )
          {
            if ( v12 != 20 )
            {
              v22 = *((unsigned __int8 *)v7 + 8);
              if ( (unsigned __int8)v22 > 0xCu )
                goto LABEL_25;
              v61 = 4109;
              if ( !_bittest64(&v61, v22) )
                goto LABEL_25;
              LOBYTE(v61) = v12 == 308;
              v62 = v12 == 308 || v12 == 250;
              if ( !v62 )
              {
                if ( v12 == 309 )
                {
                  v71 = 4;
                  if ( v176 != v186 )
                    goto LABEL_211;
                  goto LABEL_233;
                }
                if ( v12 != 310 )
                {
                  if ( v12 != 21 )
                  {
                    if ( v12 != 172 )
                    {
                      if ( v12 != 355 )
                      {
                        if ( v12 == 170 )
                        {
LABEL_292:
                          sub_9695A0((__int64)&v186);
                          v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v186);
                          goto LABEL_26;
                        }
                        if ( v12 == 2209 )
                        {
                          sub_9693D0((__int64)v189, &v186);
                          sub_969520(v189, 3u);
                          sub_9693D0((__int64)v190, &v186);
                          if ( v176 == v190[0] )
                            sub_C3D820(v190, v189, 1, v92);
                          else
                            sub_C3B1F0(v190, v189, 1, v92);
                          v194 = 1;
                          sub_975590((__int64)&v191, v186, &v194, v93, v94);
                          if ( v176 == v191 )
                            sub_C3FCD0(&v191, 1);
                          else
                            sub_C36AF0(&v191, 1);
                          sub_969B00(&v194, v190, &v191);
                          v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v194);
                          sub_91D830(&v194);
                          sub_91D830(&v191);
                          sub_91D830(v190);
                          sub_91D830(v189);
                          goto LABEL_26;
                        }
                        switch ( v12 )
                        {
                          case 0x61u:
                            v64 = 2;
                            goto LABEL_187;
                          case 0x6Au:
                            v64 = 3;
                            goto LABEL_187;
                          case 0x80u:
                          case 0x83u:
                            v164 = v21;
                            v69 = sub_B59DB0(v21);
                            v64 = v69;
                            if ( BYTE1(v69) != 1 )
                              goto LABEL_25;
                            v21 = v164;
                            if ( (_BYTE)v69 == 7 )
                              goto LABEL_25;
                            goto LABEL_187;
                          case 0x84u:
                            v64 = 4;
                            goto LABEL_187;
                          case 0x8Cu:
                            v64 = 0;
LABEL_187:
                            if ( v176 == v186 )
                            {
                              v65 = (__int64 *)v187;
                              v66 = *(_BYTE *)(v187 + 20) & 7;
                              if ( v66 == 1 )
                                goto LABEL_323;
                            }
                            else
                            {
                              v65 = &v186;
                              v66 = v188 & 7;
                              if ( (v188 & 7) == 1 )
                                goto LABEL_323;
                            }
                            if ( v66 )
                            {
                              v182 = v21;
                              v67 = sub_969520(&v186, v64);
                              if ( v12 == 131 && v67 == 16 )
                              {
                                v68 = sub_B59EF0(v182);
                                if ( HIBYTE(v68) )
                                {
                                  if ( (_BYTE)v68 == 2 )
                                    goto LABEL_25;
                                }
                              }
                              goto LABEL_586;
                            }
LABEL_323:
                            v173 = v21;
                            if ( !(unsigned __int8)sub_C35FD0(v65) )
                              goto LABEL_586;
                            v104 = sub_B59EF0(v173);
                            if ( !HIBYTE(v104) || !(_BYTE)v104 )
                            {
                              if ( v186 == v176 )
                                sub_C3C500(&v194, v176, 0);
                              else
                                sub_C373C0(&v194, v186, 0);
                              if ( v176 == v194 )
                                sub_C3D480(&v194, 0, 0, 0);
                              else
                                sub_C36070(&v194, 0, 0, 0);
                              sub_96AAC0(&v186, &v194);
                              sub_91D830(&v194);
                              goto LABEL_586;
                            }
                            goto LABEL_25;
                          default:
                            if ( v12 > 0x2171 )
                            {
                              if ( v12 - 8572 > 0xF )
                              {
LABEL_174:
                                if ( v176 == v186 )
                                {
                                  v63 = *(_BYTE *)(v187 + 20) & 7;
                                  if ( v63 == 1 )
                                    goto LABEL_25;
                                }
                                else
                                {
                                  v63 = v188 & 7;
                                  if ( (v188 & 7) == 1 )
                                    goto LABEL_25;
                                }
                                if ( !v63 )
                                  goto LABEL_25;
                                if ( v12 == 8529 )
                                {
                                  v8 = (unsigned __int8 *)sub_96A790((double (__fastcall *)(double))&exp2, v162, v7);
                                  goto LABEL_26;
                                }
                                if ( v12 <= 0x2151 )
                                {
                                  if ( v12 != 325 )
                                  {
                                    if ( v12 <= 0x145 )
                                    {
                                      if ( v12 == 90 )
                                      {
                                        v105 = 2.0;
                                        v106 = &v194;
                                        goto LABEL_337;
                                      }
                                      if ( v12 > 0x5A )
                                      {
                                        switch ( v12 )
                                        {
                                          case 0xDBu:
LABEL_596:
                                            v8 = (unsigned __int8 *)sub_96A6F0(
                                                                      (double (__fastcall *)(double))&log10,
                                                                      (__int64)v162,
                                                                      v7);
                                            goto LABEL_26;
                                          case 0xDCu:
                                            goto LABEL_592;
                                          case 0xDAu:
LABEL_333:
                                            v8 = (unsigned __int8 *)sub_96A6F0(
                                                                      (double (__fastcall *)(double))&log,
                                                                      (__int64)v162,
                                                                      v7);
                                            goto LABEL_26;
                                        }
LABEL_343:
                                        v148 = v20;
                                        v155 = v19;
                                        v107 = sub_B43CA0(v21);
                                        v108 = dest;
                                        v109 = v155;
                                        v110 = v148;
                                        v194 = (__int64)dest;
                                        v111 = *(_BYTE **)(v107 + 232);
                                        v112 = (_QWORD *)v107;
                                        v113 = *(_QWORD *)(v107 + 240);
                                        if ( &v111[v113] && !v111 )
                                          sub_426248((__int64)"basic_string::_M_construct null not valid");
                                        v191 = *(_QWORD *)(v107 + 240);
                                        if ( v113 > 0xF )
                                        {
                                          v146 = v148;
                                          v147 = v155;
                                          v151 = v113;
                                          v158 = v111;
                                          v129 = sub_22409D0(&v194, &v191, 0);
                                          v111 = v158;
                                          v113 = v151;
                                          v194 = v129;
                                          v108 = (_QWORD *)v129;
                                          v109 = v147;
                                          v110 = v146;
                                          dest[0] = v191;
                                        }
                                        else
                                        {
                                          if ( v113 == 1 )
                                          {
                                            LOBYTE(dest[0]) = *v111;
LABEL_348:
                                            v195 = v191;
                                            *(_BYTE *)(v194 + v191) = 0;
                                            v197 = v112[33];
                                            v198 = v112[34];
                                            v199 = v112[35];
                                            if ( (unsigned int)(v197 - 42) <= 1 )
                                            {
                                              v149 = v109;
                                              sub_2240A30(&v194);
                                              v156 = *a1;
                                              if ( *a1 == 95 && v149 > 2 && a1[1] == 95 )
                                                v156 = a1[2];
                                              v114 = *((_QWORD *)v8 + 3);
                                              v106 = &v191;
                                              if ( v114 == v176 )
                                                sub_C3C500(&v191, v176, 0);
                                              else
                                                sub_C373C0(&v191, v114, 0);
                                              if ( v176 == v191 )
                                                sub_C3CEB0(&v191, 0);
                                              else
                                                sub_C37310(&v191, 0);
                                              if ( v156 == 95 && a1[1] == 90 && v149 > 6 )
                                              {
                                                v115 = a1[2];
                                                switch ( v115 )
                                                {
                                                  case '3':
                                                    v137 = (_BYTE *)sub_968F90((__int64)a1, v149, 3u);
                                                    v138 = v136;
                                                    switch ( *v137 )
                                                    {
                                                      case 'c':
                                                        if ( !sub_9691B0(v137, v136, "cosf", 4)
                                                          && !sub_9691B0(v137, v138, "cosd", 4) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&cos,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        break;
                                                      case 'e':
                                                        if ( !sub_9691B0(v137, v136, "expf", 4)
                                                          && !sub_9691B0(v137, v138, "expd", 4) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&exp,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        break;
                                                      case 'l':
                                                        if ( !sub_9691B0(v137, v136, "logf", 4)
                                                          && !sub_9691B0(v137, v138, "logd", 4) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&log,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        break;
                                                      case 's':
                                                        if ( !sub_9691B0(v137, v136, "sinf", 4)
                                                          && !sub_9691B0(v137, v138, "sind", 4) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&sin,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        break;
                                                      case 't':
                                                        if ( !sub_9691B0(v137, v136, "tanf", 4)
                                                          && !sub_9691B0(v137, v138, "tand", 4) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&tan,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        break;
                                                      default:
                                                        goto LABEL_502;
                                                    }
                                                    goto LABEL_338;
                                                  case '4':
                                                    v134 = (_BYTE *)sub_968F90((__int64)a1, v149, 3u);
                                                    v135 = v133;
                                                    switch ( *v134 )
                                                    {
                                                      case 'a':
                                                        if ( sub_9691B0(v134, v133, "acosf", 5)
                                                          || sub_9691B0(v134, v135, "acosd", 5) )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&acos,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        if ( sub_9691B0(v134, v135, "asinf", 5)
                                                          || sub_9691B0(v134, v135, "asind", 5) )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&asin,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        if ( sub_9691B0(v134, v135, "atanf", 5)
                                                          || sub_9691B0(v134, v135, "atand", 5) )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&atan,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        goto LABEL_502;
                                                      case 'c':
                                                        if ( sub_9691B0(v134, v133, "ceilf", 5)
                                                          || sub_9691B0(v134, v135, "ceild", 5) )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&ceil,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        if ( sub_9691B0(v134, v135, "coshf", 5)
                                                          || sub_9691B0(v134, v135, "coshd", 5) )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&cosh,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        goto LABEL_502;
                                                      case 'e':
                                                        if ( !sub_9691B0(v134, v133, "exp2f", 5)
                                                          && !sub_9691B0(v134, v135, "exp2d", 5) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        sub_969410((__int64)&v194, 2.0);
                                                        v8 = (unsigned __int8 *)sub_96A630(
                                                                                  (double (__fastcall *)(double, double))&pow,
                                                                                  (__int64)&v194,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        sub_91D830(&v194);
                                                        goto LABEL_338;
                                                      case 'f':
                                                        if ( !sub_9691B0(v134, v133, "fabsf", 5)
                                                          && !sub_9691B0(v134, v135, "fabsd", 5) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&fabs,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        goto LABEL_338;
                                                      case 's':
                                                        if ( sub_9691B0(v134, v133, "sinhf", 5)
                                                          || sub_9691B0(v134, v135, "sinhd", 5) )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&sinh,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        if ( (sub_9691B0(v134, v135, "sqrtf", 5)
                                                           || sub_9691B0(v134, v135, "sqrtd", 5))
                                                          && (unsigned int)sub_969600(v162) - 1 <= 1 )
                                                        {
                                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                    (double (__fastcall *)(double))&sqrt,
                                                                                    (__int64)v162,
                                                                                    v7);
                                                          goto LABEL_338;
                                                        }
                                                        break;
                                                      case 't':
                                                        if ( !sub_9691B0(v134, v133, "tanhf", 5)
                                                          && !sub_9691B0(v134, v135, "tanhd", 5) )
                                                        {
                                                          goto LABEL_502;
                                                        }
                                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                  (double (__fastcall *)(double))&tanh,
                                                                                  (__int64)v162,
                                                                                  v7);
                                                        goto LABEL_338;
                                                      default:
                                                        goto LABEL_502;
                                                    }
                                                    break;
                                                  case '5':
                                                    v116 = (const void *)sub_968F90((__int64)a1, v149, 3u);
                                                    v118 = v117;
                                                    v119 = v116;
                                                    if ( sub_9691B0(v116, v117, "floorf", 6)
                                                      || sub_9691B0(v119, v118, "floord", 6) )
                                                    {
                                                      v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                (double (__fastcall *)(double))&floor,
                                                                                (__int64)v162,
                                                                                v7);
                                                      goto LABEL_338;
                                                    }
                                                    if ( sub_9691B0(v119, v118, "log10f", 6)
                                                      || sub_9691B0(v119, v118, "log10d", 6) )
                                                    {
                                                      v8 = (unsigned __int8 *)sub_96A6F0(
                                                                                (double (__fastcall *)(double))&log10,
                                                                                (__int64)v162,
                                                                                v7);
                                                      goto LABEL_338;
                                                    }
                                                    break;
                                                }
                                              }
LABEL_502:
                                              v8 = 0;
                                              sub_91D830(&v191);
                                              goto LABEL_26;
                                            }
                                            v152 = v110;
                                            v159 = v109;
                                            sub_2240A30(&v194);
                                            if ( v152 )
                                            {
                                              v139 = *v152;
                                              LODWORD(v190[0]) = 524;
                                              if ( (unsigned __int8)sub_980AF0(v139, a1, v159, v190) )
                                              {
                                                v140 = v190[0];
                                                switch ( LODWORD(v190[0]) )
                                                {
                                                  case 0x42:
                                                  case 0x43:
                                                  case 0xA0:
                                                  case 0xA1:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&acos,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0x48:
                                                  case 0x49:
                                                  case 0xA7:
                                                  case 0xA8:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&asin,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0x53:
                                                  case 0x54:
                                                  case 0xD0:
                                                  case 0xD1:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&cosh,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0x65:
                                                  case 0x66:
                                                  case 0xE7:
                                                  case 0xE8:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    sub_969410((__int64)&v194, 2.0);
                                                    v8 = (unsigned __int8 *)sub_96A630(
                                                                              (double (__fastcall *)(double, double))&pow,
                                                                              (__int64)&v194,
                                                                              (__int64)v162,
                                                                              v7);
                                                    sub_91D830(&v194);
                                                    goto LABEL_26;
                                                  case 0x68:
                                                  case 0x69:
                                                  case 0xE3:
                                                  case 0xEA:
                                                    if ( (unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_563;
                                                    goto LABEL_25;
                                                  case 0x6F:
                                                  case 0x70:
                                                  case 0x14E:
                                                  case 0x14F:
                                                    if ( !sub_9696D0(v162)
                                                      && !sub_969640(v162)
                                                      && (unsigned int)sub_96DFC0(v152, v140) )
                                                    {
                                                      goto LABEL_596;
                                                    }
                                                    goto LABEL_25;
                                                  case 0x72:
                                                  case 0x73:
                                                  case 0x154:
                                                  case 0x155:
                                                    if ( !sub_9696D0(v162)
                                                      && !sub_969640(v162)
                                                      && (unsigned int)sub_96DFC0(v152, v140) )
                                                    {
                                                      goto LABEL_592;
                                                    }
                                                    goto LABEL_25;
                                                  case 0x75:
                                                  case 0x76:
                                                  case 0x14D:
                                                  case 0x15D:
                                                    if ( !sub_9696D0(v162)
                                                      && !sub_969640(v162)
                                                      && (unsigned int)sub_96DFC0(v152, v140) )
                                                    {
                                                      goto LABEL_333;
                                                    }
                                                    goto LABEL_25;
                                                  case 0x83:
                                                  case 0x84:
                                                  case 0x1B6:
                                                  case 0x1B7:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&sinh,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0xAD:
                                                  case 0xB1:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&atan,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0xC4:
                                                  case 0xC5:
                                                    v141 = 2;
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    goto LABEL_574;
                                                  case 0xCE:
                                                  case 0xCF:
                                                    if ( (unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_380;
                                                    goto LABEL_25;
                                                  case 0xD5:
                                                  case 0xD6:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&erf,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0xEF:
                                                  case 0xF0:
                                                    if ( (unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_292;
                                                    goto LABEL_25;
                                                  case 0x102:
                                                  case 0x103:
                                                    v141 = 3;
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    goto LABEL_574;
                                                  case 0x151:
                                                  case 0x152:
                                                    if ( sub_969640(&v186) )
                                                      goto LABEL_586;
                                                    v145 = *((_QWORD *)v8 + 3);
                                                    v191 = 1;
                                                    sub_975590((__int64)&v194, v145, &v191, v143, v144);
                                                    sub_969560(&v194);
                                                    if ( (unsigned int)sub_969600(v162) != 2
                                                      || !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                    {
                                                      sub_91D830(&v194);
                                                      goto LABEL_25;
                                                    }
                                                    sub_91D830(&v194);
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&log1p,
                                                                              (__int64)v162,
                                                                              v7);
                                                    break;
                                                  case 0x157:
                                                  case 0x158:
                                                    if ( sub_969640(v162) || !(unsigned int)sub_96DFC0(v152, v140) )
                                                      goto LABEL_25;
                                                    if ( v176 == *((_QWORD *)v8 + 3) )
                                                      v162 = (_QWORD *)*((_QWORD *)v8 + 4);
                                                    v142 = sub_C3BD20(v162);
                                                    v8 = (unsigned __int8 *)sub_AD64C0(v7, v142, 1);
                                                    goto LABEL_26;
                                                  case 0x15A:
                                                  case 0x15B:
                                                    if ( sub_969640(v162) || !(unsigned int)sub_96DFC0(v152, v140) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&logb,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0x176:
                                                  case 0x177:
                                                  case 0x1A0:
                                                  case 0x1A1:
                                                    v141 = 1;
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    goto LABEL_574;
                                                  case 0x1A4:
                                                  case 0x1A8:
                                                    v141 = 4;
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    goto LABEL_574;
                                                  case 0x1B4:
                                                  case 0x1B5:
                                                    if ( (unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_416;
                                                    goto LABEL_25;
                                                  case 0x1C0:
                                                  case 0x1C1:
                                                    if ( !sub_9696D0(v162) && (unsigned int)sub_96DFC0(v152, v140) )
                                                      goto LABEL_570;
                                                    goto LABEL_25;
                                                  case 0x1EA:
                                                  case 0x1EB:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&tan,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0x1EC:
                                                  case 0x1ED:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v8 = (unsigned __int8 *)sub_96A6F0(
                                                                              (double (__fastcall *)(double))&tanh,
                                                                              (__int64)v162,
                                                                              v7);
                                                    goto LABEL_26;
                                                  case 0x1F4:
                                                  case 0x1F5:
                                                    if ( !(unsigned int)sub_96DFC0(v152, v190[0]) )
                                                      goto LABEL_25;
                                                    v141 = 0;
LABEL_574:
                                                    sub_969520(&v186, v141);
                                                    v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v186);
                                                    goto LABEL_26;
                                                  default:
                                                    goto LABEL_25;
                                                }
                                                goto LABEL_26;
                                              }
                                            }
                                            goto LABEL_25;
                                          }
                                          if ( !v113 )
                                            goto LABEL_348;
                                        }
                                        v150 = v110;
                                        v157 = v109;
                                        memcpy(v108, v111, v113);
                                        v110 = v150;
                                        v109 = v157;
                                        goto LABEL_348;
                                      }
                                      if ( v12 == 88 )
                                      {
LABEL_563:
                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                  (double (__fastcall *)(double))&exp,
                                                                  (__int64)v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      if ( v12 == 89 )
                                      {
                                        v105 = 10.0;
                                        v106 = &v194;
LABEL_337:
                                        sub_969410((__int64)&v194, v105);
                                        v8 = (unsigned __int8 *)sub_96A630(
                                                                  (double (__fastcall *)(double, double))&pow,
                                                                  (__int64)&v194,
                                                                  (__int64)v162,
                                                                  v7);
LABEL_338:
                                        sub_91D830(v106);
                                        goto LABEL_26;
                                      }
                                      if ( v12 != 63 )
                                        goto LABEL_343;
LABEL_380:
                                      v8 = (unsigned __int8 *)sub_96A6F0(
                                                                (double (__fastcall *)(double))&cos,
                                                                (__int64)v162,
                                                                v7);
                                      goto LABEL_26;
                                    }
                                    if ( v12 == 8308 )
                                      goto LABEL_380;
                                    if ( v12 > 0x2074 )
                                    {
                                      if ( v12 == 8310 )
                                      {
                                        v8 = (unsigned __int8 *)sub_96A790(
                                                                  (double (__fastcall *)(double))&cos,
                                                                  v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      if ( v12 - 8524 <= 1 )
                                      {
                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                  (double (__fastcall *)(double))&exp2,
                                                                  (__int64)v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      goto LABEL_343;
                                    }
                                    if ( v12 != 3198 )
                                    {
                                      if ( v12 > 0xC7E )
                                      {
                                        if ( v12 - 8289 <= 1 )
                                        {
                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                    (double (__fastcall *)(double))&ceil,
                                                                    (__int64)v162,
                                                                    v7);
                                          goto LABEL_26;
                                        }
                                        goto LABEL_343;
                                      }
                                      if ( v12 == 335 )
                                      {
LABEL_570:
                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                  (double (__fastcall *)(double))&sqrt,
                                                                  (__int64)v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      if ( v12 != 2087 )
                                        goto LABEL_343;
                                    }
                                    if ( *(_BYTE *)(*((_QWORD *)v8 + 1) + 8LL) > 3u )
                                    {
                                      if ( v176 == *((_QWORD *)v8 + 3) )
                                        sub_C3C790(&v194, v162);
                                      else
                                        sub_C33EB0(&v194, v162);
                                      v131 = sub_C33320(&v194);
                                      sub_C41640(&v194, v131, 1, &v191);
                                      v175 = sub_C41B00(&v194);
                                      if ( v176 == v194 )
                                      {
                                        if ( v195 )
                                        {
                                          v132 = (_QWORD *)(24LL * *(_QWORD *)(v195 - 8) + v195);
                                          while ( (_QWORD *)v195 != v132 )
                                          {
                                            v132 -= 3;
                                            if ( v176 == *v132 )
                                              sub_969EE0((__int64)v132);
                                            else
                                              sub_C338F0(v132);
                                          }
                                          j_j_j___libc_free_0_0(v132 - 1);
                                        }
                                      }
                                      else
                                      {
                                        sub_C338F0(&v194);
                                      }
                                    }
                                    else
                                    {
                                      v175 = sub_C41B00(v162);
                                    }
                                    if ( v175 >= -256.0 && v175 <= 256.0 )
                                    {
                                      if ( 4.0 * v175 == floor(4.0 * v175) )
                                      {
                                        v194 = 0;
                                        dest[0] = 0;
                                        v195 = 0x3FF0000000000000LL;
                                        dest[1] = 0xBFF0000000000000LL;
                                        v121 = *((double *)&v194
                                               + (((v12 == 2087) + (unsigned __int8)(int)(4.0 * v175)) & 3));
                                      }
                                      else
                                      {
                                        v130 = (v175 + v175) * 3.141592653589793;
                                        if ( v12 == 2087 )
                                          v121 = cos(v130);
                                        else
                                          v121 = sin(v130);
                                      }
                                      v8 = (unsigned __int8 *)sub_96A450(v7, v121);
                                      goto LABEL_26;
                                    }
                                    goto LABEL_25;
                                  }
LABEL_416:
                                  v8 = (unsigned __int8 *)sub_96A6F0(
                                                            (double (__fastcall *)(double))&sin,
                                                            (__int64)v162,
                                                            v7);
                                  goto LABEL_26;
                                }
                                if ( v12 == 9410 )
                                {
                                  sub_9693D0((__int64)&v194, v162);
                                }
                                else
                                {
                                  if ( v12 <= 0x24C2 )
                                  {
                                    if ( v12 <= 0x2304 )
                                    {
                                      if ( v12 > 0x2302 )
                                      {
LABEL_592:
                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                  (double (__fastcall *)(double))&log2,
                                                                  (__int64)v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      if ( v12 == 8592 )
                                      {
                                        v8 = (unsigned __int8 *)sub_96A790(
                                                                  (double (__fastcall *)(double))&fabs,
                                                                  v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      if ( v12 <= 0x2190 )
                                      {
                                        if ( v12 - 8589 <= 1 )
                                        {
                                          v8 = (unsigned __int8 *)sub_96A6F0(
                                                                    (double (__fastcall *)(double))&fabs,
                                                                    (__int64)v162,
                                                                    v7);
                                          goto LABEL_26;
                                        }
                                      }
                                      else if ( v12 - 8643 <= 1 )
                                      {
                                        v8 = (unsigned __int8 *)sub_96A6F0(
                                                                  (double (__fastcall *)(double))&floor,
                                                                  (__int64)v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      goto LABEL_343;
                                    }
                                    if ( v12 == 9259 )
                                      goto LABEL_628;
                                    if ( v12 <= 0x242B )
                                    {
                                      if ( v12 != 8966 )
                                        goto LABEL_343;
                                      if ( !sub_9696D0(v162) && !sub_969640(v162) )
                                      {
                                        v8 = (unsigned __int8 *)sub_96A790(
                                                                  (double (__fastcall *)(double))&log2,
                                                                  v162,
                                                                  v7);
                                        goto LABEL_26;
                                      }
                                      goto LABEL_25;
                                    }
                                    if ( v12 - 9264 > 1 )
                                      goto LABEL_343;
                                    if ( !(v176 == *((_QWORD *)v8 + 3)
                                         ? sub_C40310(v162)
                                         : (unsigned __int8)sub_C33940(v162)) )
                                    {
LABEL_628:
                                      if ( !sub_969640(v162) )
                                      {
                                        sub_C43310(&v191, *((_QWORD *)v8 + 3), "1.0", 3);
                                        sub_9693D0((__int64)&v194, &v191);
                                        sub_9694D0(&v194, (__int64)v162, 1u);
                                        v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v194);
                                        sub_91D830(&v194);
                                        sub_91D830(&v191);
                                        goto LABEL_26;
                                      }
                                    }
LABEL_25:
                                    v8 = 0;
LABEL_26:
                                    sub_91D830(&v186);
                                    return (__int64)v8;
                                  }
                                  if ( v12 == 9469 )
                                  {
                                    v8 = (unsigned __int8 *)sub_96A790((double (__fastcall *)(double))&sin, v162, v7);
                                    goto LABEL_26;
                                  }
                                  if ( v12 > 0x24FD )
                                  {
                                    if ( v12 != 9533 )
                                    {
                                      if ( v12 <= 0x253D )
                                      {
                                        if ( v12 != 9531 )
                                          goto LABEL_343;
                                      }
                                      else if ( v12 != 9534 && v12 - 9540 > 1 )
                                      {
                                        goto LABEL_343;
                                      }
                                      goto LABEL_570;
                                    }
                                    if ( !sub_9696D0(v162) )
                                    {
                                      v8 = (unsigned __int8 *)sub_96A790((double (__fastcall *)(double))&sqrt, v162, v7);
                                      goto LABEL_26;
                                    }
                                    goto LABEL_25;
                                  }
                                  if ( v12 == 9429 )
                                  {
                                    v106 = &v191;
                                    sub_9690C0(&v191, (__int64)v7, v162);
                                    if ( sub_9696D0(&v191) || sub_969640(&v191) || sub_9696A0(&v191) )
                                    {
                                      v8 = (unsigned __int8 *)sub_AD9290(v7, 0);
                                    }
                                    else
                                    {
                                      sub_969470((__int64)&v194, (__int64)v7, 1.0);
                                      v128 = sub_969600(&v191);
                                      sub_91D830(&v194);
                                      if ( v128 == 2 )
                                      {
                                        sub_969470((__int64)&v194, (__int64)&v194, 1.0);
                                        v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v194);
                                        sub_91D830(&v194);
                                      }
                                      else
                                      {
                                        v8 = (unsigned __int8 *)sub_AC8EA0(*v7, v162);
                                      }
                                    }
                                    goto LABEL_338;
                                  }
                                  if ( v12 == 9467 )
                                    goto LABEL_416;
                                  if ( v12 != 9415 )
                                    goto LABEL_343;
                                  sub_9690C0(&v194, (__int64)v7, v162);
                                }
                                v122 = sub_C41B00(&v194);
                                v123 = sqrt(v122);
                                sub_969410((__int64)v190, v123);
                                sub_C41640(v190, *((_QWORD *)v8 + 3), 1, v189);
                                sub_C43310(&v191, *((_QWORD *)v8 + 3), "1.0", 3);
                                sub_9694D0(&v191, (__int64)v190, 1u);
                                v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v191);
                                sub_91D830(&v191);
                                sub_91D830(v190);
                                sub_91D830(&v194);
                                goto LABEL_26;
                              }
                              v62 = sub_9696A0(&v186);
                              if ( !v62 )
                              {
                                v124 = sub_96E030(v12);
                                v125 = sub_96E000(v12);
                                goto LABEL_427;
                              }
                              if ( v12 <= 0x2183 )
                                goto LABEL_279;
                            }
                            else
                            {
                              if ( v12 > 0x2161 )
                              {
                                if ( !sub_9696A0(&v186) )
                                {
LABEL_423:
                                  v124 = sub_96E030(v12);
                                  v125 = sub_96E000(v12);
                                  if ( v12 > 0x2161 )
                                  {
LABEL_426:
                                    v62 = 1;
LABEL_427:
                                    v185 = v125;
                                    sub_9691E0((__int64)&v191, *((_DWORD *)v7 + 2) >> 8, 0, 0, 0);
                                    v193 = !v62;
                                    if ( v185 )
                                      sub_968FB0(&v194, &v186);
                                    else
                                      sub_9693D0((__int64)&v194, &v186);
                                    LOBYTE(v190[0]) = 0;
                                    if ( (unsigned int)sub_C41980(&v194, &v191, v124, v190, v126, v127) == 1 )
                                      v8 = 0;
                                    else
                                      v8 = (unsigned __int8 *)sub_AD8D80(v7, &v191);
                                    sub_91D830(&v194);
                                    sub_969240(&v191);
                                    goto LABEL_26;
                                  }
                                  if ( v12 > 0x2119 )
                                  {
                                    if ( v12 - 8474 <= 7 )
                                      goto LABEL_427;
                                  }
                                  else if ( v12 > 0x2111 )
                                  {
                                    goto LABEL_426;
                                  }
LABEL_625:
                                  BUG();
                                }
                                if ( v12 > 0x2169 )
                                  goto LABEL_440;
                              }
                              else
                              {
                                if ( v12 - 8466 > 0xF )
                                  goto LABEL_174;
                                if ( !sub_9696A0(&v186) )
                                  goto LABEL_423;
                              }
                              if ( v12 > 0x2121 )
                              {
                                if ( v12 - 8546 > 7 )
                                  goto LABEL_625;
LABEL_279:
                                v8 = (unsigned __int8 *)sub_AD64C0(v7, 0, 0);
                                goto LABEL_26;
                              }
                              if ( v12 <= 0x2111 )
                                goto LABEL_625;
                            }
LABEL_440:
                            sub_9691E0(
                              (__int64)&v194,
                              *((_DWORD *)v7 + 2) >> 8,
                              1LL << (BYTE1(*((_DWORD *)v7 + 2)) - 1),
                              0,
                              0);
                            v8 = (unsigned __int8 *)sub_AD8D80(v7, &v194);
                            sub_969240(&v194);
                            goto LABEL_26;
                        }
                      }
                      if ( v176 == v186 )
                      {
                        if ( (*(_BYTE *)(v187 + 20) & 7) != 1 )
                        {
LABEL_290:
                          v71 = 0;
                          goto LABEL_233;
                        }
                      }
                      else if ( (v188 & 7) != 1 )
                      {
LABEL_267:
                        v71 = 0;
                        goto LABEL_211;
                      }
                      if ( (unsigned __int8)sub_C33750(v186, v162, v61, v21, v18, v20) )
                        goto LABEL_25;
                      if ( v176 != v186 )
                        goto LABEL_267;
                      goto LABEL_290;
                    }
                    if ( v176 == v186 )
                    {
                      if ( (*(_BYTE *)(v187 + 20) & 7) != 1 )
                        goto LABEL_275;
                    }
                    else if ( (v188 & 7) != 1 )
                    {
LABEL_251:
                      sub_C3BAB0(&v186, 3);
                      goto LABEL_586;
                    }
                    if ( (unsigned __int8)sub_C33750(v186, v162, v61, v21, v18, v20) )
                      goto LABEL_25;
                    if ( v176 != v186 )
                      goto LABEL_251;
LABEL_275:
                    sub_C3E740(&v186, 3);
                    goto LABEL_586;
                  }
                  if ( v176 == v186 )
                  {
                    if ( (*(_BYTE *)(v187 + 20) & 7) != 1 )
                      goto LABEL_257;
                  }
                  else if ( (v188 & 7) != 1 )
                  {
LABEL_240:
                    sub_C3BAB0(&v186, 2);
                    goto LABEL_586;
                  }
                  if ( (unsigned __int8)sub_C33750(v186, v162, v61, v21, v18, v20) )
                    goto LABEL_25;
                  if ( v176 != v186 )
                    goto LABEL_240;
LABEL_257:
                  sub_C3E740(&v186, 2);
                  goto LABEL_586;
                }
              }
              v71 = 1;
              if ( v176 != v186 )
              {
LABEL_211:
                sub_C3BAB0(&v186, v71);
LABEL_586:
                v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v186);
                goto LABEL_26;
              }
LABEL_233:
              sub_C3E740(&v186, v71);
              goto LABEL_586;
            }
            v72 = v186;
            if ( v186 == v176 )
            {
              if ( (*(_BYTE *)(v187 + 20) & 7) != 3 )
              {
                v83 = *((_BYTE *)v7 + 8);
                if ( v83 > 3u && v83 != 5 )
                  goto LABEL_25;
                v166 = v21;
                v74 = sub_C40310(&v186);
                v75 = v166;
                goto LABEL_216;
              }
              v90 = *(_BYTE *)(v187 + 20) >> 3;
              v169 = v21;
              sub_C3C500(&v194, v176, 0);
              v86 = v90 & 1;
              v85 = v169;
            }
            else
            {
              if ( (v188 & 7) != 3 )
              {
                v73 = *((_BYTE *)v7 + 8);
                if ( v73 > 3u && v73 != 5 )
                  goto LABEL_25;
                v165 = v21;
                v74 = sub_C33940(&v186);
                v75 = v165;
LABEL_216:
                v76 = v186;
                if ( v74 )
                {
                  if ( v176 != v186 )
                  {
                    v77 = v188;
LABEL_219:
                    if ( (v77 & 7) == 0 )
                    {
LABEL_220:
                      v78 = sub_BD5C60(v75, v72, v76);
                      v8 = (unsigned __int8 *)sub_AC8EA0(v78, &v186);
                      goto LABEL_26;
                    }
                    v174 = v75;
                    v96 = sub_C33940(&v186);
                    v97 = v174;
LABEL_303:
                    if ( !v96 )
                      goto LABEL_25;
                    if ( !*(_QWORD *)(v97 + 40) )
                      goto LABEL_25;
                    v171 = v97;
                    if ( !sub_B43CB0(v97) )
                      goto LABEL_25;
                    v98 = sub_B43CB0(v171);
                    v72 = v186;
                    v99 = sub_B2DB90(v98, v186);
                    v75 = v171;
                    v76 = v99;
                    if ( (_BYTE)v99 )
                    {
                      if ( BYTE1(v99) == 3 || (_WORD)v99 == 3 )
                        goto LABEL_25;
                    }
                    else
                    {
                      if ( !BYTE1(v99) )
                        goto LABEL_220;
                      if ( BYTE1(v99) == 3 )
                        goto LABEL_25;
                    }
                    if ( v176 == v186 )
                    {
                      v100 = (*(_BYTE *)(v187 + 20) & 8) != 0 && BYTE1(v99) != 2 && ((_BYTE)v99 != 2 || BYTE1(v99) != 0);
                      sub_C3C500(&v194, v186, 0);
                      v101 = v171;
                    }
                    else
                    {
                      v100 = (v188 & 8) != 0 && BYTE1(v99) != 2 && (BYTE1(v99) != 0 || (_BYTE)v99 != 2);
                      sub_C373C0(&v194, v186, 0);
                      v101 = v171;
                    }
                    v172 = v101;
                    if ( v176 == v194 )
                      sub_C3CEB0(&v194, v100);
                    else
                      sub_C37310(&v194, v100);
                    v103 = sub_BD5C60(v172, v100, v102);
                    v8 = (unsigned __int8 *)sub_AC8EA0(v103, &v194);
                    if ( v176 == v194 )
                    {
                      if ( !v195 )
                        goto LABEL_26;
                      v91 = (_QWORD *)(24LL * *(_QWORD *)(v195 - 8) + v195);
                      while ( (_QWORD *)v195 != v91 )
                      {
                        v91 -= 3;
                        if ( v176 == *v91 )
                          sub_969EE0((__int64)v91);
                        else
                          sub_C338F0(v91);
                      }
                      goto LABEL_293;
                    }
                    goto LABEL_262;
                  }
                  v76 = v187;
                }
                else
                {
                  if ( v176 != v186 )
                  {
                    v77 = v188;
                    v76 = v188 & 7;
                    if ( (unsigned __int8)v76 > 1u && (_BYTE)v76 != 3 )
                      goto LABEL_220;
                    goto LABEL_219;
                  }
                  v76 = v187;
                  v95 = *(_BYTE *)(v187 + 20) & 7;
                  if ( v95 > 1u && v95 != 3 )
                    goto LABEL_220;
                }
                if ( (*(_BYTE *)(v76 + 20) & 7) == 0 )
                  goto LABEL_220;
                v170 = v75;
                v96 = sub_C40310(&v186);
                v97 = v170;
                goto LABEL_303;
              }
              v167 = v21;
              v84 = v188 >> 3;
              sub_C373C0(&v194, v186, 0);
              v85 = v167;
              v86 = v84 & 1;
            }
            v168 = v85;
            if ( v194 == v176 )
              sub_C3CEB0(&v194, v86);
            else
              sub_C37310(&v194, v86);
            v88 = sub_BD5C60(v168, v86, v87);
            v8 = (unsigned __int8 *)sub_AC8EA0(v88, &v194);
            if ( v194 == v176 )
            {
              if ( !v195 )
                goto LABEL_26;
              v91 = (_QWORD *)(v195 + 24LL * *(_QWORD *)(v195 - 8));
              while ( (_QWORD *)v195 != v91 )
              {
                v91 -= 3;
                if ( *v91 == v176 )
                  sub_969EE0((__int64)v91);
                else
                  sub_C338F0(v91);
              }
LABEL_293:
              j_j_j___libc_free_0_0(v91 - 1);
              goto LABEL_26;
            }
LABEL_262:
            sub_C338F0(&v194);
            goto LABEL_26;
          }
          v52 = v12 == 176;
          LODWORD(v195) = *((_DWORD *)v7 + 2) >> 8;
          if ( (unsigned int)v195 > 0x40 )
            sub_C43690(&v194, 0, 0);
          else
            v194 = 0;
          BYTE4(v195) = v52;
          sub_C41980(&v186, &v194, 0, &v191, v18, v20);
        }
        v8 = (unsigned __int8 *)sub_AD8D80(v7, &v194);
LABEL_97:
        if ( (unsigned int)v195 > 0x40 && v194 )
          j_j___libc_free_0_0(v194);
        goto LABEL_26;
      }
      if ( v16 == v17 )
        sub_C3C790(&v194, v162);
      else
        sub_C33EB0(&v194, v162);
      LOBYTE(v190[0]) = 0;
      v42 = sub_C332F0(&v194, v162, v40, v41);
      sub_C41640(&v194, v42, 1, v190);
      if ( v194 == v176 )
        sub_C3E660(&v191, &v194);
      else
        sub_C3A850(&v191, &v194);
      v8 = (unsigned __int8 *)sub_ACCFD0(*v7, &v191);
      if ( v192 > 0x40 && v191 )
        j_j___libc_free_0_0(v191);
LABEL_78:
      sub_91D830(&v194);
      return (__int64)v8;
    }
    if ( v9 != 17 )
      goto LABEL_12;
    if ( v12 == 66 )
    {
      if ( *((_DWORD *)v8 + 8) > 0x40u )
        v33 = (unsigned int)sub_C44630(v8 + 24);
      else
        v33 = (unsigned int)sub_39FAC40(*((_QWORD *)v8 + 3));
    }
    else
    {
      if ( v12 <= 0x42 )
      {
        switch ( v12 )
        {
          case 0xFu:
            sub_C496B0(&v194, v8 + 24, v14, a7);
            break;
          case 0x18u:
            v23 = sub_C332F0(v13, a2, v14, a7);
            v27 = sub_C33340(v13, a2, v24, v25, v26);
            v28 = v8 + 24;
            if ( v23 == v27 )
              sub_C3C640(&v194, v23, v28);
            else
              sub_C3B160(&v194, v23, v28);
            LOBYTE(v191) = 0;
            v29 = sub_BCAC60(v7);
            sub_C41640(&v194, v29, 1, &v191);
            v8 = (unsigned __int8 *)sub_AC8EA0(*v7, &v194);
            goto LABEL_78;
          case 0xEu:
            sub_C48440(&v194, v8 + 24, v14, a7);
            break;
          default:
            return 0;
        }
        v8 = (unsigned __int8 *)sub_ACCFD0(*v7, &v194);
        if ( (unsigned int)v195 > 0x40 && v194 )
          j_j___libc_free_0_0(v194);
        return (__int64)v8;
      }
      if ( v12 != 3165 )
      {
        if ( v12 == 3185 )
        {
          v30 = *((_QWORD *)v8 + 3);
          if ( *((_DWORD *)v8 + 8) > 0x40u )
            v30 = **((_QWORD **)v8 + 3);
          v31 = v30 | (v30 >> 1) & 0x5555555555555555LL | (2 * v30) & 0xAAAAAAAAAAAAAAAALL;
          v32 = (v31 >> 2) & 0x3333333333333333LL | (4 * v31) & 0xCCCCCCCCCCCCCCCCLL;
        }
        else
        {
          if ( v12 != 3147 )
            return 0;
          v37 = (_QWORD *)*((_QWORD *)v8 + 3);
          if ( *((_DWORD *)v8 + 8) > 0x40u )
            v37 = (_QWORD *)*v37;
          v38 = (((unsigned __int16)v37 | ((_QWORD)v37 << 16) & 0xFFFF00000000LL) << 8) & 0xFF000000FF0000LL
              | (unsigned __int8)v37
              | ((_QWORD)v37 << 16) & 0xFF00000000LL;
          v39 = (4 * ((16 * v38) & 0xF000F000F000F00LL | v38 & 0xF000F000F000FLL)) & 0x3030303030303030LL
              | (16 * v38) & 0x300030003000300LL
              | v38 & 0x3000300030003LL;
          v32 = v39 & 0x1111111111111111LL | (2 * v39) & 0x4444444444444444LL;
          v31 = 2 * v32;
        }
        v33 = v32 | v31;
        return sub_AD64C0(v7, v33, 0);
      }
      v79 = *((_QWORD *)v8 + 3);
      v80 = *((_DWORD *)v8 + 8) >> 2;
      if ( *((_DWORD *)v8 + 8) > 0x40u )
      {
        v79 = *(_QWORD *)v79;
        goto LABEL_223;
      }
      v33 = 0;
      if ( v80 )
      {
LABEL_223:
        v33 = 0;
        v81 = 0;
        do
        {
          if ( (v79 & 0xF) != 0 )
            v33 |= 1LL << v81;
          ++v81;
          v79 >>= 4;
        }
        while ( v81 < v80 );
      }
    }
    return sub_AD64C0(v7, v33, 0);
  }
  if ( !(unsigned __int8)sub_AC2F40(*a5, a2, a3, a7) )
    return 0;
  return sub_ACD6D0(*v7);
}
