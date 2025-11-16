// Function: sub_13F60E0
// Address: 0x13f60e0
//
__int64 __fastcall sub_13F60E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 *v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rax
  _QWORD *v26; // r13
  _QWORD *v27; // r12
  __int64 v28; // r11
  __int64 v29; // rcx
  __int64 v30; // r12
  _BYTE *v31; // rax
  unsigned int v33; // eax
  __int64 v34; // rax
  _QWORD *v35; // r11
  const char *v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // r11
  unsigned __int64 v39; // rax
  _BYTE *v40; // rax
  __int64 v41; // r8
  __int64 v42; // r11
  _BYTE *v43; // rax
  __int64 v44; // r8
  __int64 v45; // rax
  _QWORD *v46; // r11
  unsigned __int64 v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rdi
  __int64 v50; // r9
  int v51; // ecx
  unsigned int v52; // r8d
  __int64 v53; // rsi
  __int64 v54; // r9
  __int64 v55; // rbx
  __int64 v56; // rdi
  int v57; // ecx
  __int64 v58; // rsi
  unsigned int v59; // r8d
  const char *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rbx
  _BYTE *v63; // rax
  __int64 v64; // r11
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  __int64 v67; // rbx
  _BYTE *v68; // rax
  _QWORD *v69; // r11
  __int64 *v70; // rax
  __int64 v71; // rcx
  __int64 v72; // rsi
  int v73; // eax
  __int64 v74; // rsi
  int v75; // eax
  _QWORD *v76; // rax
  _QWORD *v77; // rax
  unsigned int v78; // eax
  unsigned __int64 v79; // rcx
  __int64 v80; // rsi
  __int64 v81; // rax
  __int64 v82; // rax
  unsigned int v83; // eax
  unsigned __int64 v84; // rcx
  __int64 v85; // rsi
  int v86; // eax
  int v87; // eax
  _BYTE *v88; // rax
  _BYTE *v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  _QWORD *v92; // rax
  __int64 v93; // rax
  _QWORD *v94; // rax
  __int64 v95; // rax
  __int64 v96; // rsi
  int v97; // eax
  __int64 v98; // rsi
  int v99; // eax
  __int64 v100; // rax
  __int64 v101; // rax
  __int64 v102; // [rsp-10h] [rbp-100h]
  __int64 v103; // [rsp-8h] [rbp-F8h]
  unsigned int v104; // [rsp+14h] [rbp-DCh]
  unsigned int v105; // [rsp+14h] [rbp-DCh]
  unsigned __int64 v106; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v107; // [rsp+18h] [rbp-D8h]
  __int64 v108; // [rsp+20h] [rbp-D0h]
  __int64 v109; // [rsp+20h] [rbp-D0h]
  int v110; // [rsp+28h] [rbp-C8h]
  int v111; // [rsp+28h] [rbp-C8h]
  __int64 v112; // [rsp+28h] [rbp-C8h]
  __int64 v113; // [rsp+28h] [rbp-C8h]
  __int64 v114; // [rsp+28h] [rbp-C8h]
  __int64 v115; // [rsp+28h] [rbp-C8h]
  unsigned int v116; // [rsp+30h] [rbp-C0h]
  unsigned int v117; // [rsp+30h] [rbp-C0h]
  __int64 v118; // [rsp+38h] [rbp-B8h]
  __int64 v119; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v120; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v121; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v122; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v123; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v124; // [rsp+38h] [rbp-B8h]
  unsigned __int64 v125; // [rsp+38h] [rbp-B8h]
  __int64 v126; // [rsp+40h] [rbp-B0h]
  __int64 v127; // [rsp+40h] [rbp-B0h]
  __int64 v128; // [rsp+40h] [rbp-B0h]
  __int64 v129; // [rsp+40h] [rbp-B0h]
  _QWORD *v130; // [rsp+40h] [rbp-B0h]
  _QWORD *v131; // [rsp+40h] [rbp-B0h]
  __int64 v132; // [rsp+40h] [rbp-B0h]
  __int64 v133; // [rsp+40h] [rbp-B0h]
  _QWORD *v134; // [rsp+40h] [rbp-B0h]
  _QWORD *v135; // [rsp+40h] [rbp-B0h]
  _QWORD *v136; // [rsp+48h] [rbp-A8h]
  int v137; // [rsp+48h] [rbp-A8h]
  int v138; // [rsp+48h] [rbp-A8h]
  int v139; // [rsp+48h] [rbp-A8h]
  int v140; // [rsp+48h] [rbp-A8h]
  __int64 v141; // [rsp+48h] [rbp-A8h]
  __int64 v142; // [rsp+48h] [rbp-A8h]
  __int64 v143; // [rsp+48h] [rbp-A8h]
  __int64 v144; // [rsp+48h] [rbp-A8h]
  _QWORD *v145; // [rsp+50h] [rbp-A0h]
  __int64 v146; // [rsp+50h] [rbp-A0h]
  __int64 v147; // [rsp+50h] [rbp-A0h]
  unsigned int v148; // [rsp+50h] [rbp-A0h]
  __int64 v149; // [rsp+50h] [rbp-A0h]
  __int64 v150; // [rsp+50h] [rbp-A0h]
  unsigned int v151; // [rsp+50h] [rbp-A0h]
  _QWORD *v152; // [rsp+50h] [rbp-A0h]
  _QWORD *v153; // [rsp+50h] [rbp-A0h]
  __int64 v154; // [rsp+50h] [rbp-A0h]
  __int64 v155; // [rsp+50h] [rbp-A0h]
  __int64 v156; // [rsp+50h] [rbp-A0h]
  __int64 v157; // [rsp+58h] [rbp-98h]
  __int64 v158; // [rsp+60h] [rbp-90h]
  _QWORD *v159; // [rsp+68h] [rbp-88h]
  const char *v160; // [rsp+70h] [rbp-80h] BYREF
  _BYTE *v161; // [rsp+78h] [rbp-78h]
  _BYTE *v162; // [rsp+80h] [rbp-70h]
  __int64 v163; // [rsp+88h] [rbp-68h]
  int v164; // [rsp+90h] [rbp-60h]
  _BYTE v165[88]; // [rsp+98h] [rbp-58h] BYREF

  v4 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 160) = v4;
  v5 = sub_1632FA0(v4);
  v6 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 168) = v5;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_195:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F96DB4 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_195;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F96DB4);
  v10 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(v9 + 160);
  v11 = *v10;
  v12 = v10[1];
  if ( v11 == v12 )
LABEL_197:
    BUG();
  while ( *(_UNKNOWN **)v11 != &unk_4F9D764 )
  {
    v11 += 16;
    if ( v12 == v11 )
      goto LABEL_197;
  }
  v13 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v11 + 8) + 104LL))(*(_QWORD *)(v11 + 8), &unk_4F9D764);
  v14 = sub_14CF090(v13, a2);
  v15 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 184) = v14;
  v16 = *v15;
  v17 = v15[1];
  if ( v16 == v17 )
LABEL_198:
    BUG();
  while ( *(_UNKNOWN **)v16 != &unk_4F9E06C )
  {
    v16 += 16;
    if ( v17 == v16 )
      goto LABEL_198;
  }
  v18 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v16 + 8) + 104LL))(*(_QWORD *)(v16 + 8), &unk_4F9E06C);
  v19 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 192) = v18 + 160;
  v20 = *v19;
  v21 = v19[1];
  if ( v20 == v21 )
LABEL_196:
    BUG();
  v22 = (__int64)&unk_4F9B6E8;
  while ( *(_UNKNOWN **)v20 != &unk_4F9B6E8 )
  {
    v20 += 16;
    if ( v21 == v20 )
      goto LABEL_196;
  }
  v23 = *(_QWORD *)(v20 + 8);
  *(_QWORD *)(a1 + 200) = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v23 + 104LL))(v23, &unk_4F9B6E8) + 360;
  if ( (*(_BYTE *)(a2 + 23) & 0x20) == 0 && (*(_BYTE *)(a2 + 32) & 0xFu) - 7 > 1 )
  {
    v160 = "Unusual: Unnamed function with non-local linkage";
    LOWORD(v162) = 259;
    sub_16E2CE0(&v160, a1 + 240);
    v88 = *(_BYTE **)(a1 + 264);
    if ( (unsigned __int64)v88 >= *(_QWORD *)(a1 + 256) )
    {
      sub_16E7DE0(a1 + 240, 10);
    }
    else
    {
      *(_QWORD *)(a1 + 264) = v88 + 1;
      *v88 = 10;
    }
    if ( *(_BYTE *)(a2 + 16) <= 0x17u )
    {
      v22 = a1 + 240;
      v23 = a2;
      sub_15537D0(a2, a1 + 240, 1);
      v89 = *(_BYTE **)(a1 + 264);
      if ( (unsigned __int64)v89 < *(_QWORD *)(a1 + 256) )
        goto LABEL_155;
    }
    else
    {
      v22 = a1 + 240;
      v23 = a2;
      sub_155C2B0(a2, a1 + 240, 0);
      v89 = *(_BYTE **)(a1 + 264);
      if ( (unsigned __int64)v89 < *(_QWORD *)(a1 + 256) )
      {
LABEL_155:
        v24 = (__int64)(v89 + 1);
        *(_QWORD *)(a1 + 264) = v89 + 1;
        *v89 = 10;
        goto LABEL_19;
      }
    }
    v22 = 10;
    v23 = a1 + 240;
    sub_16E7DE0(a1 + 240, 10);
  }
LABEL_19:
  v157 = a2 + 72;
  v158 = *(_QWORD *)(a2 + 80);
  if ( a2 + 72 != v158 )
  {
    while ( 1 )
    {
      v25 = v158;
      v159 = (_QWORD *)(v158 + 16);
      v158 = *(_QWORD *)(v158 + 8);
      v26 = *(_QWORD **)(v25 + 24);
      if ( v159 != v26 )
        break;
LABEL_25:
      if ( v157 == v158 )
        goto LABEL_26;
    }
    while ( 1 )
    {
      v27 = v26;
      v26 = (_QWORD *)v26[1];
      v28 = (__int64)(v27 - 3);
      switch ( *((_BYTE *)v27 - 8) )
      {
        case 0x18:
        case 0x1A:
        case 0x1B:
        case 0x1E:
        case 0x20:
        case 0x21:
        case 0x22:
        case 0x23:
        case 0x24:
        case 0x26:
        case 0x27:
        case 0x28:
        case 0x2B:
        case 0x2E:
        case 0x32:
        case 0x33:
        case 0x38:
        case 0x39:
        case 0x3A:
        case 0x3B:
        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x43:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x48:
        case 0x49:
        case 0x4A:
        case 0x4B:
        case 0x4C:
        case 0x4D:
        case 0x4F:
        case 0x50:
        case 0x51:
        case 0x55:
        case 0x56:
        case 0x57:
        case 0x58:
          goto LABEL_24;
        case 0x19:
          v22 = 29;
          v145 = v27 - 3;
          v23 = *(_QWORD *)(v27[2] + 56LL) + 112LL;
          if ( (unsigned __int8)sub_1560180(v23, 29) )
          {
            BYTE1(v162) = 1;
            v60 = "Unusual: Return statement in function with noreturn attribute";
LABEL_83:
            v62 = a1 + 240;
            v160 = v60;
            LOBYTE(v162) = 3;
            sub_16E2CE0(&v160, a1 + 240);
            v63 = *(_BYTE **)(a1 + 264);
            v64 = (__int64)v145;
            if ( (unsigned __int64)v63 >= *(_QWORD *)(a1 + 256) )
            {
              sub_16E7DE0(a1 + 240, 10);
              v64 = (__int64)v145;
            }
            else
            {
              *(_QWORD *)(a1 + 264) = v63 + 1;
              *v63 = 10;
            }
            if ( *((_BYTE *)v27 - 8) <= 0x17u )
            {
              v22 = a1 + 240;
              v23 = v64;
              sub_15537D0(v64, a1 + 240, 1);
              v43 = *(_BYTE **)(a1 + 264);
              if ( (unsigned __int64)v43 < *(_QWORD *)(a1 + 256) )
                goto LABEL_47;
            }
            else
            {
              v22 = a1 + 240;
              v23 = v64;
              sub_155C2B0(v64, a1 + 240, 0);
              v43 = *(_BYTE **)(a1 + 264);
              if ( (unsigned __int64)v43 < *(_QWORD *)(a1 + 256) )
                goto LABEL_47;
            }
LABEL_87:
            v22 = 10;
            v23 = v62;
            sub_16E7DE0(v62, 10);
            if ( v159 == v26 )
              goto LABEL_25;
          }
          else
          {
            v33 = *((_DWORD *)v27 - 1) & 0xFFFFFFF;
            if ( !v33 )
              goto LABEL_24;
            v24 = 4LL * v33;
            v22 = v145[-3 * v33];
            if ( !v22 )
              goto LABEL_24;
            v160 = 0;
            v161 = v165;
            v162 = v165;
            v163 = 4;
            v164 = 0;
            v34 = sub_13F3D10((_QWORD *)a1, v22, 1u, (__int64)&v160);
            v23 = (__int64)v162;
            v35 = v27 - 3;
            if ( v162 != v161 )
            {
              v146 = v34;
              _libc_free((unsigned __int64)v162);
              v35 = v27 - 3;
              v34 = v146;
            }
            if ( *(_BYTE *)(v34 + 16) == 53 )
            {
              v136 = v35;
              v36 = "Unusual: Returning alloca value";
              BYTE1(v162) = 1;
LABEL_43:
              v160 = v36;
              LOBYTE(v162) = 3;
              sub_16E2CE0(&v160, a1 + 240);
              v40 = *(_BYTE **)(a1 + 264);
              v41 = a1 + 240;
              v42 = (__int64)v136;
              if ( (unsigned __int64)v40 >= *(_QWORD *)(a1 + 256) )
              {
                sub_16E7DE0(a1 + 240, 10);
                v41 = a1 + 240;
                v42 = (__int64)v136;
              }
              else
              {
                *(_QWORD *)(a1 + 264) = v40 + 1;
                *v40 = 10;
              }
              if ( *((_BYTE *)v27 - 8) <= 0x17u )
              {
                v22 = v41;
                v23 = v42;
                v156 = v41;
                sub_15537D0(v42, v41, 1);
                v43 = *(_BYTE **)(a1 + 264);
                v44 = v156;
                if ( (unsigned __int64)v43 < *(_QWORD *)(a1 + 256) )
                  goto LABEL_47;
LABEL_148:
                v22 = 10;
                v23 = v44;
                sub_16E7DE0(v44, 10);
                if ( v159 == v26 )
                  goto LABEL_25;
              }
              else
              {
                v22 = v41;
                v23 = v42;
                v149 = v41;
                sub_155C2B0(v42, v41, 0);
                v43 = *(_BYTE **)(a1 + 264);
                v44 = v149;
                if ( (unsigned __int64)v43 >= *(_QWORD *)(a1 + 256) )
                  goto LABEL_148;
LABEL_47:
                v24 = (__int64)(v43 + 1);
                *(_QWORD *)(a1 + 264) = v43 + 1;
                *v43 = 10;
                if ( v159 == v26 )
                  goto LABEL_25;
              }
            }
            else
            {
LABEL_24:
              if ( v159 == v26 )
                goto LABEL_25;
            }
          }
          break;
        case 0x1C:
          if ( (*((_BYTE *)v27 - 1) & 0x40) != 0 )
            v70 = (__int64 *)*(v27 - 4);
          else
            v70 = (__int64 *)(v28 - 24LL * (*((_DWORD *)v27 - 1) & 0xFFFFFFF));
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          v145 = v27 - 3;
          sub_13F46D0((_QWORD *)a1, (__int64)(v27 - 3), *v70, -1, 0, 0, 8u);
          if ( (*((_DWORD *)v27 - 1) & 0xFFFFFFF) != 1 )
            goto LABEL_24;
          BYTE1(v162) = 1;
          v60 = "Undefined behavior: indirectbr with no destinations";
          goto LABEL_83;
        case 0x1D:
          v23 = a1;
          v22 = v28 & 0xFFFFFFFFFFFFFFFBLL;
          sub_13F53A0((_QWORD *)a1, v28 & 0xFFFFFFFFFFFFFFFBLL);
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x1F:
          v65 = *(_QWORD *)(v27[2] + 48LL);
          if ( v65 && v28 == v65 - 24 )
            goto LABEL_24;
          v66 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
          v67 = v66 - 24;
          if ( !v66 )
            v67 = 0;
          v23 = v67;
          if ( (unsigned __int8)sub_15F3040(v67) )
            goto LABEL_24;
          v23 = v67;
          if ( (unsigned __int8)sub_15F3330(v67) )
            goto LABEL_24;
          v62 = a1 + 240;
          v160 = "Unusual: unreachable immediately preceded by instruction without side effects";
          LOWORD(v162) = 259;
          sub_16E2CE0(&v160, a1 + 240);
          v68 = *(_BYTE **)(a1 + 264);
          v69 = v27 - 3;
          if ( (unsigned __int64)v68 >= *(_QWORD *)(a1 + 256) )
          {
            sub_16E7DE0(a1 + 240, 10);
            v69 = v27 - 3;
          }
          else
          {
            *(_QWORD *)(a1 + 264) = v68 + 1;
            *v68 = 10;
          }
          if ( *((_BYTE *)v27 - 8) <= 0x17u )
          {
            v22 = a1 + 240;
            v23 = (__int64)v69;
            sub_15537D0(v69, a1 + 240, 1);
            v43 = *(_BYTE **)(a1 + 264);
            if ( *(_QWORD *)(a1 + 256) > (unsigned __int64)v43 )
              goto LABEL_47;
          }
          else
          {
            v22 = a1 + 240;
            v23 = (__int64)v69;
            sub_155C2B0(v69, a1 + 240, 0);
            v43 = *(_BYTE **)(a1 + 264);
            if ( *(_QWORD *)(a1 + 256) > (unsigned __int64)v43 )
              goto LABEL_47;
          }
          goto LABEL_87;
        case 0x25:
          if ( *(_BYTE *)(*(v27 - 9) + 16LL) != 9 || *(_BYTE *)(*(v27 - 6) + 16LL) != 9 )
            goto LABEL_24;
          v145 = v27 - 3;
          v60 = "Undefined result: sub(undef, undef)";
          BYTE1(v162) = 1;
          goto LABEL_83;
        case 0x29:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F41E0((_QWORD *)a1, (__int64)(v27 - 3));
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x2A:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F41E0((_QWORD *)a1, (__int64)(v27 - 3));
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x2C:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F41E0((_QWORD *)a1, (__int64)(v27 - 3));
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x2D:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F41E0((_QWORD *)a1, (__int64)(v27 - 3));
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x2F:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F4520((_QWORD *)a1, (_BYTE *)v27 - 24);
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x30:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F4520((_QWORD *)a1, (_BYTE *)v27 - 24);
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x31:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F4520((_QWORD *)a1, (_BYTE *)v27 - 24);
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x34:
          if ( *(_BYTE *)(*(v27 - 9) + 16LL) != 9 || *(_BYTE *)(*(v27 - 6) + 16LL) != 9 )
            goto LABEL_24;
          v145 = v27 - 3;
          v60 = "Undefined result: xor(undef, undef)";
          BYTE1(v162) = 1;
          goto LABEL_83;
        case 0x35:
          if ( *(_BYTE *)(*(v27 - 6) + 16LL) != 13 )
            goto LABEL_24;
          v24 = v27[2];
          v61 = *(_QWORD *)(*(_QWORD *)(v24 + 56) + 80LL);
          if ( v61 )
          {
            if ( v24 == v61 - 24 )
              goto LABEL_24;
          }
          v145 = v27 - 3;
          v60 = "Pessimization: Static alloca outside of entry block";
          BYTE1(v162) = 1;
          goto LABEL_83;
        case 0x36:
          v54 = *(v27 - 3);
          v55 = 1;
          v56 = *(_QWORD *)(a1 + 168);
          v57 = *((unsigned __int16 *)v27 - 3) >> 1;
          v58 = v54;
          v59 = 1 << v57 >> 1;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v58 + 8) )
            {
              case 1:
                v29 = 16;
                goto LABEL_23;
              case 2:
                v29 = 32;
                goto LABEL_23;
              case 3:
              case 9:
                v29 = 64;
                goto LABEL_23;
              case 4:
                v29 = 80;
                goto LABEL_23;
              case 5:
              case 6:
                v29 = 128;
                goto LABEL_23;
              case 7:
                v126 = *(v27 - 3);
                v72 = 0;
                v137 = 1 << v57 >> 1;
                v152 = v27 - 3;
                goto LABEL_117;
              case 0xB:
                v29 = *(_DWORD *)(v58 + 8) >> 8;
                goto LABEL_23;
              case 0xD:
                v129 = *(v27 - 3);
                v140 = 1 << v57 >> 1;
                v77 = (_QWORD *)sub_15A9930(v56, v58);
                v28 = (__int64)(v27 - 3);
                v59 = v140;
                v54 = v129;
                v29 = 8LL * *v77;
                goto LABEL_23;
              case 0xE:
                v109 = *(v27 - 3);
                v111 = 1 << v57 >> 1;
                v119 = *(_QWORD *)(v58 + 24);
                v142 = *(_QWORD *)(v58 + 32);
                v83 = sub_15A9FE0(v56, v119);
                v54 = v109;
                v155 = 1;
                v59 = v111;
                v84 = v83;
                v85 = v119;
                v28 = (__int64)(v27 - 3);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v85 + 8) )
                  {
                    case 1:
                      v90 = 16;
                      goto LABEL_163;
                    case 2:
                      v90 = 32;
                      goto LABEL_163;
                    case 3:
                    case 9:
                      v90 = 64;
                      goto LABEL_163;
                    case 4:
                      v90 = 80;
                      goto LABEL_163;
                    case 5:
                    case 6:
                      v90 = 128;
                      goto LABEL_163;
                    case 7:
                      v115 = v109;
                      v98 = 0;
                      v117 = v59;
                      v125 = v84;
                      v135 = v27 - 3;
                      goto LABEL_178;
                    case 0xB:
                      v90 = *(_DWORD *)(v85 + 8) >> 8;
                      goto LABEL_163;
                    case 0xD:
                      v120 = v84;
                      v92 = (_QWORD *)sub_15A9930(v56, v85);
                      v28 = (__int64)(v27 - 3);
                      v84 = v120;
                      v59 = v111;
                      v54 = v109;
                      v90 = 8LL * *v92;
                      goto LABEL_163;
                    case 0xE:
                      v104 = v111;
                      v106 = v84;
                      v112 = *(_QWORD *)(v85 + 24);
                      v132 = *(_QWORD *)(v85 + 32);
                      v121 = (unsigned int)sub_15A9FE0(v56, v112);
                      v93 = sub_127FA20(v56, v112);
                      v28 = (__int64)(v27 - 3);
                      v84 = v106;
                      v59 = v104;
                      v54 = v109;
                      v90 = 8 * v121 * v132 * ((v121 + ((unsigned __int64)(v93 + 7) >> 3) - 1) / v121);
                      goto LABEL_163;
                    case 0xF:
                      v115 = v109;
                      v117 = v59;
                      v125 = v84;
                      v98 = *(_DWORD *)(v85 + 8) >> 8;
                      v135 = v27 - 3;
LABEL_178:
                      v99 = sub_15A9520(v56, v98);
                      v28 = (__int64)v135;
                      v84 = v125;
                      v59 = v117;
                      v54 = v115;
                      v90 = (unsigned int)(8 * v99);
LABEL_163:
                      v29 = 8 * v142 * v84 * ((v84 + ((unsigned __int64)(v155 * v90 + 7) >> 3) - 1) / v84);
                      goto LABEL_23;
                    case 0x10:
                      v100 = v155 * *(_QWORD *)(v85 + 32);
                      v85 = *(_QWORD *)(v85 + 24);
                      v155 = v100;
                      continue;
                    default:
                      goto LABEL_195;
                  }
                }
              case 0xF:
                v126 = *(v27 - 3);
                v137 = 1 << v57 >> 1;
                v152 = v27 - 3;
                v72 = *(_DWORD *)(v58 + 8) >> 8;
LABEL_117:
                v73 = sub_15A9520(v56, v72);
                v28 = (__int64)v152;
                v59 = v137;
                v54 = v126;
                v29 = (unsigned int)(8 * v73);
LABEL_23:
                v22 = v28;
                sub_13F46D0((_QWORD *)a1, v28, *(v27 - 6), (unsigned __int64)(v29 * v55 + 7) >> 3, v59, v54, 1u);
                v23 = v102;
                goto LABEL_24;
              case 0x10:
                v82 = *(_QWORD *)(v58 + 32);
                v58 = *(_QWORD *)(v58 + 24);
                v55 *= v82;
                continue;
              default:
                goto LABEL_195;
            }
          }
        case 0x37:
          v48 = 1;
          v49 = *(_QWORD *)(a1 + 168);
          v50 = *(_QWORD *)*(v27 - 9);
          v51 = *((unsigned __int16 *)v27 - 3) >> 1;
          v52 = 1 << v51 >> 1;
          v53 = v50;
LABEL_61:
          switch ( *(_BYTE *)(v53 + 8) )
          {
            case 1:
              v71 = 16;
              goto LABEL_112;
            case 2:
              v71 = 32;
              goto LABEL_112;
            case 3:
            case 9:
              v71 = 64;
              goto LABEL_112;
            case 4:
              v71 = 80;
              goto LABEL_112;
            case 5:
            case 6:
              v71 = 128;
              goto LABEL_112;
            case 7:
              v127 = *(_QWORD *)*(v27 - 9);
              v74 = 0;
              v138 = 1 << v51 >> 1;
              v153 = v27 - 3;
              goto LABEL_120;
            case 0xB:
              v71 = *(_DWORD *)(v53 + 8) >> 8;
              goto LABEL_112;
            case 0xD:
              v128 = *(_QWORD *)*(v27 - 9);
              v139 = 1 << v51 >> 1;
              v76 = (_QWORD *)sub_15A9930(v49, v53);
              v28 = (__int64)(v27 - 3);
              v52 = v139;
              v50 = v128;
              v71 = 8LL * *v76;
              goto LABEL_112;
            case 0xE:
              v108 = *(_QWORD *)*(v27 - 9);
              v110 = 1 << v51 >> 1;
              v118 = *(_QWORD *)(v53 + 24);
              v141 = *(_QWORD *)(v53 + 32);
              v78 = sub_15A9FE0(v49, v118);
              v50 = v108;
              v154 = 1;
              v52 = v110;
              v79 = v78;
              v80 = v118;
              v28 = (__int64)(v27 - 3);
              while ( 2 )
              {
                switch ( *(_BYTE *)(v80 + 8) )
                {
                  case 1:
                    v91 = 16;
                    goto LABEL_165;
                  case 2:
                    v91 = 32;
                    goto LABEL_165;
                  case 3:
                  case 9:
                    v91 = 64;
                    goto LABEL_165;
                  case 4:
                    v91 = 80;
                    goto LABEL_165;
                  case 5:
                  case 6:
                    v91 = 128;
                    goto LABEL_165;
                  case 7:
                    v114 = v108;
                    v96 = 0;
                    v116 = v52;
                    v124 = v79;
                    v134 = v27 - 3;
                    goto LABEL_176;
                  case 0xB:
                    v91 = *(_DWORD *)(v80 + 8) >> 8;
                    goto LABEL_165;
                  case 0xD:
                    v122 = v79;
                    v94 = (_QWORD *)sub_15A9930(v49, v80);
                    v28 = (__int64)(v27 - 3);
                    v79 = v122;
                    v52 = v110;
                    v50 = v108;
                    v91 = 8LL * *v94;
                    goto LABEL_165;
                  case 0xE:
                    v105 = v110;
                    v107 = v79;
                    v113 = *(_QWORD *)(v80 + 24);
                    v133 = *(_QWORD *)(v80 + 32);
                    v123 = (unsigned int)sub_15A9FE0(v49, v113);
                    v95 = sub_127FA20(v49, v113);
                    v28 = (__int64)(v27 - 3);
                    v79 = v107;
                    v52 = v105;
                    v50 = v108;
                    v91 = 8 * v123 * v133 * ((v123 + ((unsigned __int64)(v95 + 7) >> 3) - 1) / v123);
                    goto LABEL_165;
                  case 0xF:
                    v114 = v108;
                    v116 = v52;
                    v124 = v79;
                    v96 = *(_DWORD *)(v80 + 8) >> 8;
                    v134 = v27 - 3;
LABEL_176:
                    v97 = sub_15A9520(v49, v96);
                    v28 = (__int64)v134;
                    v79 = v124;
                    v52 = v116;
                    v50 = v114;
                    v91 = (unsigned int)(8 * v97);
LABEL_165:
                    v71 = 8 * v141 * v79 * ((v79 + ((unsigned __int64)(v154 * v91 + 7) >> 3) - 1) / v79);
                    goto LABEL_112;
                  case 0x10:
                    v101 = v154 * *(_QWORD *)(v80 + 32);
                    v80 = *(_QWORD *)(v80 + 24);
                    v154 = v101;
                    continue;
                  default:
                    goto LABEL_195;
                }
              }
            case 0xF:
              v127 = *(_QWORD *)*(v27 - 9);
              v138 = 1 << v51 >> 1;
              v153 = v27 - 3;
              v74 = *(_DWORD *)(v53 + 8) >> 8;
LABEL_120:
              v75 = sub_15A9520(v49, v74);
              v28 = (__int64)v153;
              v52 = v138;
              v50 = v127;
              v71 = (unsigned int)(8 * v75);
LABEL_112:
              v23 = a1;
              sub_13F46D0((_QWORD *)a1, v28, *(v27 - 6), (unsigned __int64)(v71 * v48 + 7) >> 3, v52, v50, 2u);
              v22 = v103;
              if ( v159 == v26 )
                goto LABEL_25;
              continue;
            case 0x10:
              v81 = *(_QWORD *)(v53 + 32);
              v53 = *(_QWORD *)(v53 + 24);
              v48 *= v81;
              goto LABEL_61;
            default:
              goto LABEL_195;
          }
        case 0x4E:
          v23 = a1;
          v22 = v28 | 4;
          sub_13F53A0((_QWORD *)a1, v28 | 4);
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x52:
          v22 = (__int64)(v27 - 3);
          v23 = a1;
          sub_13F46D0((_QWORD *)a1, (__int64)(v27 - 3), *(v27 - 6), -1, 0, 0, 3u);
          v24 = v103;
          if ( v159 == v26 )
            goto LABEL_25;
          continue;
        case 0x53:
          v22 = *(v27 - 6);
          v161 = v165;
          v160 = 0;
          v162 = v165;
          v163 = 4;
          v164 = 0;
          v45 = sub_13F3D10((_QWORD *)a1, v22, 0, (__int64)&v160);
          v23 = (__int64)v162;
          v46 = v27 - 3;
          v24 = v45;
          if ( v162 != v161 )
          {
            v150 = v45;
            _libc_free((unsigned __int64)v162);
            v46 = v27 - 3;
            v24 = v150;
          }
          if ( *(_BYTE *)(v24 + 16) != 13 )
            goto LABEL_24;
          v151 = *(_DWORD *)(v24 + 32);
          if ( v151 > 0x40 )
          {
            v23 = v24 + 24;
            v130 = v46;
            v143 = v24;
            v86 = sub_16A57B0(v24 + 24);
            v46 = v130;
            if ( v151 - v86 > 0x40 )
            {
LABEL_55:
              v136 = v46;
              v36 = "Undefined result: extractelement index out of range";
              BYTE1(v162) = 1;
              goto LABEL_43;
            }
            v47 = **(_QWORD **)(v143 + 24);
          }
          else
          {
            v47 = *(_QWORD *)(v24 + 24);
          }
          v24 = *(_QWORD *)*(v27 - 9);
          if ( *(_QWORD *)(v24 + 32) <= v47 )
            goto LABEL_55;
          goto LABEL_24;
        case 0x54:
          v22 = *(v27 - 6);
          v161 = v165;
          v160 = 0;
          v162 = v165;
          v163 = 4;
          v164 = 0;
          v37 = sub_13F3D10((_QWORD *)a1, v22, 0, (__int64)&v160);
          v23 = (__int64)v162;
          v38 = v27 - 3;
          v24 = v37;
          if ( v162 != v161 )
          {
            v147 = v37;
            _libc_free((unsigned __int64)v162);
            v38 = v27 - 3;
            v24 = v147;
          }
          if ( *(_BYTE *)(v24 + 16) != 13 )
            goto LABEL_24;
          v148 = *(_DWORD *)(v24 + 32);
          if ( v148 > 0x40 )
          {
            v23 = v24 + 24;
            v131 = v38;
            v144 = v24;
            v87 = sub_16A57B0(v24 + 24);
            v38 = v131;
            if ( v148 - v87 > 0x40 )
            {
LABEL_42:
              v136 = v38;
              v36 = "Undefined result: insertelement index out of range";
              BYTE1(v162) = 1;
              goto LABEL_43;
            }
            v39 = **(_QWORD **)(v144 + 24);
          }
          else
          {
            v39 = *(_QWORD *)(v24 + 24);
          }
          v24 = *(v27 - 3);
          if ( *(_QWORD *)(v24 + 32) <= v39 )
            goto LABEL_42;
          goto LABEL_24;
      }
    }
  }
LABEL_26:
  v30 = sub_16BA580(v23, v22, v24);
  if ( *(_QWORD *)(a1 + 264) != *(_QWORD *)(a1 + 248) )
    sub_16E7BA0(a1 + 240);
  sub_16E7EE0(v30, **(const char ***)(a1 + 280), *(_QWORD *)(*(_QWORD *)(a1 + 280) + 8LL));
  v31 = *(_BYTE **)(a1 + 208);
  *(_QWORD *)(a1 + 216) = 0;
  *v31 = 0;
  return 0;
}
