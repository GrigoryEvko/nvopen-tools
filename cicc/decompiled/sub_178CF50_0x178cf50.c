// Function: sub_178CF50
// Address: 0x178cf50
//
__int64 __fastcall sub_178CF50(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi
  __int64 *v4; // rax
  __int64 *v5; // r15
  __int64 v6; // rbx
  _QWORD *v7; // rax
  unsigned __int8 v8; // dl
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rbx
  _QWORD *v13; // r8
  int v14; // r9d
  __int64 v15; // rdi
  __int64 v16; // rsi
  __int64 v18; // rax
  unsigned int v19; // eax
  char v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rbx
  unsigned __int8 v23; // dl
  __int64 v24; // r14
  _QWORD *v25; // rax
  __int64 v26; // rdx
  char v27; // al
  __int64 v28; // rsi
  int v29; // eax
  char *v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rbx
  __int64 v33; // rax
  char *v34; // r14
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned int v37; // r12d
  unsigned int v38; // r9d
  __int64 v39; // rax
  __int64 v40; // r10
  char *v41; // r15
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rax
  char *v49; // rdx
  __int64 v50; // r12
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  char *v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rcx
  char *v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rdi
  __int64 v61; // rdi
  char *v62; // rax
  const char *v63; // rax
  int v64; // ebx
  char *v65; // rdx
  int v66; // ebx
  __int64 v67; // rax
  __int64 v68; // r14
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  char *v72; // rax
  unsigned int v73; // eax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r12
  __int64 *v77; // rdx
  _QWORD *v78; // r10
  int v79; // eax
  __int64 v80; // rsi
  __int64 v81; // rcx
  __int64 *v82; // rbx
  int v83; // eax
  __int64 v84; // rax
  int v85; // ecx
  __int64 v86; // rcx
  __int64 **v87; // rax
  __int64 *v88; // rsi
  unsigned __int64 v89; // rdi
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rcx
  __int64 v93; // rdi
  __int64 v94; // rdi
  __int64 v95; // rdi
  signed __int64 v96; // rdx
  unsigned int v97; // esi
  unsigned int v98; // eax
  unsigned int v99; // ecx
  unsigned int v100; // edi
  __int64 v101; // rdx
  unsigned __int8 v102; // dl
  __int64 v103; // rax
  __int64 v104; // rdi
  __int64 v105; // rdx
  int v106; // eax
  __int64 v107; // rax
  int v108; // ecx
  __int64 v109; // rcx
  __int64 *v110; // rsi
  unsigned __int64 v111; // rdi
  __int64 v112; // rsi
  __int64 v113; // rdi
  int v114; // edi
  signed __int64 v115; // rcx
  __int64 *v116; // [rsp+10h] [rbp-130h]
  __int64 v117; // [rsp+10h] [rbp-130h]
  __int64 *v118; // [rsp+10h] [rbp-130h]
  __int64 v119; // [rsp+20h] [rbp-120h]
  __int64 v120; // [rsp+20h] [rbp-120h]
  __int64 v121; // [rsp+20h] [rbp-120h]
  __int64 *v122; // [rsp+20h] [rbp-120h]
  __int64 *v123; // [rsp+20h] [rbp-120h]
  _QWORD *v124; // [rsp+28h] [rbp-118h]
  __int64 v125; // [rsp+28h] [rbp-118h]
  __int64 v126; // [rsp+30h] [rbp-110h]
  __int64 v128; // [rsp+40h] [rbp-100h]
  __int64 v129; // [rsp+40h] [rbp-100h]
  unsigned int v130; // [rsp+48h] [rbp-F8h]
  __int64 v131; // [rsp+48h] [rbp-F8h]
  __int64 v132; // [rsp+48h] [rbp-F8h]
  unsigned int v133; // [rsp+48h] [rbp-F8h]
  __int64 *v134; // [rsp+58h] [rbp-E8h] BYREF
  _QWORD v135[2]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 *v136; // [rsp+70h] [rbp-D0h] BYREF
  char *v137; // [rsp+78h] [rbp-C8h]
  __int16 v138; // [rsp+80h] [rbp-C0h]
  char *v139; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v140; // [rsp+98h] [rbp-A8h]
  _BYTE v141[32]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 **v142; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v143; // [rsp+C8h] [rbp-78h]
  _QWORD *v144; // [rsp+D0h] [rbp-70h] BYREF
  unsigned int v145; // [rsp+D8h] [rbp-68h]
  char v146; // [rsp+110h] [rbp-30h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11 )
    return 0;
  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return 0;
  v126 = *(_QWORD *)(v2 + 8);
  if ( v126 )
    return 0;
  v4 = sub_1648700(v2);
  v5 = v4;
  if ( *((_BYTE *)v4 + 16) != 70 )
    return 0;
  v6 = v4[1];
  if ( !v6 )
    return 0;
  while ( 1 )
  {
    v7 = sub_1648700(v6);
    v8 = *((_BYTE *)v7 + 16);
    if ( v8 <= 0x17u )
      goto LABEL_9;
    if ( v8 != 54 && v8 != 55 )
      break;
    v9 = (__int64 *)*(v7 - 3);
    if ( v9 )
      goto LABEL_13;
LABEL_9:
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      return 0;
  }
  if ( v8 != 56 )
    goto LABEL_9;
  v9 = (__int64 *)v7[-3 * (*((_DWORD *)v7 + 5) & 0xFFFFFFF)];
  if ( !v9 )
    goto LABEL_9;
LABEL_13:
  if ( v5 != v9 )
    goto LABEL_9;
  v10 = *v5;
  if ( *(_BYTE *)(*v5 + 8) == 16 )
    v10 = **(_QWORD **)(v10 + 16);
  v11 = 1;
  v12 = 8 * (unsigned int)sub_15A9520(a1[333], *(_DWORD *)(v10 + 8) >> 8);
  v15 = a1[333];
  v16 = *(_QWORD *)*(v5 - 3);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v16 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v52 = *(_QWORD *)(v16 + 32);
        v16 = *(_QWORD *)(v16 + 24);
        v11 *= v52;
        continue;
      case 1:
        v18 = 16;
        goto LABEL_23;
      case 2:
        v18 = 32;
        goto LABEL_23;
      case 3:
      case 9:
        v18 = 64;
        goto LABEL_23;
      case 4:
        v18 = 80;
        goto LABEL_23;
      case 5:
      case 6:
        v18 = 128;
        goto LABEL_23;
      case 7:
        v18 = 8 * (unsigned int)sub_15A9520(v15, 0);
        goto LABEL_23;
      case 0xB:
        v18 = *(_DWORD *)(v16 + 8) >> 8;
        goto LABEL_23;
      case 0xD:
        v18 = 8LL * *(_QWORD *)sub_15A9930(v15, v16);
        goto LABEL_23;
      case 0xE:
        v125 = a1[333];
        v50 = 1;
        v120 = *(_QWORD *)(v16 + 24);
        sub_15A9FE0(v15, v120);
        v51 = v120;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v51 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v53 = *(_QWORD *)(v51 + 32);
              v51 = *(_QWORD *)(v51 + 24);
              v50 *= v53;
              continue;
            case 1:
            case 2:
            case 4:
            case 5:
            case 6:
            case 0xB:
              goto LABEL_242;
            case 3:
            case 9:
              JUMPOUT(0x178D675);
            case 7:
              sub_15A9520(v125, 0);
              goto LABEL_242;
            case 0xD:
              sub_15A9930(v125, v51);
              goto LABEL_242;
            case 0xE:
              v117 = *(_QWORD *)(v51 + 24);
              sub_15A9FE0(v125, v117);
              sub_127FA20(v125, v117);
              goto LABEL_242;
            case 0xF:
              sub_15A9520(v125, *(_DWORD *)(v51 + 8) >> 8);
LABEL_242:
              JUMPOUT(0x178D67A);
          }
        }
      case 0xF:
        v18 = 8 * (unsigned int)sub_15A9520(v15, *(_DWORD *)(v16 + 8) >> 8);
LABEL_23:
        if ( v12 != v11 * v18 )
          return 0;
        v130 = 0;
        v139 = v141;
        v140 = 0x400000000LL;
        v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        if ( !v19 )
          goto LABEL_45;
        break;
    }
    break;
  }
  while ( 2 )
  {
    v20 = *(_BYTE *)(a2 + 23);
    if ( (v20 & 0x40) != 0 )
      v21 = *(_QWORD *)(a2 - 8);
    else
      v21 = a2 - 24LL * v19;
    v22 = *(_QWORD *)(v21 + 24LL * v130);
    v23 = *(_BYTE *)(v22 + 16);
    if ( v23 == 69 )
    {
      v22 = *(_QWORD *)(v22 - 24);
      v29 = v140;
      if ( (unsigned int)v140 < HIDWORD(v140) )
        goto LABEL_78;
      goto LABEL_86;
    }
    v24 = *(_QWORD *)(v22 + 8);
    if ( !v24 )
    {
LABEL_72:
      if ( v23 <= 0x17u )
        goto LABEL_68;
      if ( v23 != 77 )
      {
        if ( v23 != 54 )
          goto LABEL_68;
        v48 = *(_QWORD *)(v22 + 8);
        if ( !v48 || *(_QWORD *)(v48 + 8) )
          goto LABEL_68;
      }
      v29 = v140;
      if ( (unsigned int)v140 < HIDWORD(v140) )
        goto LABEL_78;
LABEL_86:
      sub_16CD150((__int64)&v139, v141, 0, 8, (int)v13, v14);
      v29 = v140;
LABEL_78:
      v49 = &v139[8 * v29];
      if ( v49 )
      {
        *(_QWORD *)v49 = v22;
        v29 = v140;
      }
      goto LABEL_44;
    }
    v128 = 8LL * v130;
    while ( 1 )
    {
      v25 = sub_1648700(v24);
      LODWORD(v13) = (_DWORD)v25;
      if ( *((_BYTE *)v25 + 16) == 70 && *v5 == *v25 )
      {
        v26 = (v20 & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v124 = v25;
        v27 = sub_15CCE20(a1[332], (__int64)v25, *(_QWORD *)(v128 + v26 + 24LL * *(unsigned int *)(a2 + 56) + 8));
        v13 = v124;
        if ( v27 )
          break;
        v20 = *(_BYTE *)(a2 + 23);
        v28 = (v20 & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        if ( *(_QWORD *)(v128 + v28 + 24LL * *(unsigned int *)(a2 + 56) + 8) == v124[5] )
          break;
      }
      v24 = *(_QWORD *)(v24 + 8);
      if ( !v24 )
      {
        v23 = *(_BYTE *)(v22 + 16);
        goto LABEL_72;
      }
    }
    v29 = v140;
    if ( (unsigned int)v140 >= HIDWORD(v140) )
    {
      sub_16CD150((__int64)&v139, v141, 0, 8, (int)v124, v14);
      v29 = v140;
      v13 = v124;
    }
    v30 = &v139[8 * v29];
    if ( v30 )
    {
      *(_QWORD *)v30 = v13;
      v29 = v140;
    }
LABEL_44:
    ++v130;
    LODWORD(v140) = v29 + 1;
    v19 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v19 != v130 )
      continue;
    break;
  }
LABEL_45:
  v31 = *(_QWORD *)(a2 + 40);
  v32 = *(_QWORD *)(v31 + 48);
  v33 = sub_157ED20(v31);
  v34 = v139;
  v35 = v33;
  v36 = v33 + 24;
  if ( !v35 )
    v36 = 0;
  v129 = v36;
  if ( v36 != v32 )
  {
    v116 = v5;
    v37 = 0;
    while ( 1 )
    {
      if ( !v32 )
        BUG();
      v119 = v32 - 24;
      if ( *(_BYTE *)(v32 - 8) == 77 && a2 != v32 - 24 && *v116 == *(_QWORD *)(v32 - 24) )
      {
        v38 = *(_DWORD *)(v32 - 4) & 0xFFFFFFF;
        if ( !v38 )
        {
LABEL_67:
          LOWORD(v144) = 257;
          v126 = sub_15FE030(v119, *(_QWORD *)*(v116 - 3), (__int64)&v142, 0);
          goto LABEL_68;
        }
        v39 = 24LL * *(unsigned int *)(a2 + 56);
        v40 = v39 + 8;
        v131 = v39 + 8LL * (v38 - 1) + 16;
        v41 = &v139[-v39];
        while ( 1 )
        {
          v42 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
          v43 = *(_QWORD *)(v42 + v40);
          v44 = 24LL * *(unsigned int *)(v32 + 32) + 8;
          v45 = 0;
          do
          {
            v46 = v119 - 24LL * v38;
            if ( (*(_BYTE *)(v32 - 1) & 0x40) != 0 )
              v46 = *(_QWORD *)(v32 - 32);
            if ( v43 == *(_QWORD *)(v46 + v44) )
            {
              v47 = 24 * v45;
              goto LABEL_65;
            }
            ++v45;
            v44 += 8;
          }
          while ( v38 != (_DWORD)v45 );
          v47 = 0x17FFFFFFE8LL;
LABEL_65:
          if ( *(_QWORD *)&v41[v40 - 8] != *(_QWORD *)(v46 + v47) )
            break;
          v40 += 8;
          if ( v131 == v40 )
            goto LABEL_67;
        }
      }
      v32 = *(_QWORD *)(v32 + 8);
      ++v37;
      if ( v129 == v32 )
        break;
      if ( v37 > dword_4FA28A0 )
      {
        v34 = v139;
        goto LABEL_69;
      }
    }
    v5 = v116;
    v34 = v139;
  }
  v54 = 8LL * (unsigned int)v140;
  v55 = &v34[v54];
  v56 = v54 >> 5;
  if ( !v56 )
  {
    v58 = v34;
LABEL_213:
    v115 = v55 - v58;
    if ( v55 - v58 != 16 )
    {
      if ( v115 != 24 )
      {
        if ( v115 != 8 )
          goto LABEL_69;
LABEL_219:
        if ( **(_QWORD **)v58 != *v5 || *(_BYTE *)(*(_QWORD *)v58 + 16LL) == 70 )
          goto LABEL_69;
LABEL_221:
        if ( v55 == v58 )
          goto LABEL_69;
        v62 = v34;
        if ( v56 )
        {
          v57 = *v5;
          goto LABEL_116;
        }
LABEL_159:
        v96 = v55 - v62;
        if ( v55 - v62 != 16 )
        {
          if ( v96 != 24 )
          {
            if ( v96 != 8 )
              goto LABEL_120;
LABEL_162:
            if ( **(_QWORD **)v62 == *v5 || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v62 + 16LL) - 25) > 9u )
              goto LABEL_120;
            goto LABEL_119;
          }
          if ( **(_QWORD **)v62 != *v5 && (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v62 + 16LL) - 25) <= 9u )
          {
LABEL_119:
            if ( v55 != v62 )
              goto LABEL_69;
LABEL_120:
            v63 = sub_1649960(a2);
            v142 = &v136;
            v64 = *(_DWORD *)(a2 + 20);
            v136 = (__int64 *)v63;
            v137 = v65;
            v66 = v64 & 0xFFFFFFF;
            LOWORD(v144) = 773;
            v143 = (__int64)".ptr";
            v132 = *v5;
            v67 = sub_1648B60(64);
            v68 = v67;
            if ( v67 )
            {
              sub_15F1EA0(v67, v132, 53, 0, 0, 0);
              *(_DWORD *)(v68 + 56) = v66;
              sub_164B780(v68, (__int64 *)&v142);
              sub_1648880(v68, *(_DWORD *)(v68 + 56), 1);
            }
            sub_157E9D0(*(_QWORD *)(a2 + 40) + 40LL, v68);
            v69 = *(_QWORD *)(a2 + 24);
            *(_QWORD *)(v68 + 32) = a2 + 24;
            v69 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v68 + 24) = v69 | *(_QWORD *)(v68 + 24) & 7LL;
            *(_QWORD *)(v69 + 8) = v68 + 24;
            *(_QWORD *)(a2 + 24) = *(_QWORD *)(a2 + 24) & 7LL | (v68 + 24);
            sub_170B990(*a1, v68);
            v72 = (char *)&v144;
            v142 = 0;
            v143 = 1;
            do
            {
              *(_QWORD *)v72 = -8;
              v72 += 16;
            }
            while ( v72 != &v146 );
            v133 = 0;
            v73 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
            if ( !v73 )
            {
LABEL_185:
              v138 = 257;
              v126 = sub_15FE030(v68, *(_QWORD *)*(v5 - 3), (__int64)&v136, 0);
              if ( (v143 & 1) == 0 )
                j___libc_free_0(v144);
LABEL_68:
              v34 = v139;
              goto LABEL_69;
            }
            while ( 1 )
            {
              if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
                v74 = *(_QWORD *)(a2 - 8);
              else
                v74 = a2 - 24LL * v73;
              v75 = 3LL * *(unsigned int *)(a2 + 56);
              v76 = *(_QWORD *)(v74 + 24LL * *(unsigned int *)(a2 + 56) + 8LL * v133 + 8);
              v77 = *(__int64 **)&v139[8 * v133];
              v134 = v77;
              if ( *v5 == *v77 )
              {
                v106 = *(_DWORD *)(v68 + 20) & 0xFFFFFFF;
                if ( v106 == *(_DWORD *)(v68 + 56) )
                {
                  v123 = v77;
                  sub_15F55D0(v68, *v77, (__int64)v77, v75, v70, v71);
                  v77 = v123;
                  v106 = *(_DWORD *)(v68 + 20) & 0xFFFFFFF;
                }
                v107 = (v106 + 1) & 0xFFFFFFF;
                v108 = v107 | *(_DWORD *)(v68 + 20) & 0xF0000000;
                *(_DWORD *)(v68 + 20) = v108;
                if ( (v108 & 0x40000000) != 0 )
                  v109 = *(_QWORD *)(v68 - 8);
                else
                  v109 = v68 - 24 * v107;
                v87 = (__int64 **)(v109 + 24LL * (unsigned int)(v107 - 1));
                if ( *v87 )
                {
                  v110 = v87[1];
                  v111 = (unsigned __int64)v87[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v111 = v110;
                  if ( v110 )
                    v110[2] = v111 | v110[2] & 3;
                }
                *v87 = v77;
                v112 = v77[1];
                v87[1] = (__int64 *)v112;
                if ( v112 )
                {
                  v71 = (__int64)(v87 + 1);
                  *(_QWORD *)(v112 + 16) = (unsigned __int64)(v87 + 1) | *(_QWORD *)(v112 + 16) & 3LL;
                }
                v87[2] = (__int64 *)((unsigned __int64)v87[2] & 3 | (unsigned __int64)(v77 + 1));
                goto LABEL_143;
              }
              v70 = v143 & 1;
              if ( (v143 & 1) != 0 )
              {
                v78 = &v144;
                v79 = 3;
              }
              else
              {
                v97 = v145;
                v78 = v144;
                v79 = v145 - 1;
                if ( !v145 )
                {
                  v98 = v143;
                  v142 = (__int64 **)((char *)v142 + 1);
                  v82 = 0;
                  v99 = ((unsigned int)v143 >> 1) + 1;
                  goto LABEL_175;
                }
              }
              v80 = (unsigned int)v77 >> 9;
              v81 = v79 & ((unsigned int)v80 ^ ((unsigned int)v77 >> 4));
              v82 = &v78[2 * v81];
              v71 = *v82;
              if ( v77 != (__int64 *)*v82 )
                break;
LABEL_131:
              v77 = (__int64 *)v82[1];
              if ( !v77 )
                goto LABEL_181;
LABEL_132:
              v83 = *(_DWORD *)(v68 + 20) & 0xFFFFFFF;
              if ( v83 == *(_DWORD *)(v68 + 56) )
              {
                v122 = v77;
                sub_15F55D0(v68, v80, (__int64)v77, v81, v70, v71);
                v77 = v122;
                v83 = *(_DWORD *)(v68 + 20) & 0xFFFFFFF;
              }
              v84 = (v83 + 1) & 0xFFFFFFF;
              v85 = v84 | *(_DWORD *)(v68 + 20) & 0xF0000000;
              *(_DWORD *)(v68 + 20) = v85;
              if ( (v85 & 0x40000000) != 0 )
                v86 = *(_QWORD *)(v68 - 8);
              else
                v86 = v68 - 24 * v84;
              v87 = (__int64 **)(v86 + 24LL * (unsigned int)(v84 - 1));
              if ( *v87 )
              {
                v88 = v87[1];
                v89 = (unsigned __int64)v87[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v89 = v88;
                if ( v88 )
                  v88[2] = v89 | v88[2] & 3;
              }
              *v87 = v77;
              if ( !v77 )
                goto LABEL_144;
              v90 = v77[1];
              v87[1] = (__int64 *)v90;
              if ( v90 )
              {
                v71 = (__int64)(v87 + 1);
                *(_QWORD *)(v90 + 16) = (unsigned __int64)(v87 + 1) | *(_QWORD *)(v90 + 16) & 3LL;
              }
              v87[2] = (__int64 *)((unsigned __int64)(v77 + 1) | (unsigned __int64)v87[2] & 3);
LABEL_143:
              v77[1] = (__int64)v87;
LABEL_144:
              v91 = *(_DWORD *)(v68 + 20) & 0xFFFFFFF;
              if ( (*(_BYTE *)(v68 + 23) & 0x40) != 0 )
                v92 = *(_QWORD *)(v68 - 8);
              else
                v92 = v68 - 24 * v91;
              *(_QWORD *)(v92 + 8LL * (unsigned int)(v91 - 1) + 24LL * *(unsigned int *)(v68 + 56) + 8) = v76;
              ++v133;
              v73 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
              if ( v133 == v73 )
                goto LABEL_185;
            }
            v80 = 0;
            v114 = 1;
            while ( v71 != -8 )
            {
              if ( !v80 && v71 == -16 )
                v80 = (__int64)v82;
              v81 = v79 & (unsigned int)(v114 + v81);
              v82 = &v78[2 * (unsigned int)v81];
              v71 = *v82;
              if ( v77 == (__int64 *)*v82 )
                goto LABEL_131;
              ++v114;
            }
            v98 = v143;
            v100 = 12;
            if ( v80 )
              v82 = (__int64 *)v80;
            v142 = (__int64 **)((char *)v142 + 1);
            v97 = 4;
            v99 = ((unsigned int)v143 >> 1) + 1;
            if ( !(_BYTE)v70 )
            {
              v97 = v145;
LABEL_175:
              v100 = 3 * v97;
            }
            if ( 4 * v99 >= v100 )
            {
              v97 *= 2;
            }
            else if ( v97 - HIDWORD(v143) - v99 > v97 >> 3 )
            {
LABEL_178:
              LODWORD(v143) = (2 * (v98 >> 1) + 2) | v98 & 1;
              if ( *v82 != -8 )
                --HIDWORD(v143);
              *v82 = (__int64)v77;
              v82[1] = 0;
LABEL_181:
              v135[0] = sub_1649960((__int64)v134);
              v135[1] = v101;
              v136 = v135;
              v137 = ".ptr";
              v138 = 773;
              v82[1] = sub_15FE030((__int64)v134, *v5, (__int64)&v136, 0);
              v102 = *((_BYTE *)v134 + 16);
              if ( v102 <= 0x17u )
              {
                v113 = *(_QWORD *)(*(_QWORD *)(v76 + 56) + 80LL);
                if ( v113 )
                  v113 -= 24;
                v103 = sub_157EE30(v113);
                if ( !v103 )
                  BUG();
              }
              else if ( v102 == 77 )
              {
                v103 = sub_157EE30(v134[5]);
                if ( !v103 )
LABEL_30:
                  BUG();
              }
              else
              {
                v103 = v134[4];
                if ( !v103 )
                  goto LABEL_30;
              }
              v118 = (__int64 *)v103;
              v121 = v82[1];
              sub_157E9D0(*(_QWORD *)(v103 + 16) + 40LL, v121);
              v80 = v121;
              v104 = *v118;
              v105 = *(_QWORD *)(v121 + 24);
              *(_QWORD *)(v121 + 32) = v118;
              v104 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v121 + 24) = v104 | v105 & 7;
              *(_QWORD *)(v104 + 8) = v121 + 24;
              *v118 = *v118 & 7 | (v121 + 24);
              sub_170B990(*a1, v121);
              v77 = (__int64 *)v82[1];
              goto LABEL_132;
            }
            sub_178CB70((__int64)&v142, v97);
            sub_178CAB0((__int64)&v142, (__int64 *)&v134, &v136);
            v82 = v136;
            v77 = v134;
            v98 = v143;
            goto LABEL_178;
          }
          v62 += 8;
        }
        if ( **(_QWORD **)v62 == *v5 || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v62 + 16LL) - 25) > 9u )
        {
          v62 += 8;
          goto LABEL_162;
        }
        goto LABEL_119;
      }
      if ( **(_QWORD **)v58 == *v5 && *(_BYTE *)(*(_QWORD *)v58 + 16LL) != 70 )
        goto LABEL_221;
      v58 += 8;
    }
    if ( **(_QWORD **)v58 == *v5 && *(_BYTE *)(*(_QWORD *)v58 + 16LL) != 70 )
      goto LABEL_221;
    v58 += 8;
    goto LABEL_219;
  }
  v57 = *v5;
  v58 = v34;
  while ( **(_QWORD **)v58 != v57 || *(_BYTE *)(*(_QWORD *)v58 + 16LL) == 70 )
  {
    v59 = *((_QWORD *)v58 + 1);
    if ( v57 == *(_QWORD *)v59 && *(_BYTE *)(v59 + 16) != 70 )
    {
      v58 += 8;
      break;
    }
    v60 = *((_QWORD *)v58 + 2);
    if ( v57 == *(_QWORD *)v60 && *(_BYTE *)(v60 + 16) != 70 )
    {
      v58 += 16;
      break;
    }
    v61 = *((_QWORD *)v58 + 3);
    if ( v57 == *(_QWORD *)v61 && *(_BYTE *)(v61 + 16) != 70 )
    {
      v58 += 24;
      break;
    }
    v58 += 32;
    if ( &v34[32 * v56] == v58 )
      goto LABEL_213;
  }
  if ( v55 != v58 )
  {
LABEL_116:
    v62 = v34;
    while ( **(_QWORD **)v62 == v57 || (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v62 + 16LL) - 25) > 9u )
    {
      v93 = *((_QWORD *)v62 + 1);
      if ( *(_QWORD *)v93 != v57 && (unsigned __int8)(*(_BYTE *)(v93 + 16) - 25) <= 9u )
      {
        v62 += 8;
        goto LABEL_119;
      }
      v94 = *((_QWORD *)v62 + 2);
      if ( *(_QWORD *)v94 != v57 && (unsigned __int8)(*(_BYTE *)(v94 + 16) - 25) <= 9u )
      {
        v62 += 16;
        goto LABEL_119;
      }
      v95 = *((_QWORD *)v62 + 3);
      if ( *(_QWORD *)v95 != v57 && (unsigned __int8)(*(_BYTE *)(v95 + 16) - 25) <= 9u )
      {
        v62 += 24;
        goto LABEL_119;
      }
      v62 += 32;
      if ( !--v56 )
        goto LABEL_159;
    }
    goto LABEL_119;
  }
LABEL_69:
  if ( v34 != v141 )
    _libc_free((unsigned __int64)v34);
  return v126;
}
