// Function: sub_DFBE30
// Address: 0xdfbe30
//
__int64 __fastcall sub_DFBE30(__int64 *a1, unsigned __int8 *a2, unsigned __int8 **a3, __int64 a4, int a5)
{
  __int64 v5; // r10
  int v9; // eax
  unsigned int v10; // r13d
  __int64 result; // rax
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rbx
  __int64 v16; // r15
  unsigned __int8 *v17; // rdi
  unsigned __int8 *v18; // r11
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // rax
  int v24; // r12d
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  int v28; // edx
  __int64 v29; // r14
  unsigned __int8 *v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // r12
  char *v33; // r13
  unsigned int v34; // ebx
  unsigned int v35; // edx
  __int64 v36; // r14
  __int64 v37; // rax
  char v38; // al
  char v39; // r9
  __int64 *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rsi
  __int64 v43; // rdi
  bool v44; // al
  _BYTE *v45; // r10
  bool v46; // zf
  unsigned __int8 v47; // al
  __int64 v48; // rdi
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rdx
  unsigned __int8 *v52; // rdi
  unsigned __int8 **v53; // rdi
  unsigned __int8 **v54; // rsi
  _QWORD *v55; // r8
  _QWORD *v56; // r9
  unsigned __int8 **v57; // rcx
  unsigned __int8 *v58; // rdx
  unsigned __int8 *v59; // r12
  bool v60; // cc
  __int64 v61; // rdi
  __int64 v62; // rdx
  __int64 v63; // r8
  __int64 v64; // rcx
  bool v65; // al
  unsigned __int8 **v66; // rcx
  _BYTE *v67; // rdi
  _BYTE *v68; // rdi
  char v69; // al
  unsigned int v70; // edx
  char v71; // al
  __int64 v72; // r8
  __int64 v73; // r15
  __int64 v74; // r10
  __int64 i; // rcx
  __int64 v76; // rdi
  int v77; // eax
  int **v78; // r9
  __int64 v79; // rax
  unsigned int v80; // ecx
  int *v81; // rax
  int v82; // edx
  unsigned int v83; // edx
  unsigned int v84; // edx
  unsigned int v85; // edx
  unsigned int v86; // edx
  char v87; // al
  __int64 v88; // r13
  unsigned __int64 v89; // rdx
  _DWORD *v90; // rax
  _DWORD *v91; // rdx
  _DWORD *v92; // rax
  _DWORD *v93; // rcx
  int v94; // edx
  __int64 v96; // [rsp+10h] [rbp-140h]
  _BYTE *v97; // [rsp+10h] [rbp-140h]
  _BYTE *v98; // [rsp+10h] [rbp-140h]
  _BYTE *v99; // [rsp+10h] [rbp-140h]
  __int64 v100; // [rsp+10h] [rbp-140h]
  __int64 v101; // [rsp+18h] [rbp-138h]
  __int64 v102; // [rsp+18h] [rbp-138h]
  __int64 v104; // [rsp+18h] [rbp-138h]
  __int64 v105; // [rsp+18h] [rbp-138h]
  __int64 v106; // [rsp+18h] [rbp-138h]
  __int64 v107; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v108; // [rsp+18h] [rbp-138h]
  _BYTE *v109; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v110; // [rsp+18h] [rbp-138h]
  __int64 v111; // [rsp+18h] [rbp-138h]
  __int64 v112; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v113; // [rsp+18h] [rbp-138h]
  __int64 v114; // [rsp+18h] [rbp-138h]
  __int64 v115; // [rsp+18h] [rbp-138h]
  __int64 v116; // [rsp+18h] [rbp-138h]
  int v117; // [rsp+20h] [rbp-130h] BYREF
  int v118; // [rsp+24h] [rbp-12Ch] BYREF
  int v119; // [rsp+28h] [rbp-128h] BYREF
  int v120; // [rsp+2Ch] [rbp-124h] BYREF
  int *v121; // [rsp+30h] [rbp-120h] BYREF
  __int64 v122; // [rsp+38h] [rbp-118h]
  _BYTE v123[64]; // [rsp+40h] [rbp-110h] BYREF
  unsigned __int64 v124; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v125; // [rsp+88h] [rbp-C8h]
  unsigned __int8 *v126; // [rsp+90h] [rbp-C0h] BYREF
  unsigned __int8 *v127; // [rsp+98h] [rbp-B8h]
  char v128; // [rsp+A0h] [rbp-B0h] BYREF
  char v129; // [rsp+A8h] [rbp-A8h] BYREF
  char *v130; // [rsp+C8h] [rbp-88h]
  char v131; // [rsp+D8h] [rbp-78h] BYREF

  v5 = (__int64)a2;
  v9 = *a2;
  if ( (unsigned __int8)v9 <= 0x1Cu )
  {
    if ( (_BYTE)v9 != 5 )
    {
      if ( a5 != 4 )
        return -(__int64)(a5 == 0) | 1;
      return 0;
    }
    v10 = *((unsigned __int16 *)a2 + 1);
    if ( a5 == 4 )
      return 0;
    v16 = *((_QWORD *)a2 + 1);
    v18 = 0;
    v17 = 0;
  }
  else
  {
    v10 = (unsigned __int8)v9 - 29;
    if ( a5 == 4 )
      return 0;
    if ( (unsigned __int8)(v9 - 34) <= 0x33u && (v12 = 0x8000000000041LL, _bittest64(&v12, (unsigned int)(v9 - 34))) )
    {
      v13 = *(_QWORD *)(v5 - 32);
      if ( (_BYTE)v9 != 85 )
      {
        if ( v13 && !*(_BYTE *)v13 )
        {
          v14 = *(_QWORD *)(v13 + 24);
LABEL_11:
          if ( v14 != *(_QWORD *)(v5 + 80) )
            goto LABEL_12;
          if ( (unsigned __int8)sub_DF7D80((__int64)a1, *(_BYTE **)(v5 - 32)) )
            return *(unsigned int *)(*(_QWORD *)(v13 + 24) + 12LL);
          return 1;
        }
LABEL_12:
        if ( (unsigned __int8)v9 == 40 )
        {
          v101 = v5;
          v19 = sub_B491D0(v5);
          v5 = v101;
          v15 = 32LL * v19;
        }
        else
        {
          v15 = 0;
          if ( (unsigned __int8)v9 != 85 )
          {
            v15 = 64;
            if ( (unsigned __int8)v9 != 34 )
LABEL_206:
              BUG();
          }
        }
        if ( *(char *)(v5 + 7) < 0 )
        {
          v102 = v5;
          v20 = sub_BD2BC0(v5);
          v5 = v102;
          v22 = v20 + v21;
          if ( *(char *)(v102 + 7) >= 0 )
          {
            if ( (unsigned int)(v22 >> 4) )
              goto LABEL_206;
          }
          else
          {
            v23 = sub_BD2BC0(v102);
            v5 = v102;
            if ( (unsigned int)((v22 - v23) >> 4) )
            {
              if ( *(char *)(v102 + 7) >= 0 )
                goto LABEL_206;
              v24 = *(_DWORD *)(sub_BD2BC0(v102) + 8);
              if ( *(char *)(v102 + 7) >= 0 )
                BUG();
              v25 = sub_BD2BC0(v102);
              v5 = v102;
              v27 = 32LL * (unsigned int)(*(_DWORD *)(v25 + v26 - 4) - v24);
              return (unsigned int)((32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) - 32 - v15 - v27) >> 5) + 1;
            }
          }
        }
        v27 = 0;
        return (unsigned int)((32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF) - 32 - v15 - v27) >> 5) + 1;
      }
      if ( !v13 || *(_BYTE *)v13 )
        goto LABEL_12;
      v14 = *(_QWORD *)(v13 + 24);
      if ( v14 != *(_QWORD *)(v5 + 80) || (*(_BYTE *)(v13 + 33) & 0x20) == 0 )
        goto LABEL_11;
      v16 = *(_QWORD *)(v5 + 8);
      v18 = (unsigned __int8 *)v5;
      v17 = (unsigned __int8 *)v5;
    }
    else
    {
      v16 = *(_QWORD *)(v5 + 8);
      v17 = (unsigned __int8 *)v5;
      v18 = 0;
    }
  }
  switch ( v10 )
  {
    case 1u:
    case 2u:
    case 3u:
    case 0x37u:
      return a5 == 0 || v10 != 55;
    case 0xCu:
    case 0xDu:
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x18u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x1Eu:
      sub_DFB770(*a3);
      if ( v10 == 12 )
        goto LABEL_44;
      sub_DFB770(a3[1]);
      if ( v10 <= 0x18 )
      {
        result = 4;
        if ( v10 > 0x12 )
          return result;
        goto LABEL_44;
      }
      if ( v10 - 28 > 1 )
        goto LABEL_44;
      v53 = a3;
      v54 = &a3[a4];
      goto LABEL_100;
    case 0x1Fu:
      if ( sub_B4D040(v5) )
        return 0;
      return -(__int64)(a5 == 0) | 1;
    case 0x20u:
      result = 4;
      if ( a5 != 1 )
        return 1;
      return result;
    case 0x21u:
      sub_DFB770(*a3);
      return 1;
    case 0x22u:
      v104 = v5;
      sub_BD36B0(v5);
      v30 = *a3;
      v31 = sub_BB5290(v104);
      return sub_DF7390(a1, v31, (__int64)v30, (__int64)(a3 + 1), a4 - 1);
    case 0x26u:
    case 0x27u:
    case 0x28u:
    case 0x29u:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x2Fu:
    case 0x30u:
    case 0x31u:
    case 0x32u:
      v29 = *((_QWORD *)*a3 + 1);
      sub_DFBCC0(v17);
      return sub_DF8590(a1, v10, v16, v29);
    case 0x35u:
    case 0x36u:
      sub_DFB770(*a3);
      sub_DFB770(a3[1]);
      return 1;
    case 0x38u:
      v41 = *(_QWORD *)(v5 - 32);
      if ( !v41 || *(_BYTE *)v41 || *(_QWORD *)(v41 + 24) != *(_QWORD *)(v5 + 80) )
        BUG();
      v42 = *(unsigned int *)(v41 + 36);
      sub_DF86E0((__int64)&v124, v42, v18, 0, 1, 0, 0);
      if ( (unsigned int)v126 > 0xD3 )
      {
        if ( (unsigned int)v126 > 0x178 )
        {
LABEL_90:
          result = 1;
        }
        else if ( (unsigned int)v126 > 0x143 )
        {
          result = ((1LL << ((unsigned __int8)v126 - 68)) & 0x10000020401001LL) == 0;
        }
        else if ( (_DWORD)v126 == 282 )
        {
LABEL_95:
          result = 0;
        }
        else
        {
          result = (unsigned int)((_DWORD)v126 - 291) >= 2;
        }
      }
      else
      {
        if ( (unsigned int)v126 <= 0x94 )
        {
          switch ( (int)v126 )
          {
            case 5:
            case 6:
            case 7:
            case 8:
            case 11:
            case 27:
            case 28:
            case 39:
            case 40:
            case 43:
            case 46:
            case 47:
            case 58:
            case 59:
            case 60:
            case 68:
            case 69:
            case 70:
            case 71:
              goto LABEL_95;
            default:
              goto LABEL_90;
          }
        }
        switch ( (int)v126 )
        {
          case 149:
          case 150:
          case 155:
          case 169:
          case 204:
          case 205:
          case 206:
          case 208:
          case 210:
          case 211:
            goto LABEL_95;
          case 161:
            result = 0;
            break;
          default:
            goto LABEL_90;
        }
      }
      if ( v130 != &v131 )
      {
        v111 = result;
        _libc_free(v130, v42);
        result = v111;
      }
      v52 = v127;
      if ( v127 != (unsigned __int8 *)&v129 )
        goto LABEL_94;
      return result;
    case 0x39u:
      if ( (unsigned __int8)v9 <= 0x1Cu )
        goto LABEL_89;
      v43 = v16;
      if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
        v43 = **(_QWORD **)(v16 + 16);
      v107 = v5;
      v44 = sub_BCAC40(v43, 1);
      v45 = (_BYTE *)v107;
      v46 = !v44;
      v47 = *(_BYTE *)v107;
      if ( v46 )
        goto LABEL_124;
      if ( v47 == 57 )
      {
        if ( (*(_BYTE *)(v107 + 7) & 0x40) != 0 )
          v57 = *(unsigned __int8 ***)(v107 - 8);
        else
          v57 = (unsigned __int8 **)(v107 - 32LL * (*(_DWORD *)(v107 + 4) & 0x7FFFFFF));
        v58 = *v57;
        if ( *v57 )
        {
          v59 = v57[4];
          if ( v59 )
            goto LABEL_113;
        }
        goto LABEL_125;
      }
      if ( v47 != 86 )
        goto LABEL_124;
      v48 = *(_QWORD *)(v107 + 8);
      v46 = *(_QWORD *)(*(_QWORD *)(v107 - 96) + 8LL) == v48;
      v108 = *(unsigned __int8 **)(v107 - 96);
      if ( v46 && **((_BYTE **)v45 - 4) <= 0x15u )
      {
        v98 = v45;
        v59 = (unsigned __int8 *)*((_QWORD *)v45 - 8);
        v65 = sub_AC30F0(*((_QWORD *)v45 - 4));
        v45 = v98;
        v58 = v108;
        if ( v65 && v59 )
          goto LABEL_113;
        v47 = *v98;
LABEL_124:
        if ( v47 <= 0x1Cu )
        {
LABEL_89:
          sub_DFB770(a3[1]);
          sub_DFB770(a3[2]);
          return 1;
        }
LABEL_125:
        v48 = *((_QWORD *)v45 + 1);
      }
      if ( (unsigned int)*(unsigned __int8 *)(v48 + 8) - 17 <= 1 )
        v48 = **(_QWORD **)(v48 + 16);
      v109 = v45;
      if ( !sub_BCAC40(v48, 1) )
        goto LABEL_89;
      v45 = v109;
      if ( *v109 == 58 )
      {
        if ( (v109[7] & 0x40) != 0 )
          v66 = (unsigned __int8 **)*((_QWORD *)v109 - 1);
        else
          v66 = (unsigned __int8 **)&v109[-32 * (*((_DWORD *)v109 + 1) & 0x7FFFFFF)];
        v58 = *v66;
        if ( !*v66 )
          goto LABEL_89;
        v59 = v66[4];
        if ( !v59 )
          goto LABEL_89;
      }
      else
      {
        if ( *v109 != 86 )
          goto LABEL_89;
        v51 = *((_QWORD *)v109 - 12);
        v46 = *(_QWORD *)(v51 + 8) == *((_QWORD *)v109 + 1);
        v110 = (unsigned __int8 *)v51;
        if ( !v46 )
          goto LABEL_89;
        v68 = (_BYTE *)*((_QWORD *)v45 - 8);
        if ( *v68 > 0x15u )
          goto LABEL_89;
        v99 = v45;
        v59 = (unsigned __int8 *)*((_QWORD *)v45 - 4);
        if ( !sub_AD7A80(v68, 1, v51, v49, v50) )
          goto LABEL_89;
        v45 = v99;
        v58 = v110;
        if ( !v59 )
          goto LABEL_89;
      }
LABEL_113:
      v97 = v45;
      v113 = v58;
      sub_DFB770(v58);
      sub_DFB770(v59);
      v124 = (unsigned __int64)&v126;
      v60 = *v97 <= 0x1Cu;
      v126 = v113;
      v127 = v59;
      v125 = 0x200000002LL;
      if ( !v60 )
      {
        v61 = *((_QWORD *)v97 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v61 + 8) - 17 <= 1 )
          v61 = **(_QWORD **)(v61 + 16);
        if ( sub_BCAC40(v61, 1) && *v97 == 86 )
        {
          v64 = *((_QWORD *)v97 + 1);
          if ( *(_QWORD *)(*((_QWORD *)v97 - 12) + 8LL) == v64 )
          {
            v67 = (_BYTE *)*((_QWORD *)v97 - 8);
            if ( *v67 <= 0x15u )
              sub_AD7A80(v67, 1, v62, v64, v63);
          }
        }
      }
      v53 = &v126;
      v54 = (unsigned __int8 **)&v128;
LABEL_100:
      v55 = sub_DF7050(v53, (__int64)v54);
      result = 0;
      if ( v56 == v55 )
      {
LABEL_44:
        result = 1;
        if ( a5 == 1 )
        {
          v28 = *(unsigned __int8 *)(v16 + 8);
          if ( (unsigned int)(v28 - 17) <= 1 )
            LOBYTE(v28) = *(_BYTE *)(**(_QWORD **)(v16 + 16) + 8LL);
          result = 3;
          if ( (unsigned __int8)v28 > 3u && (_BYTE)v28 != 5 )
            return 2LL * ((v28 & 0xFD) == 4) + 1;
        }
      }
      return result;
    case 0x3Du:
    case 0x3Eu:
      return 1;
    case 0x3Fu:
      if ( (_BYTE)v9 != 92 )
        return 1;
      v32 = *(unsigned int *)(v5 + 80);
      v33 = *(char **)(v5 + 72);
      v34 = v32;
      v35 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 - 64) + 8LL) + 32LL);
      if ( v35 == (_DWORD)v32 )
      {
        if ( *(_BYTE *)(v16 + 8) != 18 )
        {
          v114 = v5;
          v69 = sub_B4ED80(*(int **)(v5 + 72), *(unsigned int *)(v5 + 80), v35);
          v5 = v114;
          if ( v69 )
            return 0;
          v34 = *(_DWORD *)(v114 + 80);
          if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v114 - 64) + 8LL) + 32LL) != v34 )
            goto LABEL_143;
          v32 = v34;
        }
        v116 = v5;
        if ( (unsigned __int8)sub_B4EDA0(*(int **)(v5 + 72), v32, v34) )
          return 1;
        v5 = v116;
        v83 = *(_DWORD *)(v116 + 80);
        if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v116 - 64) + 8LL) + 32LL) == v83 )
        {
          if ( (unsigned __int8)sub_B4EEA0(*(int **)(v116 + 72), v83, v83) )
            return 1;
          v5 = v116;
          v84 = *(_DWORD *)(v116 + 80);
          if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v116 - 64) + 8LL) + 32LL) == v84 )
          {
            if ( (unsigned __int8)sub_B4EF10(*(_DWORD **)(v116 + 72), v84, v84) )
              return 1;
            v5 = v116;
            v85 = *(_DWORD *)(v116 + 80);
            if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v116 - 64) + 8LL) + 32LL) == v85 )
            {
              if ( (unsigned __int8)sub_B4EE20(*(int **)(v116 + 72), v85, v85) )
                return 1;
              v5 = v116;
              v86 = *(_DWORD *)(v116 + 80);
              if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v116 - 64) + 8LL) + 32LL) == v86 )
              {
                v87 = sub_B4ED30(*(int **)(v116 + 72), v86, v86);
                v5 = v116;
                if ( v87 )
                  return 1;
              }
            }
          }
        }
LABEL_143:
        v115 = v5;
        if ( !(unsigned __int8)sub_DF72F0(v5, &v117, &v118) )
        {
          v70 = *(_DWORD *)(v115 + 80);
          if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v115 - 64) + 8LL) + 32LL) == v70 )
            sub_B4EF80(*(_QWORD *)(v115 + 72), v70, v70, &v118);
          return 1;
        }
LABEL_67:
        v40 = (__int64 *)v16;
        if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
          v40 = **(__int64 ***)(v16 + 16);
        sub_BCDA70(v40, v117);
        return 1;
      }
      v36 = *((_QWORD *)*a3 + 1);
      if ( v35 >= (unsigned int)v32 )
      {
        v37 = v16;
      }
      else
      {
        v105 = v5;
        if ( (unsigned __int8)sub_B4F540(v5) )
          return 0;
        v5 = v105;
        v37 = *(_QWORD *)(v105 + 8);
      }
      if ( *(_BYTE *)(v37 + 8) == 18
        || (v96 = v5,
            v38 = sub_B4EFF0(
                    *(int **)(v5 + 72),
                    *(unsigned int *)(v5 + 80),
                    *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 - 64) + 8LL) + 32LL),
                    &v118),
            v5 = v96,
            v39 = v38,
            result = 1,
            !v39) )
      {
        v106 = v5;
        if ( (unsigned __int8)sub_DF72F0(v5, &v117, &v118) )
          goto LABEL_67;
        v42 = (__int64)&v119;
        v71 = sub_B4F660(v106, &v119, &v120);
        v73 = 4 * v32;
        v74 = v106;
        if ( v71 )
        {
          LODWORD(v125) = v32;
          if ( (unsigned int)v32 > 0x40 )
            sub_C43690((__int64)&v124, 0, 0);
          else
            v124 = 0;
          if ( v33 != &v33[v73] )
          {
            for ( i = 0; ; ++i )
            {
              if ( *(_DWORD *)&v33[4 * i] != -1 )
              {
                v76 = 1LL << i;
                if ( (unsigned int)v125 <= 0x40 )
                  v124 |= v76;
                else
                  *(_QWORD *)(v124 + 8LL * ((unsigned int)i >> 6)) |= v76;
              }
              if ( i == (unsigned __int64)(v73 - 4) >> 2 )
                break;
            }
          }
          result = 1;
          if ( (unsigned int)v125 > 0x40 && v124 )
          {
            j_j___libc_free_0_0(v124);
            return 1;
          }
          return result;
        }
        v77 = *(_DWORD *)(v36 + 32);
        v78 = &v121;
        v121 = (int *)v123;
        v117 = v77;
        v122 = 0x1000000000LL;
        if ( (unsigned __int64)v73 > 0x40 )
        {
          sub_C8D5F0((__int64)&v121, v123, v73 >> 2, 4u, v72, (__int64)&v121);
          v74 = v106;
        }
        else if ( !v73 )
        {
          goto LABEL_159;
        }
        v42 = (__int64)v33;
        v100 = v74;
        memcpy(&v121[(unsigned int)v122], v33, 4 * v32);
        v78 = &v121;
        v74 = v100;
LABEL_159:
        v79 = *(_QWORD *)(v74 - 64);
        v80 = *(_DWORD *)(v74 + 80);
        LODWORD(v122) = (v73 >> 2) + v122;
        if ( *(_DWORD *)(*(_QWORD *)(v79 + 8) + 32LL) >= v80 )
        {
          v88 = v117 - v32;
          v89 = v88 + (unsigned int)v122;
          if ( v89 > HIDWORD(v122) )
          {
            v42 = (__int64)v123;
            sub_C8D5F0((__int64)&v121, v123, v89, 4u, v72, (__int64)&v121);
          }
          if ( v88 )
          {
            v42 = 255;
            memset(&v121[(unsigned int)v122], 255, 4 * v88);
          }
          LODWORD(v122) = v88 + v122;
          v124 = (unsigned __int64)&v126;
          v125 = 0x1000000000LL;
          if ( v32 )
          {
            if ( v32 > 0x10 )
            {
              v42 = (__int64)&v126;
              sub_C8D5F0((__int64)&v124, &v126, v32, 4u, v72, (__int64)v78);
            }
            v90 = (_DWORD *)(v124 + 4LL * (unsigned int)v125);
            v91 = (_DWORD *)(v73 + v124);
            while ( v91 != v90 )
            {
              if ( v90 )
                *v90 = 0;
              ++v90;
            }
            LODWORD(v125) = v32;
          }
          v92 = (_DWORD *)v124;
          v93 = (_DWORD *)(v124 + 4LL * (unsigned int)v125);
          v94 = 0;
          while ( v92 != v93 )
            *v92++ = v94++;
          result = 2;
          if ( (unsigned __int8 **)v124 != &v126 )
          {
            _libc_free(v124, v42);
            result = 2;
          }
        }
        else
        {
          v81 = v121;
          v42 = (__int64)&v121[(unsigned int)v122];
          while ( (int *)v42 != v81 )
          {
            v82 = *v81;
            if ( *v81 >= v117 )
              v82 += v32 - v117;
            *v81++ = v82;
          }
          result = 1;
        }
        v52 = (unsigned __int8 *)v121;
        if ( v121 != (int *)v123 )
        {
LABEL_94:
          v112 = result;
          _libc_free(v52, v42);
          return v112;
        }
      }
      return result;
    case 0x40u:
    case 0x41u:
      result = 0;
      if ( v10 == 65 )
        return -(__int64)(a5 == 0) | 1;
      return result;
    case 0x43u:
      return 0;
    default:
      return -(__int64)(a5 == 0) | 1;
  }
}
