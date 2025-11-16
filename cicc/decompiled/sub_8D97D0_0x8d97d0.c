// Function: sub_8D97D0
// Address: 0x8d97d0
//
__int64 __fastcall sub_8D97D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  char v5; // r10
  __int64 v6; // r15
  char v7; // di
  int v8; // r12d
  int v9; // r13d
  int v10; // esi
  __int64 v11; // r14
  int v12; // edx
  int v13; // ebx
  char v14; // al
  char v15; // al
  char v16; // r9
  char v17; // al
  char v18; // dl
  char v19; // al
  bool v20; // dl
  char v21; // di
  unsigned int v22; // r13d
  __int64 v24; // rdx
  char v25; // al
  char v26; // al
  char v27; // dl
  unsigned int v28; // r14d
  __int64 v29; // rsi
  char v30; // dl
  __int64 v31; // rcx
  __int64 v32; // rsi
  int v33; // edx
  int v34; // eax
  __int128 v35; // rdi
  __int64 v36; // r12
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // eax
  __int64 v47; // rbx
  __int64 v48; // r12
  __int128 v49; // rdi
  unsigned int v50; // eax
  __int64 v51; // r8
  bool v52; // al
  bool v53; // dl
  __int64 v54; // r13
  __int64 v55; // r14
  __int64 v56; // r15
  __int64 v57; // rsi
  __int64 v58; // rcx
  char v59; // al
  char v60; // cl
  char v61; // al
  unsigned __int64 v62; // r15
  unsigned __int64 v63; // r14
  unsigned __int8 v64; // al
  __int64 v65; // rsi
  __int64 v66; // rcx
  __int64 v67; // rdx
  __int64 v68; // rdi
  __int64 v69; // r9
  __int64 v70; // rdx
  __int64 v71; // rcx
  _UNKNOWN *__ptr32 *v72; // r8
  char v73; // dl
  char v74; // al
  _QWORD *v75; // rcx
  _QWORD *v76; // rdx
  __int64 v77; // rbx
  __int64 v78; // r12
  int v79; // [rsp+10h] [rbp-70h]
  __int64 v80; // [rsp+10h] [rbp-70h]
  __int64 v82; // [rsp+18h] [rbp-68h]
  int v83; // [rsp+20h] [rbp-60h]
  unsigned int v84; // [rsp+20h] [rbp-60h]
  char v85; // [rsp+24h] [rbp-5Ch]
  unsigned int v86; // [rsp+28h] [rbp-58h]
  _BOOL4 v87; // [rsp+28h] [rbp-58h]
  __int64 v88; // [rsp+28h] [rbp-58h]
  __int64 v89; // [rsp+30h] [rbp-50h]
  int v90; // [rsp+30h] [rbp-50h]
  __int64 v91; // [rsp+38h] [rbp-48h]
  unsigned int v92; // [rsp+38h] [rbp-48h]
  __int64 v93; // [rsp+40h] [rbp-40h] BYREF
  __int64 v94[7]; // [rsp+48h] [rbp-38h] BYREF

  v94[0] = a1;
  v93 = a2;
  v91 = a1;
  v89 = a2;
  if ( a1 == a2 )
    return 1;
  v83 = a3 & 0x1000;
  v79 = a3 & 0x20;
  while ( 2 )
  {
    while ( 2 )
    {
      v5 = *(_BYTE *)(v91 + 140);
      v85 = *(_BYTE *)(v89 + 140);
      if ( v5 != 12 )
      {
        if ( *(_BYTE *)(v89 + 140) != 12 )
        {
          v11 = v89;
          v6 = v91;
          goto LABEL_86;
        }
        v6 = v91;
        v7 = *(_BYTE *)(v91 + 140);
        v8 = 0;
        v86 = 0;
        v9 = 0;
        v10 = v5 == 14;
        goto LABEL_6;
      }
      v6 = v91;
      v10 = 0;
      v8 = 0;
      v9 = 0;
      a5 = 6338;
      do
      {
        v19 = *(_BYTE *)(v6 + 186);
        if ( *(_QWORD *)(v6 + 8) )
          goto LABEL_38;
        a4 = *(unsigned __int8 *)(v6 + 184);
        if ( (unsigned __int8)a4 <= 0xCu )
        {
          v20 = ((0x18C2uLL >> a4) & 1) == 0;
          if ( ((0x18C2uLL >> a4) & 1) != 0 )
          {
            v8 = 1;
            goto LABEL_38;
          }
          v21 = v19 & 8;
          if ( (unsigned __int8)a4 <= 0xAu )
            v20 = ((0x71DuLL >> a4) & 1) == 0;
        }
        else
        {
          v20 = 1;
          v21 = v19 & 8;
        }
        if ( v21 && v20 )
        {
          v86 = 1;
          v7 = 12;
          goto LABEL_21;
        }
        v9 |= *(_BYTE *)(v6 + 185) & 0x7F;
LABEL_38:
        v6 = *(_QWORD *)(v6 + 160);
        if ( (v19 & 0x10) != 0 )
          v10 = 1;
        v7 = *(_BYTE *)(v6 + 140);
      }
      while ( v7 == 12 );
      v86 = 0;
      if ( v7 == 14 )
        v10 = 1;
LABEL_21:
      if ( v85 == 12 )
      {
LABEL_6:
        v11 = v89;
        v12 = 0;
        v13 = 0;
        while ( 1 )
        {
          v15 = *(_BYTE *)(v11 + 186);
          if ( !*(_QWORD *)(v11 + 8) )
            break;
LABEL_7:
          v11 = *(_QWORD *)(v11 + 160);
          if ( (v15 & 0x10) != 0 )
            v12 = 1;
          v14 = *(_BYTE *)(v11 + 140);
          if ( v14 != 12 )
            goto LABEL_23;
        }
        a4 = *(unsigned __int8 *)(v11 + 184);
        if ( (unsigned __int8)a4 <= 0xCu )
        {
          a5 = ((0x18C2uLL >> a4) & 1) == 0;
          if ( ((0x18C2uLL >> a4) & 1) != 0 )
          {
            v8 = 1;
            goto LABEL_7;
          }
          v16 = v15 & 8;
          if ( (unsigned __int8)a4 <= 0xAu )
            a5 = ((0x71DuLL >> a4) & 1) == 0;
        }
        else
        {
          a5 = 1;
          v16 = v15 & 8;
        }
        if ( !v16 || !(_BYTE)a5 )
        {
          a4 = *(_BYTE *)(v11 + 185) & 0x7F;
          v13 |= a4;
          goto LABEL_7;
        }
        v86 = 1;
        if ( v5 != 12 )
          goto LABEL_26;
        if ( v10 != v12 )
          return 0;
        v14 = 12;
        if ( v79 )
          goto LABEL_73;
LABEL_72:
        if ( ((v9 ^ v13) & 0xFFFFFF8F) != 0 )
          return 0;
LABEL_73:
        if ( v7 != 12 || v14 != 12 || *(_BYTE *)(v6 + 184) != *(_BYTE *)(v11 + 184) )
          return 0;
        v91 = *(_QWORD *)(v6 + 160);
        v94[0] = v91;
        v89 = *(_QWORD *)(v11 + 160);
        v93 = v89;
        continue;
      }
      break;
    }
    v11 = v89;
    v14 = *(_BYTE *)(v89 + 140);
    v12 = 0;
    v13 = 0;
LABEL_23:
    if ( v14 == 14 )
      v12 = 1;
    if ( v5 != 12 || v85 != 12 )
      goto LABEL_26;
    if ( v12 != v10 )
      return 0;
    a4 = v86;
    if ( v86 )
    {
      if ( !v79 )
        goto LABEL_72;
      return 0;
    }
    if ( !v12 )
    {
      if ( (a3 & 0x40) != 0 )
        goto LABEL_26;
LABEL_113:
      if ( (*(_BYTE *)(v91 + 186) & 0x20) == 0
        || (*(_BYTE *)(v93 + 186) & 0x20) == 0
        || *(_BYTE *)(v91 + 184) != 10
        || *(_BYTE *)(v93 + 184) != 10
        || sub_89AB40(
             **(_QWORD **)(v91 + 168),
             **(_QWORD **)(v93 + 168),
             2,
             *(_QWORD *)(v93 + 168),
             (_UNKNOWN *__ptr32 *)a5) )
      {
        goto LABEL_26;
      }
      return 0;
    }
    if ( (*(_BYTE *)(v91 + 89) & 4) != 0 && (*(_BYTE *)(v89 + 89) & 4) != 0 )
    {
      if ( (unsigned int)sub_8D97D0(
                           *(_QWORD *)(*(_QWORD *)(v91 + 40) + 32LL),
                           *(_QWORD *)(*(_QWORD *)(v89 + 40) + 32LL),
                           a3,
                           v86,
                           a5) )
      {
        v91 = v94[0];
        goto LABEL_57;
      }
      return 0;
    }
LABEL_57:
    if ( (a3 & 0x40) == 0 )
      goto LABEL_113;
    if ( !(unsigned int)sub_8D97D0(*(_QWORD *)(v91 + 160), *(_QWORD *)(v93 + 160), a3, a4, a5) )
      return 0;
    if ( (*(_BYTE *)(v94[0] + 186) & 0x20) != 0
      && (*(_BYTE *)(v93 + 186) & 0x20) != 0
      && *(_BYTE *)(v94[0] + 184) == 10
      && *(_BYTE *)(v93 + 184) == 10
      && v6 != v11 )
    {
      v86 = sub_8D97B0(v6);
      if ( v86 )
        return 0;
    }
LABEL_26:
    if ( !v79 && ((v9 ^ v13) & 0xFFFFFF8F) != 0 )
      return 0;
    if ( !v83 )
      goto LABEL_32;
    a4 = v93;
    v17 = *(_BYTE *)(v94[0] + 140);
    v18 = *(_BYTE *)(v93 + 140);
    if ( v17 == 12 )
    {
      if ( v18 == 14 && !*(_BYTE *)(v93 + 160) )
      {
        v24 = *(_QWORD *)(v93 + 168);
        if ( *(_DWORD *)(v24 + 28) == -1 )
        {
          v25 = *(_BYTE *)(v94[0] + 184);
          goto LABEL_80;
        }
      }
LABEL_32:
      if ( v86 )
        return 0;
      if ( !v8 )
        goto LABEL_85;
      if ( (a3 & 0x80u) != 0 && (unsigned int)sub_8D46E0(v94, &v93) )
      {
        v91 = v94[0];
        v89 = v93;
        continue;
      }
      if ( (a3 & 0x100) != 0 && dword_4F077C4 == 2 && dword_4F07588 && sub_8D1DD0(v94[0], v93, a3) )
        return 0;
LABEL_85:
      v94[0] = v6;
      v93 = v11;
LABEL_86:
      if ( v11 != v6 )
      {
        v26 = *(_BYTE *)(v6 + 140);
        v27 = *(_BYTE *)(v11 + 140);
        if ( v26 == v27 || v26 == 9 && v27 == 10 || v27 == 9 && v26 == 10 )
        {
          if ( dword_4F07588 && *qword_4D03FD0 )
          {
            if ( (unsigned int)sub_8D1330(v94, &v93, (a3 >> 2) & 1) )
              return (unsigned int)sub_8D97D0(v94[0], v93, a3, a4, a5);
            v6 = v94[0];
            v26 = *(_BYTE *)(v94[0] + 140);
          }
          v28 = a3 & 0xFFFFFFFD;
          if ( (a3 & 2) == 0 )
            v28 = a3;
          if ( (v28 & 0x20) != 0 )
          {
            if ( v26 == 8 )
            {
              a4 = (__int64)&dword_4F077C4;
              if ( dword_4F077C4 == 2 )
              {
                if ( (v28 & 0x400) == 0 )
                {
LABEL_126:
                  if ( !(unsigned int)sub_8D97D0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v93 + 160), v28, a4, a5) )
                    return 0;
                  return (unsigned int)sub_8D1590(v94[0], v93) != 0;
                }
LABEL_102:
                v90 = 1;
                v28 &= ~0x400u;
LABEL_103:
                switch ( v26 )
                {
                  case 0:
                  case 1:
                  case 20:
                  case 21:
                    return 1;
                  case 2:
                    v59 = *(_BYTE *)(v6 + 161);
                    if ( (v59 & 8) == 0 )
                    {
                      v22 = 0;
                      v60 = *(_BYTE *)(v93 + 161);
                      if ( (v60 & 8) != 0 )
                        return v22;
                      if ( *(_BYTE *)(v6 + 160) == *(_BYTE *)(v93 + 160) )
                      {
                        v61 = v60 ^ v59;
                        if ( (v61 & 0x40) == 0 && v61 >= 0 )
                          return ((*(_BYTE *)(v93 + 162) ^ *(_BYTE *)(v6 + 162)) & 7) == 0;
                      }
                      return 0;
                    }
                    if ( (v28 & 4) == 0 )
                      return 0;
                    if ( !*qword_4D03FD0 )
                      return 0;
                    v29 = v93;
                    if ( (*(_BYTE *)(v93 + 161) & 8) == 0 )
                      return 0;
                    return (unsigned int)sub_8CD200((__int64 *)v6, v29);
                  case 3:
                  case 4:
                  case 5:
                    return *(_BYTE *)(v93 + 160) == *(_BYTE *)(v6 + 160);
                  case 6:
                    if ( (a3 & 1) == 0 && ((*(_BYTE *)(v93 + 168) ^ *(_BYTE *)(v6 + 168)) & 3) != 0 )
                      return 0;
                    return (unsigned int)sub_8D97D0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v93 + 160), v28, a4, a5);
                  case 7:
                    if ( (v28 & 0x2000) != 0 )
                    {
                      v46 = v28;
                      v87 = 1;
                      BYTE1(v46) = BYTE1(v28) & 0xDF;
                      v92 = v46;
                    }
                    else
                    {
                      v92 = v28;
                      v87 = dword_4F06978 == 0;
                    }
                    v47 = *(_QWORD *)(v6 + 168);
                    v48 = *(_QWORD *)(v93 + 168);
                    *(_QWORD *)&v49 = *(_QWORD *)(v47 + 40);
                    *((_QWORD *)&v49 + 1) = *(_QWORD *)(v48 + 40);
                    if ( v49 == 0 )
                    {
                      v22 = ((*(_BYTE *)(v48 + 18) ^ *(_BYTE *)(v47 + 18)) & 0x7F) == 0;
                    }
                    else if ( (_QWORD)v49 && *((_QWORD *)&v49 + 1) )
                    {
                      v22 = 0;
                      if ( ((*(_BYTE *)(v48 + 18) ^ *(_BYTE *)(v47 + 18)) & 0x7F) == 0 )
                      {
                        v22 = 1;
                        if ( (_QWORD)v49 != *((_QWORD *)&v49 + 1) )
                          v22 = sub_8D97D0(v49, *((_QWORD *)&v49 + 1), 0, a4, a5) != 0;
                      }
                    }
                    else
                    {
                      v22 = 0;
                      if ( (a3 & 2) != 0 && (*(_BYTE *)(v47 + 18) & 0x7F) == 0 )
                        v22 = (*(_BYTE *)(v48 + 18) & 0x7F) == 0;
                    }
                    if ( ((*(_BYTE *)(v48 + 19) ^ *(_BYTE *)(v47 + 19)) & 0xC0) != 0 )
                      v22 = 0;
                    if ( dword_4F0774C )
                    {
                      v50 = v28;
                      if ( (v92 & 0x80u) != 0 )
                      {
                        BYTE1(v50) = BYTE1(v28) | 0x10;
                        v28 = v50;
                      }
                    }
                    if ( dword_4F077C4 != 2 && unk_4F07778 > 201709 )
                      v28 |= 0x20u;
                    if ( !v22
                      || !(unsigned int)sub_8D97D0(*(_QWORD *)(v94[0] + 160), *(_QWORD *)(v93 + 160), v28, a4, a5)
                      || ((*(_BYTE *)(v48 + 16) ^ *(_BYTE *)(v47 + 16)) & 3) != 0
                      || ((*(_BYTE *)(v48 + 17) >> 4) & 7) != ((*(_BYTE *)(v47 + 17) >> 4) & 7) && unk_4F06904 )
                    {
                      return 0;
                    }
                    v52 = *(_QWORD *)v47 == 0;
                    v53 = *(_QWORD *)v48 == 0;
                    if ( !*(_QWORD *)v47 )
                      goto LABEL_201;
                    if ( !*(_QWORD *)v48 )
                      return 0;
                    v82 = v47;
                    v80 = v48;
                    v84 = v22;
                    v54 = *(_QWORD *)v47;
                    v55 = *(_QWORD *)v48;
                    break;
                  case 8:
                    goto LABEL_126;
                  case 9:
                  case 10:
                  case 11:
                    if ( dword_4F077C4 == 2 )
                    {
                      if ( (*(_BYTE *)(v6 + 177) & 0x20) != 0 && (*(_BYTE *)(v93 + 177) & 0x20) != 0 )
                        return (unsigned int)sub_8DA820(
                                               v6,
                                               v93,
                                               0,
                                               (v28 >> 6) & 1,
                                               (v28 >> 9) & 1,
                                               (v28 >> 8) & 1,
                                               (v28 >> 4) & 1) != 0;
                      return 0;
                    }
                    if ( (v28 & 4) == 0 || !*qword_4D03FD0 )
                      return 0;
                    v29 = v93;
                    return (unsigned int)sub_8CD200((__int64 *)v6, v29);
                  case 13:
                    v36 = sub_8D4890(v93);
                    v37 = sub_8D4890(v94[0]);
                    v22 = sub_8D97D0(v37, v36, v28, v38, v39);
                    if ( v22 )
                    {
                      v40 = sub_8D4870(v93);
                      v41 = sub_8D4870(v94[0]);
                      return (unsigned int)sub_8D97D0(v41, v40, v28, v42, v43) != 0;
                    }
                    return v22;
                  case 14:
                    v30 = *(_BYTE *)(v6 + 160);
                    if ( v30 != *(_BYTE *)(v93 + 160)
                      || ((*(_BYTE *)(v93 + 161) ^ *(_BYTE *)(v6 + 161)) & 2) != 0
                      || (v28 & 0x40) != 0 )
                    {
                      return 0;
                    }
                    if ( v30 == 1 )
                    {
                      v22 = dword_4F07588;
                      v75 = *(_QWORD **)v6;
                      v76 = *(_QWORD **)v93;
                      if ( dword_4F07588 )
                      {
                        if ( *v75 != *v76 )
                          return 0;
                      }
                      else if ( v75 != v76 )
                      {
                        return v22;
                      }
                      v22 = 1;
                      if ( (unsigned int)sub_8D97D0(
                                           *(_QWORD *)(*(_QWORD *)(v6 + 40) + 32LL),
                                           *(_QWORD *)(*(_QWORD *)(v93 + 40) + 32LL),
                                           v28,
                                           v75,
                                           a5) )
                        return v22;
                    }
                    else
                    {
                      if ( v30 == 2 )
                        return 1;
                      if ( v30 )
LABEL_134:
                        sub_721090();
                      v31 = *(_QWORD *)(v93 + 168);
                      v32 = *(_QWORD *)(v6 + 168);
                      if ( *(_DWORD *)(v32 + 24) == *(_DWORD *)(v31 + 24) )
                      {
                        v33 = *(_DWORD *)(v32 + 28);
                        v34 = *(_DWORD *)(v31 + 28);
                        if ( (v28 & 0x10) != 0 )
                        {
                          if ( v33 != v34 )
                            return 0;
                        }
                        else if ( v33 != 0 && v33 != v34 && v34 && (v28 & 8) == 0 )
                        {
                          return 0;
                        }
                        v22 = 1;
                        if ( v33 == -2 )
                          return v22;
                        if ( v34 == -2 )
                          return v22;
                        if ( (v28 & 0x4000) == 0 )
                          return v22;
                        *(_QWORD *)&v35 = *(_QWORD *)(v32 + 32);
                        *((_QWORD *)&v35 + 1) = *(_QWORD *)(v31 + 32);
                        if ( (_QWORD)v35 == *((_QWORD *)&v35 + 1) )
                          return v22;
                        if ( (_QWORD)v35 && *((_QWORD *)&v35 + 1) )
                          return (unsigned int)sub_7386E0(v35, 0) != 0;
                      }
                    }
                    return 0;
                  case 15:
                    if ( (unsigned int)sub_8D97D0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v93 + 160), v28, a4, a5) )
                    {
                      v44 = v93;
                      v45 = v94[0];
                      if ( *(_QWORD *)(v94[0] + 128) == *(_QWORD *)(v93 + 128)
                        && *(_BYTE *)(v94[0] + 177) == *(_BYTE *)(v93 + 177) )
                      {
                        return *(_DWORD *)(v45 + 136) == *(_DWORD *)(v44 + 136);
                      }
                    }
                    return 0;
                  case 16:
                    if ( !(unsigned int)sub_8D97D0(*(_QWORD *)(v6 + 160), *(_QWORD *)(v93 + 160), v28, a4, a5) )
                      return 0;
                    v44 = v93;
                    v45 = v94[0];
                    if ( *(_BYTE *)(v94[0] + 168) != *(_BYTE *)(v93 + 168) )
                      return 0;
                    return *(_DWORD *)(v45 + 136) == *(_DWORD *)(v44 + 136);
                  case 19:
                    return (((unsigned __int8)(*(_BYTE *)(v6 + 141) ^ *(_BYTE *)(v93 + 141)) >> 5) ^ 1) & 1;
                  default:
                    goto LABEL_134;
                }
                while ( 1 )
                {
                  v56 = *(_QWORD *)(v54 + 8);
                  v57 = *(_QWORD *)(v55 + 8);
                  if ( ((*(_BYTE *)(v54 + 33) & 3) == 1) != ((*(_BYTE *)(v55 + 33) & 3) == 1) )
                    return 0;
                  v58 = v92 & 0x100;
                  if ( (v92 & 0x100) != 0 && dword_4F077C4 == 2 )
                  {
                    v77 = sub_8D72A0(v54);
                    v78 = sub_8D72A0(v55);
                    if ( sub_8D3410(v77) && sub_8D3410(v78) )
                    {
                      v57 = v78;
                      v56 = v77;
                    }
                  }
                  if ( !(unsigned int)sub_8D97D0(v56, v57, v92, v58, v51) )
                    return 0;
                  v54 = *(_QWORD *)v54;
                  v55 = *(_QWORD *)v55;
                  v52 = v54 == 0;
                  v53 = v55 == 0;
                  if ( !v54 )
                    break;
                  if ( !v55 )
                    return 0;
                }
                v47 = v82;
                v48 = v80;
                v22 = v84;
LABEL_201:
                if ( !v53 || !v52 )
                  return 0;
                if ( !dword_4F06978 )
                  goto LABEL_205;
                if ( v87 )
                  goto LABEL_205;
                v62 = *(_QWORD *)(v47 + 56);
                v63 = *(_QWORD *)(v48 + 56);
                if ( !(v63 | v62) )
                  goto LABEL_205;
                if ( !v62 )
                {
                  v74 = *(_BYTE *)v63;
                  if ( (*(_BYTE *)v63 & 4) == 0 )
                    return 0;
                  if ( (v74 & 1) == 0 )
                    goto LABEL_205;
                  if ( (v74 & 0x60) != 0 )
                    return 0;
LABEL_237:
                  if ( *(_BYTE *)(*(_QWORD *)(v63 + 8) + 173LL) == 12 )
                    return 0;
                  goto LABEL_205;
                }
                v64 = *(_BYTE *)v62;
                if ( v63 )
                {
                  v65 = *(unsigned __int8 *)v62;
                  LOBYTE(v65) = v64 & 1;
                  if ( (v64 & 1) != 0 )
                  {
                    v66 = *(unsigned __int8 *)v63;
                    if ( (v66 & 1) != 0 )
                    {
                      v67 = ((unsigned __int8)v66 ^ v64) & 4;
                      LOBYTE(v67) = (v66 | v64) & 0x60 | v67;
                      if ( (_BYTE)v67 )
                        return 0;
                      v68 = *(_QWORD *)(v62 + 8);
                      v69 = *(_QWORD *)(v63 + 8);
                      if ( v68 )
                      {
                        if ( v69 )
                        {
                          v88 = *(_QWORD *)(v63 + 8);
                          if ( sub_70FCE0(v68) && sub_70FCE0(v88) )
                          {
                            if ( ((*(_BYTE *)v63 ^ *(_BYTE *)v62) & 4) != 0 )
                              return 0;
                          }
                          else if ( !(unsigned int)sub_73A2C0(*(_QWORD *)(v62 + 8), *(_QWORD *)(v63 + 8), v70, v71, v72) )
                          {
                            return 0;
                          }
                          goto LABEL_205;
                        }
                      }
                      else
                      {
                        if ( !v69 )
                          goto LABEL_205;
                        v68 = *(_QWORD *)(v63 + 8);
                      }
                      if ( (unsigned int)sub_711520(v68, v65, v67, v66, v51) )
                        return 0;
LABEL_205:
                      if ( HIDWORD(qword_4F077B4) )
                      {
                        if ( !v90 )
                        {
                          v22 = sub_8D7310(v94[0], v93);
                          if ( !v22 )
                            return 0;
                        }
                      }
                      if ( (v92 & 0x800) != 0 && ((*(_BYTE *)(v48 + 20) ^ *(_BYTE *)(v47 + 20)) & 1) != 0
                        || HIDWORD(qword_4F077B4) && *(_DWORD *)(v94[0] + 136) != *(_DWORD *)(v93 + 136) )
                      {
                        return 0;
                      }
                      return v22;
                    }
                  }
                  if ( (v64 & 4) == 0 )
                    goto LABEL_248;
                  v73 = *(_BYTE *)v63;
                  if ( (*(_BYTE *)v63 & 4) == 0 )
                    goto LABEL_248;
                  if ( !(_BYTE)v65 )
                  {
                    if ( (v73 & 1) != 0 )
                    {
                      if ( (v73 & 0x60) != 0 )
                        return 0;
                      goto LABEL_237;
                    }
LABEL_248:
                    if ( (unsigned int)sub_8DAC40(*(_QWORD *)(v47 + 56), *(_QWORD *)(v48 + 56))
                      || (unsigned int)sub_8DAC40(v63, v62) )
                    {
                      return 0;
                    }
                    goto LABEL_205;
                  }
                }
                else
                {
                  if ( (v64 & 4) == 0 )
                    return 0;
                  if ( (v64 & 1) == 0 )
                    goto LABEL_205;
                }
                if ( (v64 & 0x60) != 0 || *(_BYTE *)(*(_QWORD *)(v62 + 8) + 173LL) == 12 )
                  return 0;
                goto LABEL_205;
              }
            }
            v28 &= ~0x20u;
          }
          v90 = 0;
          if ( (v28 & 0x400) == 0 )
            goto LABEL_103;
          goto LABEL_102;
        }
        return 0;
      }
      return 1;
    }
    break;
  }
  if ( v18 != 12 )
    goto LABEL_32;
  if ( v17 != 14 )
    goto LABEL_32;
  if ( *(_BYTE *)(v94[0] + 160) )
    goto LABEL_32;
  v24 = *(_QWORD *)(v94[0] + 168);
  if ( *(_DWORD *)(v24 + 28) != -1 )
    goto LABEL_32;
  v25 = *(_BYTE *)(v93 + 184);
LABEL_80:
  v22 = v25 == 2;
  if ( *(_DWORD *)(v24 + 24) == 1 )
    v22 = v25 == 3;
  if ( !v22 )
    goto LABEL_32;
  return v22;
}
