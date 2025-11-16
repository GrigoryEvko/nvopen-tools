// Function: sub_849040
// Address: 0x849040
//
__int64 __fastcall sub_849040(__int64 *a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 *v3; // r12
  int v4; // r13d
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // rax
  unsigned __int8 *i; // rbx
  int v9; // r15d
  _BOOL4 v10; // ecx
  __int64 v11; // r13
  int v12; // r14d
  unsigned __int8 *v13; // rsi
  unsigned __int64 v14; // rax
  int v15; // eax
  __int64 v16; // r12
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 result; // rax
  int v20; // eax
  __int64 v21; // rdi
  unsigned __int8 *v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // r12
  unsigned __int8 *v25; // rdi
  int v26; // r8d
  int v27; // esi
  int v28; // r9d
  _BOOL4 v29; // ebx
  int v30; // r12d
  int v31; // ecx
  unsigned __int8 *v32; // rax
  int v33; // edx
  int v34; // edx
  __int64 j; // rax
  _QWORD *v36; // rdi
  int v37; // ebx
  _QWORD *v38; // r12
  _QWORD *v40; // rdi
  __int64 v41; // rbx
  _BOOL4 v42; // r12d
  int v43; // r13d
  int v44; // r15d
  __int64 v45; // r10
  _DWORD *v46; // r12
  _QWORD *v47; // rax
  int v48; // eax
  _DWORD *v49; // r14
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // r14
  char v53; // al
  __int64 v54; // rdx
  _QWORD *v55; // rax
  _BOOL4 v56; // ecx
  __int64 v57; // rax
  _QWORD *v58; // rax
  _QWORD *v59; // rax
  _QWORD *v60; // rax
  int v61; // esi
  unsigned __int8 v62; // cl
  unsigned __int8 *v63; // rax
  char v64; // dl
  _QWORD *v65; // rax
  __int64 v66; // rax
  __int64 v67; // rbx
  int v68; // eax
  char v70; // al
  __int64 v71; // rax
  _QWORD *v72; // rbx
  _QWORD *v73; // r13
  _QWORD *v74; // rax
  __int64 v75; // rdx
  _DWORD *v76; // r12
  _DWORD *v77; // r12
  _DWORD *v78; // r12
  int v79; // eax
  _DWORD *v80; // r12
  _QWORD *v81; // rax
  _QWORD *v82; // rax
  _QWORD *v83; // rax
  int v84; // ebx
  __int64 v85; // r12
  __int64 *v86; // rdi
  int v87; // eax
  __int64 v88; // rax
  _QWORD *v89; // rax
  _QWORD *v90; // rax
  _QWORD *v91; // rax
  _QWORD *v92; // rax
  _BOOL4 v93; // [rsp+8h] [rbp-78h]
  _BOOL4 v94; // [rsp+8h] [rbp-78h]
  _BOOL4 v95; // [rsp+8h] [rbp-78h]
  _BOOL4 v96; // [rsp+8h] [rbp-78h]
  int v97; // [rsp+Ch] [rbp-74h]
  int v98; // [rsp+10h] [rbp-70h]
  __int64 v99; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v100; // [rsp+18h] [rbp-68h]
  __int64 v101; // [rsp+18h] [rbp-68h]
  int v102; // [rsp+20h] [rbp-60h]
  int v103; // [rsp+24h] [rbp-5Ch]
  __int64 v105; // [rsp+30h] [rbp-50h]
  char v106; // [rsp+3Ah] [rbp-46h]
  bool v107; // [rsp+3Bh] [rbp-45h]
  int v108; // [rsp+3Ch] [rbp-44h]
  _BOOL4 v109; // [rsp+3Ch] [rbp-44h]
  _BOOL4 v110; // [rsp+3Ch] [rbp-44h]
  _BOOL4 v111; // [rsp+3Ch] [rbp-44h]
  int v112; // [rsp+3Ch] [rbp-44h]
  int v113; // [rsp+3Ch] [rbp-44h]
  int v114; // [rsp+3Ch] [rbp-44h]
  unsigned __int8 *v115[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = a2;
  v105 = (__int64)a1;
  v107 = (*(_BYTE *)(qword_4D03C50 + 21LL) & 4) != 0;
  *(_BYTE *)(qword_4D03C50 + 21LL) |= 4u;
  if ( a1 )
  {
    v3 = a1;
    v4 = 0;
    while ( 1 )
    {
      sub_8489F0((__int64)v3, a2);
      v5 = *(_QWORD *)(a2 + 40);
      if ( v5 )
      {
        v6 = *(_BYTE *)(v5 + 24) == 0;
        v7 = *v3;
        if ( v6 )
          v4 = 1;
        if ( !v7 )
          goto LABEL_10;
      }
      else
      {
        v7 = *v3;
        v4 = 1;
        if ( !*v3 )
          goto LABEL_10;
      }
      if ( *(_BYTE *)(v7 + 8) == 3 )
      {
        v7 = sub_6BBB10(v3);
        if ( !v7 )
          break;
      }
      v3 = (__int64 *)v7;
    }
    i = *(unsigned __int8 **)(a2 + 72);
    if ( !i )
      goto LABEL_22;
  }
  else
  {
    v4 = 0;
LABEL_10:
    i = *(unsigned __int8 **)(a2 + 72);
    if ( !i )
      goto LABEL_22;
  }
  v108 = 0;
  v97 = 0;
  v99 = *(_QWORD *)(a2 + 64);
  v103 = v4;
  v106 = *(_BYTE *)(a2 + 23);
  v9 = 0;
LABEL_12:
  v115[0] = i;
  if ( v108 != 1 )
  {
    if ( v108 != 2 )
    {
      v98 = 0;
      v10 = 0;
      v102 = 0;
      goto LABEL_17;
    }
    v98 = 0;
    LOBYTE(v14) = *i;
    v12 = 0;
    v102 = 0;
    goto LABEL_58;
  }
  v98 = 0;
  LOBYTE(v14) = *i;
  v10 = 0;
  v12 = 0;
  v102 = 0;
LABEL_47:
  if ( v106 != 2 && (_BYTE)v14 == 46 )
  {
    v13 = i + 1;
    v115[0] = i + 1;
    LODWORD(v14) = i[1];
    if ( (unsigned int)(v14 - 48) <= 9 )
    {
      do
      {
        v115[0] = ++v13;
        LODWORD(v14) = *v13;
      }
      while ( (unsigned int)(v14 - 48) <= 9 );
      goto LABEL_56;
    }
    ++i;
    if ( (_BYTE)v14 != 42 )
      goto LABEL_58;
    v109 = v10;
    v23 = sub_72BA30(5u);
    v10 = v109;
    v11 = (__int64)v23;
    i = ++v115[0];
    if ( unk_4D04330 )
    {
      v96 = v109;
      v12 = v9;
      v87 = sub_828610(v115);
      v108 = 2;
      v10 = v96;
      v9 = v87;
      goto LABEL_85;
    }
    v108 = 2;
    v24 = v99;
    goto LABEL_126;
  }
  while ( 1 )
  {
LABEL_58:
    if ( (_BYTE)v14 != 108 )
    {
      if ( (_BYTE)v14 == 76 )
      {
        v32 = i + 2;
        v25 = i + 1;
        v115[0] = i + 2;
        switch ( i[1] )
        {
          case 'A':
          case 'E':
          case 'F':
          case 'G':
          case 'a':
          case 'e':
          case 'f':
          case 'g':
            v11 = (__int64)sub_72C610(6u);
            goto LABEL_174;
          case 'X':
          case 'o':
          case 'x':
            v102 = 1;
            goto LABEL_188;
          case '[':
LABEL_198:
            v61 = 0;
            goto LABEL_199;
          case 'c':
LABEL_186:
            v11 = (__int64)sub_72BA30(byte_4F068B0[0]);
            goto LABEL_174;
          case 'd':
          case 'i':
LABEL_290:
            v11 = (__int64)sub_72BA30(5u);
            goto LABEL_174;
          case 'n':
LABEL_289:
            v83 = sub_72BA30(5u);
            v56 = 1;
            v11 = (__int64)v83;
            goto LABEL_164;
          case 'p':
            goto LABEL_177;
          case 's':
LABEL_178:
            v59 = sub_72BA30(byte_4F068B0[0]);
            v98 = 1;
            v56 = v106 == 2;
            v11 = (__int64)v59;
            goto LABEL_164;
          case 'u':
            goto LABEL_188;
          default:
            goto LABEL_156;
        }
      }
      if ( (_BYTE)v14 == 104 )
      {
        if ( i[1] == 104 )
        {
          v25 = i + 2;
          v26 = 0;
          v29 = 0;
          v27 = 0;
          v28 = 0;
          v30 = 1;
          v31 = 0;
        }
        else
        {
          v25 = i + 1;
          v26 = 0;
          v29 = 0;
          v27 = 0;
          v28 = 0;
          v30 = 0;
          v31 = 1;
        }
      }
      else
      {
        v25 = i + 1;
        if ( (_BYTE)v14 == 106 )
        {
          v26 = 0;
          v29 = 0;
          v27 = 0;
          v28 = 1;
          v30 = 0;
          v31 = 0;
        }
        else if ( (_BYTE)v14 == 122 )
        {
          v26 = 0;
          v29 = 0;
          v27 = 1;
          v28 = 0;
          v30 = 0;
          v31 = 0;
        }
        else
        {
          v26 = 0;
          if ( (_BYTE)v14 != 116 )
            v25 = i;
          v27 = 0;
          v28 = 0;
          v29 = (_BYTE)v14 == 116;
          v30 = 0;
          v31 = 0;
        }
      }
      goto LABEL_66;
    }
    if ( i[1] == 108 )
    {
      v25 = i + 2;
      v26 = 1;
      v29 = 0;
      v27 = 0;
      v28 = 0;
      v30 = 0;
      v31 = 0;
LABEL_66:
      v32 = v25 + 1;
      v115[0] = v25 + 1;
      switch ( *v25 )
      {
        case 'A':
        case 'E':
        case 'F':
        case 'G':
        case 'a':
        case 'e':
        case 'f':
        case 'g':
          if ( v106 != 2 )
            goto LABEL_179;
          v55 = sub_72C610(2u);
          v56 = 1;
          v11 = (__int64)v55;
          goto LABEL_164;
        case 'X':
        case 'o':
        case 'x':
          v102 = 1;
          goto LABEL_168;
        case '[':
          goto LABEL_198;
        case 'c':
          goto LABEL_186;
        case 'd':
        case 'i':
          if ( v31 )
          {
            v11 = (__int64)sub_72BA30(3u);
            goto LABEL_174;
          }
          if ( v30 )
          {
            v11 = (__int64)sub_72BA30(1u);
            goto LABEL_174;
          }
          if ( v26 )
          {
            v11 = (__int64)sub_72BA30(9u);
            goto LABEL_174;
          }
          if ( v28 )
          {
            v11 = (__int64)sub_72BA30(unk_4F06AC9);
            goto LABEL_174;
          }
          if ( v27 )
          {
            v11 = (__int64)sub_72BE70(byte_4F06A51[0]);
            goto LABEL_174;
          }
          if ( v29 )
          {
            v11 = (__int64)sub_72BA30(byte_4F06A60[0]);
            goto LABEL_174;
          }
          goto LABEL_290;
        case 'n':
          if ( v31 )
          {
            v114 = v31;
            v82 = sub_72BA30(3u);
            v56 = v114;
            v11 = (__int64)v82;
          }
          else if ( v30 )
          {
            v81 = sub_72BA30(1u);
            v56 = v30;
            v11 = (__int64)v81;
          }
          else if ( v26 )
          {
            v113 = v26;
            v11 = (__int64)sub_72BA30(9u);
            v56 = v113;
          }
          else
          {
            v112 = v28;
            if ( v28 )
            {
              v11 = (__int64)sub_72BA30(unk_4F06AC9);
              v56 = v112;
            }
            else if ( v27 )
            {
              v11 = (__int64)sub_72BE70(byte_4F06A51[0]);
              v56 = v27;
            }
            else
            {
              if ( !v29 )
                goto LABEL_289;
              v60 = sub_72BA30(byte_4F06A60[0]);
              v56 = v29;
              v11 = (__int64)v60;
            }
          }
          goto LABEL_164;
        case 'p':
          goto LABEL_177;
        case 's':
          goto LABEL_178;
        case 'u':
LABEL_168:
          if ( v31 )
          {
            v11 = (__int64)sub_72BA30(4u);
            goto LABEL_174;
          }
          if ( v30 )
          {
            v11 = (__int64)sub_72BA30(2u);
            goto LABEL_174;
          }
          if ( v26 )
          {
            v11 = (__int64)sub_72BA30(0xAu);
            goto LABEL_174;
          }
          if ( v28 )
          {
            v11 = (__int64)sub_72BA30(unk_4F06AC8);
            goto LABEL_174;
          }
          if ( v27 )
          {
            v11 = (__int64)sub_72BA30(byte_4F06A51[0]);
            goto LABEL_174;
          }
          if ( v29 )
          {
            v11 = (__int64)sub_72BE70(byte_4F06A60[0]);
            goto LABEL_174;
          }
          break;
        default:
          goto LABEL_156;
      }
LABEL_188:
      v11 = (__int64)sub_72BA30(6u);
LABEL_174:
      v10 = 0;
      if ( v106 == 2 )
      {
        v56 = 1;
LABEL_164:
        v111 = v56;
        v57 = sub_72D2E0((_QWORD *)v11);
        v10 = v111;
        v11 = v57;
      }
      goto LABEL_165;
    }
    v32 = i + 2;
    v25 = i + 1;
    v115[0] = i + 2;
    switch ( i[1] )
    {
      case 'A':
      case 'E':
      case 'F':
      case 'G':
      case 'a':
      case 'e':
      case 'f':
      case 'g':
LABEL_179:
        v11 = (__int64)sub_72C610(4u);
        goto LABEL_174;
      case 'X':
      case 'o':
      case 'x':
        v102 = 1;
        goto LABEL_190;
      case '[':
        v61 = 1;
LABEL_199:
        if ( v106 != 2 )
          goto LABEL_156;
        v62 = v25[1];
        if ( v62 == 93 )
        {
          v32 = v25 + 2;
          v115[0] = v25 + 2;
          v62 = v25[2];
        }
        else if ( v62 == 94 )
        {
          if ( v25[2] != 93 )
            goto LABEL_204;
          v32 = v25 + 3;
          v115[0] = v25 + 3;
          v62 = v25[3];
        }
        if ( !v62 || v62 == 93 )
          goto LABEL_207;
LABEL_204:
        v63 = v32 + 1;
        do
        {
          v115[0] = v63;
          v64 = *v63++;
        }
        while ( v64 != 93 && v64 );
LABEL_207:
        if ( v61 )
          v65 = sub_72C270();
        else
          v65 = sub_72BA30(byte_4F068B0[0]);
        v56 = v106 == 2;
        v11 = (__int64)v65;
        goto LABEL_164;
      case 'c':
        if ( v106 == 2 )
        {
          v89 = sub_72C270();
          v56 = 1;
          v11 = (__int64)v89;
          goto LABEL_164;
        }
        v90 = sub_72BA30(unk_4F06B81);
        v10 = 0;
        v11 = (__int64)v90;
        break;
      case 'd':
      case 'i':
        v11 = (__int64)sub_72BA30(7u);
        goto LABEL_174;
      case 'n':
        v91 = sub_72BA30(7u);
        v56 = 1;
        v11 = (__int64)v91;
        goto LABEL_164;
      case 'p':
LABEL_177:
        v58 = (_QWORD *)sub_72CBE0();
        v102 = 1;
        v11 = sub_72D2E0(v58);
        goto LABEL_174;
      case 's':
        v98 = 1;
        v92 = sub_72C270();
        v56 = v106 == 2;
        v11 = (__int64)v92;
        goto LABEL_164;
      case 'u':
LABEL_190:
        v11 = (__int64)sub_72BA30(8u);
        goto LABEL_174;
      default:
LABEL_156:
        v10 = v106 == 2;
        v24 = v99;
        v9 = 0;
        v11 = 0;
        i = 0;
        goto LABEL_126;
    }
LABEL_165:
    if ( !v12 )
      break;
    v108 = 0;
    for ( i = v115[0]; ; v115[0] = i )
    {
LABEL_17:
      while ( *i != 37 )
      {
        v11 = 0;
        v12 = 0;
        v9 = 0;
        if ( !*i )
          goto LABEL_82;
        v115[0] = ++i;
      }
      v13 = i + 1;
      v115[0] = i + 1;
      v14 = i[1];
      if ( (_BYTE)v14 != 37 )
        break;
      i += 2;
    }
    v9 = unk_4D04330;
    if ( unk_4D04330 )
    {
      v93 = v10;
      v20 = sub_828610(v115);
      v13 = v115[0];
      v10 = v93;
      v9 = v20;
      v14 = *v115[0];
    }
    if ( v106 != 2 )
    {
      if ( (unsigned __int8)v14 <= 0x30u )
      {
        v21 = 0x1288900000000LL;
        if ( _bittest64(&v21, v14) )
        {
          v22 = v13 + 1;
          do
          {
            v115[0] = v22;
            v14 = *v22;
            v13 = v22;
            if ( (unsigned __int8)v14 > 0x30u )
              break;
            ++v22;
          }
          while ( _bittest64(&v21, v14) );
        }
      }
      if ( (unsigned int)(unsigned __int8)v14 - 48 <= 9 )
      {
        v12 = 0;
LABEL_45:
        i = v13;
        do
        {
          v115[0] = ++i;
          LODWORD(v14) = *i;
        }
        while ( (unsigned int)(v14 - 48) <= 9 );
        goto LABEL_47;
      }
      v12 = 0;
      i = v13;
      if ( (_BYTE)v14 != 42 )
        goto LABEL_47;
      v110 = v10;
      v47 = sub_72BA30(5u);
      v10 = v110;
      v11 = (__int64)v47;
      i = ++v115[0];
      if ( unk_4D04330 )
      {
        v95 = v110;
        v12 = v9;
        v79 = sub_828610(v115);
        v108 = 1;
        v10 = v95;
        v9 = v79;
        goto LABEL_85;
      }
      v108 = 1;
      v24 = v99;
LABEL_126:
      if ( !v24 )
      {
        v45 = v11;
        v2 = a2;
        v4 = v103;
        goto LABEL_97;
      }
      if ( *(_BYTE *)(v24 + 8) )
      {
        v4 = v103;
        v2 = a2;
        v78 = (_DWORD *)sub_6E1A20(v24);
        if ( sub_6E53E0(5, 0x922u, v78) )
          sub_684B30(0x922u, v78);
        goto LABEL_22;
      }
      if ( !i )
      {
        v4 = v103;
        v2 = a2;
        v77 = (_DWORD *)sub_6E1A20(v24);
        if ( sub_6E53E0(5, 0xE2u, v77) )
          sub_684B30(0xE2u, v77);
        goto LABEL_22;
      }
      if ( !v11 )
      {
        v4 = v103;
        v2 = a2;
        if ( !v97 )
        {
          v80 = (_DWORD *)sub_6E1A20(v24);
          if ( sub_6E53E0(5, 0xE1u, v80) )
            sub_684B30(0xE1u, v80);
        }
        goto LABEL_22;
      }
      if ( qword_4F04C50 )
      {
        v50 = *(_QWORD *)(qword_4F04C50 + 32LL);
        if ( v50 )
        {
          if ( (*(_BYTE *)(v50 + 198) & 0x10) != 0 )
          {
            v51 = *(_QWORD *)(v24 + 24);
            v101 = v51;
            v52 = *(_QWORD *)(v51 + 8);
            if ( v10 )
            {
              if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(v51 + 8)) )
                goto LABEL_214;
              v52 = sub_8D46C0(v52);
              if ( (*(_BYTE *)(v52 + 140) & 0xFB) != 8
                || (sub_8D4C10(v52, dword_4F077C4 != 2) & 1) == 0
                || (*(_BYTE *)(v11 + 140) & 0xFB) == 8 && (sub_8D4C10(v11, dword_4F077C4 != 2) & 1) != 0 )
              {
LABEL_215:
                v11 = sub_8D46C0(v11);
                goto LABEL_137;
              }
LABEL_148:
              v53 = *(_BYTE *)(v52 + 140);
              if ( v53 == 12 )
              {
                v54 = v52;
                do
                {
                  v54 = *(_QWORD *)(v54 + 160);
                  v53 = *(_BYTE *)(v54 + 140);
                }
                while ( v53 == 12 );
              }
              if ( v53 )
                sub_6E5D70(5, 0xB5u, (_DWORD *)(v101 + 76), v11, v52);
            }
            else
            {
              if ( v98 )
              {
                if ( (unsigned int)sub_8D2E30(v11) )
                {
                  if ( (unsigned int)sub_8D2E30(v52) )
                  {
                    v52 = sub_8D46C0(v52);
                    goto LABEL_215;
                  }
LABEL_214:
                  if ( (unsigned int)sub_8D3D40(v52) )
                    goto LABEL_215;
                  goto LABEL_148;
                }
              }
LABEL_137:
              while ( *(_BYTE *)(v52 + 140) == 12 )
                v52 = *(_QWORD *)(v52 + 160);
              for ( ; *(_BYTE *)(v11 + 140) == 12; v11 = *(_QWORD *)(v11 + 160) )
                ;
              if ( v52 != v11 && !(unsigned int)sub_8DED30(v11, v52, 1) && !(unsigned int)sub_8D3D40(v52) )
              {
                if ( !(unsigned int)sub_8D2E30(v52)
                  || (v88 = sub_8D46C0(v52), !(unsigned int)sub_8D3D40(v88))
                  || !(unsigned int)sub_8D2E30(v11) )
                {
                  if ( (!(v102 | v98)
                     || (!(unsigned int)sub_8D2930(v11)
                      || !(unsigned int)sub_8D2930(v52)
                      || !(unsigned int)sub_8D7480(v11, v52))
                     && (!v102 || !(unsigned int)sub_8D2E30(v11) || !(unsigned int)sub_8D2E30(v52)))
                    && (dword_4D04964
                     || !(unsigned int)sub_8D2930(v11)
                     || !(unsigned int)sub_8D2E30(v52)
                     || *(_QWORD *)(v11 + 128) != *(_QWORD *)(v52 + 128)
                     || *(_DWORD *)(v11 + 136) != *(_DWORD *)(v52 + 136)) )
                  {
                    if ( !(unsigned int)sub_8DED40(v11, v52) )
                      goto LABEL_148;
                    sub_6E5D70(4, 0xB5u, (_DWORD *)(v101 + 76), v11, v52);
                  }
                }
              }
            }
          }
        }
      }
      v99 = *(_QWORD *)v24;
      if ( *(_QWORD *)v24 && *(_BYTE *)(*(_QWORD *)v24 + 8LL) == 3 )
        v99 = sub_6BBB10((_QWORD *)v24);
      goto LABEL_12;
    }
    v12 = 0;
    if ( (_BYTE)v14 == 42 )
    {
      v12 = 1;
      v115[0] = v13 + 1;
      LOBYTE(v14) = *++v13;
    }
    if ( (unsigned int)(unsigned __int8)v14 - 48 <= 9 )
      goto LABEL_45;
LABEL_56:
    i = v13;
  }
LABEL_82:
  if ( v106 != 2 && v11 )
  {
    v94 = v10;
    v66 = sub_8D6540(v11);
    v108 = 0;
    v10 = v94;
    v11 = v66;
  }
  else
  {
    v108 = 0;
  }
LABEL_85:
  i = v115[0];
  if ( !v9 )
  {
    v24 = v99;
    v9 = v12;
    goto LABEL_126;
  }
  if ( v9 == -2 )
  {
    v2 = a2;
    v4 = v103;
    if ( sub_6E53E0(5, 0x578u, (_DWORD *)(a2 + 80)) )
      sub_684B30(0x578u, (_DWORD *)(a2 + 80));
    goto LABEL_22;
  }
  v24 = *(_QWORD *)(a2 + 64);
  if ( v9 != -1 )
  {
    if ( v9 > 1 && v24 )
    {
      v100 = v115[0];
      v40 = *(_QWORD **)(a2 + 64);
      v41 = v11;
      v42 = v10;
      v43 = v9;
      v44 = 1;
      do
      {
        if ( !*v40 )
        {
          v4 = v103;
          v2 = a2;
          v45 = v41;
          goto LABEL_97;
        }
        if ( *(_BYTE *)(*v40 + 8LL) == 3 )
          v40 = (_QWORD *)sub_6BBB10(v40);
        else
          v40 = (_QWORD *)*v40;
        ++v44;
      }
      while ( v44 < v43 && v40 );
      v11 = v41;
      v10 = v42;
      v97 = 1;
      v24 = (__int64)v40;
      i = v100;
      v9 = v12;
    }
    else
    {
      v97 = 1;
      v9 = v12;
    }
    goto LABEL_126;
  }
  v45 = v11;
  v4 = v103;
  v2 = a2;
  if ( !v24 )
    goto LABEL_97;
  v84 = 1;
  v85 = v45;
  v86 = *(__int64 **)(a2 + 64);
  while ( 1 )
  {
    if ( !*v86 )
    {
LABEL_305:
      v45 = v85;
      goto LABEL_97;
    }
    v86 = (__int64 *)(*(_BYTE *)(*v86 + 8) == 3 ? sub_6BBB10(v86) : *v86);
    if ( ++v84 > 99 )
      break;
    if ( !v86 )
      goto LABEL_305;
  }
  v45 = v85;
  if ( !v86 )
  {
LABEL_97:
    if ( v45 && sub_6E53E0(5, 0xE0u, (_DWORD *)(v2 + 80)) )
      sub_684B30(0xE0u, (_DWORD *)(v2 + 80));
  }
LABEL_22:
  v15 = *(_DWORD *)(v2 + 56);
  if ( v15 )
  {
    v33 = *(_DWORD *)(v2 + 28);
    if ( v15 > v33 )
      goto LABEL_115;
    if ( !*(_QWORD *)v2 )
      goto LABEL_23;
    v34 = v33 - v15;
    for ( j = *(_QWORD *)(*(_QWORD *)v2 + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v36 = (_QWORD *)v105;
    v37 = v34 - 1;
    v38 = **(_QWORD ***)(j + 168);
    if ( v34 )
    {
      do
      {
        if ( *v36 )
        {
          if ( *(_BYTE *)(*v36 + 8LL) == 3 )
            v36 = (_QWORD *)sub_6BBB10(v36);
          else
            v36 = (_QWORD *)*v36;
        }
        else
        {
          v36 = 0;
        }
        if ( v38 )
          v38 = (_QWORD *)*v38;
      }
      while ( v37-- != 0 );
      v105 = (__int64)v36;
    }
    if ( *(_BYTE *)(v105 + 8) )
      goto LABEL_107;
    v67 = *(_QWORD *)(v105 + 24);
    if ( !(unsigned int)sub_8D2660(*(_QWORD *)(v67 + 8)) )
    {
      if ( *(_BYTE *)(v67 + 24) != 2 )
        goto LABEL_107;
      if ( (*(_BYTE *)(v67 + 321) & 8) == 0 )
      {
        if ( *(_BYTE *)(v67 + 25) == 2 )
        {
          v68 = sub_710600(v67 + 152);
          if ( !HIDWORD(qword_4F077B4) || qword_4F077A8 <= 0x9C41u || !v38 )
          {
            if ( v68 )
              goto LABEL_23;
            goto LABEL_110;
          }
          if ( !v68 )
          {
LABEL_115:
            if ( sub_6E53E0(5, 0x60Bu, (_DWORD *)(v2 + 80)) )
              sub_684B30(0x60Bu, (_DWORD *)(v2 + 80));
            goto LABEL_23;
          }
LABEL_256:
          v76 = (_DWORD *)sub_6E1A20(v105);
          if ( sub_6E53E0(5, 0x60Cu, v76) )
            sub_684B30(0x60Cu, v76);
          goto LABEL_23;
        }
LABEL_107:
        if ( !HIDWORD(qword_4F077B4) || qword_4F077A8 <= 0x9C41u || !v38 )
        {
LABEL_110:
          v46 = (_DWORD *)sub_6E1A20(v105);
          if ( sub_6E53E0(5, 0x60Au, v46) )
            sub_684B30(0x60Au, v46);
          goto LABEL_23;
        }
        goto LABEL_115;
      }
    }
    if ( !HIDWORD(qword_4F077B4) || qword_4F077A8 <= 0x9C41u || !v38 )
      goto LABEL_23;
    goto LABEL_256;
  }
LABEL_23:
  if ( v4 )
    *(_BYTE *)(v2 + 17) = 1;
  if ( !*(_BYTE *)(v2 + 18) || *(_BYTE *)(v2 + 21) )
    goto LABEL_35;
  if ( !*(_BYTE *)(v2 + 19) )
  {
    v48 = *(_DWORD *)(v2 + 24);
    if ( v48 == -1 && *(_QWORD *)(v2 + 8) || v48 > *(_DWORD *)(v2 + 28) )
    {
      v49 = (_DWORD *)(v2 + 80);
      if ( sub_6E53E0(5, 0xA5u, v49) )
        sub_684B30(0xA5u, v49);
    }
    goto LABEL_35;
  }
  v16 = *(_QWORD *)(v2 + 8);
  if ( !v16 )
    goto LABEL_35;
  v6 = *(_QWORD *)(v16 + 40) == 0;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v2 + 80);
  if ( v6 )
  {
    v70 = *(_BYTE *)(v16 + 32);
    if ( (v70 & 0x10) != 0 )
    {
      v17 = *(_QWORD *)v2;
      if ( *(_QWORD *)v2 )
        goto LABEL_31;
    }
    if ( dword_4D04964 || (v70 & 4) == 0 )
    {
      if ( (*(_BYTE *)(v16 + 33) & 1) == 0 )
      {
LABEL_237:
        if ( (unsigned int)sub_6E5430() )
          sub_6851C0(0xA5u, (_DWORD *)(v2 + 80));
        v72 = sub_7305B0();
        v73 = v72;
        *(_QWORD *)((char *)v72 + 28) = *(_QWORD *)(v2 + 80);
        while ( 1 )
        {
          v16 = *(_QWORD *)v16;
          if ( !v16 )
            break;
          v74 = sub_7305B0();
          v75 = *(_QWORD *)(v2 + 80);
          v74[2] = v73;
          v73 = v74;
          *(_QWORD *)((char *)v74 + 28) = v75;
        }
        if ( *(_QWORD *)(v2 + 32) )
          *(_QWORD *)(*(_QWORD *)(v2 + 40) + 16LL) = v73;
        else
          *(_QWORD *)(v2 + 32) = v73;
        *(_QWORD *)(v2 + 40) = v72;
        *(_BYTE *)(v2 + 17) = 1;
        goto LABEL_34;
      }
      if ( dword_4F04C44 != -1 )
        goto LABEL_34;
      v71 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v71 + 6) & 6) != 0 )
        goto LABEL_34;
    }
    else
    {
      if ( dword_4F04C44 != -1 )
        goto LABEL_34;
      v71 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v71 + 6) & 6) != 0 || *(_BYTE *)(v71 + 4) == 12 )
        goto LABEL_34;
      if ( (*(_BYTE *)(v16 + 33) & 1) == 0 )
        goto LABEL_237;
    }
    if ( *(_BYTE *)(v71 + 4) == 12 )
      goto LABEL_34;
    goto LABEL_237;
  }
  v17 = *(_QWORD *)v2;
LABEL_31:
  v18 = sub_6E1DA0(v17, v16);
  if ( *(_QWORD *)(v2 + 32) )
    *(_QWORD *)(*(_QWORD *)(v2 + 40) + 16LL) = v18;
  else
    *(_QWORD *)(v2 + 32) = v18;
  *(_QWORD *)(v2 + 40) = v18;
LABEL_34:
  *(_QWORD *)(v2 + 72) = 0;
LABEL_35:
  result = (4 * v107) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFBu;
  *(_BYTE *)(qword_4D03C50 + 21LL) = (4 * v107) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFB;
  return result;
}
