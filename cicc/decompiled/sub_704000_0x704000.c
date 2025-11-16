// Function: sub_704000
// Address: 0x704000
//
_QWORD *__fastcall sub_704000(__int64 a1, __int64 a2)
{
  unsigned __int16 v2; // ax
  __int64 v3; // rbx
  __int64 v4; // rax
  unsigned __int16 v5; // ax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int16 v9; // ax
  const char *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // edx
  __int64 v15; // r13
  int v16; // eax
  const char *v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rsi
  char v20; // dl
  const char *v21; // rax
  char v22; // cl
  __int64 v23; // r12
  char *v24; // rax
  __int64 *v25; // r12
  int v26; // eax
  _DWORD *v27; // r8
  const char *v28; // rcx
  char v29; // dl
  int v30; // ebx
  int v31; // eax
  char v32; // r13
  __int64 v33; // rdi
  __int64 v34; // rax
  const char *v35; // rax
  char v36; // al
  char v37; // r12
  char v38; // al
  int v39; // eax
  unsigned __int8 *v40; // rcx
  unsigned __int8 v41; // r13
  __int64 v42; // rax
  char i; // dl
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  char v47; // dl
  const char *v48; // rax
  const char *v49; // r8
  char v50; // dl
  char *v51; // r13
  __int64 v52; // rax
  int v53; // eax
  __int64 v54; // rax
  char *v55; // rdx
  _DWORD *v56; // r8
  __int64 v57; // rax
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  _DWORD *v60; // r10
  __int64 v61; // r8
  const char *v62; // r13
  __int64 v63; // rax
  __int64 v64; // r8
  _DWORD *v65; // r10
  __int64 v66; // rdx
  char v67; // al
  __int64 v68; // rdx
  char j; // al
  __int64 v70; // rax
  __int64 v71; // r13
  char v72; // al
  const char *v73; // [rsp+8h] [rbp-C8h]
  size_t n; // [rsp+10h] [rbp-C0h]
  int *v75; // [rsp+18h] [rbp-B8h]
  int v76; // [rsp+20h] [rbp-B0h]
  _BYTE *v77; // [rsp+28h] [rbp-A8h]
  int v78; // [rsp+38h] [rbp-98h]
  int v79; // [rsp+3Ch] [rbp-94h]
  _QWORD *v80; // [rsp+40h] [rbp-90h]
  unsigned int v82; // [rsp+50h] [rbp-80h]
  _DWORD *v83; // [rsp+50h] [rbp-80h]
  __int64 v84; // [rsp+50h] [rbp-80h]
  char v85; // [rsp+58h] [rbp-78h]
  __int64 v86; // [rsp+58h] [rbp-78h]
  __int64 v87; // [rsp+58h] [rbp-78h]
  _DWORD *v88; // [rsp+58h] [rbp-78h]
  __int64 v89; // [rsp+60h] [rbp-70h]
  __int64 v90; // [rsp+60h] [rbp-70h]
  _DWORD *v91; // [rsp+60h] [rbp-70h]
  __int64 *v92; // [rsp+68h] [rbp-68h]
  char v93; // [rsp+7Dh] [rbp-53h] BYREF
  _BYTE v94[2]; // [rsp+7Eh] [rbp-52h] BYREF
  _QWORD *v95; // [rsp+80h] [rbp-50h] BYREF
  const char *v96; // [rsp+88h] [rbp-48h] BYREF
  __int64 v97; // [rsp+90h] [rbp-40h] BYREF
  char s[56]; // [rsp+98h] [rbp-38h] BYREF

  v75 = (int *)a2;
  v95 = 0;
  if ( dword_4D04320 )
  {
    a2 = (__int64)&dword_4F063F8;
    sub_684B30(0x64Bu, &dword_4F063F8);
  }
  sub_7BDAB0(a1);
  v79 = 1;
  v2 = word_4F06418[0];
  if ( word_4F06418[0] == 55 )
  {
    sub_7BDAB0(a1);
    v79 = 0;
    v2 = word_4F06418[0];
  }
  if ( v2 != 7 && v2 != 25 )
    return 0;
  v92 = (__int64 *)&v95;
LABEL_8:
  v3 = sub_726370();
  *v92 = v3;
  v80 = v95;
  v4 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 75LL);
  ++*(_BYTE *)(v4 + 63);
  ++*(_BYTE *)(v4 + 154);
  *(_QWORD *)(v3 + 28) = *(_QWORD *)&dword_4F063F8;
  v5 = word_4F06418[0];
  if ( word_4F06418[0] == 25 )
  {
    sub_7BDAB0(a1);
    ++*(_BYTE *)(qword_4F061C8 + 34LL);
    if ( word_4F06418[0] == 1 )
    {
      v23 = qword_4D04A00;
      v24 = (char *)sub_7247C0(*(_QWORD *)(qword_4D04A00 + 16) + 1LL);
      *(_QWORD *)(v3 + 8) = v24;
      a2 = *(_QWORD *)(v23 + 8);
      strcpy(v24, (const char *)a2);
      sub_7BDAB0(a1);
    }
    else
    {
      sub_6851D0(0x28u);
    }
    if ( word_4F06418[0] != 26 )
      sub_6851D0(0x11u);
    sub_7BDAB0(a1);
    --*(_BYTE *)(qword_4F061C8 + 34LL);
    v5 = word_4F06418[0];
  }
  if ( v5 != 7 )
  {
    v6 = 1038;
    sub_6851D0(0x40Eu);
LABEL_11:
    *(_QWORD *)(v3 + 16) = 0;
    goto LABEL_12;
  }
  if ( (unk_4F063A8 & 7) != 0 )
  {
    v6 = 2479;
    sub_6851D0(0x9AFu);
    goto LABEL_11;
  }
  v11 = (const char *)unk_4F063B8;
  sub_7BDAB0(a1);
  if ( word_4F06418[0] != 27 )
  {
    v6 = 125;
    sub_6851D0(0x7Du);
    goto LABEL_11;
  }
  sub_7B8B50(a1, a2, v12, v13);
  v14 = 0;
  LODWORD(v15) = v79 ^ 1;
  if ( v11 )
  {
    if ( v79 )
      v15 = strchr(v11, 43) != 0;
    v16 = *(unsigned __int8 *)v11;
    if ( (_BYTE)v16 )
    {
      v17 = v11;
      while ( 1 )
      {
        v18 = (unsigned int)(v16 - 60);
        if ( (unsigned __int8)v18 <= 0x3Au )
        {
          v19 = 0x40A080000000005LL;
          if ( _bittest64(&v19, v18) )
            break;
        }
        v16 = *(unsigned __int8 *)++v17;
        if ( !(_BYTE)v16 )
          goto LABEL_209;
      }
      v14 = 1;
    }
    else
    {
LABEL_209:
      v14 = 0;
    }
  }
  a2 = (unsigned int)v15;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  v77 = (_BYTE *)sub_6B7AB0(v79, v15, v14);
  if ( word_4F06418[0] == 28 )
  {
    v6 = a1;
    sub_7BDAB0(a1);
  }
  else if ( v77[24] )
  {
    v6 = 18;
    sub_6851D0(0x12u);
  }
  else
  {
    sub_7BE1A0();
    a2 = 18;
    v6 = 28;
    sub_7BE280(28, 18, 0, 0);
  }
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  *(_QWORD *)(v3 + 16) = 0;
  if ( !v11 || !v77 )
    goto LABEL_12;
  v94[1] = 0;
  v96 = v11;
  v20 = *v11;
  if ( !*v11 )
    goto LABEL_53;
  v21 = v11 + 1;
  v22 = 0;
  while ( 1 )
  {
    if ( v20 == 43 )
    {
      v22 = 3;
      goto LABEL_52;
    }
    if ( v20 != 61 )
      break;
    v22 |= 2u;
LABEL_52:
    v96 = v21;
    v20 = *v21++;
    if ( !v20 )
      goto LABEL_53;
  }
  v85 = v22;
  v25 = (__int64 *)(v3 + 16);
  v26 = sub_703A60();
  v28 = v96;
  v78 = v26;
  v29 = *v96;
  if ( !*v96 )
  {
    v76 = 1;
LABEL_71:
    v36 = v85;
    if ( !v85 )
      v36 = 1;
    v37 = v36;
    v38 = v36 & 2;
    if ( v79 )
    {
      if ( !v38 )
      {
        a2 = v3 + 28;
        v6 = 1123;
        sub_6851C0(0x463u, (_DWORD *)(v3 + 28));
        goto LABEL_12;
      }
    }
    else if ( v38 )
    {
      a2 = v3 + 28;
      v6 = 1124;
      sub_6851C0(0x464u, (_DWORD *)(v3 + 28));
      goto LABEL_12;
    }
    a2 = (__int64)v75;
    v39 = *v75;
    if ( *v75 )
    {
      if ( v39 != v76 && v39 != -1 )
      {
        a2 = v3 + 28;
        v6 = 2640;
        sub_6851C0(0xA50u, (_DWORD *)(v3 + 28));
        *v75 = -1;
        goto LABEL_12;
      }
    }
    else
    {
      *v75 = v76;
    }
    if ( v78 )
    {
      v40 = *(unsigned __int8 **)(v3 + 16);
      v41 = *v40;
      if ( !unk_4D0462C )
      {
        v42 = *(_QWORD *)v77;
        for ( i = *(_BYTE *)(*(_QWORD *)v77 + 140LL); i == 12; i = *(_BYTE *)(v42 + 140) )
          v42 = *(_QWORD *)(v42 + 160);
        a2 = v41;
        if ( v41 == 2 )
        {
          v40 = (unsigned __int8 *)*((_QWORD *)v40 + 2);
          if ( !v40 || (a2 = *v40, (_BYTE)a2 == 2) )
          {
LABEL_53:
            a2 = v3 + 28;
            v6 = 1122;
            sub_6851C0(0x462u, (_DWORD *)(v3 + 28));
            goto LABEL_12;
          }
        }
        switch ( i )
        {
          case 2:
            v58 = *(_QWORD *)(v42 + 128);
            if ( v58 > 0x10 )
              goto LABEL_162;
            if ( (_BYTE)a2 == 32 )
              break;
            if ( v58 == 1 )
            {
              LODWORD(v58) = 1;
              goto LABEL_162;
            }
            if ( (_BYTE)a2 != 21 && v58 == 2 )
            {
              LODWORD(v58) = 2;
              goto LABEL_162;
            }
            if ( (_BYTE)a2 != 20 && v58 == 4 )
            {
              LODWORD(v58) = 4;
              goto LABEL_162;
            }
            if ( v58 == 8 && (_BYTE)a2 != 22 )
              goto LABEL_161;
            if ( (_BYTE)a2 != 42 && v58 == 16 )
            {
              LODWORD(v58) = 16;
              goto LABEL_162;
            }
            break;
          case 3:
            v57 = *(_QWORD *)(v42 + 128);
            if ( v57 == 4 )
            {
              LODWORD(v58) = 4;
              if ( (_BYTE)a2 != 24 )
                goto LABEL_162;
            }
            else if ( (_BYTE)a2 != 38 )
            {
              goto LABEL_160;
            }
            break;
          case 6:
            if ( (_BYTE)a2 == 23 )
              break;
            goto LABEL_158;
          case 7:
          case 8:
LABEL_158:
            v57 = unk_4F06A68;
            if ( unk_4F06A68 == 4 )
            {
              LODWORD(v58) = 4;
              if ( (_BYTE)a2 != 20 )
                goto LABEL_162;
            }
            else
            {
              if ( (_BYTE)a2 == 22 )
                break;
LABEL_160:
              if ( v57 == 8 )
              {
LABEL_161:
                LODWORD(v58) = 8;
LABEL_162:
                LOWORD(v97) = (unsigned __int8)aXg0123456789rh[a2];
                sprintf(s, "%d", v58);
                v6 = 3533;
                a2 = v3 + 28;
                sub_686610(0xDCDu, (_DWORD *)(v3 + 28), (__int64)s, (__int64)&v97);
                goto LABEL_12;
              }
            }
            break;
          case 14:
          case 21:
            break;
          default:
            a2 = v3 + 28;
            v6 = 3530;
            sub_6851C0(0xDCAu, (_DWORD *)(v3 + 28));
            goto LABEL_12;
        }
      }
      if ( dword_4F04C44 == -1 )
      {
        v59 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v59 + 6) & 6) == 0 && *(_BYTE *)(v59 + 4) != 12 && (v41 == 32 || v41 == 23) )
        {
          a2 = sub_724DC0(v6, a2, qword_4F04C68, v40, v27, 1);
          v97 = a2;
          if ( !(unsigned int)sub_719770(v77, a2, 1, 1) )
          {
            sub_724E30(&v97);
            a2 = 3531;
            if ( v41 != 32 )
              a2 = 3722;
            v60 = (_DWORD *)(v3 + 28);
            goto LABEL_184;
          }
          v90 = v97;
          if ( v41 != 32 )
          {
            if ( *(_BYTE *)(v97 + 173) != 6 || *(_BYTE *)(v97 + 176) != 1 )
            {
              a2 = 3727;
              sub_684AA0(7u, 0xE8Fu, (_DWORD *)(v3 + 28));
              goto LABEL_235;
            }
            v61 = *(_QWORD *)(v97 + 184);
            v62 = *(const char **)(v61 + 8);
            v86 = v61;
            if ( !v62 )
              v62 = "<name not available>";
            v63 = sub_8D21C0(*(_QWORD *)(v61 + 120));
            v64 = v86;
            v65 = (_DWORD *)(v3 + 28);
            if ( *(_BYTE *)(v63 + 140) == 8 )
            {
              v66 = *(_QWORD *)(v63 + 160);
              if ( (*(_BYTE *)(v66 + 140) & 0xFB) == 8 )
              {
                v84 = v86;
                v87 = *(_QWORD *)(v63 + 160);
                v67 = sub_8D4C10(v66, dword_4F077C4 != 2);
                v64 = v84;
                v65 = (_DWORD *)(v3 + 28);
                if ( (v67 & 1) != 0 )
                {
                  v68 = v87;
                  j = *(_BYTE *)(v87 + 140);
                  if ( (j & 0xFB) != 8 )
                    goto LABEL_233;
                  v72 = sub_8D4C10(v87, dword_4F077C4 != 2);
                  v64 = v84;
                  v65 = (_DWORD *)(v3 + 28);
                  if ( (v72 & 2) == 0 )
                  {
                    v68 = v87;
                    for ( j = *(_BYTE *)(v87 + 140); j == 12; j = *(_BYTE *)(v68 + 140) )
                      v68 = *(_QWORD *)(v68 + 160);
LABEL_233:
                    if ( j == 2 && !*(_BYTE *)(v68 + 160) )
                    {
                      if ( (*(_BYTE *)(v64 + 176) & 8) != 0 || *(_BYTE *)(v64 + 136) > 2u )
                      {
                        a2 = 3724;
                        sub_6849F0(7u, 0xE8Cu, v65, (__int64)v62);
                      }
                      else if ( (*(_BYTE *)(v64 + 89) & 4) != 0 && (*(_BYTE *)(v64 + 172) & 4) == 0 )
                      {
                        a2 = 3726;
                        sub_6849F0(7u, 0xE8Eu, v65, (__int64)v62);
                      }
                      else
                      {
                        v88 = v65;
                        sub_72F9F0(v64, 0, &v93, s);
                        if ( v93 == 1 && **(_QWORD **)s )
                        {
                          v71 = sub_724D80(*(unsigned __int8 *)(**(_QWORD **)s + 173LL));
                          sub_72A510(**(_QWORD **)s, v71);
                          a2 = v90;
                          sub_72D410(v71, v90);
                          v77 = (_BYTE *)sub_73A720(v97);
                        }
                        else
                        {
                          a2 = 3725;
                          sub_6849F0(7u, 0xE8Du, v88, (__int64)v62);
                        }
                      }
                      goto LABEL_235;
                    }
                  }
                }
              }
            }
            v91 = v65;
            v70 = sub_8D21C0(*(_QWORD *)(v64 + 120));
            a2 = 3723;
            sub_686310(7u, 0xE8Bu, v91, (__int64)v62, v70);
LABEL_235:
            sub_724E30(&v97);
            goto LABEL_157;
          }
          if ( !(unsigned int)sub_8D2780(*(_QWORD *)(v97 + 128)) || *(_BYTE *)(v97 + 173) != 1 )
          {
            sub_724E30(&v97);
            v60 = (_DWORD *)(v3 + 28);
            a2 = 3531;
LABEL_184:
            v6 = 7;
            sub_684AA0(7u, a2, v60);
            goto LABEL_12;
          }
          v77 = (_BYTE *)sub_73A720(v97);
          sub_724E30(&v97);
        }
      }
    }
LABEL_157:
    *(_BYTE *)(v3 + 24) = v37;
    *(_QWORD *)(v3 + 40) = v77;
    goto LABEL_13;
  }
  v82 = 0;
  v76 = 1;
  v89 = v3;
  v30 = v26;
  while ( 2 )
  {
    switch ( v29 )
    {
      case '!':
        v33 = 7;
        v32 = 7;
        goto LABEL_66;
      case '#':
        v47 = v28[1];
        v48 = v28 + 1;
        if ( v47 && v47 != 44 )
        {
          do
          {
            v49 = v48;
            v50 = *++v48;
          }
          while ( v50 && v50 != 44 );
          v51 = 0;
          if ( v49 != v28 )
          {
            v73 = v49;
            n = v49 - v28;
            v51 = (char *)sub_7247C0(v49 - v28 + 1);
            a2 = (__int64)(v96 + 1);
            strncpy(v51, v96 + 1, n);
            v51[n] = 0;
            v96 = v73;
          }
        }
        else
        {
          v51 = 0;
        }
        v52 = sub_726340(4);
        *v25 = v52;
        *(_QWORD *)(v52 + 8) = v51;
        v32 = 4;
        v44 = *v25;
        goto LABEL_84;
      case '%':
        v33 = 3;
        v32 = 3;
        goto LABEL_66;
      case '&':
        v33 = 2;
        v32 = 2;
        goto LABEL_66;
      case '*':
        v33 = 5;
        v32 = 5;
        goto LABEL_66;
      case ',':
        ++v76;
        v33 = 1;
        v32 = 1;
        goto LABEL_66;
      case '0':
        v33 = 10;
        v32 = 10;
        goto LABEL_66;
      case '1':
        v33 = 11;
        v32 = 11;
        goto LABEL_66;
      case '2':
        v33 = 12;
        v32 = 12;
        goto LABEL_66;
      case '3':
        v33 = 13;
        v32 = 13;
        goto LABEL_66;
      case '4':
        v33 = 14;
        v32 = 14;
        goto LABEL_66;
      case '5':
        v33 = 15;
        v32 = 15;
        goto LABEL_66;
      case '6':
        v33 = 16;
        v32 = 16;
        goto LABEL_66;
      case '7':
        v33 = 17;
        v32 = 17;
        goto LABEL_66;
      case '8':
        v33 = 18;
        v32 = 18;
        goto LABEL_66;
      case '9':
        v33 = 19;
        v32 = 19;
        goto LABEL_66;
      case '<':
        v33 = 29;
        v32 = 29;
        goto LABEL_66;
      case '>':
        v33 = 30;
        v32 = 30;
        goto LABEL_66;
      case '?':
        v33 = 6;
        v32 = 6;
        goto LABEL_66;
      case 'A':
        v32 = 44;
        v44 = sub_726340(44);
        *v25 = v44;
        goto LABEL_84;
      case 'C':
        v33 = 23;
        v32 = 23;
        goto LABEL_66;
      case 'D':
        v33 = 40;
        v32 = 40;
        goto LABEL_66;
      case 'E':
      case 'F':
        v33 = 34;
        v32 = 34;
        goto LABEL_66;
      case 'G':
        v32 = 56;
        v44 = sub_726340(56);
        *v25 = v44;
        goto LABEL_84;
      case 'H':
        v32 = 57;
        v44 = sub_726340(57);
        *v25 = v44;
        goto LABEL_84;
      case 'I':
        v32 = 50;
        v44 = sub_726340(50);
        *v25 = v44;
        goto LABEL_84;
      case 'J':
        v32 = 51;
        v44 = sub_726340(51);
        *v25 = v44;
        goto LABEL_84;
      case 'K':
        v32 = 53;
        v44 = sub_726340(53);
        *v25 = v44;
        goto LABEL_84;
      case 'L':
        v32 = 55;
        v44 = sub_726340(55);
        *v25 = v44;
        goto LABEL_84;
      case 'M':
        v32 = 52;
        v44 = sub_726340(52);
        *v25 = v44;
        goto LABEL_84;
      case 'N':
        v32 = 54;
        v44 = sub_726340(54);
        *v25 = v44;
        goto LABEL_84;
      case 'Q':
        v33 = 43;
        v32 = 43;
        goto LABEL_66;
      case 'R':
        v33 = 41;
        v32 = 41;
        goto LABEL_66;
      case 'S':
        v33 = 39;
        v32 = 39;
        goto LABEL_66;
      case 'V':
        v33 = 28;
        v32 = 28;
        goto LABEL_66;
      case 'X':
        v33 = 8;
        v32 = 8;
        goto LABEL_66;
      case 'Y':
        v32 = 48;
        v44 = sub_726340(48);
        *v25 = v44;
        goto LABEL_84;
      case 'Z':
        v32 = 59;
        v44 = sub_726340(59);
        *v25 = v44;
        goto LABEL_84;
      case '[':
        a2 = (__int64)v80;
        v53 = sub_703AC0(&v96, v80, 0, 0, (_DWORD *)(v89 + 28));
        v27 = (_DWORD *)(v89 + 28);
        if ( v53 < 0 )
        {
          if ( !v30 )
            goto LABEL_174;
LABEL_139:
          v35 = v96;
          v82 = 1;
          if ( v96[1] )
          {
            a2 = (__int64)v27;
            sub_6851C0(0xDC9u, v27);
            v35 = v96;
          }
          goto LABEL_69;
        }
        if ( v53 > 9 )
        {
          a2 = v89 + 28;
          sub_6851C0(0x587u, (_DWORD *)(v89 + 28));
          v27 = (_DWORD *)(v89 + 28);
          if ( !v30 )
          {
LABEL_174:
            v82 = 1;
            v35 = v96;
            goto LABEL_69;
          }
          goto LABEL_139;
        }
        v32 = v53 + 10;
        v54 = sub_726340((unsigned __int8)(v53 + 10));
        v27 = (_DWORD *)(v89 + 28);
        *v25 = v54;
        v25 = (__int64 *)(v54 + 16);
        if ( v30 )
        {
          if ( v96[1] )
          {
LABEL_87:
            a2 = (__int64)v27;
            sub_6851C0(0xDC9u, v27);
            v82 = 1;
            if ( (unsigned __int8)v85 > 1u && v32 == 23 )
              goto LABEL_64;
          }
          else
          {
            v82 = 0;
LABEL_63:
            if ( (unsigned __int8)v85 > 1u && v32 == 23 )
            {
LABEL_64:
              a2 = 3721;
              sub_684AA0(7u, 0xE89u, (_DWORD *)(v89 + 28));
            }
          }
        }
        else
        {
          v82 = 0;
        }
LABEL_88:
        v35 = v96;
LABEL_69:
        v28 = v35 + 1;
        v96 = v35 + 1;
        v29 = v35[1];
        if ( v29 )
          continue;
        v6 = v82;
        v3 = v89;
        if ( !v82 )
          goto LABEL_71;
LABEL_12:
        v7 = sub_7305B0(v6, a2);
        *(_BYTE *)(v3 + 24) = 0;
        *(_QWORD *)(v3 + 40) = v7;
LABEL_13:
        v8 = qword_4F061C8;
        --*(_BYTE *)(qword_4F061C8 + 75LL);
        --*(_BYTE *)(v8 + 63);
        --*(_BYTE *)(v8 + 154);
        v92 = (__int64 *)*v92;
        v9 = word_4F06418[0];
        if ( word_4F06418[0] != 55 )
        {
          if ( word_4F06418[0] == 67 )
          {
            sub_7BDAB0(a1);
            if ( word_4F06418[0] != 25 && word_4F06418[0] != 7 )
            {
              sub_6851D0(0x46Du);
              v9 = word_4F06418[0];
              goto LABEL_15;
            }
          }
          else
          {
LABEL_15:
            if ( v9 != 7 && v9 != 25 )
              return v80;
          }
          goto LABEL_8;
        }
        if ( v79 )
        {
          sub_7BDAB0(a1);
          v79 = 0;
          v9 = word_4F06418[0];
          goto LABEL_15;
        }
        return v80;
      case 'a':
        v33 = 35;
        v32 = 35;
        goto LABEL_66;
      case 'b':
        v33 = 36;
        v32 = 36;
        goto LABEL_66;
      case 'c':
        v33 = 37;
        v32 = 37;
        goto LABEL_66;
      case 'd':
        v33 = 38;
        v32 = 38;
        goto LABEL_66;
      case 'e':
        v32 = 58;
        v44 = sub_726340(58);
        *v25 = v44;
        goto LABEL_84;
      case 'f':
        v33 = 24;
        v32 = 24;
        goto LABEL_66;
      case 'g':
        v33 = 9;
        v32 = 9;
        goto LABEL_66;
      case 'h':
        v33 = 21;
        v32 = 21;
        goto LABEL_66;
      case 'i':
        v45 = 31;
        v32 = 31;
        if ( !v30 )
          goto LABEL_90;
        v55 = "i";
        v56 = (_DWORD *)(v89 + 28);
        goto LABEL_138;
      case 'l':
        v33 = 22;
        v32 = 22;
        goto LABEL_66;
      case 'm':
        v45 = 25;
        v32 = 25;
        if ( !v30 )
          goto LABEL_90;
        v55 = "m";
        v56 = (_DWORD *)(v89 + 28);
        goto LABEL_138;
      case 'n':
        v33 = 32;
        v32 = 32;
        goto LABEL_66;
      case 'o':
        v33 = 27;
        v32 = 27;
        goto LABEL_66;
      case 'p':
        v33 = 26;
        v32 = 26;
        goto LABEL_66;
      case 'q':
        v33 = 42;
        v32 = 42;
        goto LABEL_66;
      case 'r':
        v33 = 20;
        v32 = 20;
LABEL_66:
        v34 = sub_726340(v33);
        *v25 = v34;
        v25 = (__int64 *)(v34 + 16);
        if ( !v30 )
          goto LABEL_91;
        v35 = v96;
        if ( !v96[1] )
          goto LABEL_63;
        if ( v32 == 2 )
          goto LABEL_69;
        goto LABEL_86;
      case 's':
        v45 = 33;
        v32 = 33;
        if ( v30 )
        {
          v55 = "s";
          v56 = (_DWORD *)(v89 + 28);
LABEL_138:
          a2 = (__int64)v56;
          v83 = v56;
          sub_6851A0(0xDCCu, v56, (__int64)v55);
          v27 = v83;
          goto LABEL_139;
        }
LABEL_90:
        v46 = sub_726340(v45);
        *v25 = v46;
        v25 = (__int64 *)(v46 + 16);
LABEL_91:
        if ( v32 == 23 )
        {
          a2 = 3720;
          sub_684AA0(7u, 0xE88u, (_DWORD *)(v89 + 28));
          v35 = v96;
          goto LABEL_69;
        }
        goto LABEL_88;
      case 't':
        v32 = 45;
        v44 = sub_726340(45);
        *v25 = v44;
        goto LABEL_84;
      case 'u':
        v32 = 46;
        v44 = sub_726340(46);
        *v25 = v44;
        goto LABEL_84;
      case 'x':
        v32 = 47;
        v44 = sub_726340(47);
        *v25 = v44;
        goto LABEL_84;
      case 'y':
        v32 = 49;
        v44 = sub_726340(49);
        *v25 = v44;
LABEL_84:
        v25 = (__int64 *)(v44 + 16);
        if ( !v30 )
          goto LABEL_88;
        if ( !v96[1] )
          goto LABEL_63;
LABEL_86:
        v27 = (_DWORD *)(v89 + 28);
        goto LABEL_87;
      default:
        if ( !v30 )
          goto LABEL_88;
        v94[0] = v29;
        v31 = ispunct(*(unsigned __int8 *)v28);
        a2 = v89 + 28;
        v32 = 0;
        sub_6851A0((v31 == 0) + 1120, (_DWORD *)(v89 + 28), (__int64)v94);
        v27 = (_DWORD *)(v89 + 28);
        if ( v96[1] )
          goto LABEL_87;
        v82 = 1;
        goto LABEL_63;
    }
  }
}
