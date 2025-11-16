// Function: sub_7BC390
// Address: 0x7bc390
//
_DWORD *sub_7BC390()
{
  _BYTE *v0; // rdx
  char v1; // al
  unsigned __int8 v3; // r12
  char *v4; // rdx
  _BYTE *v5; // rdi
  char v6; // bl
  _BYTE *v7; // r9
  char v8; // al
  unsigned __int8 *v9; // r14
  unsigned __int8 v10; // dl
  int v11; // eax
  _BYTE *v12; // r12
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 *v15; // rax
  unsigned __int64 v16; // rax
  _BYTE *v17; // rdi
  int v18; // r14d
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned __int16 v23; // ax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  _BYTE *v28; // r12
  __int64 v29; // rax
  unsigned int v30; // r13d
  char v31; // al
  __int64 v32; // rdx
  int v33; // eax
  bool v34; // zf
  int v35; // eax
  unsigned __int8 v36; // cl
  _BYTE *v37; // rdi
  char v38; // r13
  unsigned __int8 *v39; // rsi
  __int64 v40; // rsi
  __int64 v41; // rax
  unsigned __int8 *v42; // rcx
  __int64 v43; // rax
  unsigned __int8 *v44; // rdi
  __int64 v45; // rsi
  __int64 v46; // rax
  __int16 v47; // r9
  __int16 v48; // ax
  __int64 *v49; // rax
  __int64 v50; // rcx
  __int64 *v51; // rdx
  _BYTE *i; // rax
  __int64 *v53; // rax
  char v54; // dl
  __int64 v55; // rdx
  __int64 v56; // rax
  int v57; // eax
  unsigned __int8 *v58; // rdx
  _BYTE *v59; // r8
  int j; // eax
  _BYTE *v61; // rdx
  int v62; // ecx
  int v63; // ebx
  __int64 v64; // rax
  int v65; // eax
  int v66; // eax
  __int64 v67; // rdi
  __int64 *v68; // rax
  __int64 v69; // rcx
  __int64 *v70; // rdx
  int v71; // eax
  unsigned __int8 *v72; // rax
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // rax
  __int16 v76; // r10
  __int16 v77; // dx
  __int16 v78; // r10
  __int16 v79; // dx
  __int16 v80; // r10
  __int16 v81; // dx
  __int16 v82; // r10
  __int16 v83; // dx
  __int16 v84; // r10
  __int16 v85; // dx
  __int16 v86; // r10
  __int16 v87; // dx
  unsigned int v88; // [rsp+4h] [rbp-7Ch]
  int v89; // [rsp+8h] [rbp-78h]
  int v90; // [rsp+Ch] [rbp-74h]
  _BYTE *v91; // [rsp+10h] [rbp-70h]
  unsigned __int64 v92; // [rsp+20h] [rbp-60h]
  _BYTE *v93; // [rsp+20h] [rbp-60h]
  unsigned __int64 v94; // [rsp+20h] [rbp-60h]
  _BYTE *v95; // [rsp+20h] [rbp-60h]
  unsigned __int64 v96; // [rsp+20h] [rbp-60h]
  _BYTE *v97; // [rsp+20h] [rbp-60h]
  unsigned __int64 v98; // [rsp+20h] [rbp-60h]
  _BYTE *v99; // [rsp+20h] [rbp-60h]
  unsigned __int64 v100; // [rsp+20h] [rbp-60h]
  unsigned __int64 v101; // [rsp+20h] [rbp-60h]
  _BYTE *v102; // [rsp+20h] [rbp-60h]
  unsigned int v103; // [rsp+28h] [rbp-58h]
  unsigned __int8 v104; // [rsp+28h] [rbp-58h]
  unsigned __int8 v105; // [rsp+28h] [rbp-58h]
  _BYTE *v106; // [rsp+28h] [rbp-58h]
  int v107; // [rsp+3Ch] [rbp-44h] BYREF
  __int64 v108; // [rsp+40h] [rbp-40h] BYREF
  unsigned __int64 v109[7]; // [rsp+48h] [rbp-38h] BYREF

  qword_4F06410 = 0;
  v90 = unk_4F063E8;
  if ( unk_4F063E8 )
    v90 = dword_4F063EC;
  LOBYTE(v89) = 0;
  v91 = 0;
  v0 = qword_4F06460;
  v1 = *qword_4F06460;
  while ( 2 )
  {
    if ( v1 == 32 )
    {
LABEL_12:
      v90 |= 2u;
      v4 = v0 + 1;
      do
      {
        qword_4F06460 = v4;
        v1 = *v4++;
      }
      while ( v1 == 32 );
    }
LABEL_5:
    switch ( v1 )
    {
      case 0:
        v5 = qword_4F06460;
        v6 = qword_4F06460[1];
        if ( v6 != 2 )
        {
          if ( (v6 & 0xFD) == 1 )
          {
            v28 = qword_4F06420;
            if ( qword_4F06420 )
            {
              if ( qword_4F06460 - qword_4F06420 > 0 )
              {
                v56 = sub_7AEE00(qword_4F06420, qword_4F06460 - qword_4F06420, 0, 0);
                v5 = qword_4F06460;
                *(_BYTE *)(v56 + 48) &= 0x7Du;
                *(_WORD *)(v56 + 51) = 768;
                *(_QWORD *)(v56 + 64) = v56 + 51;
                *(_QWORD *)(v56 + 56) = v56 + 51;
              }
              qword_4F06420 = 0;
            }
            if ( v6 == 1 )
            {
              unk_4F06470 = &v5[-qword_4F06498];
              if ( dword_4F17FA8 || sub_7B2B10(1, 0) )
                goto LABEL_7;
              v0 = qword_4F06460;
            }
            else
            {
              v53 = sub_7AEFF0((unsigned __int64)v5);
              v54 = *((_BYTE *)v53 + 48);
              if ( (v54 & 8) != 0 )
              {
                if ( qword_4D03BD8 <= 1u )
                  goto LABEL_7;
                *((_BYTE *)v53 + 48) = v54 & 0xF7;
              }
              else if ( (v53[6] & 1) != 0 )
              {
                goto LABEL_7;
              }
              v55 = v53[2];
              if ( !v55 )
              {
                v55 = qword_4F06498;
                if ( unk_4F06478 )
                  v55 = unk_4F06470 + qword_4F06498;
              }
              v0 = (_BYTE *)(v53[4] + v55);
              qword_4F06460 = v0;
              unk_4F063D0 = v53;
            }
            if ( v28 )
              goto LABEL_58;
LABEL_83:
            v1 = *v0;
          }
          else
          {
            switch ( v6 )
            {
              case 4:
                goto LABEL_234;
              case 5:
              case 8:
                goto LABEL_7;
              case 6:
                if ( unk_4D04328 )
                {
                  sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)dword_4F07508);
                  sub_684AC0(5u, 0x4A8u);
                }
                else
                {
                  sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)dword_4F07508);
                  sub_684AC0(7u, 0x35Du);
                }
                v90 |= 2u;
                v0 = qword_4F06460 + 2;
                qword_4F06460 = v0;
                v1 = *v0;
                break;
              case 7:
                unk_4F063E4 = 1;
LABEL_234:
                v0 = qword_4F06460 + 2;
                qword_4F06460 += 2;
                v1 = v5[2];
                continue;
              case 9:
                qword_4F061D0 = *(_QWORD *)&dword_4F077C8;
                v0 = qword_4F06460 + 2;
                qword_4F06460 += 2;
                v1 = v5[2];
                break;
              case 10:
                sub_81B840();
                v0 = qword_4F06460 + 2;
                qword_4F06460 = v0;
                v1 = *v0;
                break;
              case 11:
                v0 = qword_4F06460 + 2;
                qword_4F06460 += 2;
                if ( unk_4D03BD0 )
                {
                  v1 = v5[2];
                  if ( v1 == 44 )
                  {
                    unk_4F063E0 = 1;
                    v1 = v5[2];
                  }
                }
                else
                {
                  v0 = v5 + 3;
                  v90 |= 2u;
                  qword_4F06460 = v5 + 3;
                  v1 = v5[3];
                }
                break;
              case 12:
                v0 = qword_4F06460 + 2;
                qword_4F06460 += 2;
                unk_4F063DC = 1;
                v1 = v5[2];
                break;
              case 13:
                v0 = qword_4F06460 + 2;
                unk_4F063D8 = 1;
                qword_4F06460 += 2;
                v1 = v5[2];
                break;
              default:
                sub_721090();
            }
          }
          continue;
        }
        if ( !(unk_4F061F8 | dword_4D03D18) )
        {
          v0 = qword_4F06460 + 2;
          v90 |= 2u;
          qword_4F06460 += 2;
          v1 = v5[2];
          continue;
        }
LABEL_7:
        dword_4F063EC = v90;
        return &dword_4F063EC;
      case 9:
      case 11:
      case 12:
        goto LABEL_11;
      case 10:
        v12 = qword_4F06460;
        v13 = sub_7AF1D0((unsigned __int64)qword_4F06460);
        v0 = *(_BYTE **)(v13 + 56);
        if ( v0 == *(_BYTE **)(v13 + 64) )
          v0 = &qword_4F06460[*(_QWORD *)(v13 + 32)];
        qword_4F06460 = v0;
        if ( (*(_BYTE *)(v13 + 48) & 1) != 0 )
        {
          qword_4F06460 = v12;
          goto LABEL_7;
        }
        if ( !qword_4F06420 )
          goto LABEL_83;
        v14 = v12 - qword_4F06420;
        if ( v14 > 0 )
        {
          v29 = sub_7AEE00(qword_4F06420, v14, 0, 0);
          *(_BYTE *)(v29 + 48) &= 0x7Du;
          *(_QWORD *)(v29 + 64) = v29 + 51;
          *(_QWORD *)(v29 + 56) = v29 + 51;
          v0 = qword_4F06460;
          *(_WORD *)(v29 + 51) = 768;
          qword_4F06420 = v0;
          v1 = *v0;
        }
        else
        {
LABEL_58:
          qword_4F06420 = v0;
          v1 = *v0;
        }
        continue;
      case 13:
        v3 = 4;
        if ( dword_4D04964 )
          v3 = unk_4F07471;
        sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)dword_4F07508);
        sub_684AC0(v3, 0x565u);
LABEL_11:
        v90 |= 2u;
        v0 = qword_4F06460 + 1;
        qword_4F06460 = v0;
        v1 = *v0;
        if ( *v0 == 32 )
          goto LABEL_12;
        goto LABEL_5;
      case 47:
        v7 = qword_4F06460;
        v8 = qword_4F06460[1];
        if ( v8 != 42 && (!unk_4D042A8 || v8 != 47) )
          goto LABEL_7;
        if ( (unsigned __int64)qword_4F06460 < qword_4F06498 || (unsigned __int64)qword_4F06460 >= qword_4F06490 )
        {
          if ( !unk_4D04954 || dword_4F084D8 || v8 != 42 )
            goto LABEL_7;
          if ( (unsigned __int64)qword_4F06460 >= qword_4F06498 && (unsigned __int64)qword_4F06460 < qword_4F06490 )
            goto LABEL_37;
        }
        else if ( !unk_4D04954 || (unsigned __int64)qword_4F06460 < qword_4F06490 || v8 != 42 )
        {
LABEL_37:
          if ( v8 == 47 )
          {
            dword_4F17F58 = 0;
            for ( i = ++qword_4F06460; *i || i[1] != 2; qword_4F06460 = i )
              ++i;
            if ( (dword_4D0493C && !dword_4D03CF4 || qword_4D04908) && !unk_4D04938 && i - v7 > 0 )
            {
              v106 = v7;
              v74 = sub_7AEE00(v7, i - v7, 0, 0);
              *(_WORD *)(v74 + 51) = 768;
              *(_QWORD *)(v74 + 64) = v74 + 51;
              *(_QWORD *)(v74 + 56) = v74 + 51;
              v91 = v106;
              *(_BYTE *)(v74 + 48) = *(_BYTE *)(v74 + 48) & 0x7D | 2;
            }
            else
            {
              v91 = v7;
            }
            goto LABEL_164;
          }
LABEL_38:
          v9 = v7 + 2;
          qword_4F06460 = v7 + 2;
          v103 = dword_4D03CE4 | dword_4D03D18;
          if ( dword_4D03CE4 | dword_4D03D18 )
          {
            v103 = 0;
            goto LABEL_88;
          }
          v10 = v7[2];
          if ( v10 == 32 || v10 == 9 )
          {
            v72 = v7 + 3;
            do
            {
              do
              {
                qword_4F06460 = v72;
                v10 = *v72;
                v9 = v72++;
              }
              while ( v10 == 9 );
            }
            while ( v10 == 32 );
          }
          if ( v10 == 78 )
          {
            if ( v9[1] == 79 && !memcmp(v9 + 2, "TREACHED", 8u) )
            {
              v101 = (unsigned __int64)v7;
              v73 = isalpha(v9[10]);
              v103 = 0;
              v7 = (_BYTE *)v101;
              if ( !v73 )
              {
                if ( !dword_4F17FA0 && (qword_4F06498 > v101 || qword_4F06490 <= v101 || unk_4F06458 || dword_4F17F78) )
                {
                  if ( (_DWORD)qword_4F061D0 )
                  {
                    v108 = qword_4F061D0;
                  }
                  else
                  {
                    sub_7B0EB0(v101, (__int64)&v108);
                    v7 = (_BYTE *)v101;
                  }
                }
                else
                {
                  v84 = v101 - qword_4F06498;
                  LODWORD(v108) = unk_4F0647C;
                  v85 = word_4F06480;
                  if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > v101 )
                    v85 = sub_7AB680(v101);
                  WORD2(v108) = v84 + 1 - v85;
                }
                v102 = v7;
                sub_8540A0(5, &v108);
                v103 = 1;
                v7 = v102;
                v9 = qword_4F06460 + 10;
                qword_4F06460 += 10;
              }
            }
          }
          else if ( v10 == 65 )
          {
            if ( v9[1] == 82 && !memcmp(v9 + 2, "GSUSED", 6u) )
            {
              v92 = (unsigned __int64)v7;
              v11 = isalpha(v9[8]);
              v103 = 0;
              v7 = (_BYTE *)v92;
              if ( !v11 )
              {
                if ( !dword_4F17FA0 && (qword_4F06498 > v92 || qword_4F06490 <= v92 || unk_4F06458 || dword_4F17F78) )
                {
                  if ( (_DWORD)qword_4F061D0 )
                  {
                    v108 = qword_4F061D0;
                  }
                  else
                  {
                    sub_7B0EB0(v92, (__int64)&v108);
                    v7 = (_BYTE *)v92;
                  }
                }
                else
                {
                  v78 = v92 - qword_4F06498;
                  LODWORD(v108) = unk_4F0647C;
                  v79 = word_4F06480;
                  if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > v92 )
                    v79 = sub_7AB680(v92);
                  WORD2(v108) = v78 + 1 - v79;
                }
                v93 = v7;
                sub_8540A0(3, &v108);
                v103 = 1;
                v7 = v93;
                v9 = qword_4F06460 + 8;
                qword_4F06460 += 8;
              }
            }
          }
          else if ( v10 == 86 )
          {
            if ( v9[1] == 65 && !memcmp(v9 + 2, "RARGS", 5u) )
            {
              v94 = (unsigned __int64)v7;
              v57 = isalpha(v9[7]);
              v103 = 0;
              v7 = (_BYTE *)v94;
              if ( !v57 )
              {
                v58 = v9 + 7;
                v59 = v9 + 8;
                qword_4F06460 = v9 + 7;
                for ( j = v9[7]; (_BYTE)j == 32; ++v59 )
                {
                  qword_4F06460 = v59;
                  v58 = v59;
                  LOBYTE(j) = *v59;
                }
                v61 = v58 + 1;
                j = (char)j;
                v62 = 0;
                if ( (unsigned int)(unsigned __int8)j - 48 > 9 )
                {
LABEL_197:
                  LOWORD(v63) = 0;
                }
                else
                {
                  while ( 1 )
                  {
                    v63 = j + v62 - 48;
                    qword_4F06460 = v61;
                    j = (char)*v61;
                    if ( (unsigned int)(unsigned __int8)*v61 - 48 > 9 )
                      break;
                    if ( v63 <= 3276 )
                    {
                      ++v61;
                      v62 = 10 * v63;
                      if ( 32815 - j >= 10 * v63 )
                        continue;
                    }
                    goto LABEL_197;
                  }
                }
                if ( !dword_4F17FA0 && (qword_4F06498 > v94 || qword_4F06490 <= v94 || unk_4F06458 || dword_4F17F78) )
                {
                  if ( (_DWORD)qword_4F061D0 )
                  {
                    v108 = qword_4F061D0;
                  }
                  else
                  {
                    sub_7B0EB0(v94, (__int64)&v108);
                    v7 = (_BYTE *)v94;
                  }
                }
                else
                {
                  v82 = v94 - qword_4F06498;
                  LODWORD(v108) = unk_4F0647C;
                  v83 = word_4F06480;
                  if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > v94 )
                    v83 = sub_7AB680(v94);
                  WORD2(v108) = v82 + 1 - v83;
                }
                v95 = v7;
                v64 = sub_8540A0(4, &v108);
                v103 = 1;
                v9 = qword_4F06460;
                *(_WORD *)(v64 + 96) = v63;
                v7 = v95;
              }
            }
          }
          else
          {
            if ( v10 != 68 )
            {
              if ( v10 == 84 )
              {
                if ( v9[1] != 69 )
                  goto LABEL_88;
                if ( memcmp(v9 + 2, "XTURE_TYPE", 0xAu) )
                  goto LABEL_88;
                v98 = (unsigned __int64)v7;
                v66 = isalpha(v9[12]);
                v103 = 0;
                v7 = (_BYTE *)v98;
                if ( v66 )
                  goto LABEL_88;
                if ( !dword_4F17FA0 && (qword_4F06498 > v98 || qword_4F06490 <= v98 || unk_4F06458 || dword_4F17F78) )
                {
                  if ( (_DWORD)qword_4F061D0 )
                  {
                    v108 = qword_4F061D0;
                  }
                  else
                  {
                    sub_7B0EB0(v98, (__int64)&v108);
                    v7 = (_BYTE *)v98;
                  }
                }
                else
                {
                  v86 = v98 - qword_4F06498;
                  LODWORD(v108) = unk_4F0647C;
                  v87 = word_4F06480;
                  if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > v98 )
                    v87 = sub_7AB680(v98);
                  WORD2(v108) = v86 + 1 - v87;
                }
                v99 = v7;
                v67 = 12;
              }
              else
              {
                if ( v10 != 83 )
                  goto LABEL_88;
                if ( v9[1] != 85 )
                  goto LABEL_88;
                if ( memcmp(v9 + 2, "RFACE_TYPE", 0xAu) )
                  goto LABEL_88;
                v100 = (unsigned __int64)v7;
                v71 = isalpha(v9[12]);
                v103 = 0;
                v7 = (_BYTE *)v100;
                if ( v71 )
                  goto LABEL_88;
                if ( !dword_4F17FA0 && (qword_4F06498 > v100 || qword_4F06490 <= v100 || unk_4F06458 || dword_4F17F78) )
                {
                  if ( (_DWORD)qword_4F061D0 )
                  {
                    v108 = qword_4F061D0;
                  }
                  else
                  {
                    sub_7B0EB0(v100, (__int64)&v108);
                    v7 = (_BYTE *)v100;
                  }
                }
                else
                {
                  v76 = v100 - qword_4F06498;
                  LODWORD(v108) = unk_4F0647C;
                  v77 = word_4F06480;
                  if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > v100 )
                    v77 = sub_7AB680(v100);
                  WORD2(v108) = v76 + 1 - v77;
                }
                v99 = v7;
                v67 = 13;
              }
              sub_8540A0(v67, &v108);
              v103 = 1;
              v7 = v99;
              v9 = qword_4F06460 + 12;
              qword_4F06460 += 12;
              goto LABEL_88;
            }
            if ( v9[1] == 69 && !memcmp(v9 + 2, "VICE_BUILTIN", 0xCu) )
            {
              v96 = (unsigned __int64)v7;
              v65 = isalpha(v9[14]);
              v103 = 0;
              v7 = (_BYTE *)v96;
              if ( !v65 )
              {
                if ( !dword_4F17FA0 && (qword_4F06498 > v96 || qword_4F06490 <= v96 || unk_4F06458 || dword_4F17F78) )
                {
                  if ( (_DWORD)qword_4F061D0 )
                  {
                    v108 = qword_4F061D0;
                  }
                  else
                  {
                    sub_7B0EB0(v96, (__int64)&v108);
                    v7 = (_BYTE *)v96;
                  }
                }
                else
                {
                  v80 = v96 - qword_4F06498;
                  LODWORD(v108) = unk_4F0647C;
                  v81 = word_4F06480;
                  if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > v96 )
                    v81 = sub_7AB680(v96);
                  WORD2(v108) = v80 + 1 - v81;
                }
                v97 = v7;
                sub_8540A0(11, &v108);
                v103 = 1;
                v7 = v97;
                v9 = qword_4F06460 + 14;
                qword_4F06460 += 14;
              }
            }
          }
LABEL_88:
          v91 = v7;
          v89 = 0;
          while ( 1 )
          {
LABEL_89:
            while ( 1 )
            {
              v30 = dword_4D0432C;
              v31 = *v9;
              if ( *v9 != 42 )
                break;
LABEL_103:
              if ( *(v9 - 1) != 47 || dword_4D03CB0[0] )
              {
                if ( v9[1] == 47 )
                  goto LABEL_143;
              }
              else
              {
                sub_7B0EB0((unsigned __int64)(v9 - 1), (__int64)dword_4F07508);
                sub_684AC0(5u, 9u);
                if ( v9[1] == 47 )
                {
LABEL_143:
                  qword_4F06460 = v9 + 2;
                  if ( (dword_4D0493C && !dword_4D03CF4 || qword_4D04908) && !unk_4D04938 && !qword_4F06420 )
                  {
                    v45 = v9 + 2 - v91;
                    if ( unk_4D04954 )
                    {
                      if ( v45 > 0 )
                      {
                        v46 = sub_7AEE00(v91, v45, 0, 0);
                        *(_WORD *)(v46 + 51) = 768;
                        *(_QWORD *)(v46 + 64) = v46 + 51;
                        *(_QWORD *)(v46 + 56) = v46 + 51;
                        *(_BYTE *)(v46 + 48) = *(_BYTE *)(v46 + 48) & 0x7D | 2;
                      }
                    }
                    else
                    {
                      v75 = sub_7AEE00(v91, v45, 0, 0);
                      *(_BYTE *)(v75 + 48) |= 2u;
                      *(_QWORD *)(v75 + 56) = v75 + 51;
                      *(_BYTE *)(v75 + 51) = 32;
                      *(_WORD *)(v75 + 52) = 768;
                      *(_QWORD *)(v75 + 64) = v75 + 52;
                    }
                  }
                  goto LABEL_164;
                }
              }
              qword_4F06460 = ++v9;
            }
            while ( v31 )
            {
              if ( v30 && v31 < 0 )
              {
                if ( unk_4D041A0 && unk_4F064A8 )
                {
                  v32 = (unsigned int)sub_722680(v9, v109, &v107, 0);
                  if ( !v107 && v109[0] > 0x2000 )
                  {
                    v88 = v32;
                    v33 = sub_7AB890(v109[0], 0);
                    v32 = v88;
                    v34 = v33 == 0;
                    v35 = 1;
                    if ( v34 )
                      v35 = v89;
                    v89 = v35;
                  }
                }
                else
                {
                  v32 = (unsigned int)sub_721AB0((char *)v9, 0, 0);
                }
                if ( (int)v32 > 1 && qword_4F06498 <= (unsigned __int64)v9 && qword_4F06490 > (unsigned __int64)v9 )
                {
                  v42 = v9 + 1;
                  v9 += v32;
                  do
                  {
                    v43 = *(int *)&word_4F06480;
                    v44 = v42++;
                    ++*(_DWORD *)&word_4F06480;
                    qword_4F06488[v43] = v44;
                  }
                  while ( v9 != v42 );
                }
                else
                {
                  v9 += (int)v32;
                }
                v31 = *v9;
                if ( *v9 == 42 )
                  goto LABEL_103;
              }
              else
              {
                v31 = *++v9;
                if ( v31 == 42 )
                  goto LABEL_103;
              }
            }
            v36 = v9[1];
            if ( v36 != 6 )
              break;
            v9 += 2;
            qword_4F06460 = v9;
          }
          if ( v103 )
          {
LABEL_113:
            v37 = qword_4F06420;
            if ( qword_4F06420 )
              goto LABEL_114;
LABEL_134:
            if ( (dword_4D0493C && !dword_4D03CF4 || qword_4D04908) && !unk_4D04938 && v91 )
            {
              v37 = v91;
              v38 = 1;
              goto LABEL_115;
            }
          }
          else
          {
            if ( dword_4F17FA0
              || (unsigned __int64)v91 >= qword_4F06498
              && (unsigned __int64)v91 < qword_4F06490
              && !unk_4F06458
              && !dword_4F17F78 )
            {
              v47 = (_WORD)v91 - qword_4F06498;
              LODWORD(v108) = unk_4F0647C;
              v48 = word_4F06480;
              if ( *(_DWORD *)&word_4F06480 && (unsigned __int64)v91 < qword_4F06488[*(int *)&word_4F06480 - 1] )
              {
                v105 = v36;
                v48 = sub_7AB680((unsigned __int64)v91);
                v36 = v105;
              }
              WORD2(v108) = v47 + 1 - v48;
              goto LABEL_113;
            }
            if ( !(_DWORD)qword_4F061D0 )
            {
              v104 = v9[1];
              sub_7B0EB0((unsigned __int64)v91, (__int64)&v108);
              v36 = v104;
              goto LABEL_113;
            }
            v108 = qword_4F061D0;
            v37 = qword_4F06420;
            if ( !qword_4F06420 )
              goto LABEL_134;
LABEL_114:
            v38 = 0;
            qword_4F06420 = (_BYTE *)qword_4F06498;
LABEL_115:
            if ( !dword_4D03D18 || (v39 = v9 + 2, v36 != 2) )
              v39 = v9;
            v40 = v39 - v37;
            if ( v40 > 0 )
            {
              v41 = sub_7AEE00(v37, v40, 0, 0);
              *(_WORD *)(v41 + 51) = 768;
              *(_QWORD *)(v41 + 64) = v41 + 51;
              *(_QWORD *)(v41 + 56) = v41 + 51;
              *(_BYTE *)(v41 + 48) = *(_BYTE *)(v41 + 48) & 0x7D | (2 * v38);
            }
          }
          qword_4F06460 = v9;
          dword_4F17F58 = 0;
          if ( unk_4D042B8 || sub_7B2B10(0, 0) )
          {
            if ( !dword_4D03CB0[0] )
            {
              *(_QWORD *)dword_4F07508 = v108;
              sub_6851C0(6u, dword_4F07508);
            }
            goto LABEL_164;
          }
          dword_4F17F5C = 1;
          v91 = (_BYTE *)qword_4F06498;
          if ( unk_4D041A0 && unk_4F064A8 )
          {
            v49 = (__int64 *)qword_4F084D0;
            v50 = qword_4F084C8;
            if ( qword_4F084D0 )
            {
              while ( 1 )
              {
                v51 = (__int64 *)*v49;
                *v49 = v50;
                v50 = (__int64)v49;
                if ( !v51 )
                  break;
                v49 = v51;
              }
              qword_4F084C8 = (__int64)v49;
              qword_4F084D0 = 0;
            }
            v89 = 0;
            v9 = qword_4F06460;
            v103 = 1;
          }
          else
          {
            v103 = 1;
            v9 = qword_4F06460;
          }
          goto LABEL_89;
        }
        v15 = sub_7AEFF0((unsigned __int64)qword_4F06460);
        v7 = qword_4F06460;
        v16 = v15[12];
        if ( v16 && v16 <= (unsigned __int64)qword_4F06460 )
          goto LABEL_38;
        v17 = qword_4F06460;
        dword_4F084D8 = 1;
        v18 = unk_4D03D20;
        sub_7B0EB0((unsigned __int64)qword_4F06460, (__int64)v109);
        qword_4F06460 += 2;
        unk_4D03D20 = 1;
        while ( 1 )
        {
          sub_7B8B50((unsigned __int64)v17, (unsigned int *)v109, v19, v20, v21, v22);
          v23 = word_4F06418[0];
          if ( word_4F06418[0] == 34 )
            break;
LABEL_70:
          if ( v23 == 9 )
          {
            if ( !dword_4D03CB0[0] )
            {
              *(_QWORD *)dword_4F07508 = v109[0];
              sub_6851C0(6u, dword_4F07508);
            }
            goto LABEL_73;
          }
        }
        while ( 1 )
        {
          sub_7BC390(v17, v109, v19, v20);
          sub_7B8B50((unsigned __int64)v17, (unsigned int *)v109, v24, v25, v26, v27);
          if ( !dword_4F063EC && *qword_4F06410 == 47 )
            break;
          v23 = word_4F06418[0];
          if ( word_4F06418[0] != 34 )
            goto LABEL_70;
        }
        qword_4F06460 = qword_4F06410 + 1;
LABEL_73:
        unk_4D03D20 = v18;
        dword_4F084D8 = 0;
LABEL_164:
        if ( unk_4D041A0 && dword_4D0432C && unk_4F064A8 && (qword_4F084D0 || (v89 & 1) != 0) )
        {
          sub_7B0EB0((unsigned __int64)v91, (__int64)dword_4F07508);
          sub_684AC0(5u, 0xC9Cu);
          v68 = (__int64 *)qword_4F084D0;
          if ( qword_4F084D0 )
          {
            v69 = qword_4F084C8;
            while ( 1 )
            {
              v70 = (__int64 *)*v68;
              *v68 = v69;
              v69 = (__int64)v68;
              if ( !v70 )
                break;
              v68 = v70;
            }
            qword_4F084C8 = (__int64)v68;
            qword_4F084D0 = 0;
          }
        }
        v0 = qword_4F06460;
        v90 |= 1u;
        v1 = *qword_4F06460;
        continue;
      default:
        goto LABEL_7;
    }
  }
}
