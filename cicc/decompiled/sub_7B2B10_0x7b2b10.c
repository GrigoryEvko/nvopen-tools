// Function: sub_7B2B10
// Address: 0x7b2b10
//
_BOOL8 __fastcall sub_7B2B10(__int64 a1, __int64 a2)
{
  int v2; // r12d
  int v3; // r13d
  int v4; // eax
  int v5; // eax
  char *v6; // r14
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  _QWORD *v9; // r15
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  _QWORD *v13; // rsi
  int v14; // eax
  __int64 v15; // rax
  char *v16; // r12
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  char *v19; // rax
  char v20; // al
  _BOOL4 v21; // r12d
  __int64 v22; // rax
  int v24; // eax
  int v25; // ecx
  int v26; // esi
  char *v27; // rdx
  int v28; // r15d
  int v29; // eax
  bool v30; // si
  int v31; // ecx
  _QWORD *v32; // rax
  _QWORD *v33; // rsi
  __int64 v34; // rdx
  int v35; // eax
  int v36; // ecx
  __int64 v37; // rdx
  __int64 v38; // r13
  int v39; // eax
  _WORD *v40; // rcx
  unsigned __int64 v41; // rax
  __int64 v42; // rdx
  char *v43; // r15
  __int64 v44; // rax
  bool v45; // cc
  int v46; // eax
  int v47; // r12d
  char *v48; // rcx
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // r15
  int v53; // eax
  char *v54; // rax
  __int64 v55; // rcx
  char *v56; // r14
  char v57; // r13
  char v58; // [rsp+8h] [rbp-68h]
  int v59; // [rsp+10h] [rbp-60h]
  int v60; // [rsp+18h] [rbp-58h]
  int v61; // [rsp+1Ch] [rbp-54h]
  char *v62; // [rsp+20h] [rbp-50h]
  char *v63; // [rsp+20h] [rbp-50h]
  int v64; // [rsp+20h] [rbp-50h]
  unsigned __int64 v65; // [rsp+28h] [rbp-48h]
  _QWORD *v66; // [rsp+38h] [rbp-38h] BYREF

  v60 = a2;
  v65 = unk_4F06490 - 4LL;
  qword_4F061D0 = *(_QWORD *)&dword_4F077C8;
  if ( (_DWORD)a2 )
  {
    dword_4F17F58 = 0;
    v6 = (char *)unk_4F06460;
    if ( unk_4F06460 > v65 )
    {
      sub_7ABFD0();
      v6 = (char *)unk_4F06460;
      v65 = unk_4F06490 - 4LL;
    }
    v61 = dword_4F17FC0;
    if ( !dword_4F17FC0 )
    {
      if ( dword_4F17FBC <= 1 )
      {
        v25 = getc(qword_4F17FC8);
      }
      else
      {
        LOBYTE(v24) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
        v25 = v24;
      }
      if ( !dword_4F17FC0 && v25 != -1 )
        goto LABEL_106;
    }
    dword_4F17FC0 = 1;
    unk_4F06478 = 1;
    *(_DWORD *)v6 = (_DWORD)&loc_1000200;
LABEL_76:
    if ( unk_4F06478 )
      goto LABEL_77;
    v21 = dword_4F17FA8 != 0;
LABEL_66:
    v22 = (unsigned __int16)word_4F0851C;
    if ( word_4F0851C )
      goto LABEL_78;
    return v21;
  }
  v2 = a1;
  if ( dword_4F17F58 )
  {
    sub_7B0EB0(qword_4F06498 + qword_4F17F50, (__int64)dword_4F07508);
    a2 = 1750;
    a1 = 5;
    sub_684AC0(5u, 0x6D6u);
    dword_4F17F58 = 0;
  }
  if ( unk_4D0493C )
    sub_7B1260();
  if ( qword_4D04908 && byte_4F17F98 )
    sub_7AFA40(a1, a2);
  dword_4F17F5C = 0;
  dword_4F04D98 = unk_4F04D9C;
  if ( dword_4F17FC0 )
  {
    v3 = 0;
  }
  else
  {
    if ( dword_4F17FBC <= 1 )
    {
      v3 = getc(qword_4F17FC8);
    }
    else
    {
      LOBYTE(v5) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
      v3 = v5;
    }
    if ( !dword_4F17FC0 && v3 != -1 )
      goto LABEL_25;
  }
  dword_4F17FC0 = 1;
  unk_4F06478 = 1;
  if ( !v2 )
    goto LABEL_63;
  while ( 1 )
  {
    if ( (qword_4F064B0[11] & 0x20) != 0 )
      goto LABEL_62;
    sub_7B2450();
    if ( dword_4F17FD8 < 0 )
      break;
    if ( dword_4F17FBC > 1 )
    {
      LOBYTE(v4) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
      v3 = v4;
    }
    else
    {
      v3 = getc(qword_4F17FC8);
    }
    if ( !dword_4F17FC0 && v3 != -1 )
      goto LABEL_25;
    dword_4F17FC0 = 1;
    unk_4F06478 = 1;
  }
  dword_4F17FA8 = 1;
LABEL_25:
  v6 = (char *)qword_4F06498;
  unk_4F0647C = ++unk_4F06468;
  v7 = (_QWORD *)unk_4F06458;
  if ( unk_4F06458 )
  {
    v8 = qword_4F06448;
    while ( 1 )
    {
      unk_4F06458 = *v7;
      *v7 = v8;
      v8 = v7;
      qword_4F06448 = v7;
      if ( !unk_4F06458 )
        break;
      v7 = (_QWORD *)unk_4F06458;
    }
    qword_4F084E0 = 0;
    qword_4F06450 = 0;
  }
  v9 = (_QWORD *)qword_4F06440;
  if ( qword_4F06440 )
  {
    v66 = (_QWORD *)qword_4F06440;
    do
    {
      while ( 1 )
      {
        v11 = (__int64)v9;
        v9 = (_QWORD *)*v9;
        if ( (*(_BYTE *)(v11 + 48) & 0x20) != 0 )
        {
          if ( unk_4D03BD8 )
            break;
        }
        sub_7AEF90(v11);
        sub_7AEF30((__int64)&v66);
        v66 = v9;
        if ( !v9 )
          goto LABEL_40;
      }
      v10 = *(_QWORD *)(v11 + 16);
      if ( v10 && v10 >= qword_4F06498 && v10 < unk_4F06490 )
      {
        sub_7AED90(v11);
        v32 = v66;
        v66[2] = 0;
        v32[4] = 0;
      }
      v66 = v9;
    }
    while ( v9 );
LABEL_40:
    v9 = (_QWORD *)qword_4F06440;
  }
  dword_4F17FA0 = v9 == 0;
  v61 = dword_4F17FA8;
  if ( dword_4F17FA8 )
  {
LABEL_60:
    *(_WORD *)v6 = 256;
    goto LABEL_61;
  }
  v12 = qword_4F17FD0;
  v13 = qword_4F064B0;
  ++*((_DWORD *)qword_4F064B0 + 10);
  v14 = *(_DWORD *)(v12 + 80) + 1;
  *(_DWORD *)(v12 + 80) = v14;
  *((_DWORD *)v13 + 20) = v14;
  if ( v14 == *(_DWORD *)(v12 + 84) )
  {
    v49 = ftell(*(FILE **)v12);
    v50 = qword_4F17FD0;
    *(_DWORD *)(v50 + 84) = sub_67D160(*(_QWORD *)(qword_4F17FD0 + 64), *(_DWORD *)(qword_4F17FD0 + 80), v49 - 1);
  }
  if ( v3 == 10 )
  {
LABEL_59:
    v6 += 2;
    *((_WORD *)v6 - 1) = 512;
    goto LABEL_60;
  }
  LODWORD(v15) = v3;
  v16 = v6;
  do
  {
    if ( (int)v15 > 63 )
      goto LABEL_52;
    switch ( (_DWORD)v15 )
    {
      case 0x3F:
        if ( (char *)qword_4F06498 != v16 && *(v16 - 1) == 63 )
        {
          v6 = v16;
          v26 = 0;
          v27 = v16;
          v28 = (_DWORD)v16 - qword_4F06498 + 1;
          goto LABEL_88;
        }
        break;
      case 0xA:
        v6 = v16;
        if ( (char *)qword_4F06498 != v16 )
        {
          v18 = (unsigned __int8)*(v16 - 1);
          if ( !unk_4D04204 )
          {
            while ( 1 )
            {
              v18 = (unsigned __int8)*(v6 - 1);
              v54 = v6--;
              if ( (_BYTE)v18 != 13 )
                break;
              if ( (char *)qword_4F06498 == v6 )
                goto LABEL_134;
            }
            v6 = v54;
          }
          v19 = v6 - 1;
          if ( (unsigned __int8)v18 <= 0x20u )
          {
            v17 = 0x100001A00LL;
            while ( _bittest64(&v17, v18) )
            {
              if ( (char *)qword_4F06498 == v19 )
                goto LABEL_134;
              v18 = (unsigned __int8)*--v19;
              if ( (unsigned __int8)v18 > 0x20u )
                goto LABEL_165;
            }
          }
          else
          {
LABEL_165:
            if ( (_BYTE)v18 == 92 )
            {
              v42 = (unsigned int)((_DWORD)v6 - (_DWORD)v19 - 1);
              goto LABEL_146;
            }
          }
        }
        goto LABEL_59;
      case 0:
        v6 = v16;
        v61 = 0;
        v28 = (_DWORD)v16 - qword_4F06498 + 1;
        goto LABEL_138;
    }
LABEL_52:
    if ( v16 == (char *)v65 )
    {
      v37 = qword_4F06498;
      v36 = 0;
      v28 = v65 - qword_4F06498 + 1;
      while ( 1 )
      {
        v58 = v15;
        v59 = v36;
        v38 = v65 - v37;
        sub_7ABFD0();
        v36 = v59;
        v27 = (char *)(v38 + qword_4F06498);
        v65 = unk_4F06490 - 4LL;
        LOBYTE(v15) = v58;
        while ( 1 )
        {
          *v27 = v15;
          v26 = v61;
          v6 = v27 + 1;
          if ( !v36 )
            goto LABEL_129;
LABEL_115:
          if ( v61 == -1 )
            goto LABEL_133;
          if ( v61 == 10 )
            break;
          ++v28;
          if ( v61 != 63 )
          {
            v25 = v61;
            v61 = v26;
LABEL_111:
            if ( v25 )
            {
              LOBYTE(v15) = v25;
              v27 = v6;
              v36 = 0;
            }
            else
            {
LABEL_138:
              if ( v6 == (char *)v65 )
              {
                v56 = &v6[-qword_4F06498];
                sub_7ABFD0();
                v6 = &v56[qword_4F06498];
                v65 = unk_4F06490 - 4LL;
              }
              sub_7ABAA0(3, (__int64)v6);
              *v6 = 0;
              v27 = v6 + 1;
              v36 = 0;
              LOBYTE(v15) = 6;
            }
            goto LABEL_113;
          }
          if ( v28 != 1 && *v27 == 63 )
          {
            v27 = v6;
LABEL_88:
            if ( dword_4F077C4 == 1 )
            {
              v61 = v26;
              LOBYTE(v15) = 63;
              v36 = 0;
            }
            else
            {
              v62 = v27;
              if ( dword_4F17FBC <= 1 )
                v29 = getc(qword_4F17FC8);
              else
                LOBYTE(v29) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
              v27 = v62;
              v61 = v29;
              switch ( v29 )
              {
                case '!':
                  v57 = 124;
                  goto LABEL_184;
                case '\'':
                  v57 = 94;
                  goto LABEL_184;
                case '(':
                  v57 = 91;
                  goto LABEL_184;
                case ')':
                  v57 = 93;
                  goto LABEL_184;
                case '-':
                  v57 = 126;
                  goto LABEL_184;
                case '/':
                  v57 = 92;
                  goto LABEL_184;
                case '<':
                  v57 = 123;
                  goto LABEL_184;
                case '=':
                  v57 = 35;
                  goto LABEL_184;
                case '>':
                  v57 = 125;
LABEL_184:
                  if ( unk_4D04384 )
                  {
                    ++v28;
                    v15 = sub_7ABAA0(0, (__int64)(v6 - 1));
                    v27 = v6 - 1;
                    *(_BYTE *)(v15 + 24) = v61;
                    LOBYTE(v15) = v57;
                    v36 = 0;
                  }
                  else if ( dword_4F08520 )
                  {
LABEL_126:
                    LOBYTE(v15) = 63;
                    v36 = 1;
                  }
                  else
                  {
                    v36 = 1;
                    dword_4F08520 = 1;
                    LOBYTE(v15) = 63;
                    word_4F0851C = (_WORD)v6 - qword_4F06498;
                  }
                  break;
                default:
                  goto LABEL_126;
              }
            }
          }
          else
          {
            v61 = v26;
            v27 = v6;
            LOBYTE(v15) = 63;
            v36 = 0;
          }
LABEL_113:
          if ( v27 == (char *)v65 )
            goto LABEL_120;
        }
        if ( v6 == (char *)qword_4F06498 )
          goto LABEL_134;
        v41 = (unsigned __int8)*v27;
        if ( !unk_4D04204 )
        {
          while ( 1 )
          {
            v41 = (unsigned __int8)*(v6 - 1);
            v48 = v6--;
            if ( (_BYTE)v41 != 13 )
              break;
            if ( !--v28 )
              goto LABEL_134;
          }
          v27 = v6;
          v6 = v48;
        }
        if ( (unsigned __int8)v41 <= 0x20u )
        {
          v55 = 0x100001A00LL;
          if ( !_bittest64(&v55, v41) || v28 == 1 )
            goto LABEL_134;
          while ( 1 )
          {
            v41 = (unsigned __int8)*--v27;
            if ( (unsigned __int8)v41 > 0x20u )
              break;
            if ( !_bittest64(&v55, v41) || v28 == (_DWORD)v6 - (_DWORD)v27 )
              goto LABEL_134;
          }
        }
        if ( (_BYTE)v41 != 92 )
          goto LABEL_134;
        v61 = v26;
        v42 = (unsigned int)((_DWORD)v6 - (_DWORD)v27 - 1);
LABEL_146:
        if ( (_DWORD)v42 )
        {
          if ( !HIDWORD(qword_4F077B4) )
          {
            v40 = v6 + 2;
            dword_4F17F58 = 1;
            *(_WORD *)v6 = 512;
            qword_4F17F50 = (__int64)&v6[-qword_4F06498 + 2 - v42 - 3];
            goto LABEL_135;
          }
          *(_DWORD *)v6 = (_DWORD)&loc_1000200;
          v43 = &v6[-v42];
          sub_7B0EB0((unsigned __int64)&v6[-v42], (__int64)dword_4F07508);
          sub_684AC0(5u, 0x574u);
        }
        else
        {
          v43 = v6;
        }
        v6 = v43 - 1;
        v44 = sub_7ABAA0(1, (__int64)(v43 - 1));
        v45 = dword_4F17FBC <= 1;
        *(_DWORD *)(v44 + 24) = unk_4F06468 + 1;
        if ( v45 )
        {
          v25 = getc(qword_4F17FC8);
        }
        else
        {
          LOBYTE(v46) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
          v25 = v46;
        }
        if ( v25 == -1 )
        {
          *(v43 - 1) = 0;
          *(_WORD *)v43 = 2;
          v43[2] = 1;
          dword_4F17FC0 = 1;
          v47 = HIDWORD(qword_4F077B4) == 0 ? 3 : 0;
          sub_7B0EB0((unsigned __int64)(v43 - 1), (__int64)dword_4F07508);
          sub_684AC0(v47 + 5, 2u);
          goto LABEL_134;
        }
LABEL_106:
        v33 = qword_4F064B0;
        v34 = qword_4F17FD0;
        ++*((_DWORD *)qword_4F064B0 + 10);
        ++unk_4F06468;
        v35 = *(_DWORD *)(v34 + 80) + 1;
        *(_DWORD *)(v34 + 80) = v35;
        *((_DWORD *)v33 + 20) = v35;
        if ( v35 == *(_DWORD *)(v34 + 84) )
        {
          v64 = v25;
          v51 = ftell(*(FILE **)v34);
          v52 = qword_4F17FD0;
          v53 = sub_67D160(*(_QWORD *)(qword_4F17FD0 + 64), *(_DWORD *)(qword_4F17FD0 + 80), v51 - 1);
          v25 = v64;
          *(_DWORD *)(v52 + 84) = v53;
        }
        if ( v25 == 10 )
          goto LABEL_134;
        if ( v25 != 63 )
        {
          v28 = 1;
          goto LABEL_111;
        }
        if ( v6 != (char *)v65 )
        {
          *v6 = 63;
          v27 = v6;
          v28 = 1;
          ++v6;
LABEL_129:
          v63 = v27;
          if ( dword_4F17FBC <= 1 )
            v39 = getc(qword_4F17FC8);
          else
            LOBYTE(v39) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
          v26 = v61;
          v27 = v63;
          v61 = v39;
          goto LABEL_115;
        }
        LOBYTE(v15) = 63;
        v36 = 0;
        v28 = 1;
LABEL_120:
        v37 = qword_4F06498;
      }
    }
    *v16++ = v15;
    if ( dword_4F17FBC > 1 )
      LOBYTE(v15) = sub_722840(qword_4F17FC8, (__int64)&unk_4F17FB0);
    else
      LODWORD(v15) = getc(qword_4F17FC8);
  }
  while ( (_DWORD)v15 != -1 );
  v6 = v16;
LABEL_133:
  dword_4F17FC0 = 1;
  *(_DWORD *)v6 = (_DWORD)&loc_1000200;
LABEL_134:
  v40 = v6 + 2;
  *(_WORD *)v6 = 512;
LABEL_135:
  *v40 = 256;
  if ( v60 )
    goto LABEL_76;
LABEL_61:
  dword_4F17FA4 = 0;
  unk_4F06460 = qword_4F06498;
  *(_DWORD *)&word_4F06480 = 0;
LABEL_62:
  if ( !unk_4F06478 )
  {
    v30 = dword_4F17FA8 != 0;
    v21 = dword_4F17FA8 != 0;
    if ( !unk_4D0493C )
    {
      if ( !qword_4D04908 )
        goto LABEL_66;
      if ( dword_4F17FA8 )
      {
        v20 = 0;
        goto LABEL_65;
      }
      goto LABEL_97;
    }
    v31 = 1;
    if ( unk_4D03CE4 )
      goto LABEL_95;
    v31 = unk_4D03D18;
    if ( !unk_4D03D18 )
      goto LABEL_95;
    if ( !unk_4D03CF0 )
    {
      v31 = 1;
LABEL_95:
      dword_4F17F9C = v31;
      dword_4D03CF4 = v31;
      if ( !v30 )
      {
        if ( !qword_4D04908 )
          goto LABEL_66;
LABEL_97:
        v20 = unk_4D03CE4 == 0 ? 78 : 83;
        goto LABEL_65;
      }
LABEL_81:
      v20 = 0;
      dword_4D03CF4 = 1;
      if ( !qword_4D04908 )
        goto LABEL_66;
LABEL_65:
      byte_4F17F98 = v20;
      goto LABEL_66;
    }
LABEL_101:
    v31 = 0;
    goto LABEL_95;
  }
LABEL_63:
  if ( unk_4D0493C )
  {
    if ( unk_4D03CE4 )
    {
LABEL_80:
      dword_4F17F9C = 1;
      v21 = 1;
      goto LABEL_81;
    }
    if ( !unk_4D03D18 )
    {
      dword_4F17F9C = 0;
      v21 = 1;
      goto LABEL_81;
    }
    if ( !unk_4D03CF0 )
      goto LABEL_80;
    v21 = 1;
    v30 = 1;
    goto LABEL_101;
  }
  v20 = 0;
  v21 = 1;
  if ( qword_4D04908 )
    goto LABEL_65;
LABEL_77:
  v22 = (unsigned __int16)word_4F0851C;
  v21 = 1;
  if ( word_4F0851C )
  {
LABEL_78:
    sub_7B0EB0(qword_4F06498 + v22 - 1, (__int64)dword_4F07508);
    sub_684AC0(5u, 0x57Eu);
    word_4F0851C = 0;
  }
  return v21;
}
