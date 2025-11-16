// Function: sub_5DF1B0
// Address: 0x5df1b0
//
int __fastcall sub_5DF1B0(__int64 a1)
{
  char *v2; // r12
  int v3; // edi
  char v4; // al
  int v5; // edi
  char *v6; // r12
  __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int64 i; // r14
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r15
  int v13; // edi
  char *v14; // r12
  __int64 **v15; // r13
  FILE *v16; // rsi
  const char *v17; // r12
  const char *v18; // r15
  int v19; // eax
  int v20; // edi
  int v21; // edx
  int v22; // r14d
  unsigned __int8 *j; // r15
  _BYTE *v24; // rax
  int v25; // edi
  _BYTE *v26; // r14
  char *v27; // r12
  int v28; // edi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rdi
  char *v34; // r12
  int v35; // edi
  char *v36; // r12
  int v37; // edi
  char *v38; // r12
  int v39; // edi
  __int64 **v40; // r12
  __int64 v41; // rax
  __int64 *v42; // rdx
  int v43; // edi
  _BYTE *v44; // r13
  int v45; // edi
  char *v46; // rbx
  int result; // eax
  char *v48; // r12
  int v49; // edi
  _BYTE *v50; // rax
  int v51; // edi
  _BYTE *v52; // r13
  char *v53; // r15
  int v54; // edi
  _QWORD *v55; // rbx
  __int64 v56; // rdi
  char v57; // al
  char *v58; // r12
  char v59; // al
  char *v60; // r15
  _QWORD *v61; // r14
  const char *v62; // r15
  int v63; // edi
  char *v64; // r12
  __int64 v65; // r12
  __int64 v66; // rsi
  __int64 v67; // rax
  char *s1; // [rsp+0h] [rbp-50h]
  __int64 v69; // [rsp+8h] [rbp-48h]
  int na; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+10h] [rbp-40h]
  int v72; // [rsp+18h] [rbp-38h]
  unsigned __int64 v73; // [rsp+18h] [rbp-38h]

  if ( *(char *)(a1 + 88) < 0 )
  {
    v65 = 0;
    while ( 1 )
    {
      v66 = unk_4F04C50;
      if ( unk_4F04C50 )
        v66 = qword_4CF7E98;
      v67 = sub_732D20(a1, v66, 0, v65);
      v65 = v67;
      if ( !v67 )
        break;
      sub_5D52E0(v67, v66);
    }
  }
  v2 = "_asm";
  sub_5D45D0((unsigned int *)(a1 + 64));
  v3 = 95;
  do
  {
    ++v2;
    putc(v3, stream);
    v3 = *(v2 - 1);
  }
  while ( *(v2 - 1) );
  v4 = *(_BYTE *)(a1 + 128);
  dword_4CF7F40 += 5;
  if ( (v4 & 4) != 0 )
  {
    v5 = 32;
    v6 = "volatile";
    if ( *(_QWORD *)(a1 + 136) || *(_QWORD *)(a1 + 144) || (v4 & 2) != 0 )
    {
      do
      {
        ++v6;
        putc(v5, stream);
        v5 = *(v6 - 1);
      }
      while ( *(v6 - 1) );
      dword_4CF7F40 += 9;
      v4 = *(_BYTE *)(a1 + 128);
    }
  }
  if ( (v4 & 0x20) != 0 )
  {
    v63 = 32;
    v64 = "goto";
    do
    {
      ++v64;
      putc(v63, stream);
      v63 = *(v64 - 1);
    }
    while ( *(v64 - 1) );
    dword_4CF7F40 += 5;
  }
  putc(40, stream);
  ++dword_4CF7F40;
  if ( (*(_BYTE *)(a1 + 128) & 0x20) != 0 )
  {
    putc(32, stream);
    ++dword_4CF7F40;
    putc(34, stream);
    v7 = *(_QWORD *)(a1 + 120);
    ++dword_4CF7F40;
    v8 = *(_QWORD *)(v7 + 176);
    v9 = *(_QWORD *)(v7 + 184);
    if ( v8 )
    {
      for ( i = 0; ; i = v12 )
      {
        v11 = (unsigned int)*(char *)(v9 + i);
        v12 = i + 1;
        if ( i + 5 < v8 && *(_BYTE *)(v9 + i) == 37 )
        {
          if ( *(_BYTE *)(v9 + v12) == 108 && *(_BYTE *)(v9 + i + 2) == 91 )
          {
            v60 = "l[";
            do
            {
              ++v60;
              putc(v11, stream);
              v11 = (unsigned int)*(v60 - 1);
              ++dword_4CF7F40;
            }
            while ( (_BYTE)v11 );
            v7 = *(_QWORD *)(a1 + 120);
            v12 = i + 3;
            v8 = *(_QWORD *)(v7 + 176);
            if ( i + 3 >= v8 )
              break;
            v73 = i + 3;
            while ( *(_BYTE *)(v9 + v73) != 93 )
            {
              if ( ++v73 >= v8 )
                goto LABEL_14;
            }
            v61 = *(_QWORD **)(a1 + 152);
            n = v73 - v12;
            if ( !v61 )
LABEL_127:
              sub_721090(v11);
            s1 = (char *)(v9 + v12);
            while ( 1 )
            {
              v11 = (unsigned __int64)s1;
              v62 = *(const char **)(v61[1] + 8LL);
              v69 = v61[1];
              if ( !strncmp(s1, v62, n) )
              {
                v11 = (unsigned __int64)v62;
                if ( n == strlen(v62) )
                  break;
              }
              v61 = (_QWORD *)*v61;
              if ( !v61 )
                goto LABEL_127;
            }
            sub_5D5A80(v69, 0);
            v12 = v73;
            v7 = *(_QWORD *)(a1 + 120);
LABEL_13:
            v8 = *(_QWORD *)(v7 + 176);
            if ( v8 <= v12 )
              break;
            continue;
          }
        }
        else if ( v12 >= v8 && !*(_BYTE *)(v9 + i) )
        {
          goto LABEL_13;
        }
        sub_746F50(v11, &qword_4CF7CE0);
        v7 = *(_QWORD *)(a1 + 120);
        v8 = *(_QWORD *)(v7 + 176);
        if ( v8 <= v12 )
          break;
LABEL_14:
        ;
      }
    }
    putc(34, stream);
    ++dword_4CF7F40;
  }
  else
  {
    sub_5D5250(*(_QWORD *)(a1 + 120));
  }
  v13 = 32;
  v14 = ":";
  if ( !*(_QWORD *)(a1 + 136) && !*(_QWORD *)(a1 + 144) && (*(_BYTE *)(a1 + 128) & 6) == 4 )
    goto LABEL_88;
  do
  {
    ++v14;
    putc(v13, stream);
    v13 = *(v14 - 1);
  }
  while ( *(v14 - 1) );
  v15 = *(__int64 ***)(a1 + 136);
  dword_4CF7F40 += 2;
  if ( !v15 || (v72 = 1, ((_BYTE)v15[3] & 2) == 0) )
  {
    v48 = ":";
    v49 = 32;
    do
    {
      ++v48;
      putc(v49, stream);
      v49 = *(v48 - 1);
    }
    while ( *(v48 - 1) );
    dword_4CF7F40 += 2;
    v15 = *(__int64 ***)(a1 + 136);
    v72 = 0;
  }
LABEL_25:
  v16 = stream;
  if ( v15 )
  {
    while ( 1 )
    {
      putc(32, v16);
      ++dword_4CF7F40;
      if ( v15[1] )
      {
        putc(91, stream);
        v17 = (const char *)v15[1];
        v18 = v17 + 1;
        na = ++dword_4CF7F40;
        v19 = strlen(v17);
        v20 = *v17;
        v21 = na;
        v22 = v19;
        if ( *v17 )
        {
          do
          {
            ++v18;
            putc(v20, stream);
            v20 = *(v18 - 1);
          }
          while ( *(v18 - 1) );
          v21 = dword_4CF7F40;
        }
        dword_4CF7F40 = v21 + v22;
        putc(93, stream);
        ++dword_4CF7F40;
      }
      putc(34, stream);
      ++dword_4CF7F40;
      if ( v72 )
      {
        if ( ((_BYTE)v15[3] & 1) != 0 )
          putc(43, stream);
        else
          putc(61, stream);
        ++dword_4CF7F40;
      }
      for ( j = (unsigned __int8 *)v15[2]; j; j = (unsigned __int8 *)*((_QWORD *)j + 2) )
      {
        while ( 1 )
        {
          putc(aXg0123456789rh[*j], stream);
          ++dword_4CF7F40;
          if ( *j == 4 )
          {
            v24 = (_BYTE *)*((_QWORD *)j + 1);
            if ( v24 )
            {
              v25 = (char)*v24;
              v26 = v24 + 1;
              if ( *v24 )
                break;
            }
          }
          j = (unsigned __int8 *)*((_QWORD *)j + 2);
          if ( !j )
            goto LABEL_43;
        }
        do
        {
          ++v26;
          putc(v25, stream);
          v25 = (char)*(v26 - 1);
          ++dword_4CF7F40;
        }
        while ( (_BYTE)v25 );
      }
LABEL_43:
      v27 = "(";
      putc(34, stream);
      ++dword_4CF7F40;
      v28 = 32;
      do
      {
        ++v27;
        putc(v28, stream);
        v28 = *(v27 - 1);
      }
      while ( *(v27 - 1) );
      v33 = (__int64)v15[5];
      dword_4CF7F40 += 2;
      sub_5DBFC0(v33, 0, v29, v30, v31, v32);
      putc(41, stream);
      v15 = (__int64 **)*v15;
      ++dword_4CF7F40;
      if ( !v15 )
        break;
      if ( ((_BYTE)v15[3] & 2) != 0 || !v72 )
      {
        putc(44, stream);
        ++dword_4CF7F40;
        goto LABEL_25;
      }
      v34 = ":";
      v35 = 32;
      do
      {
        ++v34;
        putc(v35, stream);
        v35 = *(v34 - 1);
      }
      while ( *(v34 - 1) );
      dword_4CF7F40 += 2;
      v16 = stream;
      v72 = 0;
    }
    v16 = stream;
  }
  if ( v72 )
  {
    if ( !*(_QWORD *)(a1 + 144) )
    {
      v57 = *(_BYTE *)(a1 + 128);
      if ( (v57 & 2) == 0 )
      {
        if ( (v57 & 0x20) == 0 )
          goto LABEL_69;
        goto LABEL_94;
      }
    }
    v36 = ":";
    v37 = 32;
    while ( 1 )
    {
      putc(v37, v16);
      v37 = *v36++;
      if ( !(_BYTE)v37 )
        break;
      v16 = stream;
    }
    dword_4CF7F40 += 2;
    v16 = stream;
  }
  v38 = ":";
  v39 = 32;
  if ( *(_QWORD *)(a1 + 144) )
  {
    while ( 1 )
    {
      putc(v39, v16);
      v39 = *v38++;
      if ( !(_BYTE)v39 )
        break;
      v16 = stream;
    }
    v40 = *(__int64 ***)(a1 + 144);
    for ( dword_4CF7F40 += 2; v40; ++dword_4CF7F40 )
    {
      putc(32, stream);
      ++dword_4CF7F40;
      putc(34, stream);
      v41 = *((unsigned __int8 *)v40 + 8);
      ++dword_4CF7F40;
      if ( (_BYTE)v41 == 58 && (v42 = v40[2]) != 0 )
      {
        v43 = *(char *)v42;
        v44 = (char *)v42 + 1;
        if ( *(_BYTE *)v42 )
        {
          do
          {
            ++v44;
            putc(v43, stream);
            v43 = (char)*(v44 - 1);
            ++dword_4CF7F40;
          }
          while ( (_BYTE)v43 );
        }
      }
      else
      {
        v50 = *(&off_4B6DCE0 + v41);
        v51 = (char)*v50;
        v52 = v50 + 1;
        if ( *v50 )
        {
          do
          {
            ++v52;
            putc(v51, stream);
            v51 = (char)*(v52 - 1);
            ++dword_4CF7F40;
          }
          while ( (_BYTE)v51 );
        }
      }
      putc(34, stream);
      ++dword_4CF7F40;
      if ( !*v40 )
        break;
      putc(44, stream);
      v40 = (__int64 **)*v40;
    }
    v16 = stream;
    if ( (*(_BYTE *)(a1 + 128) & 0x20) != 0 )
      goto LABEL_82;
    goto LABEL_69;
  }
  if ( (*(_BYTE *)(a1 + 128) & 0x20) == 0 )
    goto LABEL_69;
LABEL_94:
  v58 = ":";
  v59 = 32;
  while ( 1 )
  {
    ++v58;
    putc(v59, v16);
    v59 = *(v58 - 1);
    if ( !v59 )
      break;
    v16 = stream;
  }
  dword_4CF7F40 += 2;
  v16 = stream;
  if ( (*(_BYTE *)(a1 + 128) & 0x20) != 0 )
  {
LABEL_82:
    v53 = ":";
    v54 = 32;
    do
    {
      ++v53;
      putc(v54, stream);
      v54 = *(v53 - 1);
    }
    while ( *(v53 - 1) );
    v55 = *(_QWORD **)(a1 + 152);
    for ( dword_4CF7F40 += 2; v55; ++dword_4CF7F40 )
    {
      putc(32, stream);
      v56 = v55[1];
      ++dword_4CF7F40;
      sub_5D5A80(v56, 0);
      if ( !*v55 )
        break;
      putc(44, stream);
      v55 = (_QWORD *)*v55;
    }
LABEL_88:
    v16 = stream;
  }
LABEL_69:
  v45 = 41;
  v46 = ";";
  while ( 1 )
  {
    result = putc(v45, v16);
    v45 = *v46++;
    if ( !(_BYTE)v45 )
      break;
    v16 = stream;
  }
  dword_4CF7F40 += 2;
  return result;
}
