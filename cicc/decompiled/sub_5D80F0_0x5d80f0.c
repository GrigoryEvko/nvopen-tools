// Function: sub_5D80F0
// Address: 0x5d80f0
//
void __fastcall sub_5D80F0(__int64 a1, __int64 a2, __int64 a3, int *a4, _QWORD *a5, _DWORD *a6)
{
  __int64 i; // r12
  int v9; // ebx
  int v10; // eax
  unsigned __int64 v11; // rsi
  _QWORD *j; // r15
  FILE *v13; // r12
  FILE *v14; // rdi
  char *v15; // rbx
  char v16; // al
  char v17; // al
  __int64 v18; // r15
  __int64 m; // rbx
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // r14
  char v28; // al
  char v29; // al
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rcx
  char v34; // al
  char *v35; // r12
  __int64 v36; // rax
  char v37; // al
  char *v38; // rbx
  char *v39; // rbx
  char v40; // al
  unsigned __int64 v41; // rdi
  __int64 v42; // rdi
  unsigned __int64 jj; // rbx
  int v44; // edi
  char *v45; // r13
  int v46; // eax
  __int64 ii; // rax
  char v48; // al
  char *v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // r15
  unsigned __int64 v52; // r13
  int v53; // r12d
  __int64 v54; // rax
  __int64 n; // r12
  char v56; // al
  char *v57; // r14
  __int64 v58; // rbx
  __int64 v59; // rsi
  __int64 v60; // rax
  char v62; // al
  char *v63; // r12
  char *v64; // r12
  char v65; // al
  char v66; // al
  unsigned __int64 v67; // r12
  unsigned int v68; // r15d
  unsigned __int64 k; // rbx
  unsigned __int64 v70; // rax
  __int64 v71; // r13
  char *v73; // r12
  char v74; // al
  unsigned __int64 v75; // rax
  bool v76; // zf
  FILE *v77; // rax
  char v78; // al
  const char *v79; // r12
  char v80; // al
  const char *v81; // r12
  __int64 v82; // [rsp+8h] [rbp-B8h]
  _QWORD *v83; // [rsp+10h] [rbp-B0h]
  __int64 v84; // [rsp+18h] [rbp-A8h]
  int v85; // [rsp+20h] [rbp-A0h]
  _BOOL4 v86; // [rsp+24h] [rbp-9Ch]
  __int64 v87; // [rsp+28h] [rbp-98h]
  int v88; // [rsp+28h] [rbp-98h]
  __int64 v89; // [rsp+30h] [rbp-90h]
  unsigned __int64 v93; // [rsp+58h] [rbp-68h] BYREF
  _QWORD v94[3]; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v95; // [rsp+78h] [rbp-48h]
  __int64 v96; // [rsp+80h] [rbp-40h]
  unsigned __int64 *v97; // [rsp+88h] [rbp-38h]

  for ( i = a2; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( a3 )
    v9 = *(_BYTE *)(a3 + 173) == 10;
  else
    v9 = sub_8D3B80(i);
  v10 = sub_8D2B80(i);
  if ( !v9 || v10 )
  {
    if ( *a4 )
    {
      v11 = (unsigned int)a6[2];
      if ( !(_DWORD)v11 )
      {
        v11 = (unsigned __int64)a6;
        sub_5D6A30(a1, a6);
      }
      if ( a3 )
      {
        for ( j = 0; a5; a5 = (_QWORD *)*a5 )
          j = a5;
        v13 = stream;
        if ( !dword_4CF7CD4 )
        {
          v14 = qword_4CF7EA8;
          if ( !qword_4CF7EA8 )
          {
            v77 = (FILE *)sub_721330(0, v11);
            qword_4CF7F00 = 0;
            qword_4CF7EA8 = v77;
            v14 = v77;
            qword_4CF7F08 = 0;
            dword_4CF7F10 = 0;
          }
          sub_5D3B20(v14);
        }
        sub_5D45D0((unsigned int *)(a1 + 64));
        if ( *(_BYTE *)(a3 + 173) == 2 )
        {
          v37 = 109;
          v38 = "emcpy(";
          do
          {
            ++v38;
            putc(v37, stream);
            v37 = *(v38 - 1);
          }
          while ( v37 );
          dword_4CF7F40 += 7;
          v39 = " (char *)";
          sub_5D7F90(a1, j);
          v40 = 44;
          do
          {
            ++v39;
            putc(v40, stream);
            v40 = *(v39 - 1);
          }
          while ( v40 );
          dword_4CF7F40 += 10;
          sub_5D5250(a3);
          putc(44, stream);
          v41 = *(_QWORD *)(a3 + 176);
          ++dword_4CF7F40;
          sub_5D32F0(v41);
          putc(41, stream);
          ++dword_4CF7F40;
        }
        else
        {
          v15 = "= ";
          sub_5D7F90(a1, j);
          v16 = 32;
          do
          {
            ++v15;
            putc(v16, stream);
            v16 = *(v15 - 1);
          }
          while ( v16 );
          dword_4CF7F40 += 3;
          sub_5D5250(a3);
        }
        putc(59, stream);
        ++dword_4CF7F40;
        if ( v13 != stream )
          sub_5D3B20(v13);
      }
      return;
    }
    if ( !*a6 )
      sub_5D4790(a6);
    if ( a3 )
    {
      if ( *(_BYTE *)(a3 + 173) != 2 )
      {
LABEL_27:
        sub_5D5250(a3);
        return;
      }
      if ( (*(_BYTE *)(a3 + 168) & 7) != 0 )
      {
        putc(123, stream);
        v66 = *(_BYTE *)(a3 + 168);
        v67 = *(_QWORD *)(a3 + 176);
        ++dword_4CF7F40;
        v68 = qword_4F06B40[v66 & 7];
        if ( v67 )
        {
          for ( k = 0; k < v67; k += v68 )
          {
            v70 = sub_722AB0(k + *(_QWORD *)(a3 + 184), v68);
            sub_5D32F0(v70);
            if ( v67 - v68 != k )
            {
              putc(44, stream);
              ++dword_4CF7F40;
            }
          }
        }
      }
      else
      {
        if ( !*(_BYTE *)(*(_QWORD *)(a3 + 184) + *(_QWORD *)(a3 + 176) - 1LL) )
          goto LABEL_27;
        v21 = 0;
        putc(123, stream);
        v22 = *(_QWORD *)(a3 + 176);
        ++dword_4CF7F40;
        if ( v22 )
        {
          do
          {
            putc(39, stream);
            v23 = *(_QWORD *)(a3 + 184);
            ++dword_4CF7F40;
            sub_746F50((unsigned int)*(char *)(v23 + v21), &qword_4CF7CE0);
            putc(39, stream);
            ++dword_4CF7F40;
            if ( v21 != v22 - 1 )
            {
              putc(44, stream);
              ++dword_4CF7F40;
            }
            ++v21;
          }
          while ( v22 != v21 );
        }
      }
    }
    else
    {
      if ( !(unsigned int)sub_8D2B80(i) )
      {
        putc(48, stream);
        ++dword_4CF7F40;
        return;
      }
      putc(123, stream);
      ++dword_4CF7F40;
      v58 = sub_8D4620(i);
      if ( v58 )
      {
        while ( 1 )
        {
          putc(48, stream);
          ++dword_4CF7F40;
          if ( v58 == 1 )
            break;
          --v58;
          putc(44, stream);
          ++dword_4CF7F40;
        }
      }
    }
    putc(125, stream);
    ++dword_4CF7F40;
    return;
  }
  v94[0] = a5;
  v94[1] = 0;
  v94[2] = i;
  v96 = 0;
  if ( a5 )
  {
    a5[1] = v94;
    v17 = *(_BYTE *)(i + 140);
    if ( v17 != 8 )
    {
      v97 = 0;
      if ( a3 )
      {
        v18 = *(_QWORD *)(a3 + 176);
        goto LABEL_32;
      }
LABEL_178:
      if ( (unsigned __int8)(v17 - 10) <= 1u )
      {
        v18 = 0;
LABEL_33:
        v96 = sub_72FD90(*(_QWORD *)(i + 160), 11);
        m = v96;
        if ( v96 )
        {
          if ( unk_4F068C0 )
            qword_4CF7C90 = 0;
          v89 = *(_QWORD *)(v96 + 120);
          if ( m == sub_72FD90(*(_QWORD *)(i + 160), 10) )
          {
            v84 = 0;
            m = 0;
          }
          else
          {
            for ( m = sub_72FD90(*(_QWORD *)(i + 160), 10); ; m = sub_72FD90(*(_QWORD *)(m + 112), 10) )
            {
              v20 = sub_72FD90(*(_QWORD *)(m + 112), 10);
              if ( v96 == v20 )
                break;
              if ( unk_4F068C0 )
                sub_5D3540(m);
            }
            v84 = 0;
          }
        }
        else
        {
          v89 = 0;
          v84 = 0;
        }
        goto LABEL_55;
      }
LABEL_245:
      sub_721090(i);
    }
    v97 = (unsigned __int64 *)a5[5];
    if ( a3 )
    {
      v18 = *(_QWORD *)(a3 + 176);
      goto LABEL_52;
    }
LABEL_193:
    v60 = *(_QWORD *)(i + 160);
    v95 = 0;
    m = 0;
    v18 = 0;
    v89 = v60;
    v84 = *(_QWORD *)(i + 176);
    goto LABEL_55;
  }
  v97 = 0;
  v17 = *(_BYTE *)(i + 140);
  if ( !a3 )
  {
    if ( v17 != 8 )
      goto LABEL_178;
    goto LABEL_193;
  }
  v18 = *(_QWORD *)(a3 + 176);
  if ( v17 != 8 )
  {
LABEL_32:
    if ( (unsigned __int8)(v17 - 10) <= 1u )
      goto LABEL_33;
    goto LABEL_245;
  }
LABEL_52:
  v24 = *(_QWORD *)(i + 160);
  m = (__int64)v97;
  v95 = 0;
  v89 = v24;
  v84 = *(_QWORD *)(i + 176);
  if ( v97 )
  {
    m = 0;
    if ( !(unsigned int)sub_8D3410(*(_QWORD *)(a3 + 128)) )
      v18 = a3;
  }
LABEL_55:
  v86 = 0;
  if ( a5 )
  {
    v25 = a5[4];
    if ( v25 )
      v86 = (*(_BYTE *)(v25 + 145) & 0x10) != 0;
  }
  v85 = *a4 | v86;
  if ( v85 )
  {
    v85 = 0;
  }
  else
  {
    if ( a3 && (*(_BYTE *)(a3 + 170) & 8) != 0 )
    {
      if ( v89 )
        goto LABEL_62;
LABEL_123:
      if ( !*a6 )
        sub_5D4790(a6);
      if ( (!unk_4F072D8 || !unk_4F068C4) && (*(_BYTE *)(i + 142) & 0x10) == 0 )
      {
        putc(48, stream);
        ++dword_4CF7F40;
      }
      goto LABEL_99;
    }
    if ( *a6 )
    {
      putc(123, stream);
      ++dword_4CF7F40;
      v85 = 1;
    }
    else
    {
      v85 = 1;
      ++a6[1];
    }
  }
  if ( !v89 )
  {
    if ( *a4 )
      goto LABEL_99;
    goto LABEL_123;
  }
LABEL_62:
  if ( v18 )
  {
    v26 = *a4;
  }
  else if ( (unsigned int)sub_8D3410(i) && (*(_BYTE *)(i + 169) & 0x20) != 0
         || (v76 = (unsigned int)sub_8D2B80(v89) == 0, v26 = *a4, !v76) && !v26 )
  {
    if ( !*a6 )
      sub_5D4790(a6);
    goto LABEL_99;
  }
  v83 = a5;
  v27 = i;
  if ( v18 && *(_BYTE *)(v18 + 173) == 13 )
    goto LABEL_83;
LABEL_66:
  v28 = *(_BYTE *)(v27 + 140);
  if ( v26 )
    goto LABEL_96;
  if ( (unsigned __int8)(v28 - 9) <= 2u )
  {
    v82 = v18;
    v87 = v27;
    while ( !m )
    {
      while ( 1 )
      {
        v50 = v96;
LABEL_159:
        v51 = sub_72FD90(v50, 10);
        v52 = sub_5D3810(m, v51, v87);
        if ( v52 )
        {
          if ( !*a6 )
            sub_5D4790(a6);
          for ( n = 0; n != v52; ++n )
          {
            v56 = 39;
            v57 = "\\0',";
            do
            {
              ++v57;
              putc(v56, stream);
              v56 = *(v57 - 1);
            }
            while ( v56 );
            dword_4CF7F40 += 5;
          }
        }
        v53 = 0;
        if ( v96 != v51 )
        {
          v53 = 1;
          if ( unk_4F068C0 )
            sub_5D3540(m);
        }
        if ( !m || (v54 = sub_72FD90(*(_QWORD *)(m + 112), 10), v96 == v54) )
        {
          v27 = v87;
          v18 = v82;
          v28 = *(_BYTE *)(v87 + 140);
          goto LABEL_96;
        }
        if ( !v53 )
          break;
        m = sub_72FD90(*(_QWORD *)(m + 112), 10);
        if ( m )
          goto LABEL_158;
      }
    }
LABEL_158:
    v50 = *(_QWORD *)(m + 112);
    goto LABEL_159;
  }
  if ( dword_4CF7EA0 && v28 == 8 )
  {
    if ( !dword_4CF7EA4++ )
    {
      v78 = 47;
      v79 = "*";
      do
      {
        ++v79;
        putc(v78, stream);
        v78 = *(v79 - 1);
        ++dword_4CF7F40;
      }
      while ( v78 );
    }
    v62 = 32;
    v63 = "[";
    do
    {
      ++v63;
      putc(v62, stream);
      v62 = *(v63 - 1);
    }
    while ( v62 );
    dword_4CF7F40 += 2;
    v64 = ": ";
    sub_5D32F0(v95);
    v65 = 93;
    do
    {
      ++v64;
      putc(v65, stream);
      v65 = *(v64 - 1);
    }
    while ( v65 );
    dword_4CF7F40 += 3;
    if ( !--dword_4CF7EA4 )
    {
      v80 = 42;
      v81 = "/";
      do
      {
        ++v81;
        putc(v80, stream);
        v80 = *(v81 - 1);
        ++dword_4CF7F40;
      }
      while ( v80 );
    }
    v28 = *(_BYTE *)(v27 + 140);
    while ( 1 )
    {
LABEL_96:
      if ( (unsigned __int8)(v28 - 9) > 2u )
        break;
      v89 = *(_QWORD *)(v96 + 120);
      if ( !v18 )
        goto LABEL_98;
LABEL_71:
      if ( *(_BYTE *)(v18 + 173) == 11 )
      {
        v71 = *(_QWORD *)(v18 + 176);
        v93 = *(_QWORD *)(v18 + 184);
        v88 = sub_8D44E0(v71, v89);
        if ( v88 )
          v88 = 1;
        else
          v97 = &v93;
        if ( dword_4CF7EA0 )
        {
          if ( !dword_4CF7EA4++ )
            sub_5D3190("/*");
          v73 = "repetitions: ";
          putc(32, stream);
          ++dword_4CF7F40;
          sub_5D32F0(v93);
          v74 = 32;
          do
          {
            ++v73;
            putc(v74, stream);
            v74 = *(v73 - 1);
          }
          while ( v74 );
          dword_4CF7F40 += 14;
          if ( !--dword_4CF7EA4 )
            sub_5D3190("*/");
        }
        v75 = v93;
        while ( v75 )
        {
          sub_5D80F0(a1, v89, v71, a4, v94, a6);
          v75 = v93;
          if ( v88 )
            v75 = --v93;
          if ( !v75 )
            break;
          if ( !*a4 )
          {
            putc(44, stream);
            ++dword_4CF7F40;
            v75 = v93;
          }
          ++v95;
        }
        v97 = 0;
      }
      else
      {
        sub_5D80F0(a1, v89, v18, a4, v94, a6);
        if ( *(_BYTE *)(v27 + 140) == 8 && v97 )
        {
          if ( !(unsigned int)sub_8D3410(v89) )
            --*v97;
          if ( --v84 )
          {
            if ( *v97 )
              goto LABEL_74;
          }
        }
      }
      v18 = *(_QWORD *)(v18 + 120);
      if ( !v18 )
      {
        i = v27;
        a5 = v83;
        goto LABEL_99;
      }
LABEL_74:
      if ( !*a4 )
      {
        putc(44, stream);
        ++dword_4CF7F40;
      }
      if ( *(_BYTE *)(v18 + 173) != 13 )
      {
        v29 = *(_BYTE *)(v27 + 140);
        if ( v29 == 15 || v29 == 8 )
        {
          ++v95;
        }
        else
        {
          m = v96;
          v30 = v96;
          if ( unk_4F068C0 )
          {
            sub_5D3540(v96);
            v30 = v96;
          }
          v96 = sub_72FD90(*(_QWORD *)(v30 + 112), 11);
        }
      }
      v26 = *a4;
      if ( *(_BYTE *)(v18 + 173) != 13 )
        goto LABEL_66;
LABEL_83:
      if ( !v26 )
      {
        if ( !*a6 )
          sub_5D4790(a6);
        if ( (*(_BYTE *)(v18 + 176) & 1) != 0 )
        {
          putc(46, stream);
          v59 = *(_QWORD *)(v18 + 184);
          ++dword_4CF7F40;
          sub_5D4E40(*(_BYTE **)(v59 + 8), v59);
        }
        else
        {
          putc(91, stream);
          v31 = *(_QWORD *)(v18 + 184);
          ++dword_4CF7F40;
          sub_5D32F0(v31);
          putc(93, stream);
          ++dword_4CF7F40;
        }
        v32 = *(_QWORD *)(v18 + 120);
        if ( (*(_QWORD *)(v32 + 168) & 0xFF0000020000LL) == 0xA0000020000LL )
        {
          v33 = *(_QWORD *)(v32 + 176);
          v34 = 32;
          v35 = "= ";
          if ( v33 && *(_BYTE *)(v33 + 173) == 13 && (*(_BYTE *)(v32 + 169) & 0x20) == 0 )
          {
            *(_BYTE *)(v32 + 170) |= 8u;
            goto LABEL_93;
          }
        }
        else
        {
          v34 = 32;
          v35 = "= ";
        }
        do
        {
          ++v35;
          putc(v34, stream);
          v34 = *(v35 - 1);
        }
        while ( v34 );
        dword_4CF7F40 += 3;
      }
LABEL_93:
      v28 = *(_BYTE *)(v27 + 140);
      if ( v28 == 8 )
        v95 = *(_QWORD *)(v18 + 184);
      else
        v96 = *(_QWORD *)(v18 + 184);
      v18 = *(_QWORD *)(v18 + 120);
    }
  }
  if ( v18 )
    goto LABEL_71;
LABEL_98:
  i = v27;
  a5 = v83;
  sub_5D80F0(a1, v89, 0, a4, v94, a6);
LABEL_99:
  if ( v85 )
  {
    v46 = a6[1];
    if ( v46 )
    {
      a6[1] = v46 - 1;
    }
    else
    {
      putc(125, stream);
      ++dword_4CF7F40;
    }
    if ( (*(_BYTE *)(i + 142) & 0x10) != 0
      && v94[0]
      && !(unsigned int)sub_8D3410(*(_QWORD *)(v94[0] + 16LL))
      && !(unsigned int)sub_8D3B10(*(_QWORD *)(v94[0] + 16LL)) )
    {
      goto LABEL_142;
    }
    if ( !(unsigned int)sub_8D3410(i) || !v94[0] || (unsigned int)sub_8D3410(*(_QWORD *)(v94[0] + 16LL)) )
      goto LABEL_103;
    for ( ii = sub_8D40F0(i); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
      ;
    if ( (*(_BYTE *)(ii + 142) & 0x10) == 0 )
      goto LABEL_103;
    while ( *(_BYTE *)(i + 140) == 12 )
      i = *(_QWORD *)(i + 160);
    if ( *(_QWORD *)(i + 128) <= 1u )
    {
LABEL_142:
      sub_5D31F0(",0");
    }
    else
    {
      v48 = 44;
      v49 = "{0}";
      do
      {
        ++v49;
        putc(v48, stream);
        v48 = *(v49 - 1);
      }
      while ( v48 );
      dword_4CF7F40 += 4;
    }
LABEL_103:
    if ( unk_4F068C0 )
      qword_4CF7C90 = 0;
    goto LABEL_105;
  }
  if ( v86 )
  {
    v36 = v96;
    if ( v96 )
    {
      if ( !*a4 )
      {
        do
        {
          v42 = v36;
          v36 = *(_QWORD *)(v36 + 112);
        }
        while ( v36 && (*(_BYTE *)(v36 + 146) & 8) == 0 );
        for ( jj = sub_5D35E0(v42); *(_QWORD *)(i + 128) > jj; ++jj )
        {
          v44 = 44;
          v45 = "'\\0'";
          do
          {
            ++v45;
            putc(v44, stream);
            v44 = *(v45 - 1);
          }
          while ( *(v45 - 1) );
          dword_4CF7F40 += 5;
        }
      }
      goto LABEL_103;
    }
  }
LABEL_105:
  if ( a5 )
    a5[1] = 0;
}
