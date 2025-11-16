// Function: sub_5DEB70
// Address: 0x5deb70
//
int __fastcall sub_5DEB70(__int64 *a1, int a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r15
  int v8; // r14d
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int result; // eax
  char v15; // al
  __int64 v16; // r14
  __int64 i; // rax
  __int64 j; // rax
  int v19; // edi
  char *v20; // rbx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // ebx
  int v26; // eax
  int v27; // edi
  int v28; // r12d
  char *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rax
  char v32; // dl
  __int64 v33; // rax
  char v34; // dl
  unsigned __int64 v35; // [rsp+0h] [rbp-60h]
  int v36; // [rsp+Ch] [rbp-54h]
  char s; // [rsp+10h] [rbp-50h] BYREF
  char v38; // [rsp+11h] [rbp-4Fh] BYREF

  v2 = a1[9];
  putc(40, stream);
  ++dword_4CF7F40;
  if ( *(_BYTE *)(v2 + 24) != 1 || (unsigned __int8)(*(_BYTE *)(v2 + 56) - 94) > 1u )
  {
    v35 = 0;
    v7 = 0;
    v8 = 0;
    v36 = 0;
    goto LABEL_3;
  }
  v7 = *(_QWORD *)(v2 + 72);
  v16 = 0;
  for ( i = *(_QWORD *)(*(_QWORD *)(v7 + 16) + 56LL);
        (*(_BYTE *)(i + 145) & 0x10) != 0;
        i = sub_72FD90(*(_QWORD *)(j + 160), 11) )
  {
    v16 += *(_QWORD *)(i + 128);
    for ( j = *(_QWORD *)(i + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
  }
  if ( (*(_BYTE *)(i + 144) & 4) == 0 )
  {
    if ( *(_BYTE *)(v2 + 56) == 94 )
    {
      if ( (*(_BYTE *)(v7 + 25) & 1) != 0 )
      {
        if ( *(_BYTE *)(v2 + 24) == 1 )
        {
          v31 = v2;
          do
          {
            v31 = *(_QWORD *)(v31 + 72);
            if ( *(_BYTE *)(v31 + 24) != 1 )
              break;
            v32 = *(_BYTE *)(v31 + 56);
            if ( v32 == 91 )
              goto LABEL_58;
          }
          while ( v32 == 94 );
        }
      }
      else
      {
        v33 = *(_QWORD *)(v2 + 72);
        v34 = *(_BYTE *)(v33 + 24);
        if ( v34 != 3 && (v34 != 1 || *(_BYTE *)(v33 + 56) != 3) )
        {
LABEL_58:
          *(_BYTE *)(v2 + 59) |= 0x80u;
          v8 = 0;
          v35 = 0;
          v36 = 1;
          goto LABEL_3;
        }
      }
    }
    v35 = 0;
    v8 = 0;
    v36 = 0;
    goto LABEL_3;
  }
  if ( (*((_BYTE *)a1 + 25) & 3) == 0 )
  {
    sub_5DBFC0(v2, 0, v3, v4, v5, v6);
    goto LABEL_12;
  }
  v36 = 0;
  v35 = *(_QWORD *)(i + 128) + v16;
  v8 = 1;
LABEL_3:
  if ( !a2 )
  {
    putc(42, stream);
    ++dword_4CF7F40;
  }
  v9 = a1[1];
  if ( v9 )
    goto LABEL_6;
  if ( (*((_BYTE *)a1 + 25) & 1) != 0
    || !(unsigned int)sub_8D2310(*(_QWORD *)v2)
    || !(unsigned int)sub_8D2E30(*a1)
    || (v30 = sub_8D46C0(*a1), !(unsigned int)sub_8D2310(v30)) )
  {
    v9 = *a1;
LABEL_6:
    putc(40, stream);
    ++dword_4CF7F40;
    sub_5D4320(v9, 1);
    putc(41, stream);
    ++dword_4CF7F40;
    goto LABEL_7;
  }
  sub_5D3E00(*a1);
LABEL_7:
  if ( !v8 )
  {
    if ( *(_BYTE *)(v2 + 24) == 1 )
    {
      v15 = *(_BYTE *)(v2 + 56);
      if ( v15 == 3 )
      {
        v2 = *(_QWORD *)(v2 + 72);
LABEL_11:
        sub_5DBFC0(v2, (FILE *)1, v10, v11, v12, v13);
        goto LABEL_12;
      }
      if ( ((v15 - 6) & 0xFD) == 0 )
      {
        sub_5DEB70(v2, 1);
        goto LABEL_12;
      }
    }
    if ( !v36 )
    {
      putc(38, stream);
      ++dword_4CF7F40;
    }
    goto LABEL_11;
  }
  v19 = 40;
  v20 = "((char *)";
  do
  {
    ++v20;
    putc(v19, stream);
    v19 = *(v20 - 1);
  }
  while ( *(v20 - 1) );
  dword_4CF7F40 += 10;
  if ( *(_BYTE *)(v2 + 56) == 94 )
  {
    putc(38, stream);
    ++dword_4CF7F40;
  }
  sub_5DBFC0(v7, (FILE *)1, v21, v22, v23, v24);
  putc(41, stream);
  ++dword_4CF7F40;
  if ( v35 )
  {
    if ( v35 > 9 )
    {
      sub_622470(v35, &s);
    }
    else
    {
      v38 = 0;
      s = v35 + 48;
    }
    putc(43, stream);
    v25 = ++dword_4CF7F40;
    v26 = strlen(&s);
    v27 = s;
    v28 = v26;
    if ( s )
    {
      v29 = &v38;
      do
      {
        ++v29;
        putc(v27, stream);
        v27 = *(v29 - 1);
      }
      while ( *(v29 - 1) );
      v25 = dword_4CF7F40;
    }
    dword_4CF7F40 = v28 + v25;
  }
  putc(41, stream);
  ++dword_4CF7F40;
LABEL_12:
  result = putc(41, stream);
  ++dword_4CF7F40;
  return result;
}
