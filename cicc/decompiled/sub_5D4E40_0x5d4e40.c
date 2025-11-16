// Function: sub_5D4E40
// Address: 0x5d4e40
//
void __fastcall sub_5D4E40(_BYTE *a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // rax
  _BYTE *v5; // rax
  int v6; // edi
  _BYTE *v7; // r14
  int v8; // edi
  _BYTE *v9; // rbx
  __int64 v10; // rax
  unsigned __int64 i; // r12
  int v12; // edi
  char *v13; // rbx
  char *v14; // rbx
  __int64 v15; // rax
  int v16; // edi
  int v17; // edi
  char *v18; // r14
  char s; // [rsp+0h] [rbp-60h] BYREF
  char v20; // [rsp+1h] [rbp-5Fh] BYREF

  v3 = (_QWORD *)qword_4CF7CB0;
  if ( !qword_4CF7CB0 )
  {
    if ( a1 )
    {
LABEL_7:
      v8 = (char)*a1;
      v9 = a1 + 1;
      if ( *a1 )
      {
        do
        {
          ++v9;
          putc(v8, stream);
          v8 = (char)*(v9 - 1);
          ++dword_4CF7F40;
        }
        while ( (_BYTE)v8 );
      }
      if ( a2 && (*(_BYTE *)(a2 + 145) & 2) != 0 )
      {
        v10 = *(_QWORD *)(a2 + 112);
        for ( i = 1; v10; ++i )
        {
          if ( (*(_BYTE *)(v10 + 145) & 2) == 0 )
            break;
          v10 = *(_QWORD *)(v10 + 112);
        }
        v12 = 95;
        v13 = (char *)&unk_42B6DB3;
        do
        {
          ++v13;
          putc(v12, stream);
          v12 = *(v13 - 1);
          ++dword_4CF7F40;
        }
        while ( (_BYTE)v12 );
        sub_5D32F0(i);
      }
      return;
    }
LABEL_26:
    sub_5D34A0();
    return;
  }
  do
  {
    v4 = v3[2];
    if ( (*(_BYTE *)(v4 + 146) & 4) != 0 )
    {
      v17 = 95;
      v18 = (char *)&unk_42B6DB3;
      do
      {
        ++v18;
        putc(v17, stream);
        v17 = *(v18 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v17 );
      v4 = v3[2];
    }
    v5 = *(_BYTE **)(v4 + 8);
    v6 = (char)*v5;
    v7 = v5 + 1;
    if ( *v5 )
    {
      do
      {
        ++v7;
        putc(v6, stream);
        v6 = (char)*(v7 - 1);
        ++dword_4CF7F40;
      }
      while ( (_BYTE)v6 );
    }
    putc(95, stream);
    v3 = (_QWORD *)*v3;
    ++dword_4CF7F40;
  }
  while ( v3 );
  if ( a1 )
    goto LABEL_7;
  if ( !qword_4CF7CB0 )
    goto LABEL_26;
  v14 = &v20;
  v15 = sub_737880(a2);
  snprintf(&s, 0x32u, "__T%llu", v15);
  v16 = s;
  if ( s )
  {
    do
    {
      ++v14;
      putc(v16, stream);
      v16 = *(v14 - 1);
      ++dword_4CF7F40;
    }
    while ( (_BYTE)v16 );
  }
}
