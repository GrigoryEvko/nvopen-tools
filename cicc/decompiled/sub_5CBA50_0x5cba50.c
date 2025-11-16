// Function: sub_5CBA50
// Address: 0x5cba50
//
__int64 __fastcall sub_5CBA50(char a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  char v4; // al
  char *v5; // r9
  char v6; // r12
  const char *v7; // rax
  char v8; // r8
  _QWORD *v9; // rax
  _QWORD *v10; // rbx
  char *v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned int v15; // edx
  int v16; // eax
  int v18; // eax
  unsigned __int8 v19; // r15
  __int64 v20; // rdx
  char *v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rax

  if ( !sub_5CBA20(word_4F06418[0]) )
  {
    v3 = 0;
    sub_6851D0(40);
    return v3;
  }
  v2 = sub_727670();
  *(_BYTE *)(v2 + 9) = a1;
  v3 = v2;
  *(_QWORD *)(v2 + 56) = unk_4F063F8;
  sub_5C6A20(v2);
  sub_7B8B50();
  if ( word_4F06418[0] == 146 && a1 == 1 )
  {
    sub_7B8B50();
    if ( sub_5CBA20(word_4F06418[0]) )
    {
      if ( a2 )
      {
        v23 = v3 + 56;
        v3 = 0;
        sub_6851C0(2802, v23);
        sub_7B8B50();
        return v3;
      }
      sub_5C9440(v3);
      v24 = *(_QWORD *)(v3 + 16);
      *(_QWORD *)(v3 + 16) = 0;
      *(_QWORD *)(v3 + 24) = v24;
      sub_5C6A20(v3);
      sub_7B8B50();
    }
    else
    {
      sub_6851D0(40);
    }
  }
  else if ( a2 )
  {
    *(_QWORD *)(v3 + 24) = *(_QWORD *)(a2 + 16);
    v4 = *(_BYTE *)(v3 + 11) | 0x40;
    *(_BYTE *)(v3 + 11) = v4;
    *(_BYTE *)(v3 + 11) = *(_BYTE *)(a2 + 11) & 0x80 | v4 & 0x7F;
  }
  v5 = *(char **)(v3 + 16);
  v6 = *(_BYTE *)(v3 + 9);
  if ( !unk_4F077B8 || unk_4F077A8 <= 0x9F5Fu )
  {
    if ( unk_4F077B4 && v6 == 1 )
    {
      v7 = *(const char **)(v3 + 24);
      if ( !v7 )
        goto LABEL_52;
      goto LABEL_50;
    }
    goto LABEL_16;
  }
  if ( v6 != 1 )
  {
LABEL_16:
    v8 = *(_BYTE *)(v3 + 9);
    goto LABEL_17;
  }
  v7 = *(const char **)(v3 + 24);
  v8 = 1;
  if ( v7 )
  {
    if ( !strcmp(*(const char **)(v3 + 24), "gnu") || !strcmp(*(const char **)(v3 + 24), "__gnu__") )
      goto LABEL_51;
    if ( !unk_4F077B4 )
      goto LABEL_17;
LABEL_50:
    if ( !strcmp(v7, "clang") )
    {
LABEL_51:
      *(_BYTE *)(v3 + 11) |= 0x10u;
      v8 = 2;
      v6 = 2;
      goto LABEL_17;
    }
LABEL_52:
    v8 = 1;
    v6 = 1;
  }
LABEL_17:
  v9 = (_QWORD *)sub_5C7880(v5, v8);
  if ( v9 )
  {
    v10 = (_QWORD *)*v9;
    if ( *v9 )
    {
      do
      {
        while ( 1 )
        {
          v11 = (char *)((**(_BYTE **)(v10[1] + 16LL) == 49) + *(_QWORD *)(v10[1] + 16LL));
          if ( v6 != 2 )
            break;
          if ( sub_5CAA00(v11, v3) )
            goto LABEL_27;
LABEL_21:
          v10 = (_QWORD *)*v10;
          if ( !v10 )
            goto LABEL_32;
        }
        if ( v6 == 3 )
        {
          if ( (unsigned int)sub_5C97E0(v11) )
            goto LABEL_27;
          goto LABEL_21;
        }
        if ( v6 != 1 )
          sub_721090(v11);
        if ( sub_5CAB70(v11, v3) )
        {
LABEL_27:
          v12 = v10[1];
          v13 = *(unsigned __int8 *)(v12 + 24);
          *(_BYTE *)(v3 + 8) = v13;
          *(_BYTE *)(v3 + 11) = (2 * (**((_BYTE **)&unk_496EE40 + 3 * v13 + 1) == 84)) | *(_BYTE *)(v3 + 11) & 0xFD;
          sub_5C9ED0(v3, *(_BYTE **)(v12 + 8));
          v14 = *(unsigned __int8 *)(v3 + 8);
          v15 = *(unsigned __int8 *)(v3 + 9);
          v16 = dword_4CF6E60[v14];
          if ( _bittest(&v16, v15) )
          {
            v22 = 4;
            if ( **(_BYTE **)(v12 + 16) == 49 )
            {
              *(_BYTE *)(v3 + 8) = 0;
              v22 = 8;
            }
            sub_684AA0(v22, 1834, v3 + 56);
          }
          else
          {
            dword_4CF6E60[v14] = v16 | (1 << v15);
          }
          return v3;
        }
        v10 = (_QWORD *)*v10;
      }
      while ( v10 );
    }
  }
LABEL_32:
  sub_5C9ED0(v3, "?(*)");
  v18 = unk_4F0751C;
  if ( unk_4F0751C )
  {
    if ( *(_BYTE *)(v3 + 9) != 3 )
      return v3;
    v19 = 7;
  }
  else
  {
    v19 = 2 * (*(_BYTE *)(v3 + 9) == 3) + 5;
  }
  if ( *(char *)(v3 + 11) < 0 )
    v19 = 4;
  v20 = *(_QWORD *)(unk_4F04C68 + 776LL * unk_4F04C64 + 624);
  if ( v20 )
  {
    if ( sub_736C60(88, *(_QWORD *)(v20 + 184))
      || sub_736C60(88, *(_QWORD *)(*(_QWORD *)(unk_4F04C68 + 776LL * unk_4F04C64 + 624) + 200LL)) )
    {
      v21 = sub_5C79F0(v3);
      sub_6849F0(v19, 1097, v3 + 56, v21);
    }
    v18 = unk_4F0751C;
  }
  if ( !v18 )
    return 0;
  return v3;
}
