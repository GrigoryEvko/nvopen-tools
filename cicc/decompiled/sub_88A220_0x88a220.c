// Function: sub_88A220
// Address: 0x88a220
//
const char *__fastcall sub_88A220(__int64 a1, __int64 a2, char a3)
{
  __int64 i; // rbx
  _BYTE *v6; // rax
  char v7; // dl
  _QWORD *v8; // rcx
  __int64 v9; // r8
  int v11; // esi
  unsigned __int8 *v12; // rcx
  __int64 v13; // rax
  unsigned __int8 *v14; // r9
  unsigned __int64 v15; // r8

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (unsigned int)sub_8D2AC0(i) )
  {
    v6 = &unk_4B7DF80;
    if ( qword_4F06A7C )
      v6 = &unk_4B7E000;
    while ( 1 )
    {
      v7 = *v6;
      if ( *v6 == 14 )
        break;
      v8 = v6;
      v6 += 32;
      if ( *(_BYTE *)(i + 160) == v7 )
      {
        v9 = v8[1] == a2 ? v8[2] : v8[3];
        if ( v9 )
          return (const char *)v9;
      }
    }
    return 0;
  }
  if ( (unsigned int)sub_8D2780(i) )
  {
    v11 = sub_8D27E0(i);
    if ( a3 == 3 )
    {
      v12 = (unsigned __int8 *)&unk_4B7DC40;
      if ( qword_4F06A7C )
        v12 = (unsigned __int8 *)&unk_4B7DCC0;
    }
    else
    {
      v12 = (unsigned __int8 *)&unk_4B7DD40;
      if ( qword_4F06A7C )
        v12 = (unsigned __int8 *)&unk_4B7DE60;
    }
    while ( 1 )
    {
      v13 = *v12;
      if ( (_BYTE)v13 == 13 )
        break;
      while ( 1 )
      {
        v14 = v12;
        v12 += 32;
        if ( byte_4B6DF90[v13] == v11 )
        {
          v15 = *((_QWORD *)v12 - 3);
          if ( *(_QWORD *)(i + 128) == 8 / v15 )
            break;
        }
        v13 = *v12;
        if ( (_BYTE)v13 == 13 )
          return 0;
      }
      if ( v15 == a2 )
        v9 = *((_QWORD *)v14 + 2);
      else
        v9 = *((_QWORD *)v14 + 3);
      if ( v9 )
        return (const char *)v9;
    }
    return 0;
  }
  if ( *(_BYTE *)(i + 140) != 18 )
    goto LABEL_36;
  if ( a2 != 8 )
  {
    if ( a2 == 16 )
      return "__Mfloat8x16_t";
LABEL_36:
    sub_721090();
  }
  return "__Mfloat8x8_t";
}
