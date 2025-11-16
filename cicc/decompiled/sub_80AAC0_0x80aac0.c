// Function: sub_80AAC0
// Address: 0x80aac0
//
__int64 __fastcall sub_80AAC0(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  char v4; // al
  unsigned __int64 v6; // r15
  __int64 v7; // r14
  bool v8; // zf
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // al
  __int64 v14; // rax

  v2 = a1;
  if ( !a1 )
    return v2;
  v3 = 2106130;
  while ( 1 )
  {
    v4 = *(_BYTE *)(v2 + 24);
    if ( v4 != 1 )
    {
      if ( v4 != 5 )
        return v2;
      v9 = *(_QWORD *)(v2 + 56);
      if ( v9 )
      {
        while ( 1 )
        {
          v10 = sub_730290(v9);
          if ( (*(_BYTE *)(v10 + 51) & 0x40) != 0 )
          {
            v10 = sub_730770(v10, 0);
            if ( v10 == v9 )
              break;
          }
          else if ( v10 == v9 )
          {
            break;
          }
          v9 = v10;
        }
      }
      if ( !sub_7307F0(v9) )
        return v2;
      if ( (*(_BYTE *)(v9 + 48) & 0xFB) == 2 || *(_BYTE *)(v9 + 48) == 8 )
      {
        v7 = (__int64)sub_726700(2);
        v12 = *(_QWORD *)(v9 + 56);
        *(_QWORD *)(v7 + 56) = v12;
        *(_QWORD *)v7 = *(_QWORD *)(v12 + 128);
      }
      else
      {
        v7 = sub_6E3F50(v9);
      }
      goto LABEL_9;
    }
    v6 = *(unsigned __int8 *)(v2 + 56);
    v7 = *(_QWORD *)(v2 + 72);
    if ( (_BYTE)v6 != 116 )
      break;
    if ( *(_BYTE *)(v7 + 24) == 2 )
    {
      v11 = *(_QWORD *)(v7 + 56);
      if ( *(_BYTE *)(v11 + 173) == 12 && (*(_BYTE *)(v11 + 176) & 0xF7) == 3 )
        goto LABEL_25;
    }
LABEL_9:
    v8 = v7 == v2;
    v2 = v7;
    if ( v8 || !v7 )
      return v2;
  }
  if ( (unsigned __int8)v6 <= 0x15u )
  {
    if ( _bittest64(&v3, v6) || *(char *)(v2 + 58) < 0 )
      goto LABEL_9;
  }
  else if ( *(char *)(v2 + 58) < 0 )
  {
    goto LABEL_9;
  }
  if ( sub_730740(v2) )
  {
    if ( (*(_BYTE *)(v2 + 27) & 2) == 0 )
      return v2;
    goto LABEL_9;
  }
  if ( (*(_BYTE *)(v2 + 27) & 2) == 0 )
    return v2;
  if ( (*(_BYTE *)(v2 + 59) & 1) != 0 )
    goto LABEL_44;
  v13 = *(_BYTE *)(v7 + 24);
  if ( (_BYTE)v6 == 94 )
  {
    if ( v13 == 3 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v7 + 56) + 172LL) & 2) == 0 )
        return v2;
LABEL_44:
      v7 = *(_QWORD *)(v7 + 16);
      goto LABEL_9;
    }
LABEL_37:
    if ( v13 != 1 )
      return v2;
LABEL_38:
    if ( (*(_BYTE *)(v7 + 27) & 2) == 0 )
      return v2;
    if ( (_BYTE)v6 )
    {
      if ( (_BYTE)v6 != 3 || *(_BYTE *)(v7 + 56) )
        return v2;
    }
    else if ( *(_BYTE *)(v7 + 56) != 3 )
    {
      return v2;
    }
    v7 = *(_QWORD *)(v7 + 72);
    goto LABEL_9;
  }
  if ( (_BYTE)v6 != 3 )
  {
    if ( v13 != 1 )
    {
      if ( (_BYTE)v6 != 31 )
        return v2;
      goto LABEL_9;
    }
    goto LABEL_38;
  }
  if ( v13 != 2 )
    goto LABEL_37;
  v14 = *(_QWORD *)(v7 + 56);
  if ( *(_BYTE *)(v14 + 173) == 12 && *(_BYTE *)(v14 + 176) == 9 )
  {
LABEL_25:
    *a2 = 1;
    return v7;
  }
  return v2;
}
