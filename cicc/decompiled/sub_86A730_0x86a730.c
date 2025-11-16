// Function: sub_86A730
// Address: 0x86a730
//
_QWORD *__fastcall sub_86A730(__int64 a1)
{
  char v1; // al
  _QWORD *v2; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  char v5; // al
  _BOOL4 v6; // r13d
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  char v11; // al
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rdx

  v1 = *(_BYTE *)(a1 + 16);
  v2 = *(_QWORD **)a1;
  if ( v1 != 53 )
  {
    if ( v1 != 54 )
      return v2;
    v3 = *(_QWORD *)(a1 + 24);
    if ( *(_BYTE *)(v3 + 8) != 6 )
      return v2;
    v4 = *(_QWORD *)(v3 + 16);
    v5 = *(_BYTE *)(v4 + 140);
    if ( v5 == 12 )
    {
      v13 = *(unsigned __int8 *)(v4 + 184);
      if ( (unsigned __int8)v13 <= 0xCu )
      {
        v14 = 6338;
        if ( _bittest64(&v14, v13) )
          return v2;
      }
      if ( (*(_DWORD *)(v4 + 140) & 0x8008000) != 0 )
        return v2;
    }
    else
    {
      if ( (*(_DWORD *)(v4 + 140) & 0x8008000) != 0 )
        return v2;
      if ( v5 == 2 && (*(_BYTE *)(v4 + 161) & 8) != 0 )
      {
        v6 = *(_QWORD *)(v4 + 8) == 0;
        v7 = 0;
        goto LABEL_9;
      }
    }
    v6 = 0;
    v7 = 0;
    goto LABEL_9;
  }
  v7 = *(_QWORD *)(a1 + 24);
  if ( (*(_BYTE *)(v7 + 57) & 9) != 0 )
    return v2;
  if ( *(_BYTE *)(v7 + 16) != 6 )
    return v2;
  v4 = *(_QWORD *)(v7 + 24);
  v6 = 0;
  if ( *(_BYTE *)(v4 + 140) == 12 )
    return v2;
LABEL_9:
  if ( !(v6 | (unsigned int)qword_4D0495C | HIDWORD(qword_4D0495C))
    && (!*(_QWORD *)(v4 + 8) || (unsigned __int8)(*(_BYTE *)(v4 + 140) - 9) <= 2u && (*(_BYTE *)(v4 + 177) & 4) != 0) )
  {
    return v2;
  }
LABEL_13:
  if ( v2 )
  {
    while ( 1 )
    {
      if ( *((_BYTE *)v2 + 16) == 58 )
      {
        v2 = (_QWORD *)*v2;
        goto LABEL_13;
      }
      v8 = sub_86A430((__int64)v2);
      if ( !v8 )
        break;
      v9 = sub_8D4940(v8);
      if ( v9 != v4 )
      {
        if ( !v9 )
          break;
        if ( !dword_4F07588 )
          break;
        v10 = *(_QWORD *)(v9 + 32);
        if ( *(_QWORD *)(v4 + 32) != v10 || !v10 )
          break;
      }
      v11 = *((_BYTE *)v2 - 8);
      if ( !v6 )
      {
        if ( v11 >= 0 )
          goto LABEL_25;
        return v2;
      }
      if ( v11 < 0 )
        return v2;
      v2 = (_QWORD *)sub_86A660((_QWORD **)v2);
      if ( !v2 )
        goto LABEL_24;
    }
    if ( *((_BYTE *)v2 + 16) == 54 )
    {
      v15 = v2[3];
      if ( *(_BYTE *)(v15 + 8) == 6 )
      {
        v16 = *(_QWORD *)(v15 + 16);
        if ( *(_BYTE *)(v16 + 140) == 12 )
        {
          v17 = *(unsigned __int8 *)(v16 + 184);
          if ( (unsigned __int8)v17 <= 0xCu )
          {
            v18 = 6338;
            if ( _bittest64(&v18, v17) )
              return v2;
          }
        }
      }
    }
  }
  else
  {
LABEL_24:
    v2 = 0;
  }
LABEL_25:
  if ( v7 )
  {
    *(_BYTE *)(v7 + 57) = *(_BYTE *)(v7 + 57) & 0xF6 | 1;
    return v2;
  }
  *(_BYTE *)(v4 + 143) |= 8u;
  return v2;
}
