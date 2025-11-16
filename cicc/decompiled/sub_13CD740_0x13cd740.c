// Function: sub_13CD740
// Address: 0x13cd740
//
__int64 __fastcall sub_13CD740(unsigned int a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  if ( a1 == 140 )
  {
LABEL_13:
    if ( *(_BYTE *)(a2 + 16) != 78 )
      return 0;
    goto LABEL_14;
  }
  if ( a1 > 0x8C )
  {
    if ( a1 > 0xBC )
    {
      if ( a1 != 206 )
        return 0;
    }
    else if ( a1 <= 0xBA )
    {
      return 0;
    }
    goto LABEL_13;
  }
  if ( a1 <= 8 )
  {
    if ( a1 <= 6 )
      goto LABEL_5;
    if ( *(_BYTE *)(a2 + 16) != 78 )
      return 0;
LABEL_14:
    v4 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v4 + 16) )
    {
      if ( (*(_BYTE *)(v4 + 33) & 0x20) != 0 && *(_DWORD *)(v4 + 36) == a1 )
        return a2;
      if ( a1 != 55 )
      {
        if ( a1 <= 0x37 )
        {
          if ( a1 == 54 )
            goto LABEL_38;
          return 0;
        }
        goto LABEL_29;
      }
LABEL_43:
      if ( !(unsigned __int8)sub_15F24A0(*(_QWORD *)(a3 + 32)) )
        return 0;
      if ( *(_BYTE *)(a2 + 16) != 78 )
        return 0;
      v7 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v7 + 16) || *(_DWORD *)(v7 + 36) != 124 )
        return 0;
      goto LABEL_22;
    }
    if ( a1 == 55 )
      goto LABEL_43;
    if ( a1 <= 0x37 )
    {
      if ( a1 != 54 )
        return 0;
      goto LABEL_38;
    }
LABEL_29:
    if ( a1 == 122 )
    {
      if ( !(unsigned __int8)sub_15F24A0(*(_QWORD *)(a3 + 32)) )
        return 0;
      if ( *(_BYTE *)(a2 + 16) != 78 )
        return 0;
      v8 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v8 + 16) || *(_DWORD *)(v8 + 36) != 54 )
        return 0;
      goto LABEL_22;
    }
    if ( a1 == 124 )
    {
      if ( !(unsigned __int8)sub_15F24A0(*(_QWORD *)(a3 + 32)) )
        return 0;
      if ( *(_BYTE *)(a2 + 16) != 78 )
        return 0;
      v5 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(v5 + 16) || *(_DWORD *)(v5 + 36) != 55 )
        return 0;
      goto LABEL_22;
    }
    goto LABEL_49;
  }
  if ( a1 - 96 <= 1 )
  {
    if ( *(_BYTE *)(a2 + 16) != 78 )
    {
LABEL_49:
      if ( a1 != 96 || !(unsigned __int8)sub_14ABE20(a2, *(_QWORD *)(a3 + 8)) )
        return 0;
      return a2;
    }
    goto LABEL_14;
  }
  if ( a1 == 55 )
    goto LABEL_43;
  if ( a1 > 0x37 )
    goto LABEL_29;
LABEL_5:
  if ( a1 == 6 )
  {
    if ( *(_BYTE *)(a2 + 16) == 78 )
    {
      v9 = *(_QWORD *)(a2 - 24);
      if ( !*(_BYTE *)(v9 + 16) && *(_DWORD *)(v9 + 36) == 6 )
        return *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL)
                         - 24LL * (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
    }
    return 0;
  }
  if ( a1 == 54 )
  {
LABEL_38:
    if ( !(unsigned __int8)sub_15F24A0(*(_QWORD *)(a3 + 32)) )
      return 0;
    if ( *(_BYTE *)(a2 + 16) != 78 )
      return 0;
    v6 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v6 + 16) || *(_DWORD *)(v6 + 36) != 122 )
      return 0;
    goto LABEL_22;
  }
  if ( a1 != 5 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) != 78 )
    return 0;
  v10 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v10 + 16) || *(_DWORD *)(v10 + 36) != 5 )
    return 0;
LABEL_22:
  result = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( !result )
    return 0;
  return result;
}
