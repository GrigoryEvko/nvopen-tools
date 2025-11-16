// Function: sub_70E2C0
// Address: 0x70e2c0
//
_BOOL8 __fastcall sub_70E2C0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // rax
  _BOOL8 result; // rax
  char v7; // al
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx

  *a3 = 0;
  if ( *(_BYTE *)(a1 + 173) == 1 )
  {
    if ( *(_BYTE *)(a2 + 173) != 1 )
    {
      v4 = 0;
LABEL_4:
      switch ( *(_BYTE *)(a2 + 176) )
      {
        case 0:
          v5 = *(_QWORD *)(a2 + 184);
          if ( (*(_BYTE *)(v5 + 200) & 0x20) == 0 )
            goto LABEL_6;
          goto LABEL_18;
        case 1:
          v5 = *(_QWORD *)(a2 + 184);
          if ( (*(_BYTE *)(v5 + 168) & 8) == 0 )
            goto LABEL_6;
LABEL_18:
          *a3 = 1;
          return 0;
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
          v5 = *(_QWORD *)(a2 + 184);
LABEL_6:
          if ( !*a3 )
            goto LABEL_7;
          return 0;
        default:
          goto LABEL_34;
      }
    }
    return 1;
  }
  v4 = *(_QWORD *)(a1 + 184);
  switch ( *(_BYTE *)(a1 + 176) )
  {
    case 0:
      if ( (*(_BYTE *)(v4 + 200) & 0x20) != 0 )
        goto LABEL_12;
      goto LABEL_13;
    case 1:
      if ( (*(_BYTE *)(v4 + 168) & 8) != 0 )
LABEL_12:
        *a3 = 1;
LABEL_13:
      if ( *(_BYTE *)(a2 + 173) != 1 )
        goto LABEL_4;
      return 0;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      if ( *(_BYTE *)(a2 + 173) != 1 )
        goto LABEL_4;
      v5 = 0;
LABEL_7:
      if ( v5 == v4 )
        return 1;
      if ( *(_BYTE *)(a1 + 173) != 6 )
        return 0;
      if ( *(_BYTE *)(a2 + 173) != 6 )
        return 0;
      v7 = *(_BYTE *)(a1 + 176);
      if ( v7 != *(_BYTE *)(a2 + 176) )
        return 0;
      if ( v7 == 2 )
      {
        v10 = *(_QWORD *)(a1 + 184);
        if ( *(_BYTE *)(v10 + 173) != 2 )
          return 0;
        v11 = *(_QWORD *)(a2 + 184);
        return *(_BYTE *)(v11 + 173) == 2 && *(_QWORD *)(v10 + 184) == *(_QWORD *)(v11 + 184);
      }
      if ( (unsigned __int8)(v7 - 4) > 1u )
        return 0;
      v8 = *(_QWORD *)(a1 + 184);
      v9 = *(_QWORD *)(a2 + 184);
      result = 1;
      if ( v8 != v9 )
        return (unsigned int)sub_8D97D0(v8, v9, 0, a4, v4) != 0;
      return result;
    default:
LABEL_34:
      sub_721090(a1);
  }
}
