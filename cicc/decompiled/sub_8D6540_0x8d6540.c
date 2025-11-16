// Function: sub_8D6540
// Address: 0x8d6540
//
_QWORD *__fastcall sub_8D6540(__int64 a1)
{
  char v1; // cl
  _QWORD *v2; // r8
  __int64 v3; // rax
  char v4; // dl
  _QWORD *result; // rax
  char v6; // dl
  char v7; // si
  _QWORD *v8; // rax

  v1 = *(_BYTE *)(a1 + 140);
  v2 = (_QWORD *)a1;
  v3 = a1;
  if ( v1 == 12 )
  {
    do
    {
      v3 = *(_QWORD *)(v3 + 160);
      v4 = *(_BYTE *)(v3 + 140);
    }
    while ( v4 == 12 );
  }
  else
  {
    v4 = *(_BYTE *)(a1 + 140);
  }
  if ( v4 != 2 )
    return (_QWORD *)a1;
  v6 = *(_BYTE *)(v3 + 161);
  if ( (v6 & 0x10) != 0 )
    return (_QWORD *)a1;
  if ( (*(_BYTE *)(v3 + 162) & 4) != 0 )
    return sub_72BA30(5u);
  v7 = *(_BYTE *)(v3 + 160);
  if ( dword_4F077C4 != 2 )
  {
    switch ( v7 )
    {
      case 0:
        goto LABEL_20;
      case 1:
      case 3:
        goto LABEL_11;
      case 2:
        goto LABEL_22;
      case 4:
        if ( dword_4F077C4 != 1 )
          goto LABEL_10;
        goto LABEL_24;
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
        goto LABEL_13;
      default:
        goto LABEL_35;
    }
  }
  if ( (*(_DWORD *)(v3 + 160) & 0x34800) != 0 )
  {
    if ( *(_QWORD *)&dword_4F06B20 != unk_4F06B10 )
    {
      switch ( v7 )
      {
        case 0:
          goto LABEL_20;
        case 1:
        case 3:
          goto LABEL_11;
        case 2:
          goto LABEL_23;
        case 4:
          goto LABEL_10;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
          goto LABEL_16;
        default:
          goto LABEL_35;
      }
    }
    if ( (unsigned __int8)(v7 - 7) > 1u )
    {
      switch ( v7 )
      {
        case 0:
          goto LABEL_20;
        case 1:
        case 3:
          goto LABEL_11;
        case 2:
          goto LABEL_23;
        case 4:
          goto LABEL_10;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
          goto LABEL_16;
        default:
          goto LABEL_35;
      }
    }
    if ( v7 != 7 )
      return sub_72BA30(6u);
    return sub_72BA30(5u);
  }
  switch ( v7 )
  {
    case 0:
LABEL_20:
      if ( dword_4F06B98 && v6 >= 0 )
        goto LABEL_11;
LABEL_22:
      if ( dword_4F077C4 == 1 )
        goto LABEL_24;
LABEL_23:
      if ( *(_QWORD *)&dword_4F06B20 > 1u )
LABEL_11:
        v2 = sub_72BA30(5u);
      else
LABEL_24:
        v2 = sub_72BA30(6u);
LABEL_12:
      if ( dword_4F077C4 != 2 )
        goto LABEL_13;
      v1 = *((_BYTE *)v2 + 140);
LABEL_16:
      v8 = v2;
      if ( v1 == 12 )
      {
        do
          v8 = (_QWORD *)v8[20];
        while ( *((_BYTE *)v8 + 140) == 12 );
      }
      if ( (v8[20] & 0x34800) != 0 )
        result = sub_72BA30(*((_BYTE *)v8 + 160));
      else
LABEL_13:
        result = v2;
      break;
    case 1:
    case 3:
      goto LABEL_11;
    case 2:
      goto LABEL_23;
    case 4:
LABEL_10:
      if ( *(_QWORD *)&dword_4F06B20 > unk_4F06B30 )
        goto LABEL_11;
      goto LABEL_24;
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
      goto LABEL_12;
    default:
LABEL_35:
      sub_721090();
  }
  return result;
}
