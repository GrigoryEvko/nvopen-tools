// Function: sub_98D0D0
// Address: 0x98d0d0
//
__int64 __fastcall sub_98D0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char *v5; // rcx
  unsigned __int8 v6; // al
  unsigned __int16 v7; // ax
  unsigned int v8; // r8d
  __int64 result; // rax
  __int64 v10; // rax
  unsigned int v11; // eax

  v5 = *(char **)(a1 + 24);
  v6 = *v5;
  if ( (unsigned __int8)*v5 <= 0x1Cu )
  {
    v7 = *((_WORD *)v5 + 1);
    if ( v7 == 56 )
      return 0;
    if ( v7 <= 0x38u )
    {
      if ( v7 == 34 )
        return 1;
      LOBYTE(a5) = (unsigned __int16)(v7 - 53) <= 1u;
    }
    else
    {
      a5 = 0;
      if ( v7 == 57 )
      {
LABEL_6:
        LOBYTE(v8) = (unsigned int)sub_BD2910(a1) == 0;
        return v8;
      }
    }
    return a5;
  }
  switch ( v6 )
  {
    case '"':
    case 'T':
    case '`':
      return 0;
    case '?':
    case 'R':
    case 'S':
      return 1;
    case 'U':
      a5 = 0;
      if ( v6 != 85 )
        return a5;
      v10 = *((_QWORD *)v5 - 4);
      if ( !v10 )
        return a5;
      if ( *(_BYTE *)v10 )
        return a5;
      if ( *(_QWORD *)(v10 + 24) != *((_QWORD *)v5 + 10) )
        return a5;
      if ( (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
        return a5;
      v11 = *(_DWORD *)(v10 + 36);
      if ( v11 > 0x174 )
        return a5;
      if ( v11 > 0x136 )
      {
        switch ( v11 )
        {
          case 0x137u:
          case 0x138u:
          case 0x149u:
          case 0x14Au:
          case 0x14Du:
          case 0x151u:
          case 0x152u:
          case 0x153u:
          case 0x167u:
          case 0x168u:
          case 0x16Du:
          case 0x16Eu:
          case 0x171u:
          case 0x172u:
          case 0x173u:
          case 0x174u:
            return 1;
          default:
            return 0;
        }
      }
      if ( v11 > 0xF )
      {
        LOBYTE(a5) = v11 - 65 < 3;
        result = a5;
      }
      else
      {
        a5 = 1;
        if ( v11 > 0xD )
          return a5;
        LOBYTE(a5) = v11 == 1;
        result = a5;
      }
      break;
    case 'V':
      goto LABEL_6;
    default:
      a5 = 1;
      if ( (unsigned __int8)(v6 - 41) > 0x12u )
        LOBYTE(a5) = (unsigned int)v6 - 67 <= 0xC;
      return a5;
  }
  return result;
}
