// Function: sub_15C8A80
// Address: 0x15c8a80
//
__int64 __fastcall sub_15C8A80(char *a1, unsigned __int64 a2)
{
  int v3; // edi
  unsigned int v4; // r13d
  _BYTE *v5; // r8
  unsigned __int64 v6; // rdx
  char v7; // al
  char *i; // rdx
  char v9; // al
  _BYTE *v11; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v12; // [rsp+18h] [rbp-38h]

  v11 = a1;
  v12 = a2;
  if ( !a2 )
    return 1;
  v3 = *a1;
  if ( isspace(v3) )
    return 1;
  v4 = isspace(a1[a2 - 1]);
  if ( v4 )
    return 1;
  if ( a2 == 4 )
  {
    if ( *(_DWORD *)a1 != 1819047278
      && *(_DWORD *)a1 != 1819047246
      && *(_DWORD *)a1 != 1280070990
      && *(_DWORD *)a1 != 1702195828
      && *(_DWORD *)a1 != 1702195796
      && *(_DWORD *)a1 != 1163219540 )
    {
      goto LABEL_7;
    }
    return 1;
  }
  if ( a2 == 1 )
  {
    if ( (_BYTE)v3 != 126 )
      goto LABEL_7;
    return 1;
  }
  if ( a2 == 5
    && (*(_DWORD *)a1 == 1936482662 && a1[4] == 101
     || *(_DWORD *)a1 == 1936482630 && a1[4] == 101
     || *(_DWORD *)a1 == 1397506374 && a1[4] == 69) )
  {
    return 1;
  }
LABEL_7:
  v5 = v11;
  v6 = v12;
  if ( (((_BYTE)v3 - 43) & 0xFD) == 0 )
  {
    v7 = sub_15C88E0(a1 + 1, a2 - 1);
    v5 = v11;
    v6 = v12;
    if ( v7 )
      return 1;
  }
  if ( (unsigned __int8)sub_15C88E0(v5, v6)
    || a2 == 4 && (*(_DWORD *)a1 == 1851878958 || *(_DWORD *)a1 == 1314999854 || *(_DWORD *)a1 == 1312902702)
    || !sub_16D23E0(&v11, "-?:\\,[]{}#&*!|>'\"%@`", 20, 0) )
  {
    return 1;
  }
  for ( i = v11; &v11[v12] != i; ++i )
  {
    v9 = *i;
    if ( (unsigned __int8)((*i & 0xDF) - 65) > 0x19u && (unsigned __int8)(v9 - 48) > 9u )
    {
      if ( v9 > 47 )
      {
        if ( v9 <= 95 )
        {
          if ( v9 <= 93 )
            v4 = 1;
          continue;
        }
        if ( v9 == 127 )
          return 2;
      }
      else
      {
        if ( v9 > 8 )
        {
          switch ( v9 )
          {
            case 9:
            case 32:
            case 44:
            case 45:
            case 46:
            case 47:
              continue;
            case 10:
            case 13:
              goto LABEL_37;
            default:
              break;
          }
        }
        if ( v9 <= 31 )
          return 2;
      }
LABEL_37:
      v4 = 1;
    }
  }
  return v4;
}
