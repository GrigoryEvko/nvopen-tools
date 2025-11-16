// Function: sub_C2FE50
// Address: 0xc2fe50
//
__int64 __fastcall sub_C2FE50(char *a1, __int64 a2, char a3)
{
  _BOOL4 v3; // r14d
  char v4; // r12
  char v6; // al
  char *v7; // rsi
  char *i; // rdi
  char v10; // al

  v3 = 1;
  if ( !a2 )
    return v3;
  v4 = *a1;
  if ( *a1 == 32 )
  {
    if ( !a3 )
      goto LABEL_9;
  }
  else
  {
    if ( (unsigned __int8)(v4 - 9) > 4u )
    {
      v6 = a1[a2 - 1];
      if ( v6 != 32 )
        v3 = (unsigned __int8)(v6 - 9) <= 4u;
    }
    if ( !a3 )
      goto LABEL_7;
  }
  if ( a2 == 4 )
  {
    if ( *(_DWORD *)a1 == 1819047278 || *(_DWORD *)a1 == 1819047246 || *(_DWORD *)a1 == 1280070990 )
      v3 = 1;
    if ( *(_DWORD *)a1 != 1702195828 && *(_DWORD *)a1 != 1702195796 && *(_DWORD *)a1 != 1163219540 )
      goto LABEL_25;
    goto LABEL_24;
  }
  if ( a2 == 1 )
  {
    if ( *a1 == 126 )
      v3 = 1;
    goto LABEL_25;
  }
  if ( a2 == 5
    && (*(_DWORD *)a1 == 1936482662 && a1[4] == 101
     || *(_DWORD *)a1 == 1936482630 && a1[4] == 101
     || *(_DWORD *)a1 == 1397506374 && a1[4] == 69) )
  {
LABEL_24:
    v3 = 1;
  }
LABEL_25:
  v10 = sub_C2FB70(a1, a2);
  v4 = *a1;
  if ( v10 )
    v3 = 1;
LABEL_7:
  if ( strchr("-?:\\,[]{}#&*!|>'\"%@`", v4) )
    v3 = 1;
LABEL_9:
  v7 = &a1[a2];
  if ( a1 != &a1[a2] )
  {
    for ( i = a1 + 1; ; ++i )
    {
      if ( (unsigned __int8)((v4 & 0xDF) - 65) > 0x19u && (unsigned __int8)(v4 - 48) > 9u )
      {
        if ( v4 <= 46 )
        {
          if ( v4 > 8 )
          {
            switch ( v4 )
            {
              case 9:
              case 32:
              case 44:
              case 45:
              case 46:
                goto LABEL_29;
              case 10:
              case 13:
                return 2;
              default:
                break;
            }
          }
          if ( v4 <= 31 )
            return 2;
          goto LABEL_33;
        }
        if ( v4 > 95 )
        {
          if ( v4 == 127 )
            return 2;
LABEL_33:
          v3 = 1;
          goto LABEL_29;
        }
        if ( v4 <= 93 )
          v3 = 1;
      }
LABEL_29:
      if ( i == v7 )
        return v3;
      v4 = *i;
    }
  }
  return v3;
}
