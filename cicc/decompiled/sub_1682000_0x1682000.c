// Function: sub_1682000
// Address: 0x1682000
//
__int64 __fastcall sub_1682000(char *a1, _BYTE *a2, __int64 a3, __int64 a4, unsigned int a5)
{
  char v5; // al
  __int64 v7; // rdx
  __int64 v8; // r8
  unsigned int v10; // eax

  v5 = *a1;
  if ( *a1 )
  {
    while ( 1 )
    {
      v7 = (unsigned __int8)*a2;
      if ( (_BYTE)v7 != v5 || !(_BYTE)v7 )
        break;
      v5 = *++a1;
      ++a2;
      if ( !v5 )
        goto LABEL_8;
    }
    LODWORD(v8) = 0;
    if ( v5 == 42 )
    {
      v8 = 1;
      if ( a1[1] )
      {
        while ( 1 )
        {
          v10 = sub_1682000(a1 + 1, a2, v7, a4, v8);
          v8 = v10;
          if ( (_BYTE)v10 )
            break;
          if ( *a2 )
          {
            if ( *++a2 )
              continue;
          }
          return (unsigned int)v8;
        }
        LODWORD(v8) = 1;
      }
    }
    return (unsigned int)v8;
  }
  else
  {
LABEL_8:
    LOBYTE(a5) = *a2 == 0;
    return a5;
  }
}
