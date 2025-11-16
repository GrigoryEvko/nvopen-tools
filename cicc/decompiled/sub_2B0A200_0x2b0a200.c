// Function: sub_2B0A200
// Address: 0x2b0a200
//
char *__fastcall sub_2B0A200(char *a1, char *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  char *v4; // rax
  unsigned __int8 v5; // dl
  unsigned __int8 v6; // dl
  unsigned __int8 v7; // dl
  char *result; // rax
  unsigned __int8 v9; // dl
  unsigned __int8 v10; // al
  unsigned __int8 v11; // dl
  unsigned __int8 v12; // al
  unsigned __int8 v13; // dl
  unsigned __int8 v14; // dl

  v2 = (a2 - a1) >> 5;
  v3 = (a2 - a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_14:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return a2;
LABEL_28:
        v14 = **(_BYTE **)a1;
        result = a2;
        if ( v14 <= 0x15u && (unsigned __int8)(v14 - 12) >= 2u )
          return a1;
        return result;
      }
      v10 = **(_BYTE **)a1;
      if ( v10 <= 0x15u )
      {
        v11 = v10 - 12;
        result = a1;
        if ( v11 > 1u )
          return result;
      }
      a1 += 8;
    }
    v12 = **(_BYTE **)a1;
    if ( v12 <= 0x15u )
    {
      v13 = v12 - 12;
      result = a1;
      if ( v13 > 1u )
        return result;
    }
    a1 += 8;
    goto LABEL_28;
  }
  v4 = &a1[32 * v2];
  while ( 1 )
  {
    v5 = **(_BYTE **)a1;
    if ( v5 <= 0x15u && (unsigned __int8)(v5 - 12) > 1u )
      return a1;
    v6 = **((_BYTE **)a1 + 1);
    if ( v6 <= 0x15u && (unsigned __int8)(v6 - 12) > 1u )
      return a1 + 8;
    v7 = **((_BYTE **)a1 + 2);
    if ( v7 <= 0x15u && (unsigned __int8)(v7 - 12) > 1u )
      return a1 + 16;
    v9 = **((_BYTE **)a1 + 3);
    if ( v9 <= 0x15u && (unsigned __int8)(v9 - 12) > 1u )
      return a1 + 24;
    a1 += 32;
    if ( v4 == a1 )
    {
      v3 = (a2 - a1) >> 3;
      goto LABEL_14;
    }
  }
}
