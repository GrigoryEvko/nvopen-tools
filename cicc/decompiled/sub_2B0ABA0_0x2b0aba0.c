// Function: sub_2B0ABA0
// Address: 0x2b0aba0
//
char *__fastcall sub_2B0ABA0(char *a1, char *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  char *v4; // rax
  char *result; // rax

  v2 = (a2 - a1) >> 5;
  v3 = (a2 - a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_9:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return a2;
LABEL_22:
        result = a2;
        if ( **(_BYTE **)a1 == 12 )
          return a1;
        return result;
      }
      result = a1;
      if ( **(_BYTE **)a1 == 12 )
        return result;
      a1 += 8;
    }
    result = a1;
    if ( **(_BYTE **)a1 == 12 )
      return result;
    a1 += 8;
    goto LABEL_22;
  }
  v4 = &a1[32 * v2];
  while ( 1 )
  {
    if ( **(_BYTE **)a1 == 12 )
      return a1;
    if ( **((_BYTE **)a1 + 1) == 12 )
      return a1 + 8;
    if ( **((_BYTE **)a1 + 2) == 12 )
      return a1 + 16;
    if ( **((_BYTE **)a1 + 3) == 12 )
      return a1 + 24;
    a1 += 32;
    if ( v4 == a1 )
    {
      v3 = (a2 - a1) >> 3;
      goto LABEL_9;
    }
  }
}
