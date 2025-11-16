// Function: sub_721AB0
// Address: 0x721ab0
//
__int64 __fastcall sub_721AB0(char *a1, _DWORD *a2, int a3)
{
  char v3; // al
  unsigned int v4; // r8d
  __int64 v6; // rax
  char v7; // dl
  unsigned int v8; // r8d

  if ( a2 )
    *a2 = 0;
  if ( a3 )
    return 1;
  v3 = *a1;
  if ( *a1 >= 0 )
    return 1;
  if ( (v3 & 0xE0) == 0xC0 )
  {
    v4 = 2;
    if ( (a1[1] & 0xC0) == 0x80 )
      return v4;
    goto LABEL_13;
  }
  if ( (v3 & 0xF0) != 0xE0 )
  {
    if ( (v3 & 0xF8) != 0xF0 )
    {
      if ( !a2 )
      {
LABEL_15:
        if ( (a1[1] & 0xC0) == 0x80 )
          goto LABEL_16;
        return 1;
      }
LABEL_14:
      *a2 = 1;
      goto LABEL_15;
    }
    if ( (a1[1] & 0xC0) == 0x80 )
    {
      if ( (a1[2] & 0xC0) == 0x80 )
      {
        v4 = 4;
        if ( (a1[3] & 0xC0) == 0x80 )
          return v4;
      }
      goto LABEL_21;
    }
LABEL_13:
    if ( !a2 )
      return 1;
    goto LABEL_14;
  }
  if ( (a1[1] & 0xC0) != 0x80 )
    goto LABEL_13;
  v4 = 3;
  if ( (a1[2] & 0xC0) == 0x80 )
    return v4;
LABEL_21:
  if ( a2 )
    goto LABEL_14;
LABEL_16:
  v6 = 2;
  do
  {
    v7 = a1[v6];
    v8 = v6++;
  }
  while ( (v7 & 0xC0) == 0x80 );
  return v8;
}
