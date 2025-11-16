// Function: sub_8EE4D0
// Address: 0x8ee4d0
//
__int64 __fastcall sub_8EE4D0(_BYTE *a1, int a2)
{
  int v2; // edx
  _BYTE *v3; // rdx
  int v4; // ecx
  __int64 result; // rax
  int v6; // r8d
  _BYTE *v7; // rdx

  v2 = a2 + 14;
  if ( a2 + 7 >= 0 )
    v2 = a2 + 7;
  v3 = &a1[v2 >> 3];
  if ( (a2 & 7) != 0 )
  {
    v6 = (unsigned __int8)*(v3 - 1);
    result = (unsigned int)(a2 % 8);
    v4 = v6 & ~(-1 << result);
    if ( a1 == v3 )
      goto LABEL_16;
  }
  else
  {
    if ( a1 == v3 )
      return 0;
    LOBYTE(v4) = *(v3 - 1);
    result = 8;
    LOBYTE(v6) = v4;
  }
  v7 = v3 - 1;
  if ( !(_BYTE)v6 )
  {
    while ( 1 )
    {
      result = (unsigned int)(result + 8);
      if ( a1 == v7 )
        break;
      LOBYTE(v4) = *--v7;
      if ( (_BYTE)v4 )
        goto LABEL_9;
    }
LABEL_16:
    if ( !*a1 )
      return (unsigned int)(result - 8);
  }
LABEL_9:
  if ( (v4 & 0xF0) != 0 )
  {
    LOBYTE(v4) = (unsigned __int8)v4 >> 4;
    result = (unsigned int)(result - 4);
  }
  if ( (v4 & 0xC) != 0 )
  {
    LODWORD(result) = result - 2;
    if ( (((unsigned __int8)v4 >> 2) & 2) == 0 )
      return (unsigned int)(result - 1);
LABEL_13:
    LODWORD(result) = result - 1;
    return (unsigned int)(result - 1);
  }
  if ( (v4 & 2) != 0 )
    goto LABEL_13;
  if ( (_BYTE)v4 )
    return (unsigned int)(result - 1);
  return result;
}
