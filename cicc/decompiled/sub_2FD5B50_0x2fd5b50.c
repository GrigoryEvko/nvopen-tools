// Function: sub_2FD5B50
// Address: 0x2fd5b50
//
_BYTE *__fastcall sub_2FD5B50(_BYTE *a1, __int64 a2)
{
  signed __int64 v2; // rax
  _BYTE *v3; // rax
  _BYTE *result; // rax

  v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)a1) >> 3);
  if ( v2 >> 2 <= 0 )
  {
LABEL_11:
    if ( v2 != 2 )
    {
      if ( v2 != 3 )
      {
        if ( v2 != 1 )
          return (_BYTE *)a2;
LABEL_19:
        result = a1;
        if ( *a1 != 8 )
          return (_BYTE *)a2;
        return result;
      }
      result = a1;
      if ( *a1 == 8 )
        return result;
      a1 += 40;
    }
    result = a1;
    if ( *a1 == 8 )
      return result;
    a1 += 40;
    goto LABEL_19;
  }
  v3 = &a1[160 * (v2 >> 2)];
  while ( 1 )
  {
    if ( *a1 == 8 )
      return a1;
    if ( a1[40] == 8 )
      return a1 + 40;
    if ( a1[80] == 8 )
      return a1 + 80;
    if ( a1[120] == 8 )
      return a1 + 120;
    a1 += 160;
    if ( a1 == v3 )
    {
      v2 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)a1) >> 3);
      goto LABEL_11;
    }
  }
}
