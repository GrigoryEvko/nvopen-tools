// Function: sub_D322A0
// Address: 0xd322a0
//
_BYTE *__fastcall sub_D322A0(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _BYTE *v4; // rax
  _BYTE *result; // rax

  v2 = (a2 - (__int64)a1) >> 5;
  v3 = (a2 - (__int64)a1) >> 3;
  if ( v2 <= 0 )
  {
LABEL_11:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return (_BYTE *)a2;
LABEL_19:
        result = a1;
        if ( (*a1 & 4) == 0 )
          return (_BYTE *)a2;
        return result;
      }
      result = a1;
      if ( (*a1 & 4) != 0 )
        return result;
      a1 += 8;
    }
    result = a1;
    if ( (*a1 & 4) != 0 )
      return result;
    a1 += 8;
    goto LABEL_19;
  }
  v4 = &a1[32 * v2];
  while ( 1 )
  {
    if ( (*a1 & 4) != 0 )
      return a1;
    if ( (a1[8] & 4) != 0 )
      return a1 + 8;
    if ( (a1[16] & 4) != 0 )
      return a1 + 16;
    if ( (a1[24] & 4) != 0 )
      return a1 + 24;
    a1 += 32;
    if ( v4 == a1 )
    {
      v3 = (a2 - (__int64)a1) >> 3;
      goto LABEL_11;
    }
  }
}
