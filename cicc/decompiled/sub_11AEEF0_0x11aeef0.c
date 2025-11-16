// Function: sub_11AEEF0
// Address: 0x11aeef0
//
_BYTE *__fastcall sub_11AEEF0(_BYTE *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _BYTE *v4; // rax
  _BYTE *result; // rax

  v2 = (a2 - (__int64)a1) >> 6;
  v3 = (a2 - (__int64)a1) >> 4;
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
        if ( a1[8] )
          return (_BYTE *)a2;
        return result;
      }
      result = a1;
      if ( !a1[8] )
        return result;
      a1 += 16;
    }
    result = a1;
    if ( !a1[8] )
      return result;
    a1 += 16;
    goto LABEL_19;
  }
  v4 = &a1[64 * v2];
  while ( 1 )
  {
    if ( !a1[8] )
      return a1;
    if ( !a1[24] )
      return a1 + 16;
    if ( !a1[40] )
      return a1 + 32;
    if ( !a1[56] )
      return a1 + 48;
    a1 += 64;
    if ( a1 == v4 )
    {
      v3 = (a2 - (__int64)a1) >> 4;
      goto LABEL_11;
    }
  }
}
