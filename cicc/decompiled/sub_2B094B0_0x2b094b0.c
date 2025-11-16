// Function: sub_2B094B0
// Address: 0x2b094b0
//
_DWORD *__fastcall sub_2B094B0(_DWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _DWORD *v4; // rax
  _DWORD *result; // rax

  v2 = (a2 - (__int64)a1) >> 4;
  v3 = (a2 - (__int64)a1) >> 2;
  if ( v2 <= 0 )
  {
LABEL_11:
    if ( v3 != 2 )
    {
      if ( v3 != 3 )
      {
        if ( v3 != 1 )
          return (_DWORD *)a2;
LABEL_19:
        result = a1;
        if ( *a1 == -1 )
          return (_DWORD *)a2;
        return result;
      }
      result = a1;
      if ( *a1 != -1 )
        return result;
      ++a1;
    }
    result = a1;
    if ( *a1 != -1 )
      return result;
    ++a1;
    goto LABEL_19;
  }
  v4 = &a1[4 * v2];
  while ( 1 )
  {
    if ( *a1 != -1 )
      return a1;
    if ( a1[1] != -1 )
      return a1 + 1;
    if ( a1[2] != -1 )
      return a1 + 2;
    if ( a1[3] != -1 )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v4 )
    {
      v3 = (a2 - (__int64)a1) >> 2;
      goto LABEL_11;
    }
  }
}
