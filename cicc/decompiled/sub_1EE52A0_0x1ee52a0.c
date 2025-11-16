// Function: sub_1EE52A0
// Address: 0x1ee52a0
//
_DWORD *__fastcall sub_1EE52A0(_DWORD *a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  _DWORD *v5; // rax
  _DWORD *result; // rax

  v3 = (a2 - (__int64)a1) >> 5;
  v4 = (a2 - (__int64)a1) >> 3;
  if ( v3 <= 0 )
  {
LABEL_11:
    if ( v4 != 2 )
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 )
          return (_DWORD *)a2;
LABEL_19:
        result = a1;
        if ( a3 != *a1 )
          return (_DWORD *)a2;
        return result;
      }
      result = a1;
      if ( a3 == *a1 )
        return result;
      a1 += 2;
    }
    result = a1;
    if ( a3 == *a1 )
      return result;
    a1 += 2;
    goto LABEL_19;
  }
  v5 = &a1[8 * v3];
  while ( 1 )
  {
    if ( a3 == *a1 )
      return a1;
    if ( a3 == a1[2] )
      return a1 + 2;
    if ( a3 == a1[4] )
      return a1 + 4;
    if ( a3 == a1[6] )
      return a1 + 6;
    a1 += 8;
    if ( a1 == v5 )
    {
      v4 = (a2 - (__int64)a1) >> 3;
      goto LABEL_11;
    }
  }
}
