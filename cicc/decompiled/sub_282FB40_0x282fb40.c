// Function: sub_282FB40
// Address: 0x282fb40
//
_QWORD *__fastcall sub_282FB40(_QWORD *a1, __int64 a2, __int64 a3)
{
  signed __int64 v3; // rcx
  _QWORD *v4; // rcx
  _QWORD *result; // rax

  v3 = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3);
  if ( v3 >> 2 > 0 )
  {
    v4 = &a1[12 * (v3 >> 2)];
    while ( a3 != *a1 )
    {
      if ( a3 == a1[3] )
        return a1 + 3;
      if ( a3 == a1[6] )
        return a1 + 6;
      if ( a3 == a1[9] )
        return a1 + 9;
      a1 += 12;
      if ( a1 == v4 )
      {
        v3 = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3);
        goto LABEL_10;
      }
    }
    return a1;
  }
LABEL_10:
  if ( v3 == 2 )
    goto LABEL_16;
  if ( v3 == 3 )
  {
    if ( a3 == *a1 )
      return a1;
    a1 += 3;
LABEL_16:
    if ( a3 != *a1 )
    {
      a1 += 3;
      goto LABEL_18;
    }
    return a1;
  }
  if ( v3 != 1 )
    return (_QWORD *)a2;
LABEL_18:
  result = a1;
  if ( a3 != *a1 )
    return (_QWORD *)a2;
  return result;
}
