// Function: sub_14530E0
// Address: 0x14530e0
//
_QWORD *__fastcall sub_14530E0(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rdx
  _QWORD *result; // rax
  __int64 v9; // rdx

  v4 = (a2 - (__int64)a1) >> 6;
  v5 = (a2 - (__int64)a1) >> 4;
  if ( v4 > 0 )
  {
    v6 = *a3;
    v7 = &a1[8 * v4];
    while ( *a1 != v6 || a1[1] != a3[1] )
    {
      if ( v6 == a1[2] && a1[3] == a3[1] )
        return a1 + 2;
      if ( v6 == a1[4] && a1[5] == a3[1] )
        return a1 + 4;
      if ( v6 == a1[6] && a1[7] == a3[1] )
        return a1 + 6;
      a1 += 8;
      if ( v7 == a1 )
      {
        v5 = (a2 - (__int64)a1) >> 4;
        goto LABEL_9;
      }
    }
    return a1;
  }
LABEL_9:
  if ( v5 == 2 )
  {
    v9 = *a3;
    goto LABEL_27;
  }
  if ( v5 != 3 )
  {
    if ( v5 != 1 )
      return (_QWORD *)a2;
    v9 = *a3;
LABEL_22:
    if ( *a1 == v9 && a1[1] == a3[1] )
      return a1;
    return (_QWORD *)a2;
  }
  v9 = *a3;
  if ( *a1 != *a3 || (result = a1, a1[1] != a3[1]) )
  {
    a1 += 2;
LABEL_27:
    if ( *a1 != v9 || (result = a1, a1[1] != a3[1]) )
    {
      a1 += 2;
      goto LABEL_22;
    }
  }
  return result;
}
