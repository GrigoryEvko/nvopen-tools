// Function: sub_2538140
// Address: 0x2538140
//
_QWORD *__fastcall sub_2538140(_QWORD *a1, __int64 a2, __int64 a3)
{
  signed __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  _QWORD *result; // rax
  __int64 v9; // rax
  bool v10; // zf

  v4 = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3);
  v5 = v4 >> 2;
  if ( v4 >> 2 > 0 )
  {
    v6 = *(_QWORD *)(a3 + 16);
    v7 = &a1[12 * v5];
    while ( a1[2] != v6 )
    {
      if ( v6 == a1[5] )
        return a1 + 3;
      if ( v6 == a1[8] )
        return a1 + 6;
      if ( v6 == a1[11] )
        return a1 + 9;
      a1 += 12;
      if ( a1 == v7 )
      {
        v4 = 0xAAAAAAAAAAAAAAABLL * ((a2 - (__int64)a1) >> 3);
        goto LABEL_10;
      }
    }
    return a1;
  }
LABEL_10:
  if ( v4 == 2 )
  {
    v9 = *(_QWORD *)(a3 + 16);
    goto LABEL_21;
  }
  if ( v4 == 3 )
  {
    v9 = *(_QWORD *)(a3 + 16);
    if ( a1[2] == v9 )
      return a1;
    a1 += 3;
LABEL_21:
    if ( a1[2] != v9 )
    {
      a1 += 3;
      goto LABEL_17;
    }
    return a1;
  }
  if ( v4 != 1 )
    return (_QWORD *)a2;
  v9 = *(_QWORD *)(a3 + 16);
LABEL_17:
  v10 = a1[2] == v9;
  result = a1;
  if ( !v10 )
    return (_QWORD *)a2;
  return result;
}
