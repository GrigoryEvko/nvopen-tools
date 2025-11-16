// Function: sub_2A5D2A0
// Address: 0x2a5d2a0
//
_QWORD *__fastcall sub_2A5D2A0(unsigned __int128 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rcx
  _QWORD *v3; // r8
  unsigned __int64 v4; // rdx
  _QWORD *v5; // rdx
  _QWORD *i; // rcx

  result = *(_QWORD **)(a1 + 16);
  v2 = *((_QWORD *)&a1 + 1);
  v3 = (_QWORD *)(a1 + 8);
  if ( !result )
    return v3;
  *((_QWORD *)&a1 + 1) = **((_QWORD **)&a1 + 1);
  while ( 1 )
  {
    v4 = result[4];
    if ( *((_QWORD *)&a1 + 1) <= v4 && (*((_QWORD *)&a1 + 1) != v4 || result[5] >= *(_QWORD *)(v2 + 8)) )
      break;
    result = (_QWORD *)result[3];
LABEL_19:
    if ( !result )
      return v3;
  }
  if ( *((_QWORD *)&a1 + 1) < v4 || (*(_QWORD *)&a1 = *(_QWORD *)(v2 + 8), (unsigned __int64)a1 < result[5]) )
  {
    v3 = result;
    result = (_QWORD *)result[2];
    goto LABEL_19;
  }
  v5 = (_QWORD *)result[3];
  for ( i = (_QWORD *)result[2]; v5; v5 = (_QWORD *)v5[2] )
  {
    while ( a1 >= __PAIR128__(v5[4], v5[5]) )
    {
      v5 = (_QWORD *)v5[3];
      if ( !v5 )
        goto LABEL_14;
    }
  }
LABEL_14:
  while ( i )
  {
    while ( a1 <= __PAIR128__(i[4], i[5]) )
    {
      result = i;
      i = (_QWORD *)i[2];
      if ( !i )
        return result;
    }
    i = (_QWORD *)i[3];
  }
  return result;
}
