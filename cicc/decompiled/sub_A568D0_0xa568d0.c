// Function: sub_A568D0
// Address: 0xa568d0
//
_QWORD *__fastcall sub_A568D0(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // r8
  unsigned __int64 v4; // rcx
  _QWORD *v5; // rdx
  _QWORD *v6; // rsi
  __int64 v7; // r9
  __int64 v8; // rdi
  _QWORD *v9; // rdi
  _QWORD *i; // rsi

  result = *(_QWORD **)(a1 + 16);
  v3 = (_QWORD *)(a1 + 8);
  if ( !result )
    return v3;
  v4 = *a2;
  while ( 1 )
  {
    while ( v4 > result[4] )
    {
      result = (_QWORD *)result[3];
      if ( !result )
        return v3;
    }
    v5 = (_QWORD *)result[2];
    if ( v4 >= result[4] )
      break;
    v3 = result;
    result = (_QWORD *)result[2];
    if ( !v5 )
      return v3;
  }
  v6 = (_QWORD *)result[3];
  if ( v6 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v6[2];
        v8 = v6[3];
        if ( v4 < v6[4] )
          break;
        v6 = (_QWORD *)v6[3];
        if ( !v8 )
          goto LABEL_14;
      }
      v6 = (_QWORD *)v6[2];
    }
    while ( v7 );
  }
LABEL_14:
  while ( v5 )
  {
    v9 = (_QWORD *)v5[2];
    for ( i = (_QWORD *)v5[3]; v4 > v5[4]; i = (_QWORD *)i[3] )
    {
      v5 = i;
      if ( !i )
        return result;
      v9 = (_QWORD *)i[2];
    }
    result = v5;
    v5 = v9;
  }
  return result;
}
