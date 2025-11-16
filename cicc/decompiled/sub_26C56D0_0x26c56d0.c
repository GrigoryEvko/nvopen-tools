// Function: sub_26C56D0
// Address: 0x26c56d0
//
_QWORD *__fastcall sub_26C56D0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // r8
  unsigned __int64 v4; // rdi
  _QWORD *v5; // r9
  unsigned __int64 v6; // r10
  _QWORD *v7; // rax
  _QWORD *v8; // rsi
  _QWORD *result; // rax

  v2 = *a2;
  v4 = a1[1];
  v5 = *(_QWORD **)(*a1 + 8 * (*a2 % v4));
  v6 = *a2 % v4;
  if ( v5 )
  {
    v7 = (_QWORD *)*v5;
    if ( v2 == *(_QWORD *)(*v5 + 8LL) )
    {
LABEL_6:
      result = (_QWORD *)*v5;
      if ( *v5 )
        return result;
    }
    else
    {
      while ( 1 )
      {
        v8 = (_QWORD *)*v7;
        if ( !*v7 )
          break;
        v5 = v7;
        if ( v6 != v8[1] % v4 )
          break;
        v7 = (_QWORD *)*v7;
        if ( v2 == v8[1] )
          goto LABEL_6;
      }
    }
  }
  return 0;
}
