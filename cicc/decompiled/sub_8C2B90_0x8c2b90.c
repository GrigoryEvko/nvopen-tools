// Function: sub_8C2B90
// Address: 0x8c2b90
//
_QWORD *__fastcall sub_8C2B90(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rdx
  _QWORD *i; // rcx
  _QWORD **v5; // rdx
  _QWORD *v6; // rdx
  _QWORD *v7; // rcx
  _QWORD *v8; // rdx

  result = *(_QWORD **)(a1 + 104);
  if ( result )
  {
    *(_QWORD *)(a1 + 104) = 0;
    v3 = *(_QWORD **)(a2 + 104);
    for ( i = 0; v3; v3 = *v5 )
    {
      i = v3;
      v5 = (_QWORD **)*v3;
      if ( !v5 )
        break;
      i = v5;
    }
    do
    {
      v6 = result;
      result = (_QWORD *)*result;
      if ( (*((_BYTE *)v6 + 11) & 4) != 0 )
      {
        if ( i )
          *i = v6;
        else
          *(_QWORD *)(a2 + 104) = v6;
        *v6 = 0;
        i = v6;
      }
    }
    while ( result );
  }
  v7 = *(_QWORD **)(a1 + 80);
  if ( v7 )
  {
    result = *(_QWORD **)(a2 + 80);
    if ( result )
    {
      while ( 1 )
      {
        v8 = result;
        result = (_QWORD *)*result;
        if ( !result )
          break;
        if ( v7 == v8 )
          goto LABEL_15;
      }
      if ( v7 != v8 )
      {
        *v8 = v7;
        *(_QWORD *)(a1 + 80) = 0;
        return result;
      }
    }
    else
    {
      *(_QWORD *)(a2 + 80) = v7;
    }
LABEL_15:
    *(_QWORD *)(a1 + 80) = 0;
  }
  return result;
}
