// Function: sub_15CC2D0
// Address: 0x15cc2d0
//
_QWORD *__fastcall sub_15CC2D0(__int64 a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rax
  _QWORD *v4; // rdx

  result = *(_QWORD **)(a1 + 8);
  if ( *(_QWORD **)(a1 + 16) == result )
  {
    v4 = &result[*(unsigned int *)(a1 + 28)];
    if ( v4 == result )
    {
      return v4;
    }
    else
    {
      while ( *result != a2 )
      {
        if ( v4 == ++result )
          return v4;
      }
    }
  }
  else
  {
    result = (_QWORD *)sub_16CC9F0(a1, a2);
    if ( *result != a2 )
    {
      v3 = *(_QWORD *)(a1 + 16);
      if ( v3 == *(_QWORD *)(a1 + 8) )
        return (_QWORD *)(v3 + 8LL * *(unsigned int *)(a1 + 28));
      else
        return (_QWORD *)(v3 + 8LL * *(unsigned int *)(a1 + 24));
    }
  }
  return result;
}
