// Function: sub_D57AF0
// Address: 0xd57af0
//
_QWORD *__fastcall sub_D57AF0(__int64 a1, _QWORD *a2, __int64 *a3)
{
  _QWORD *v4; // r9
  _QWORD *v5; // rdi
  unsigned __int64 v7; // r10
  __int64 v8; // rsi
  _QWORD *result; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rdx
  _QWORD **i; // rsi
  __int64 v13; // rcx

  v4 = (_QWORD *)*a2;
  v5 = (_QWORD *)a2[1];
  v7 = a2[3];
  v8 = *(_QWORD *)(a1 + 24);
  result = *(_QWORD **)a1;
  v10 = *(_QWORD **)(a1 + 16);
  v11 = *a3;
  if ( v7 == v8 )
  {
    while ( v4 != result )
      *result++ = v11;
  }
  else
  {
    if ( result != v10 )
    {
      do
        *result++ = v11;
      while ( v10 != result );
      v11 = *a3;
    }
    for ( i = (_QWORD **)(v8 + 8); v7 > (unsigned __int64)i; v11 = *a3 )
    {
      result = *i;
      v13 = (__int64)(*i + 64);
      do
        *result++ = v11;
      while ( (_QWORD *)v13 != result );
      ++i;
    }
    for ( ; v4 != v5; ++v5 )
      *v5 = v11;
  }
  return result;
}
