// Function: sub_168FB40
// Address: 0x168fb40
//
_QWORD *__fastcall sub_168FB40(__int64 a1)
{
  _QWORD *result; // rax
  _QWORD *v2; // r13
  __int64 v3; // r12
  _QWORD *v4; // rbx

  result = *(_QWORD **)(a1 + 8);
  v2 = &result[*(unsigned int *)(a1 + 16)];
  if ( result != v2 )
  {
    while ( 1 )
    {
      v3 = *result;
      v4 = result;
      if ( *result )
        break;
      if ( v2 == ++result )
        return result;
    }
    while ( v2 != result )
    {
      sub_16934D0(v3);
      j_j___libc_free_0(v3, 80);
      result = v4 + 1;
      if ( v2 == v4 + 1 )
        break;
      while ( 1 )
      {
        v3 = *result;
        v4 = result;
        if ( *result )
          break;
        if ( v2 == ++result )
          return result;
      }
    }
  }
  return result;
}
