// Function: sub_2EBDFC0
// Address: 0x2ebdfc0
//
_QWORD *__fastcall sub_2EBDFC0(__int64 a1, unsigned int a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r13
  __int64 v5; // rdi
  _QWORD *v6; // rbx

  result = *(_QWORD **)(a1 + 16);
  if ( *(_BYTE *)(a1 + 36) )
    v3 = *(unsigned int *)(a1 + 28);
  else
    v3 = *(unsigned int *)(a1 + 24);
  v4 = &result[v3];
  if ( result != v4 )
  {
    while ( 1 )
    {
      v5 = *result;
      v6 = result;
      if ( *result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v4 == ++result )
        return result;
    }
    while ( v4 != v6 )
    {
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 24LL))(v5, a2);
      result = v6 + 1;
      if ( v6 + 1 == v4 )
        break;
      while ( 1 )
      {
        v5 = *result;
        v6 = result;
        if ( *result < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v4 == ++result )
          return result;
      }
    }
  }
  return result;
}
