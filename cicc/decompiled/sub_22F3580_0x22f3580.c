// Function: sub_22F3580
// Address: 0x22f3580
//
unsigned __int64 *__fastcall sub_22F3580(__int64 a1)
{
  unsigned __int64 *result; // rax
  unsigned __int64 *v2; // r13
  unsigned __int64 v3; // r12
  unsigned __int64 *v4; // rbx

  result = *(unsigned __int64 **)(a1 + 8);
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
      sub_314D410(v3);
      j_j___libc_free_0(v3);
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
