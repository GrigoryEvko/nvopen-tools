// Function: sub_3176100
// Address: 0x3176100
//
__int64 *__fastcall sub_3176100(__int64 a1)
{
  __int64 *result; // rax
  __int64 v2; // rdx
  __int64 *v3; // r12
  __int64 v4; // rdi
  __int64 *v5; // rbx

  result = *(__int64 **)(a1 + 160);
  if ( *(_BYTE *)(a1 + 180) )
    v2 = *(unsigned int *)(a1 + 172);
  else
    v2 = *(unsigned int *)(a1 + 168);
  v3 = &result[v2];
  if ( result != v3 )
  {
    while ( 1 )
    {
      v4 = *result;
      v5 = result;
      if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v3 == ++result )
        return result;
    }
    while ( v3 != v5 )
    {
      sub_3174F80(v4);
      result = v5 + 1;
      if ( v5 + 1 == v3 )
        break;
      while ( 1 )
      {
        v4 = *result;
        v5 = result;
        if ( (unsigned __int64)*result < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v3 == ++result )
          return result;
      }
    }
  }
  return result;
}
