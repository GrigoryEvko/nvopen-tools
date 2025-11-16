// Function: sub_D67DF0
// Address: 0xd67df0
//
__int64 __fastcall sub_D67DF0(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rax
  __int64 v3; // r8

  v1 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)a1 + 7) & 0x40) != 0 )
  {
    v2 = (__int64 *)*(a1 - 1);
    a1 = &v2[(unsigned __int64)v1 / 8];
  }
  else
  {
    v2 = &a1[v1 / 0xFFFFFFFFFFFFFFF8LL];
  }
  v3 = 0;
  if ( a1 == v2 )
    return v3;
  while ( 1 )
  {
    while ( !v3 )
    {
      v3 = *v2;
      v2 += 4;
      if ( a1 == v2 )
        return v3;
    }
    if ( *v2 != v3 )
      break;
    v2 += 4;
    if ( a1 == v2 )
      return v3;
  }
  return 0;
}
