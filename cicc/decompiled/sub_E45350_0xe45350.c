// Function: sub_E45350
// Address: 0xe45350
//
__int64 __fastcall sub_E45350(__int64 a1)
{
  unsigned __int8 **v1; // rax
  unsigned __int8 *v2; // r8
  __int64 v3; // rsi
  unsigned __int8 *v4; // rdx

  if ( *(_BYTE *)a1 != 84 )
    return 0;
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 0 )
    return 1;
  v1 = *(unsigned __int8 ***)(a1 - 8);
  v2 = 0;
  v3 = (__int64)&v1[4 * (*(_DWORD *)(a1 + 4) & 0x7FFFFFFu)];
  while ( 1 )
  {
    while ( 1 )
    {
      v4 = *v1;
      if ( !*v1 )
        BUG();
      if ( (unsigned __int8 *)a1 != v4 && (unsigned int)*v4 - 12 > 1 )
        break;
      v1 += 4;
      if ( (unsigned __int8 **)v3 == v1 )
        return 1;
    }
    if ( v4 != v2 && v2 )
      break;
    v1 += 4;
    v2 = v4;
    if ( (unsigned __int8 **)v3 == v1 )
      return 1;
  }
  return 0;
}
