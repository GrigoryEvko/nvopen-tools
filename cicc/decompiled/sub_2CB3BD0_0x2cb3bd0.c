// Function: sub_2CB3BD0
// Address: 0x2cb3bd0
//
__int64 __fastcall sub_2CB3BD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r12
  __int64 v5; // rax
  __int64 *v6; // rbx

  v4 = (__int64 *)a2;
  if ( !(unsigned __int8)sub_B19DB0(a3, a1, a2) )
    return 0;
  v5 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(__int64 **)(a2 - 8);
    v4 = &v6[v5];
  }
  else
  {
    v6 = (__int64 *)(a2 - v5 * 8);
  }
  if ( v4 != v6 )
  {
    while ( (unsigned __int8)sub_B19DB0(a3, *v6, a1) )
    {
      v6 += 4;
      if ( v4 == v6 )
        return a1;
    }
    return 0;
  }
  return a1;
}
