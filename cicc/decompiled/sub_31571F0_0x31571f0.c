// Function: sub_31571F0
// Address: 0x31571f0
//
__int64 __fastcall sub_31571F0(__int64 *a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int v5; // r8d
  unsigned int i; // edx

  v3 = *a1;
  v4 = a3;
  v5 = *(_DWORD *)(*a1 + 4LL * a2);
  for ( i = *(_DWORD *)(*a1 + 4LL * a3); i != v5; v5 = *(_DWORD *)(*a1 + 4LL * v5) )
  {
    while ( v5 < i )
    {
      *(_DWORD *)(v3 + 4 * v4) = v5;
      v3 = *a1;
      v4 = i;
      i = *(_DWORD *)(*a1 + 4LL * i);
      if ( i == v5 )
        return v5;
    }
    *(_DWORD *)(v3 + 4LL * a2) = i;
    v3 = *a1;
    a2 = v5;
  }
  return v5;
}
