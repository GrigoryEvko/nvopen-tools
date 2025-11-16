// Function: sub_33E2340
// Address: 0x33e2340
//
__int64 __fastcall sub_33E2340(__int64 a1, int a2)
{
  __int64 v2; // rax
  int v3; // ecx
  unsigned int v4; // edx
  int v5; // eax

  if ( a2 )
  {
    v2 = 0;
    while ( 1 )
    {
      v3 = *(_DWORD *)(a1 + 4 * v2);
      v4 = v2;
      if ( v3 >= 0 )
        break;
      if ( ++v2 == a2 )
        return 1;
    }
    while ( a2 != ++v4 )
    {
      v5 = *(_DWORD *)(a1 + 4LL * v4);
      if ( v3 != v5 && v5 >= 0 )
        return 0;
    }
  }
  return 1;
}
