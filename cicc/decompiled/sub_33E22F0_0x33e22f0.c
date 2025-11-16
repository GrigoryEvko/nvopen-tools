// Function: sub_33E22F0
// Address: 0x33e22f0
//
__int64 __fastcall sub_33E22F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  int v3; // edx

  v1 = *(_QWORD *)(a1 + 40);
  v2 = v1 + 40LL * *(unsigned int *)(a1 + 64);
  if ( v1 == v2 )
    return 1;
  while ( 1 )
  {
    v3 = *(_DWORD *)(*(_QWORD *)v1 + 24LL);
    if ( v3 != 51 && (unsigned int)(v3 - 11) > 1 )
      break;
    v1 += 40;
    if ( v2 == v1 )
      return 1;
  }
  return 0;
}
