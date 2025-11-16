// Function: sub_2939FC0
// Address: 0x2939fc0
//
__int64 __fastcall sub_2939FC0(__int64 a1)
{
  int v1; // eax
  unsigned int v2; // r8d
  __int64 v3; // rcx
  int v4; // esi
  __int64 v5; // rdx
  __int64 v6; // rcx

  v1 = *(_DWORD *)(a1 + 12);
  v2 = 0;
  if ( !v1 )
    return v2;
  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) != 17 )
    return v2;
  v4 = *(_DWORD *)(*(_QWORD *)v3 + 32LL);
  if ( v1 == 1 )
    return 1;
  v5 = v3 + 8;
  v6 = v3 + 8LL * (unsigned int)(v1 - 2) + 16;
  while ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 17 && v4 == *(_DWORD *)(*(_QWORD *)v5 + 32LL) )
  {
    v5 += 8;
    if ( v5 == v6 )
      return 1;
  }
  return 0;
}
