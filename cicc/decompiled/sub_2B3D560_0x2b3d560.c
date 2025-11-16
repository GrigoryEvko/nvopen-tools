// Function: sub_2B3D560
// Address: 0x2b3d560
//
__int64 *__fastcall sub_2B3D560(__int64 a1, __int64 *a2)
{
  __int64 v2; // r9
  int v3; // edx
  __int64 v4; // rcx
  int v5; // edi
  unsigned int v6; // eax
  __int64 *v7; // r8
  __int64 v8; // rsi
  int v10; // edx

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = a1 + 16;
    v3 = 3;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v2 = *(_QWORD *)(a1 + 16);
    if ( !v10 )
      return 0;
    v3 = v10 - 1;
  }
  v4 = *a2;
  v5 = 1;
  v6 = v3 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v7 = (__int64 *)(v2 + 72LL * v6);
  v8 = *v7;
  if ( *v7 == v4 )
    return v7;
  while ( v8 != -4096 )
  {
    v6 = v3 & (v5 + v6);
    v7 = (__int64 *)(v2 + 72LL * v6);
    v8 = *v7;
    if ( *v7 == v4 )
      return v7;
    ++v5;
  }
  return 0;
}
