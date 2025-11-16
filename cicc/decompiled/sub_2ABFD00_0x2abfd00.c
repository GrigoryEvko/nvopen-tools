// Function: sub_2ABFD00
// Address: 0x2abfd00
//
__int64 __fastcall sub_2ABFD00(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 v3; // r9
  char v4; // di
  int v5; // r10d
  int v6; // edx
  unsigned int i; // eax
  __int64 v8; // r8
  unsigned int v9; // eax

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( !v2 )
    return 0;
  v4 = *(_BYTE *)(a2 + 4);
  v5 = 1;
  v6 = v2 - 1;
  for ( i = v6 & ((v4 == 0) + 37 * *(_DWORD *)a2 - 1); ; i = v6 & v9 )
  {
    v8 = v3 + 72LL * i;
    if ( *(_DWORD *)a2 == *(_DWORD *)v8 && v4 == *(_BYTE *)(v8 + 4) )
      break;
    if ( *(_DWORD *)v8 == -1 && *(_BYTE *)(v8 + 4) )
      return 0;
    v9 = v5 + i;
    ++v5;
  }
  return v3 + 72LL * i;
}
