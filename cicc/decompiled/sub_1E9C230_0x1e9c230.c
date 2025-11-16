// Function: sub_1E9C230
// Address: 0x1e9c230
//
__int64 __fastcall sub_1E9C230(__int64 a1, _QWORD *a2, _DWORD *a3)
{
  int v5; // esi
  unsigned int v6; // edx
  __int64 v7; // r8
  unsigned __int8 v8; // cl
  unsigned int *v9; // rdx
  unsigned int v10; // eax

  v5 = *(_DWORD *)(a1 + 20);
  v6 = *(_DWORD *)(a1 + 16);
  if ( v6 == v5 )
    return 0;
  v7 = *(_QWORD *)(a1 + 8);
  while ( 1 )
  {
    v8 = *(_BYTE *)(*(_QWORD *)(v7 + 32) + 40LL * v6 + 3);
    if ( (((v8 & 0x40) != 0) & (v8 >> 4)) == 0 )
      break;
    *(_DWORD *)(a1 + 16) = ++v6;
    if ( v5 == v6 )
      return 0;
  }
  *a2 = 0;
  v9 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL * *(unsigned int *)(a1 + 16));
  v10 = *v9;
  *a3 = v9[2];
  a3[1] = (v10 >> 8) & 0xFFF;
  ++*(_DWORD *)(a1 + 16);
  return 1;
}
