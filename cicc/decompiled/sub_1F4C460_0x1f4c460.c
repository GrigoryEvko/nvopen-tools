// Function: sub_1F4C460
// Address: 0x1f4c460
//
__int64 __fastcall sub_1F4C460(__int64 a1, unsigned int a2, _DWORD *a3, __int64 a4, __int64 a5, _BYTE *a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  unsigned int v10; // esi

  v7 = *(unsigned int *)(a1 + 40);
  if ( !(_DWORD)v7 )
    return 0;
  v8 = a2;
  v9 = *(_QWORD *)(a1 + 32);
  v10 = 0;
  while ( *(_BYTE *)v9
       || (*(_BYTE *)(v9 + 3) & 0x10) != 0
       || (_DWORD)v8 != *(_DWORD *)(v9 + 8)
       || (*(_WORD *)(v9 + 2) & 0xFF0) == 0 )
  {
    ++v10;
    v9 += 40;
    if ( (_DWORD)v7 == v10 )
      return 0;
  }
  *a3 = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 40LL * (unsigned int)sub_1E16AB0(a1, v10, v7, v8, a5, a6) + 8);
  return 1;
}
