// Function: sub_1E9BED0
// Address: 0x1e9bed0
//
__int64 __fastcall sub_1E9BED0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  int v6; // eax
  unsigned int *v7; // rcx
  unsigned int v8; // eax

  if ( *(_DWORD *)(a1 + 16) == 1 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 16) = 1;
  v4 = *(_QWORD *)(v3 + 32);
  if ( (*(_DWORD *)(v4 + 40) & 0xFFF00) != 0 )
    return 0;
  v5 = *(_QWORD *)(v4 + 104);
  v6 = *(_DWORD *)(v4 + 48);
  a2[1] = v5;
  *a2 = v6;
  v7 = *(unsigned int **)(*(_QWORD *)(a1 + 8) + 32LL);
  v8 = *v7;
  *a3 = v7[2];
  a3[1] = (v8 >> 8) & 0xFFF;
  return 1;
}
