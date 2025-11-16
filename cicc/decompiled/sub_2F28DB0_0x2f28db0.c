// Function: sub_2F28DB0
// Address: 0x2f28db0
//
__int64 __fastcall sub_2F28DB0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax

  if ( *(_DWORD *)(a1 + 16) == 2 )
    return 0;
  v3 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 16) = 2;
  v4 = *(_QWORD *)(v3 + 32);
  LODWORD(v3) = *(_DWORD *)(v4 + 80);
  *a2 = *(_DWORD *)(v4 + 88);
  a2[1] = ((unsigned int)v3 >> 8) & 0xFFF;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  if ( (*(_DWORD *)v5 & 0xFFF00) != 0 )
    return 0;
  v6 = *(_QWORD *)(v5 + 144);
  v7 = *(_DWORD *)(v5 + 8);
  a3[1] = v6;
  *a3 = v7;
  return 1;
}
