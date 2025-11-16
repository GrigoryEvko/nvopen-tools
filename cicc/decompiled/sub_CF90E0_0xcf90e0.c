// Function: sub_CF90E0
// Address: 0xcf90e0
//
__int64 __fastcall sub_CF90E0(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // eax
  int v7; // edx
  int v8; // ecx
  __int64 v9; // rax
  unsigned int v10; // edi
  int v11; // r8d
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax

  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v6 = sub_A6E860(*(_QWORD *)a3 + 16LL, **(_QWORD **)a3);
  v7 = *(_DWORD *)(a3 + 12);
  *(_DWORD *)a1 = v6;
  v8 = v6;
  v9 = *(unsigned int *)(a3 + 8);
  v10 = v7 - v9;
  if ( (_DWORD)v9 != v7 )
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 32
                                          * ((unsigned int)v9 - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( v10 <= 1 )
    return a1;
  v11 = *(_DWORD *)(a2 + 4);
  v12 = *(_QWORD *)(32 * (v9 - (v11 & 0x7FFFFFF)) + a2 + 32);
  v13 = 1;
  if ( *(_BYTE *)v12 == 17 )
  {
    v13 = *(_QWORD *)(v12 + 24);
    if ( *(_DWORD *)(v12 + 32) > 0x40u )
      v13 = *(_QWORD *)v13;
  }
  *(_QWORD *)(a1 + 8) = v13;
  if ( v10 <= 2 || v8 != 86 )
    return a1;
  v15 = *(_QWORD *)(32 * (v9 - (v11 & 0x7FFFFFF)) + a2 + 64);
  v16 = 1;
  if ( *(_BYTE *)v15 == 17 )
  {
    v16 = *(_QWORD *)(v15 + 24);
    if ( *(_DWORD *)(v15 + 32) > 0x40u )
      v16 = *(_QWORD *)v16;
  }
  *(_QWORD *)(a1 + 8) = -(*(_QWORD *)(a1 + 8) | v16) & (*(_QWORD *)(a1 + 8) | v16);
  return a1;
}
