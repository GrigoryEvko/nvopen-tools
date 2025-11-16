// Function: sub_279EB90
// Address: 0x279eb90
//
__int64 __fastcall sub_279EB90(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v6; // rax
  bool v7; // zf
  int v8; // eax
  __int64 v9; // rcx
  int v10; // edx
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  int v17; // edi

  if ( !a2 )
    return 0;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 1 )
    return 0;
  v3 = *(_QWORD *)(a2 - 64);
  if ( *(_QWORD *)(a2 - 32) == v3 )
    return 0;
  v4 = *(_QWORD *)(a2 - 96);
  if ( *(_BYTE *)v4 != 17 )
    return 0;
  if ( *(_DWORD *)(v4 + 32) <= 0x40u )
    v6 = *(_QWORD *)(v4 + 24);
  else
    v6 = **(_QWORD **)(v4 + 24);
  v7 = v6 == 0;
  v8 = *(_DWORD *)(a1 + 72);
  v9 = *(_QWORD *)(a1 + 56);
  if ( v7 )
    v3 = *(_QWORD *)(a2 - 32);
  if ( v8 )
  {
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v12 = *(_QWORD *)(v9 + 8LL * v11);
    if ( v3 == v12 )
      return 0;
    v17 = 1;
    while ( v12 != -4096 )
    {
      v11 = v10 & (v17 + v11);
      v12 = *(_QWORD *)(v9 + 8LL * v11);
      if ( v3 == v12 )
        return 0;
      ++v17;
    }
  }
  if ( !sub_AA54C0(v3) )
    v3 = sub_278C0E0(a1, *(_QWORD *)(a2 + 40), v3);
  sub_279D530(a1, v3, v14, v15, v16);
  return 1;
}
