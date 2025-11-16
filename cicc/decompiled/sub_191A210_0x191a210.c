// Function: sub_191A210
// Address: 0x191a210
//
__int64 __fastcall sub_191A210(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  _QWORD *v5; // rcx
  int v7; // eax
  int v8; // edx
  __int64 v9; // rsi
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *v16; // r9
  int v17; // edi

  if ( !a2 )
    return 0;
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) == 1 )
    return 0;
  v3 = *(_QWORD *)(a2 - 48);
  if ( *(_QWORD *)(a2 - 24) == v3 )
    return 0;
  v4 = *(_QWORD *)(a2 - 72);
  if ( *(_BYTE *)(v4 + 16) != 13 )
    return 0;
  v5 = *(_QWORD **)(v4 + 24);
  if ( *(_DWORD *)(v4 + 32) <= 0x40u )
  {
    v7 = *(_DWORD *)(a1 + 72);
    if ( !v5 )
      v3 = *(_QWORD *)(a2 - 24);
    if ( v7 )
      goto LABEL_9;
  }
  else
  {
    v7 = *(_DWORD *)(a1 + 72);
    if ( !*v5 )
      v3 = *(_QWORD *)(a2 - 24);
    if ( v7 )
    {
LABEL_9:
      v8 = v7 - 1;
      v9 = *(_QWORD *)(a1 + 56);
      v10 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v11 = *(_QWORD *)(v9 + 8LL * v10);
      if ( v3 == v11 )
        return 0;
      v17 = 1;
      while ( v11 != -8 )
      {
        v10 = v8 & (v17 + v10);
        v11 = *(_QWORD *)(v9 + 8LL * v10);
        if ( v3 == v11 )
          return 0;
        ++v17;
      }
    }
  }
  if ( !sub_157F0B0(v3) )
    v3 = sub_190B590(a1, *(_QWORD *)(a2 + 40), v3);
  sub_1919320(a1, v3, v13, v14, v15, v16);
  return 1;
}
