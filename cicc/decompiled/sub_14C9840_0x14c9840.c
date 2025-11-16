// Function: sub_14C9840
// Address: 0x14c9840
//
__int64 __fastcall sub_14C9840(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  _QWORD *v4; // rcx
  unsigned int v5; // edi
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v9; // rsi

  v3 = sub_14C8690(*(_QWORD *)a2 + 16LL, **(_QWORD **)a2);
  v4 = (_QWORD *)*(unsigned int *)(a2 + 8);
  v5 = *(_DWORD *)(a2 + 12) - (_DWORD)v4;
  v6 = 0;
  if ( v5 > 1 )
  {
    v7 = *(_QWORD *)(a1 + 8 * (3LL * ((_QWORD)v4 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) + 3));
    v6 = 1;
    if ( *(_BYTE *)(v7 + 16) == 13 )
    {
      v6 = *(_QWORD *)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) > 0x40u )
        v6 = *(_QWORD *)v6;
    }
    if ( v5 > 2 && v3 == 1 )
    {
      v9 = *(_QWORD *)(a1 + 8 * (3LL * ((_QWORD)v4 - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) + 6));
      LODWORD(v4) = 1;
      if ( *(_BYTE *)(v9 + 16) == 13 )
      {
        v4 = *(_QWORD **)(v9 + 24);
        if ( *(_DWORD *)(v9 + 32) > 0x40u )
          v4 = (_QWORD *)*v4;
      }
      v6 = -((unsigned int)v4 | (unsigned int)v6) & ((unsigned int)v4 | (unsigned int)v6);
    }
  }
  return (v6 << 32) | v3;
}
