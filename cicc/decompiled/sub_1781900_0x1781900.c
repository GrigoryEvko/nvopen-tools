// Function: sub_1781900
// Address: 0x1781900
//
__int64 __fastcall sub_1781900(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  char v5; // al
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx

  v2 = *(_QWORD *)(a2 + 8);
  v3 = 0;
  if ( v2 && !*(_QWORD *)(v2 + 8) )
  {
    v5 = *(_BYTE *)(a2 + 16);
    if ( v5 == 40 )
    {
      v8 = *(_QWORD *)(a2 - 48);
      v9 = *(_QWORD *)(a2 - 24);
      if ( v8 == *(_QWORD *)a1 && v9 )
      {
        v3 = 1;
        **(_QWORD **)(a1 + 8) = v9;
        return v3;
      }
      if ( *(_QWORD *)a1 == v9 && v8 )
      {
        v3 = 1;
        **(_QWORD **)(a1 + 8) = v8;
        return v3;
      }
    }
    else
    {
      if ( v5 != 5 || *(_WORD *)(a2 + 18) != 16 )
        return v3;
      v6 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
      v7 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
      if ( v6 == *(_QWORD *)a1 && v7 )
      {
        v3 = 1;
        **(_QWORD **)(a1 + 8) = v7;
        return v3;
      }
      if ( *(_QWORD *)a1 == v7 && v6 )
      {
        **(_QWORD **)(a1 + 8) = v6;
        return 1;
      }
    }
    return 0;
  }
  return v3;
}
