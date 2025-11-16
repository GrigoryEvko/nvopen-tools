// Function: sub_1454270
// Address: 0x1454270
//
__int64 __fastcall sub_1454270(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 v4; // r8
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rcx

  v2 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v2 )
  {
    v4 = 0;
    v5 = a1 - 24LL * v2;
    v6 = 8LL * v2;
    v7 = 0;
    while ( 1 )
    {
      v8 = 24LL * *(unsigned int *)(a1 + 56) + 8 + v7;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
        v9 = *(_QWORD *)(a1 - 8);
        if ( a2 == *(_QWORD *)(v9 + v8) )
          goto LABEL_9;
        v10 = *(_QWORD *)(v9 + 3 * v7);
        if ( *(_BYTE *)(v10 + 16) > 0x10u )
          return 0;
      }
      else
      {
        if ( a2 == *(_QWORD *)(v5 + v8) )
          goto LABEL_9;
        v10 = *(_QWORD *)(v5 + 3 * v7);
        if ( *(_BYTE *)(v10 + 16) > 0x10u )
          return 0;
      }
      if ( v10 != v4 )
      {
        if ( v4 )
          return 0;
        v4 = v10;
      }
LABEL_9:
      v7 += 8;
      if ( v6 == v7 )
        return v4;
    }
  }
  return 0;
}
