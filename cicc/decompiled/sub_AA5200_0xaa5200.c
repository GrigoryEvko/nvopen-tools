// Function: sub_AA5200
// Address: 0xaa5200
//
void __fastcall sub_AA5200(__int64 a1)
{
  __int64 v2; // rdi
  __int64 i; // r8
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rdx

  v2 = *(_QWORD *)(a1 + 56);
  for ( i = a1 + 48; i != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    if ( !v2 )
      BUG();
    v4 = 32LL * (*(_DWORD *)(v2 - 20) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v2 - 17) & 0x40) != 0 )
    {
      v5 = *(_QWORD *)(v2 - 32);
      v6 = v5 + v4;
    }
    else
    {
      v6 = v2 - 24;
      v5 = v2 - 24 - v4;
    }
    for ( ; v5 != v6; v5 += 32 )
    {
      if ( *(_QWORD *)v5 )
      {
        v7 = *(_QWORD *)(v5 + 8);
        **(_QWORD **)(v5 + 16) = v7;
        if ( v7 )
          *(_QWORD *)(v7 + 16) = *(_QWORD *)(v5 + 16);
      }
      *(_QWORD *)v5 = 0;
    }
  }
}
