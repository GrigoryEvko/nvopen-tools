// Function: sub_A3C900
// Address: 0xa3c900
//
__int64 __fastcall sub_A3C900(__int64 a1, __int64 a2)
{
  __int64 k; // r12
  __int64 v5; // rdi
  __int64 i; // r12
  __int64 v7; // rdi
  __int64 j; // r12
  __int64 v9; // rdi

  if ( *(_BYTE *)(a2 + 872) )
  {
    if ( unk_4F80E08 )
    {
      sub_BA8950(a2);
    }
    else
    {
      for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        v7 = i - 56;
        if ( !i )
          v7 = 0;
        sub_B2B9A0(v7);
      }
      *(_BYTE *)(a2 + 872) = 0;
    }
    sub_A3ACE0(a2, *(_QWORD *)(a1 + 176), *(_BYTE *)(a1 + 184), 0, 0, 0);
    if ( !*(_BYTE *)(a2 + 872) )
    {
      for ( j = *(_QWORD *)(a2 + 32); a2 + 24 != j; j = *(_QWORD *)(j + 8) )
      {
        v9 = j - 56;
        if ( !j )
          v9 = 0;
        sub_B2B950(v9);
      }
      *(_BYTE *)(a2 + 872) = 1;
    }
  }
  else
  {
    sub_A3ACE0(a2, *(_QWORD *)(a1 + 176), *(_BYTE *)(a1 + 184), 0, 0, 0);
    if ( *(_BYTE *)(a2 + 872) )
    {
      for ( k = *(_QWORD *)(a2 + 32); a2 + 24 != k; k = *(_QWORD *)(k + 8) )
      {
        v5 = k - 56;
        if ( !k )
          v5 = 0;
        sub_B2B9A0(v5);
      }
      *(_BYTE *)(a2 + 872) = 0;
    }
  }
  return 0;
}
