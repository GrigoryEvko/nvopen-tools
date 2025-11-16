// Function: sub_1422930
// Address: 0x1422930
//
void __fastcall sub_1422930(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 j; // r15
  __int64 v6; // rsi
  __int64 v7; // r15
  __int64 i; // [rsp+8h] [rbp-48h]
  __int64 v9; // [rsp+18h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 80);
  for ( i = a2 + 72; i != v2; v2 = *(_QWORD *)(v2 + 8) )
  {
    v3 = v2 - 24;
    if ( !v2 )
      v3 = 0;
    v4 = sub_14228C0(a1, v3);
    if ( v4 && (*(_DWORD *)(v4 + 20) & 0xFFFFFFF) != 0 )
    {
      v7 = 0;
      v9 = 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF);
      do
      {
        v7 += 24;
        nullsub_535();
      }
      while ( v7 != v9 );
    }
    for ( j = *(_QWORD *)(v3 + 48); v3 + 40 != j; j = *(_QWORD *)(j + 8) )
    {
      v6 = j - 24;
      if ( !j )
        v6 = 0;
      if ( sub_1422850(a1, v6) )
        nullsub_535();
    }
  }
}
