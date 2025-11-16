// Function: sub_2EBEE90
// Address: 0x2ebee90
//
unsigned __int64 __fastcall sub_2EBEE90(__int64 a1, int a2)
{
  __int64 v2; // r8
  unsigned __int64 v3; // rsi
  unsigned __int64 i; // rcx
  __int64 v6; // rax
  unsigned __int64 j; // rdx

  if ( a2 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  if ( !v2 )
    return v2;
  if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
  {
    v6 = *(_QWORD *)(v2 + 32);
    v2 = 0;
    if ( !v6 || (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
      return v2;
    v2 = v6;
  }
  v3 = *(_QWORD *)(v2 + 16);
  for ( i = v3; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
    ;
  while ( 1 )
  {
    v2 = *(_QWORD *)(v2 + 32);
    if ( !v2 || (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
      break;
    for ( j = *(_QWORD *)(v2 + 16); (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
      ;
    if ( j != i )
      return 0;
  }
  for ( ; (*(_BYTE *)(v3 + 44) & 4) != 0; v3 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  return v3;
}
