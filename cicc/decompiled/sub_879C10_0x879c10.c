// Function: sub_879C10
// Address: 0x879c10
//
__int64 __fastcall sub_879C10(__int64 a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rax
  __int64 v5; // rax

  v2 = 0;
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
    return v2;
  v3 = *(_QWORD *)(a1 + 64);
  if ( !v3 )
    return v2;
  do
  {
    if ( (*(_BYTE *)(v3 + 124) & 2) == 0 )
      return *(_QWORD *)(a2 + 88) == v3;
    v5 = *(_QWORD *)(v3 + 40);
    if ( !v5 )
      break;
    if ( *(_BYTE *)(v5 + 28) != 3 )
      break;
    v3 = *(_QWORD *)(v5 + 32);
  }
  while ( v3 );
  return 0;
}
