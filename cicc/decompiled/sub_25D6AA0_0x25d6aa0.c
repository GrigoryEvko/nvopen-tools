// Function: sub_25D6AA0
// Address: 0x25d6aa0
//
bool __fastcall sub_25D6AA0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rbx
  bool result; // al

  if ( sub_B2FC80(a3) )
    return 0;
  v4 = *(_QWORD *)(a3 + 80);
  if ( !v4 )
    BUG();
  v5 = *(_QWORD *)(v4 + 32);
  v6 = v4 + 24;
  if ( v5 == v6 )
    return 0;
  while ( 1 )
  {
    v7 = v5 - 24;
    if ( !v5 )
      v7 = 0;
    result = sub_B46AA0(v7);
    if ( !result )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v6 == v5 )
      return 0;
  }
  if ( *(_BYTE *)v7 == 30 )
  {
    result = 1;
    if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
      return *(_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF)) == 0;
  }
  return result;
}
