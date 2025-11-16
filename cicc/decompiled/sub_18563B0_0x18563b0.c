// Function: sub_18563B0
// Address: 0x18563b0
//
bool __fastcall sub_18563B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx
  bool result; // al

  v1 = *(_QWORD *)(a1 + 80);
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)(v1 + 24);
  v3 = v1 + 16;
  if ( v2 == v1 + 16 )
    return 0;
  v4 = *(_QWORD *)(v1 + 24);
  v5 = 0;
  do
  {
    v4 = *(_QWORD *)(v4 + 8);
    ++v5;
  }
  while ( v4 != v3 );
  if ( v5 != 1 )
    return 0;
  if ( !v2 )
    BUG();
  result = 0;
  if ( *(_BYTE *)(v2 - 8) == 25 )
  {
    result = 1;
    if ( (*(_DWORD *)(v2 - 4) & 0xFFFFFFF) != 0 )
      return *(_QWORD *)(v2 - 24 * ((*(_DWORD *)(v2 - 4) & 0xFFFFFFF) + 1LL)) == 0;
  }
  return result;
}
