// Function: sub_21C5290
// Address: 0x21c5290
//
bool __fastcall sub_21C5290(__int64 a1, __int64 a2, int a3)
{
  int v3; // eax
  unsigned __int64 v4; // rcx
  bool result; // al
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rcx

  v3 = *(unsigned __int16 *)(a2 + 24);
  v4 = (unsigned int)(v3 - 185);
  if ( (unsigned __int16)(v3 - 185) <= 0x35u )
  {
    v6 = 0x3FFFFD00000003LL;
    if ( !_bittest64(&v6, v4) )
      return 0;
  }
  else if ( (unsigned __int16)(v3 - 44) <= 1u )
  {
    if ( (*(_BYTE *)(a2 + 26) & 2) == 0 )
      return 0;
  }
  else if ( (__int16)v3 <= 658 )
  {
    return 0;
  }
  v7 = **(_QWORD **)(a2 + 104);
  if ( a3 )
  {
    if ( (v7 & 4) != 0 )
      return 0;
  }
  else if ( (v7 & 4) != 0 )
  {
    return (v7 & 0xFFFFFFFFFFFFFFF8LL) != 0;
  }
  v8 = (__int64 *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
  if ( !v8 )
    return 0;
  v9 = *v8;
  result = 0;
  if ( *(_BYTE *)(v9 + 8) == 15 )
    return *(_DWORD *)(v9 + 8) >> 8 == a3;
  return result;
}
