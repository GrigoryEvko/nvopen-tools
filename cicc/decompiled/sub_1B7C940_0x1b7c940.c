// Function: sub_1B7C940
// Address: 0x1b7c940
//
__int64 __fastcall sub_1B7C940(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v3; // rax
  __int64 v4; // rax
  int v5; // eax
  bool v6; // dl
  __int64 v7; // rax
  __int64 *v8; // rcx

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 <= 0x17u )
    return 0xFFFFFFFFLL;
  if ( v1 == 54 || v1 == 55 )
  {
    v3 = **(_QWORD **)(a1 - 24);
    if ( *(_BYTE *)(v3 + 8) == 16 )
      v3 = **(_QWORD **)(v3 + 16);
    return *(_DWORD *)(v3 + 8) >> 8;
  }
  if ( v1 != 78 )
    return 0xFFFFFFFFLL;
  v4 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v4 + 16) )
    return 0xFFFFFFFFLL;
  v5 = *(_DWORD *)(v4 + 36);
  if ( !v5 )
    return 0xFFFFFFFFLL;
  v6 = v5 == 4492 || v5 == 4503;
  if ( v5 != 4085 && v5 != 4057 )
  {
    if ( v6 )
    {
      v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
      goto LABEL_15;
    }
    return 0xFFFFFFFFLL;
  }
  v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v8 = *(__int64 **)(a1 + 24 * (1 - v7));
  if ( v8 )
  {
    if ( !v6 )
      goto LABEL_16;
  }
  else if ( !v6 )
  {
    return 0xFFFFFFFFLL;
  }
LABEL_15:
  v8 = *(__int64 **)(a1 + 24 * (2 - v7));
  if ( !v8 )
    return 0xFFFFFFFFLL;
LABEL_16:
  v3 = *v8;
  return *(_DWORD *)(v3 + 8) >> 8;
}
