// Function: sub_172F970
// Address: 0x172f970
//
__int64 __fastcall sub_172F970(__int64 a1, __int64 a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rax

  v2 = *(_BYTE *)(a2 + 16);
  v3 = 0;
  if ( v2 != 50 )
  {
    if ( v2 != 5 )
      return v3;
    if ( *(_WORD *)(a2 + 18) != 26 )
      return v3;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
      return v3;
    v5 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v5 + 16) != 13 )
      return v3;
LABEL_7:
    v3 = 1;
    **(_QWORD **)(a1 + 8) = v5;
    return v3;
  }
  if ( *(_QWORD *)a1 == *(_QWORD *)(a2 - 48) )
  {
    v5 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v5 + 16) == 13 )
      goto LABEL_7;
  }
  return v3;
}
