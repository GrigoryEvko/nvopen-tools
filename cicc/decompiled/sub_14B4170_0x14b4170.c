// Function: sub_14B4170
// Address: 0x14b4170
//
__int64 __fastcall sub_14B4170(_QWORD **a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 35 )
  {
    if ( *(_QWORD *)(a2 - 48) != **a1 )
      return 0;
    v4 = *(_QWORD *)(a2 - 24);
    if ( !v4 )
      return 0;
  }
  else
  {
    if ( v2 != 5 )
      return 0;
    if ( *(_WORD *)(a2 + 18) != 11 )
      return 0;
    if ( **a1 != *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
      return 0;
    v4 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( !v4 )
      return 0;
  }
  *a1[1] = v4;
  return 1;
}
