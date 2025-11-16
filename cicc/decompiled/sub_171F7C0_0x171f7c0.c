// Function: sub_171F7C0
// Address: 0x171f7c0
//
__int64 __fastcall sub_171F7C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  int v5; // eax

  if ( *(_BYTE *)(a2 + 16) != 75 )
    return 0;
  v3 = *(_QWORD *)(a2 - 48);
  if ( !v3 )
    return 0;
  **(_QWORD **)(a1 + 8) = v3;
  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v4 + 16) != 13 )
    return 0;
  **(_QWORD **)(a1 + 16) = v4;
  v5 = *(unsigned __int16 *)(a2 + 18);
  BYTE1(v5) &= ~0x80u;
  **(_DWORD **)a1 = v5;
  return 1;
}
