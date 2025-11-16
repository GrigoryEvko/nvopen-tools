// Function: sub_17200C0
// Address: 0x17200c0
//
bool __fastcall sub_17200C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v4; // al
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 51 )
  {
    if ( v4 != 5 || *(_WORD *)(a2 + 18) != 27 )
      return 0;
    v5 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    if ( !v5 )
    {
      v6 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
LABEL_9:
      if ( v6 )
      {
        **(_QWORD **)a1 = v6;
        return *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) == *(_QWORD *)(a1 + 8);
      }
      return 0;
    }
    **(_QWORD **)a1 = v5;
    v6 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    if ( v6 != *(_QWORD *)(a1 + 8) )
      goto LABEL_9;
    return 1;
  }
  v7 = *(_QWORD *)(a2 - 48);
  if ( v7 )
  {
    **(_QWORD **)a1 = v7;
    v8 = *(_QWORD *)(a2 - 24);
    if ( v8 == *(_QWORD *)(a1 + 8) )
      return 1;
  }
  else
  {
    v8 = *(_QWORD *)(a2 - 24);
  }
  if ( !v8 )
    return 0;
  **(_QWORD **)a1 = v8;
  return *(_QWORD *)(a2 - 48) == *(_QWORD *)(a1 + 8);
}
