// Function: sub_13D7B90
// Address: 0x13d7b90
//
char __fastcall sub_13D7B90(__int64 *a1, __int64 a2)
{
  char v2; // al
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 50 )
  {
    v4 = *a1;
    if ( *a1 == *(_QWORD *)(a2 - 48) )
    {
      if ( sub_13D1F50(a1 + 1, *(_QWORD *)(a2 - 24)) )
        return 1;
      v4 = *a1;
    }
    if ( *(_QWORD *)(a2 - 24) == v4 )
      return sub_13D1F50(a1 + 1, *(_QWORD *)(a2 - 48));
    return 0;
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 26 )
    return 0;
  v5 = *a1;
  v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( *a1 != *(_QWORD *)(a2 - 24 * v6) )
  {
LABEL_9:
    if ( *(_QWORD *)(a2 + 24 * (1 - v6)) == v5 )
      return sub_13D77F0(a1 + 1, *(_QWORD *)(a2 - 24 * v6));
    return 0;
  }
  if ( !sub_13D77F0(a1 + 1, *(_QWORD *)(a2 + 24 * (1 - v6))) )
  {
    v5 = *a1;
    v6 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    goto LABEL_9;
  }
  return 1;
}
