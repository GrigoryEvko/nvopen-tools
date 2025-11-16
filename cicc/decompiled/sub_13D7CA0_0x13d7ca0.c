// Function: sub_13D7CA0
// Address: 0x13d7ca0
//
bool __fastcall sub_13D7CA0(__int64 *a1, __int64 a2)
{
  char v2; // al
  char v5; // al
  __int64 v6; // rsi
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // rsi

  v2 = *(_BYTE *)(a2 + 16);
  if ( v2 == 50 )
  {
    v5 = sub_13D1F50(a1, *(_QWORD *)(a2 - 48));
    v6 = *(_QWORD *)(a2 - 24);
    if ( !v5 || v6 != a1[2] )
    {
      if ( sub_13D1F50(a1, v6) )
        return a1[2] == *(_QWORD *)(a2 - 48);
      return 0;
    }
    return 1;
  }
  if ( v2 != 5 || *(_WORD *)(a2 + 18) != 26 )
    return 0;
  v7 = sub_13D77F0(a1, *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v8 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( !v7 )
  {
    v9 = *(_QWORD *)(a2 + 24 * (1 - v8));
    goto LABEL_12;
  }
  v9 = *(_QWORD *)(a2 + 24 * (1 - v8));
  if ( a1[2] == v9 )
    return 1;
LABEL_12:
  if ( !sub_13D77F0(a1, v9) )
    return 0;
  return *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) == a1[2];
}
