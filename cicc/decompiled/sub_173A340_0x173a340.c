// Function: sub_173A340
// Address: 0x173a340
//
bool __fastcall sub_173A340(__int64 **a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  char v4; // al
  char v7; // al
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  char v11; // al
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rsi

  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 == 50 )
  {
    v7 = sub_1738BE0(a1, *(_QWORD *)(a2 - 48), a3, a4);
    v10 = *(_QWORD *)(a2 - 24);
    if ( !v7 || v10 != *a1[2] )
    {
      if ( sub_1738BE0(a1, v10, v8, v9) )
        return *a1[2] == *(_QWORD *)(a2 - 48);
      return 0;
    }
    return 1;
  }
  if ( v4 != 5 || *(_WORD *)(a2 + 18) != 26 )
    return 0;
  v11 = sub_1738E40(
          a1,
          *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
          4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
          a4);
  v13 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( !v11 )
  {
    v14 = *(_QWORD *)(a2 + 24 * (1 - v13));
    goto LABEL_12;
  }
  v14 = *(_QWORD *)(a2 + 24 * (1 - v13));
  if ( *a1[2] == v14 )
    return 1;
LABEL_12:
  if ( !sub_1738E40(a1, v14, v13, v12) )
    return 0;
  return *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) == *a1[2];
}
