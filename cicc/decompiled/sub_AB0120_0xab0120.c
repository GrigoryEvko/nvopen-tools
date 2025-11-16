// Function: sub_AB0120
// Address: 0xab0120
//
bool __fastcall sub_AB0120(__int64 a1)
{
  int v1; // r8d
  bool result; // al
  unsigned int v3; // eax
  __int64 v4; // rsi
  unsigned int v5; // r13d

  v1 = sub_C4C880(a1, a1 + 16);
  result = 0;
  if ( v1 > 0 )
  {
    v3 = *(_DWORD *)(a1 + 24);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = v3 - 1;
    if ( v3 <= 0x40 )
    {
      return v4 != 1LL << v5;
    }
    else
    {
      result = 1;
      if ( (*(_QWORD *)(v4 + 8LL * (v5 >> 6)) & (1LL << v5)) != 0 )
        return (unsigned int)sub_C44590(a1 + 16) != v5;
    }
  }
  return result;
}
