// Function: sub_6DFA00
// Address: 0x6dfa00
//
__int64 __fastcall sub_6DFA00(__int64 a1, unsigned int a2)
{
  __int64 v3; // rdx
  char v4; // al

  while ( 1 )
  {
    while ( 1 )
    {
      v3 = a1 + 24LL * (int)a2;
      v4 = *(_BYTE *)v3 & 3;
      if ( v4 != 3 )
        break;
      ++a2;
      if ( *(_DWORD *)(v3 + 8) )
        a2 = *(_DWORD *)v3 >> 2;
    }
    if ( !v4 )
      break;
    ++a2;
  }
  return a2;
}
