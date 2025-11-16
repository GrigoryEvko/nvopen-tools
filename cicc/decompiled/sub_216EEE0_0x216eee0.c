// Function: sub_216EEE0
// Address: 0x216eee0
//
bool __fastcall sub_216EEE0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = **(_WORD **)(a1 + 16) != 3066 && **(_WORD **)(a1 + 16) != 3060;
  if ( result )
    return 0;
  v2 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(v2 + 80) == 1 )
    return *(_QWORD *)(v2 + 104) == 4;
  return result;
}
