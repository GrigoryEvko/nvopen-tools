// Function: sub_253A760
// Address: 0x253a760
//
bool __fastcall sub_253A760(__int64 a1)
{
  bool result; // al

  if ( *(_DWORD *)(a1 + 24) <= 0x40u )
  {
    if ( *(_QWORD *)(a1 + 16) != *(_QWORD *)(a1 + 48) )
      return 0;
  }
  else
  {
    result = sub_C43C50(a1 + 16, (const void **)(a1 + 48));
    if ( !result )
      return result;
  }
  if ( *(_DWORD *)(a1 + 40) <= 0x40u )
    return *(_QWORD *)(a1 + 32) == *(_QWORD *)(a1 + 64);
  else
    return sub_C43C50(a1 + 32, (const void **)(a1 + 64));
}
