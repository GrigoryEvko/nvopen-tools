// Function: sub_157E950
// Address: 0x157e950
//
bool __fastcall sub_157E950(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // rdx

  result = 1;
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v3 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v3 + 16) && (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
      return (unsigned int)(*(_DWORD *)(v3 + 36) - 35) > 3;
  }
  return result;
}
