// Function: sub_91CC00
// Address: 0x91cc00
//
bool __fastcall sub_91CC00(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)(a1 + 40) == 6 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 128LL);
    if ( *(_BYTE *)(v2 + 40) == 7 )
      return (*(_BYTE *)(*(_QWORD *)(v2 + 72) + 120LL) & 0xA) != 0;
  }
  return result;
}
