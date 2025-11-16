// Function: sub_AAFBB0
// Address: 0xaafbb0
//
bool __fastcall sub_AAFBB0(__int64 a1)
{
  int v1; // r8d
  bool result; // al
  unsigned int v3; // r13d

  v1 = sub_C49970(a1, a1 + 16);
  result = 0;
  if ( v1 > 0 )
  {
    v3 = *(_DWORD *)(a1 + 24);
    if ( v3 <= 0x40 )
      return *(_QWORD *)(a1 + 16) != 0;
    else
      return v3 != (unsigned int)sub_C444A0(a1 + 16);
  }
  return result;
}
