// Function: sub_25F6110
// Address: 0x25f6110
//
bool __fastcall sub_25F6110(_QWORD *a1, __int64 a2)
{
  bool result; // al
  _BYTE *v4; // rdi

  result = 0;
  v4 = *(_BYTE **)(a2 + 24);
  if ( *v4 > 0x1Cu )
    return *a1 == sub_B43CB0((__int64)v4);
  return result;
}
