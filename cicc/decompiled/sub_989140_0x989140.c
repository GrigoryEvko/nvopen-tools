// Function: sub_989140
// Address: 0x989140
//
bool __fastcall sub_989140(_DWORD *a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 v4; // rax

  result = 0;
  if ( (*a1 & 0x60) == 0 )
  {
    result = 1;
    if ( (*a1 & 0x90) != 0 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        a3 = **(_QWORD **)(a3 + 16);
      v4 = sub_BCAC60(a3);
      return (unsigned __int16)sub_B2DB90(a2, v4) >> 8 == 0;
    }
  }
  return result;
}
