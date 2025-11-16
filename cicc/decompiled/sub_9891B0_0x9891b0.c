// Function: sub_9891B0
// Address: 0x9891b0
//
bool __fastcall sub_9891B0(_DWORD *a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 v4; // rax

  result = 0;
  if ( (*a1 & 0x20) == 0 )
  {
    result = 1;
    if ( (*a1 & 0x10) != 0 )
    {
      if ( (unsigned int)*(unsigned __int8 *)(a3 + 8) - 17 <= 1 )
        a3 = **(_QWORD **)(a3 + 16);
      v4 = sub_BCAC60(a3);
      return (sub_B2DB90(a2, v4) & 0xFD00) == 0;
    }
  }
  return result;
}
