// Function: sub_6DEC10
// Address: 0x6dec10
//
__int64 __fastcall sub_6DEC10(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(qword_4D03C50 + 16LL) & 0x200000200LL;
  if ( result == 512 )
  {
    result = qword_4F04C50;
    if ( !qword_4F04C50 || (result = *(_QWORD *)(qword_4F04C50 + 32LL), (*(_BYTE *)(result + 193) & 4) == 0) )
    {
      if ( (*(_BYTE *)(a1 + 206) & 0x10) == 0 )
        *(_BYTE *)(qword_4D03C50 + 20LL) |= 4u;
    }
  }
  return result;
}
