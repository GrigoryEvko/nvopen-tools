// Function: sub_325F590
// Address: 0x325f590
//
__int64 __fastcall sub_325F590(unsigned __int16 a1)
{
  if ( a1 <= 1u || (unsigned __int16)(a1 - 504) <= 7u )
    BUG();
  return *(_QWORD *)&byte_444C4A0[16 * a1 - 16];
}
