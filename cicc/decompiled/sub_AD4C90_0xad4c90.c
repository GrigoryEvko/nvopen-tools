// Function: sub_AD4C90
// Address: 0xad4c90
//
__int64 __fastcall sub_AD4C90(unsigned __int64 a1, __int64 **a2, char a3)
{
  if ( *(__int64 ***)(a1 + 8) == a2 )
    return a1;
  else
    return sub_AD4B40(0x31u, a1, a2, a3);
}
