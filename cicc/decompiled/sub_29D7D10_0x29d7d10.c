// Function: sub_29D7D10
// Address: 0x29d7d10
//
__int64 __fastcall sub_29D7D10(__int64 a1, char a2, char a3)
{
  __int64 result; // rax

  result = 1LL << a3 < (unsigned __int64)(1LL << a2);
  if ( 1LL << a3 > (unsigned __int64)(1LL << a2) )
    return 0xFFFFFFFFLL;
  return result;
}
