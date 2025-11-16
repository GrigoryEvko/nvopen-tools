// Function: sub_AD6840
// Address: 0xad6840
//
__int64 __fastcall sub_AD6840(unsigned int a1, __int64 a2, char a3)
{
  if ( a1 == 28 )
    return sub_AD6530(a2, a2);
  if ( a1 == 29 )
    return sub_AD62B0(a2);
  if ( a1 == 17 || a3 && a1 <= 0x1B && ((1LL << a1) & 0xED80000) != 0 )
    return sub_AD6530(a2, a2);
  return 0;
}
