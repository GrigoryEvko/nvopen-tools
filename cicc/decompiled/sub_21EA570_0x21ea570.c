// Function: sub_21EA570
// Address: 0x21ea570
//
__int64 __fastcall sub_21EA570(signed int a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // eax

  if ( a1 >= 0 )
    return 0;
  if ( (unsigned int)sub_1F4B530(a3, a1, a2) == 1 )
    return 0x100000000LL;
  v5 = (unsigned int)sub_1F4B530(a3, a1, a2) >> 5;
  if ( !v5 )
    v5 = 1;
  return v5 & 0x7FFFFFF;
}
