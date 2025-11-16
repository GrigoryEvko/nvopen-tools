// Function: sub_C7F4B0
// Address: 0xc7f4b0
//
__int64 __fastcall sub_C7F4B0(__int64 a1, signed int a2, unsigned __int64 a3, int a4)
{
  if ( a2 >= 0 )
    return sub_C7F1D0(a1, a2, a3, a4, 0);
  else
    return sub_C7F1D0(a1, -a2, a3, a4, 1);
}
