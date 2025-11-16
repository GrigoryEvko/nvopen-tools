// Function: sub_B1A0F0
// Address: 0xb1a0f0
//
__int64 __fastcall sub_B1A0F0(__int64 a1, __int64 *a2, __int64 *a3)
{
  if ( *a2 == *a3 && a2[1] == a3[1] )
    return 1;
  else
    return sub_B19C20(a1, a2, *a3);
}
