// Function: sub_38D7200
// Address: 0x38d7200
//
__int64 __fastcall sub_38D7200(__int64 a1, unsigned int a2)
{
  int v2; // eax

  v2 = sub_38D71A0(a1, a2);
  if ( v2 == -1 )
    return a2;
  else
    return sub_38D70E0(a1, v2, 0);
}
