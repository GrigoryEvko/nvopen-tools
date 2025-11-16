// Function: sub_1B16050
// Address: 0x1b16050
//
__int64 __fastcall sub_1B16050(int a1)
{
  int v2; // eax

  if ( !a1 )
    return 0;
  LOBYTE(v2) = sub_1B16040(a1);
  return v2 ^ 1u;
}
