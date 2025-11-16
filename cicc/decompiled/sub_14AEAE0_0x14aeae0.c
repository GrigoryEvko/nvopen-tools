// Function: sub_14AEAE0
// Address: 0x14aeae0
//
__int64 __fastcall sub_14AEAE0(int a1, char a2)
{
  int v2; // eax

  switch ( a1 )
  {
    case 1:
      return 40;
    case 2:
      return 36;
    case 3:
      return 38;
    case 4:
      return 34;
  }
  v2 = a2 == 0 ? 8 : 0;
  if ( a1 == 5 )
    return (unsigned int)(v2 + 4);
  else
    return (unsigned int)(v2 + 2);
}
