// Function: sub_887620
// Address: 0x887620
//
__int64 __fastcall sub_887620(_BYTE *a1)
{
  int v1; // eax
  unsigned int i; // r8d

  v1 = (char)*a1;
  for ( i = 0; *a1; v1 = (char)*a1 )
  {
    ++a1;
    i += 32 * i + v1;
  }
  return i;
}
