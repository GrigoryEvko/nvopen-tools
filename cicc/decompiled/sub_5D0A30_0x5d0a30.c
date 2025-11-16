// Function: sub_5D0A30
// Address: 0x5d0a30
//
__int64 __fastcall sub_5D0A30(const char *a1)
{
  unsigned int v1; // r8d

  v1 = 1;
  if ( strcmp(a1, "hidden") )
  {
    v1 = 2;
    if ( strcmp(a1, "protected") )
    {
      v1 = 3;
      if ( strcmp(a1, "internal") )
      {
        LOBYTE(v1) = strcmp(a1, "default") == 0;
        v1 *= 4;
      }
    }
  }
  return v1;
}
