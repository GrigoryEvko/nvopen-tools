// Function: sub_3905F30
// Address: 0x3905f30
//
__int64 __fastcall sub_3905F30(void *s1, size_t n, void *s2, size_t a4)
{
  unsigned int v4; // r15d

  if ( a4 > n || (v4 = 1, a4) && memcmp(s1, s2, a4) )
  {
    LOBYTE(v4) = n == 0;
    if ( a4 )
    {
      v4 = 0;
      if ( a4 - 1 == n )
      {
        if ( n )
          LOBYTE(v4) = memcmp(s1, s2, n) == 0;
        else
          return 1;
      }
    }
  }
  return v4;
}
