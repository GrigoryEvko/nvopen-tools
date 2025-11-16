// Function: sub_29D7F50
// Address: 0x29d7f50
//
__int64 __fastcall sub_29D7F50(__int64 a1, const void *a2, unsigned __int64 a3, const void *a4, unsigned __int64 a5)
{
  unsigned int v8; // r15d
  size_t v9; // rdx
  int v10; // eax

  v8 = sub_29D7CF0(a1, a3, a5);
  if ( !v8 )
  {
    v9 = a5;
    if ( a3 <= a5 )
      v9 = a3;
    if ( v9 )
    {
      v10 = memcmp(a2, a4, v9);
      if ( v10 )
      {
        if ( v10 < 0 )
          return (unsigned int)-1;
        return 1;
      }
    }
    if ( a3 != a5 )
    {
      if ( a3 < a5 )
        return (unsigned int)-1;
      return 1;
    }
  }
  return v8;
}
