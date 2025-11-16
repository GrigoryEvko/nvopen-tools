// Function: sub_16D2350
// Address: 0x16d2350
//
__int64 __fastcall sub_16D2350(_QWORD *a1, const void *a2, size_t a3)
{
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rbx
  unsigned __int64 v7; // rdi

  v3 = -1;
  v4 = a1[1];
  if ( v4 >= a3 )
  {
    v3 = v4 - a3 + 1;
    if ( v4 - a3 == -1 )
    {
      return -1;
    }
    else
    {
      while ( 1 )
      {
        v7 = --v3;
        if ( v4 <= v3 )
          v7 = v4;
        if ( v4 - v7 >= a3 && (!a3 || !memcmp((const void *)(*a1 + v7), a2, a3)) )
          break;
        if ( !v3 )
          return -1;
      }
    }
  }
  return v3;
}
