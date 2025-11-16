// Function: sub_22419D0
// Address: 0x22419d0
//
__int64 __fastcall sub_22419D0(__int64 *a1, const void *a2, unsigned __int64 a3, size_t a4)
{
  unsigned __int64 v4; // r13
  __int64 v5; // r14
  unsigned __int64 v6; // rbx

  v4 = a1[1];
  if ( a3 >= v4 )
    return -1;
  v5 = *a1;
  v6 = a3;
  if ( a4 )
  {
    while ( memchr(a2, *(char *)(v5 + v6), a4) )
    {
      if ( ++v6 == v4 )
        return -1;
    }
  }
  return v6;
}
