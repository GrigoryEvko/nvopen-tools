// Function: sub_1ACABE0
// Address: 0x1acabe0
//
__int64 __fastcall sub_1ACABE0(__int64 a1, const void *a2, unsigned __int64 a3, const void *a4, unsigned __int64 a5)
{
  unsigned int v8; // r12d
  int v9; // eax

  v8 = sub_1ACA9E0(a1, a3, a5);
  if ( v8 )
    return v8;
  if ( a3 > a5 )
  {
    v8 = 1;
    if ( !a5 )
      return v8;
    v9 = memcmp(a2, a4, a5);
    if ( !v9 )
      return a3 < a5 ? -1 : 1;
    return (v9 >> 31) | 1u;
  }
  if ( a3 )
  {
    v9 = memcmp(a2, a4, a3);
    if ( v9 )
      return (v9 >> 31) | 1u;
  }
  if ( a3 != a5 )
    return a3 < a5 ? -1 : 1;
  return v8;
}
