// Function: sub_2E5E7F0
// Address: 0x2e5e7f0
//
__int64 __fastcall sub_2E5E7F0(__int64 *a1, __int64 *a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r8d

  if ( !a2 )
    return 0;
  v2 = *((_DWORD *)a1 + 42);
  v3 = 0;
  if ( v2 <= *((_DWORD *)a2 + 42) )
  {
    if ( v2 != *((_DWORD *)a2 + 42) )
    {
      do
        a2 = (__int64 *)*a2;
      while ( v2 < *((_DWORD *)a2 + 42) );
    }
    LOBYTE(v3) = a1 == a2;
  }
  return v3;
}
