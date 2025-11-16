// Function: sub_C446F0
// Address: 0xc446f0
//
__int64 __fastcall sub_C446F0(__int64 *a1, __int64 *a2)
{
  unsigned __int64 v2; // rcx
  __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rdx

  v2 = ((unsigned __int64)*((unsigned int *)a1 + 2) + 63) >> 6;
  if ( !v2 )
    return 1;
  v3 = *a1;
  v4 = *a2;
  v5 = 0;
  while ( (*(_QWORD *)(v3 + 8 * v5) & ~*(_QWORD *)(v4 + 8 * v5)) == 0 )
  {
    if ( v2 == ++v5 )
      return 1;
  }
  return 0;
}
