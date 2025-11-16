// Function: sub_C45DF0
// Address: 0xc45df0
//
__int64 __fastcall sub_C45DF0(__int64 a1, int a2)
{
  __int64 v2; // rax

  if ( !a2 )
    return 0xFFFFFFFFLL;
  v2 = 0;
  while ( !*(_QWORD *)(a1 + 8 * v2) )
  {
    if ( a2 == ++v2 )
      return 0xFFFFFFFFLL;
  }
  __asm { tzcnt   rdx, rdx }
  return (unsigned int)(((_DWORD)v2 << 6) + _RDX);
}
