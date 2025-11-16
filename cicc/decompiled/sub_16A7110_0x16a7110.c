// Function: sub_16A7110
// Address: 0x16a7110
//
__int64 __fastcall sub_16A7110(__int64 a1, int a2)
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
