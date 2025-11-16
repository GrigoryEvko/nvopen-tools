// Function: sub_10E00E0
// Address: 0x10e00e0
//
__int64 __fastcall sub_10E00E0(__int64 a1)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    return sub_C445E0(a1);
  _RAX = ~*(_QWORD *)a1;
  __asm { tzcnt   rdx, rax }
  result = 64;
  if ( *(_QWORD *)a1 != -1 )
    return (unsigned int)_RDX;
  return result;
}
