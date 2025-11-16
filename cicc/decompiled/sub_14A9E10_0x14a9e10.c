// Function: sub_14A9E10
// Address: 0x14a9e10
//
__int64 __fastcall sub_14A9E10(__int64 a1)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    return sub_16A58F0(a1);
  _RAX = ~*(_QWORD *)a1;
  __asm { tzcnt   rdx, rax }
  result = 64;
  if ( *(_QWORD *)a1 != -1 )
    return (unsigned int)_RDX;
  return result;
}
