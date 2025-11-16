// Function: sub_15FA1F0
// Address: 0x15fa1f0
//
__int64 __fastcall sub_15FA1F0(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned int v3; // ebx
  __int64 v4; // rdi
  unsigned int v5; // r14d

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v1 == 1 )
    return 1;
  v3 = 1;
  while ( 1 )
  {
    v4 = *(_QWORD *)(a1 + 24 * (v3 - (unsigned __int64)v1));
    if ( *(_BYTE *)(v4 + 16) != 13 )
      break;
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
    {
      if ( *(_QWORD *)(v4 + 24) )
        return 0;
    }
    else if ( v5 != (unsigned int)sub_16A57B0(v4 + 24) )
    {
      return 0;
    }
    if ( v1 == ++v3 )
      return 1;
  }
  return 0;
}
