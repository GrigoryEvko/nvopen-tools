// Function: sub_16A58F0
// Address: 0x16a58f0
//
__int64 __fastcall sub_16A58F0(__int64 a1)
{
  _QWORD *v1; // rax
  unsigned int v2; // r8d

  if ( ((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6 )
  {
    v1 = *(_QWORD **)a1;
    v2 = 0;
    while ( *v1 == -1 )
    {
      ++v1;
      v2 += 64;
      if ( v1 == (_QWORD *)(*(_QWORD *)a1
                          + 8LL * ((unsigned int)(((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6) - 1)
                          + 8) )
        return v2;
    }
    _RDX = ~*v1;
    __asm { tzcnt   rdx, rdx }
    v2 += _RDX;
  }
  else
  {
    return 0;
  }
  return v2;
}
