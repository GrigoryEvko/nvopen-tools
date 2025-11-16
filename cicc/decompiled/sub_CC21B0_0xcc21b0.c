// Function: sub_CC21B0
// Address: 0xcc21b0
//
__int64 sub_CC21B0()
{
  int v5; // r9d
  unsigned int v11; // r8d
  char v12; // al
  char v13; // r10

  _RAX = 0;
  __asm { cpuid }
  v5 = _RAX;
  _RAX = 1;
  __asm { cpuid }
  v11 = (_RCX & 1) == 0 ? 1 : 3;
  v12 = v11 | 4;
  if ( (_RCX & 0x80000) != 0 )
    v11 = ((_RCX & 1) == 0 ? 1 : 3) | 4;
  if ( (_RCX & 0x8000000) != 0 )
  {
    __asm { xgetbv }
    v13 = v12;
    if ( (((_RCX & 1) == 0 ? 1 : 3) & 2 | 4) == 6 )
    {
      if ( (_RCX & 0x10000000) != 0 )
        v11 |= 8u;
      if ( v5 > 6 )
      {
        _RAX = 7;
        __asm { cpuid }
        if ( (_RBX & 0x20) != 0 )
          v11 |= 0x10u;
        if ( (v13 & 0xE0) == 0xE0 )
        {
          if ( (int)_RBX < 0 )
            v11 |= 0x40u;
          if ( (_RBX & 0x10000) != 0 )
            v11 |= 0x20u;
        }
      }
    }
  }
  dword_4C5D058 = v11;
  return v11;
}
