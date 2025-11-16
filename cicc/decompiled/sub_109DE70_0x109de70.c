// Function: sub_109DE70
// Address: 0x109de70
//
__int64 __fastcall sub_109DE70(__int64 a1)
{
  unsigned int v1; // ebx
  __int64 v2; // rdx
  unsigned int v4; // r8d
  int v5; // r13d
  unsigned __int64 v6; // r13
  int v9; // r13d
  unsigned int v10; // r8d

  v1 = *(_DWORD *)(a1 + 8);
  v2 = 1LL << ((unsigned __int8)v1 - 1);
  _RAX = *(_QWORD *)a1;
  if ( v1 <= 0x40 )
  {
    v4 = 0;
    if ( (v2 & _RAX) != 0 )
    {
      if ( v1 )
      {
        v5 = 64;
        if ( _RAX << (64 - (unsigned __int8)v1) != -1 )
        {
          _BitScanReverse64(&v6, ~(_RAX << (64 - (unsigned __int8)v1)));
          v5 = v6 ^ 0x3F;
        }
      }
      else
      {
        v5 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v1 )
        LODWORD(_RAX) = *(_DWORD *)(a1 + 8);
      LOBYTE(v4) = v5 + (_DWORD)_RAX == v1;
    }
    return v4;
  }
  v4 = 0;
  if ( (*(_QWORD *)(_RAX + 8LL * ((v1 - 1) >> 6)) & v2) == 0 )
    return v4;
  v9 = sub_C44500(a1);
  LOBYTE(v10) = v9 + (unsigned int)sub_C44590(a1) == v1;
  return v10;
}
