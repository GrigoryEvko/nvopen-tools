// Function: sub_2F4F510
// Address: 0x2f4f510
//
__int64 __fastcall sub_2F4F510(__int64 a1, _QWORD *a2, int a3)
{
  unsigned int v3; // r11d
  unsigned int v4; // r14d
  unsigned int v7; // edi
  __int64 v8; // rdx
  __int64 v9; // r10
  unsigned int i; // eax
  _DWORD *v14; // rdx
  unsigned int v15; // eax
  unsigned int v16; // r9d
  unsigned int v17; // esi
  int v18; // ecx
  __int64 v19; // r10
  __int64 v20; // rdx
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // r8

  v3 = *(_DWORD *)(a1 + 88);
  v4 = v3;
  if ( v3 )
  {
    v7 = (v3 - 1) >> 6;
    v8 = 0;
    v9 = *(_QWORD *)(a1 + 24);
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v9 + 8 * v8);
      if ( v7 == (_DWORD)v8 )
        _RCX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v3) & *(_QWORD *)(v9 + 8 * v8);
      if ( _RCX )
        break;
      if ( v7 + 1 == ++v8 )
        return 0;
    }
    __asm { tzcnt   rcx, rcx }
    v4 = 0;
    for ( i = _RCX + ((_DWORD)v8 << 6); i != -1; i = ((_DWORD)v20 << 6) + _RAX )
    {
      v14 = (_DWORD *)(*a2 + 4LL * i);
      if ( *v14 == -1 )
      {
        ++v4;
        *v14 = a3;
        v3 = *(_DWORD *)(a1 + 88);
      }
      v15 = i + 1;
      if ( v3 == v15 )
        break;
      v16 = v15 >> 6;
      v17 = (v3 - 1) >> 6;
      if ( v15 >> 6 > v17 )
        break;
      v18 = 64 - (v15 & 0x3F);
      v19 = *(_QWORD *)(a1 + 24);
      v20 = v16;
      v21 = 0xFFFFFFFFFFFFFFFFLL >> v18;
      if ( v18 == 64 )
        v21 = 0;
      v22 = ~v21;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v19 + 8 * v20);
        if ( v16 == (_DWORD)v20 )
          _RAX = v22 & *(_QWORD *)(v19 + 8 * v20);
        if ( v17 == (_DWORD)v20 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
        if ( _RAX )
          break;
        if ( v17 < (unsigned int)++v20 )
          return v4;
      }
      __asm { tzcnt   rax, rax }
    }
  }
  return v4;
}
