// Function: sub_1EBBD70
// Address: 0x1ebbd70
//
__int64 __fastcall sub_1EBBD70(__int64 a1, _QWORD *a2, int a3)
{
  int v3; // ecx
  unsigned int v5; // edi
  __int64 v6; // r10
  unsigned __int64 v8; // r8
  __int64 v9; // rdx
  unsigned int i; // eax
  _DWORD *v14; // rdx
  int v15; // edx
  unsigned int v16; // eax
  unsigned int v17; // r9d
  unsigned int v18; // esi
  int v19; // eax
  __int64 v20; // r10
  unsigned __int64 v21; // r8
  __int64 v22; // rdx
  unsigned __int64 v23; // r8
  unsigned int v27; // [rsp+4h] [rbp-2Ch]

  v3 = *(_DWORD *)(a1 + 40);
  if ( v3 )
  {
    v5 = (unsigned int)(v3 - 1) >> 6;
    v6 = *(_QWORD *)(a1 + 24);
    v8 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
    v9 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v6 + 8 * v9);
      if ( v5 == (_DWORD)v9 )
        _RCX = v8 & *(_QWORD *)(v6 + 8 * v9);
      if ( _RCX )
        break;
      if ( v5 + 1 == ++v9 )
        return 0;
    }
    __asm { tzcnt   rcx, rcx }
    v27 = 0;
    for ( i = _RCX + ((_DWORD)v9 << 6); i != -1; i = ((_DWORD)v22 << 6) + _RAX )
    {
      v14 = (_DWORD *)(*a2 + 4LL * i);
      if ( *v14 == -1 )
      {
        ++v27;
        *v14 = a3;
      }
      v15 = *(_DWORD *)(a1 + 40);
      v16 = i + 1;
      if ( v15 == v16 )
        break;
      v17 = v16 >> 6;
      v18 = (unsigned int)(v15 - 1) >> 6;
      if ( v16 >> 6 > v18 )
        break;
      v19 = v16 & 0x3F;
      v20 = *(_QWORD *)(a1 + 24);
      v21 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v19);
      v22 = v17;
      if ( v19 == 0 )
        v21 = 0;
      v23 = ~v21;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v20 + 8 * v22);
        if ( v17 == (_DWORD)v22 )
          _RAX = v23 & *(_QWORD *)(v20 + 8 * v22);
        if ( v18 == (_DWORD)v22 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(a1 + 40);
        if ( _RAX )
          break;
        if ( v18 < (unsigned int)++v22 )
          return v27;
      }
      __asm { tzcnt   rax, rax }
    }
  }
  else
  {
    return 0;
  }
  return v27;
}
