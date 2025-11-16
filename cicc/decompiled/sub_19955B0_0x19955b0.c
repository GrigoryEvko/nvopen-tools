// Function: sub_19955B0
// Address: 0x19955b0
//
__int64 __fastcall sub_19955B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned int v4; // r8d
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  unsigned __int64 v10; // rax
  int v11; // r11d
  unsigned int v12; // r9d
  __int64 v13; // rdi
  __int64 v14; // rax
  int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // ebx
  int v20; // ecx
  unsigned __int64 v21; // r10
  __int64 v22; // rdx
  unsigned __int64 v23; // r10
  unsigned __int64 v24; // r11
  unsigned __int64 v28; // rdi
  unsigned __int64 v31; // rcx
  int v32; // edx
  int v33; // r11d

  v3 = *(unsigned int *)(a1 + 24);
  v4 = 0;
  if ( !(_DWORD)v3 )
    return v4;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v32 = 1;
    while ( v9 != -8 )
    {
      v33 = v32 + 1;
      v7 = (v3 - 1) & (v32 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v32 = v33;
    }
    return 0;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v6 + 16 * v3) )
    return 0;
  v10 = v8[1];
  v4 = v10 & 1;
  if ( (v10 & 1) != 0 )
  {
    v28 = v10 >> 58;
    _RAX = ~(-1LL << (v10 >> 58)) & (v10 >> 1);
    if ( _RAX )
    {
      __asm { tzcnt   rcx, rax }
      if ( a3 != (int)_RCX )
        return v4;
      v31 = (unsigned int)(_RCX + 1);
      if ( (_RAX & (-1LL << v31)) != 0 )
      {
        LOBYTE(v4) = v28 > v31;
        return v4;
      }
    }
    return 0;
  }
  v11 = *(_DWORD *)(v10 + 16);
  if ( !v11 )
    return v4;
  v12 = (unsigned int)(v11 - 1) >> 6;
  v13 = *(_QWORD *)v10;
  v14 = 0;
  while ( 1 )
  {
    _RDX = *(_QWORD *)(v13 + 8 * v14);
    if ( (unsigned int)(v11 - 1) >> 6 == v14 )
      break;
    if ( _RDX )
      goto LABEL_11;
    if ( v12 + 1 == ++v14 )
      return v4;
  }
  _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
  if ( !_RDX )
    return v4;
LABEL_11:
  __asm { tzcnt   rdx, rdx }
  v17 = ((_DWORD)v14 << 6) + _RDX;
  if ( v17 != -1 )
  {
    if ( a3 == v17 )
    {
      v18 = v17 + 1;
      if ( v11 != v18 )
      {
        v19 = v18 >> 6;
        if ( v12 >= v18 >> 6 )
        {
          v20 = 64 - (v18 & 0x3F);
          v21 = 0xFFFFFFFFFFFFFFFFLL >> v20;
          if ( v20 == 64 )
            v21 = 0;
          v22 = v19;
          v23 = ~v21;
          v24 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
          while ( 1 )
          {
            _RAX = *(_QWORD *)(v13 + 8 * v22);
            if ( v19 == (_DWORD)v22 )
              _RAX = v23 & *(_QWORD *)(v13 + 8 * v22);
            if ( (_DWORD)v22 == v12 )
              _RAX &= v24;
            if ( _RAX )
              break;
            if ( v12 < (unsigned int)++v22 )
              return v4;
          }
          __asm { tzcnt   rax, rax }
          LOBYTE(v4) = ((_DWORD)v22 << 6) + (_DWORD)_RAX != -1;
        }
      }
    }
    else
    {
      return 1;
    }
  }
  return v4;
}
