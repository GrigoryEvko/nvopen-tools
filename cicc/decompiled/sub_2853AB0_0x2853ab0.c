// Function: sub_2853AB0
// Address: 0x2853ab0
//
__int64 __fastcall sub_2853AB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r10
  unsigned __int64 v9; // rax
  unsigned int v10; // r10d
  int v11; // r11d
  unsigned int v12; // r9d
  __int64 v13; // r8
  __int64 v14; // rax
  int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // ebx
  int v20; // ecx
  unsigned __int64 v21; // rdi
  __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r11
  unsigned __int64 v28; // r8
  int v31; // edx
  unsigned __int64 v32; // rcx
  int v33; // r11d

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v3 )
    return 0;
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v31 = 1;
    while ( v8 != -4096 )
    {
      v33 = v31 + 1;
      v6 = (v3 - 1) & (v31 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      v31 = v33;
    }
    return 0;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v4 + 16 * v3) )
    return 0;
  v9 = v7[1];
  v10 = v9 & 1;
  if ( (v9 & 1) != 0 )
  {
    v28 = v9 >> 58;
    _RAX = ~(-1LL << (v9 >> 58)) & (v9 >> 1);
    if ( _RAX )
    {
      __asm { tzcnt   rcx, rax }
      if ( a3 != (int)_RCX )
        return v10;
      v32 = (unsigned int)(_RCX + 1);
      if ( (_RAX & (-1LL << v32)) != 0 )
      {
        LOBYTE(v10) = v28 > v32;
        return v10;
      }
    }
    return 0;
  }
  v11 = *(_DWORD *)(v9 + 64);
  if ( !v11 )
    return v10;
  v12 = (unsigned int)(v11 - 1) >> 6;
  v13 = *(_QWORD *)v9;
  v14 = 0;
  while ( 1 )
  {
    _RDX = *(_QWORD *)(v13 + 8 * v14);
    if ( (unsigned int)(v11 - 1) >> 6 == v14 )
      break;
    if ( _RDX )
      goto LABEL_11;
    if ( v12 + 1 == ++v14 )
      return v10;
  }
  _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
  if ( !_RDX )
    return v10;
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
            if ( v12 == (_DWORD)v22 )
              _RAX &= v24;
            if ( _RAX )
              break;
            if ( v12 < (unsigned int)++v22 )
              return v10;
          }
          __asm { tzcnt   rax, rax }
          LOBYTE(v10) = ((_DWORD)v22 << 6) + (_DWORD)_RAX != -1;
        }
      }
    }
    else
    {
      return 1;
    }
  }
  return v10;
}
