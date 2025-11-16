// Function: sub_13AE060
// Address: 0x13ae060
//
__int64 __fastcall sub_13AE060(__int64 a1, __int64 *a2, __int64 *a3, unsigned __int64 *a4, _QWORD *a5, _BYTE *a6)
{
  unsigned __int64 v9; // r11
  __int64 v10; // r14
  int *v13; // rcx
  int v14; // eax
  unsigned __int64 v15; // rax
  int v17; // ecx
  __int64 v18; // r9
  unsigned int v19; // edi
  unsigned __int64 v20; // r8
  __int64 v21; // rdx
  int v24; // edx
  unsigned int v25; // r9d
  unsigned int v26; // esi
  __int64 v27; // r10
  int v28; // ecx
  unsigned __int64 v29; // r8
  __int64 v30; // rdx
  unsigned __int64 v31; // r8

  v9 = *a4;
  v10 = *a4 & 1;
  if ( (*a4 & 1) != 0 )
  {
    if ( !((v9 >> 1) & ~(-1LL << (v9 >> 58))) )
    {
      LODWORD(v10) = 0;
      return (unsigned int)v10;
    }
    __asm { tzcnt   r15, r15 }
    goto LABEL_4;
  }
  v17 = *(_DWORD *)(v9 + 16);
  if ( v17 )
  {
    v18 = *(_QWORD *)v9;
    v19 = (unsigned int)(v17 - 1) >> 6;
    v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
    v21 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v18 + 8 * v21);
      if ( v19 == (_DWORD)v21 )
        _RCX = v20 & *(_QWORD *)(v18 + 8 * v21);
      if ( _RCX )
        break;
      if ( v19 + 1 == ++v21 )
        return (unsigned int)v10;
    }
    __asm { tzcnt   rcx, rcx }
    LODWORD(_R15) = ((_DWORD)v21 << 6) + _RCX;
    if ( (_DWORD)_R15 != -1 )
    {
LABEL_4:
      LODWORD(v10) = 0;
      do
      {
        while ( 1 )
        {
          v13 = (int *)(*a5 + 48LL * (unsigned int)_R15);
          v14 = *v13;
          if ( *v13 == 2 )
          {
            LODWORD(v10) = sub_13AD7D0(a1, a2, a3, (__int64)v13, a6) | v10;
            v9 = *a4;
          }
          else if ( v14 == 3 )
          {
            LODWORD(v10) = sub_13AD8F0(a1, a2, a3, (__int64)v13, a6) | v10;
            v9 = *a4;
          }
          else if ( v14 == 1 )
          {
            LODWORD(v10) = sub_13ADEB0(a1, a2, a3, (__int64)v13) | v10;
            v9 = *a4;
          }
          v15 = (unsigned int)(_R15 + 1);
          if ( (v9 & 1) == 0 )
            break;
          _R15 = (-1LL << ((unsigned __int8)_R15 + 1)) & (v9 >> 1) & ~(-1LL << (v9 >> 58));
          if ( !_R15 || v9 >> 58 <= v15 )
            return (unsigned int)v10;
          __asm { tzcnt   r15, r15 }
        }
        v24 = *(_DWORD *)(v9 + 16);
        if ( v24 == (_DWORD)v15 )
          break;
        v25 = (unsigned int)v15 >> 6;
        v26 = (unsigned int)(v24 - 1) >> 6;
        if ( (unsigned int)v15 >> 6 > v26 )
          break;
        v27 = *(_QWORD *)v9;
        v28 = 64 - (((_BYTE)_R15 + 1) & 0x3F);
        v29 = 0xFFFFFFFFFFFFFFFFLL >> v28;
        v30 = v25;
        if ( v28 == 64 )
          v29 = 0;
        v31 = ~v29;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v27 + 8 * v30);
          if ( v25 == (_DWORD)v30 )
            _RAX = v31 & *(_QWORD *)(v27 + 8 * v30);
          if ( v26 == (_DWORD)v30 )
            _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*(_DWORD *)(v9 + 16);
          if ( _RAX )
            break;
          if ( v26 < (unsigned int)++v30 )
            return (unsigned int)v10;
        }
        __asm { tzcnt   rax, rax }
        LODWORD(_R15) = ((_DWORD)v30 << 6) + _RAX;
      }
      while ( (_DWORD)_R15 != -1 );
    }
  }
  return (unsigned int)v10;
}
