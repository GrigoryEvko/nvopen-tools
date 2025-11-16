// Function: sub_1F6D610
// Address: 0x1f6d610
//
unsigned __int64 __fastcall sub_1F6D610(__int64 a1, unsigned int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 *v7; // rsi
  __int64 v8; // r9
  int v9; // eax
  __int64 v10; // r12
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v18; // rcx
  int v19; // ecx
  int v20; // r14d
  unsigned int v24; // eax
  unsigned int v25; // ebx
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // [rsp-38h] [rbp-38h]
  __int64 v30; // [rsp-30h] [rbp-30h]

  if ( *(_WORD *)(a1 + 24) != 118 )
    return 0;
  v7 = *(__int64 **)(a1 + 32);
  v8 = v7[5];
  v9 = *(unsigned __int16 *)(v8 + 24);
  if ( v9 != 10 && v9 != 32 )
    return 0;
  v10 = *v7;
  if ( *(_WORD *)(*v7 + 24) != 185 )
    return 0;
  if ( (*(_BYTE *)(v10 + 27) & 0xC) != 0 )
    return 0;
  if ( (*(_WORD *)(v10 + 26) & 0x380) != 0 )
    return 0;
  v11 = *(_QWORD *)(v10 + 32);
  if ( *(_QWORD *)(v11 + 40) != a3 )
    return 0;
  if ( *(_DWORD *)(v11 + 48) != a4 )
    return 0;
  v12 = *(_BYTE *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
  if ( (unsigned __int8)(v12 - 4) > 2u )
    return 0;
  v13 = *(_QWORD *)(v8 + 88);
  v14 = *(_DWORD *)(v13 + 32);
  v15 = *(__int64 **)(v13 + 24);
  v16 = v14 > 0x40 ? *v15 : (__int64)((_QWORD)v15 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
  _RAX = ~v16;
  if ( !_RAX )
    return 0;
  _BitScanReverse64(&v18, _RAX);
  v19 = v18 ^ 0x3F;
  v20 = v19;
  if ( (v19 & 7) != 0 )
    return 0;
  __asm { tzcnt   r13, rax }
  if ( (_R13 & 7) != 0 )
    return 0;
  if ( ~(_RAX >> _R13) )
  {
    __asm { tzcnt   rax, rax }
    _RAX = (int)_RAX;
  }
  else
  {
    _RAX = 64;
  }
  if ( v19 + (__int64)(int)_R13 + _RAX != 64 )
    return 0;
  if ( v12 != 6 && v19 )
  {
    v29 = a5;
    v28 = sub_1F6D5D0(a1, a2);
    a5 = v29;
    v20 = v20 + v28 - 64;
  }
  v30 = a5;
  v24 = sub_1F6D5D0(a1, a2) - _R13 - v20;
  v25 = v24 >> 3;
  if ( v24 > 0x17 )
  {
    if ( v25 == 4 )
      goto LABEL_25;
    return 0;
  }
  if ( !v25 )
    return 0;
LABEL_25:
  if ( (_DWORD)_R13 && ((unsigned int)_R13 >> 3) % v25 )
    return 0;
  if ( v10 != v30 )
  {
    if ( *(_WORD *)(v30 + 24) == 2 && sub_1D18C00(v10, 1, 1) )
    {
      v26 = *(_QWORD *)(v30 + 32);
      v27 = v26 + 40LL * *(unsigned int *)(v30 + 56);
      while ( v27 != v26 )
      {
        v26 += 40;
        if ( v10 == *(_QWORD *)(v26 - 40) )
          return ((unsigned __int64)((unsigned int)_R13 >> 3) << 32) | v25 & 0x1FFFFFFF;
      }
    }
    return 0;
  }
  return ((unsigned __int64)((unsigned int)_R13 >> 3) << 32) | v25 & 0x1FFFFFFF;
}
