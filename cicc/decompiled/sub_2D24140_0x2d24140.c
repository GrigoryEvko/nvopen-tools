// Function: sub_2D24140
// Address: 0x2d24140
//
__int64 __fastcall sub_2D24140(__int64 **a1, __int64 *a2, __int64 *a3)
{
  unsigned int v3; // eax
  __int64 *v4; // r11
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  int v11; // edx
  unsigned int v12; // eax
  unsigned int v13; // edi
  unsigned int v14; // esi
  int v15; // ecx
  __int64 v16; // r8
  unsigned __int64 v17; // r10
  __int64 v18; // rdx
  unsigned __int64 v19; // r10

  v3 = *((_DWORD *)a1 + 2);
  v4 = *a1;
  if ( v3 == -1 )
    return 1;
  v5 = *a2;
  v6 = *a3;
  do
  {
    v7 = 24LL * v3;
    v8 = v5 + v7;
    v9 = v6 + v7;
    if ( *(_DWORD *)v8 != *(_DWORD *)v9 || *(_QWORD *)(v8 + 8) != *(_QWORD *)(v9 + 8) )
      return 0;
    v11 = *((_DWORD *)v4 + 16);
    v12 = v3 + 1;
    if ( v11 == v12 )
      break;
    v13 = v12 >> 6;
    v14 = (unsigned int)(v11 - 1) >> 6;
    if ( v12 >> 6 > v14 )
      break;
    v15 = 64 - (v12 & 0x3F);
    v16 = *v4;
    v17 = 0xFFFFFFFFFFFFFFFFLL >> v15;
    v18 = v13;
    if ( v15 == 64 )
      v17 = 0;
    v19 = ~v17;
    while ( 1 )
    {
      _RAX = *(_QWORD *)(v16 + 8 * v18);
      if ( v13 == (_DWORD)v18 )
        _RAX = v19 & *(_QWORD *)(v16 + 8 * v18);
      if ( v14 == (_DWORD)v18 )
        _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*((_DWORD *)v4 + 16);
      if ( _RAX )
        break;
      if ( v14 < (unsigned int)++v18 )
        return 1;
    }
    __asm { tzcnt   rax, rax }
    v3 = ((_DWORD)v18 << 6) + _RAX;
  }
  while ( v3 != -1 );
  return 1;
}
