// Function: sub_1F12DF0
// Address: 0x1f12df0
//
bool __fastcall sub_1F12DF0(__int64 a1)
{
  __int64 *v1; // r14
  int v2; // ecx
  unsigned int v4; // edi
  __int64 v5; // r10
  unsigned __int64 v6; // r8
  __int64 v7; // rdx
  unsigned int v11; // r12d
  __int64 v12; // r15
  _QWORD *v13; // rbx
  int v14; // r8d
  int v15; // r9d
  int v16; // edx
  unsigned int v17; // eax
  unsigned int v18; // r10d
  unsigned int v19; // esi
  __int64 v20; // r11
  int v21; // ecx
  unsigned __int64 v22; // r8
  __int64 v23; // rdx
  unsigned __int64 v24; // r8
  __int64 v27; // rax

  v1 = *(__int64 **)(a1 + 272);
  *(_DWORD *)(a1 + 336) = 0;
  v2 = *((_DWORD *)v1 + 4);
  if ( !v2 )
    return 0;
  v4 = (unsigned int)(v2 - 1) >> 6;
  v5 = *v1;
  v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
  v7 = 0;
  while ( 1 )
  {
    _RCX = *(_QWORD *)(v5 + 8 * v7);
    if ( v4 == (_DWORD)v7 )
      _RCX = v6 & *(_QWORD *)(v5 + 8 * v7);
    if ( _RCX )
      break;
    if ( v4 + 1 == ++v7 )
      return 0;
  }
  __asm { tzcnt   rcx, rcx }
  v11 = ((_DWORD)v7 << 6) + _RCX;
  if ( v11 == -1 )
    return 0;
  do
  {
    sub_1F12B80(a1, v11);
    v12 = 112LL * v11;
    v13 = (_QWORD *)(v12 + *(_QWORD *)(a1 + 264));
    if ( (unsigned __int64)sub_16AF590(v13 + 1, v13[13]) > *v13 && *(int *)(*(_QWORD *)(a1 + 264) + v12 + 16) > 0 )
    {
      v27 = *(unsigned int *)(a1 + 336);
      if ( (unsigned int)v27 >= *(_DWORD *)(a1 + 340) )
      {
        sub_16CD150(a1 + 328, (const void *)(a1 + 344), 0, 4, v14, v15);
        v27 = *(unsigned int *)(a1 + 336);
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 328) + 4 * v27) = v11;
      ++*(_DWORD *)(a1 + 336);
    }
    v16 = *((_DWORD *)v1 + 4);
    v17 = v11 + 1;
    if ( v16 == v11 + 1 )
      break;
    v18 = v17 >> 6;
    v19 = (unsigned int)(v16 - 1) >> 6;
    if ( v17 >> 6 > v19 )
      break;
    v20 = *v1;
    v21 = 64 - (v17 & 0x3F);
    v22 = 0xFFFFFFFFFFFFFFFFLL >> v21;
    v23 = v18;
    if ( v21 == 64 )
      v22 = 0;
    v24 = ~v22;
    while ( 1 )
    {
      _RAX = *(_QWORD *)(v20 + 8 * v23);
      if ( v18 == (_DWORD)v23 )
        _RAX = v24 & *(_QWORD *)(v20 + 8 * v23);
      if ( v19 == (_DWORD)v23 )
        _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*((_DWORD *)v1 + 4);
      if ( _RAX )
        break;
      if ( v19 < (unsigned int)++v23 )
        return *(_DWORD *)(a1 + 336) != 0;
    }
    __asm { tzcnt   rax, rax }
    v11 = ((_DWORD)v23 << 6) + _RAX;
  }
  while ( v11 != -1 );
  return *(_DWORD *)(a1 + 336) != 0;
}
