// Function: sub_2FAFD80
// Address: 0x2fafd80
//
bool __fastcall sub_2FAFD80(__int64 a1)
{
  __int64 *v1; // r14
  int v2; // ecx
  unsigned int v4; // edi
  __int64 v5; // r10
  unsigned __int64 v6; // r8
  __int64 v7; // rdx
  unsigned int v11; // ebx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // rdx
  bool v16; // cf
  unsigned __int64 v17; // rdx
  int v18; // edx
  unsigned int v19; // eax
  unsigned int v20; // r9d
  unsigned int v21; // esi
  int v22; // ecx
  __int64 v23; // r11
  unsigned __int64 v24; // r8
  __int64 v25; // rdx
  unsigned __int64 v26; // r8
  __int64 v29; // rax

  v1 = *(__int64 **)(a1 + 32);
  *(_DWORD *)(a1 + 96) = 0;
  v2 = *((_DWORD *)v1 + 16);
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
    sub_2FAFB50(a1, v11);
    v14 = *(_QWORD *)(a1 + 24) + 112LL * v11;
    v15 = *(_QWORD *)(v14 + 104);
    v16 = __CFADD__(*(_QWORD *)(v14 + 8), v15);
    v17 = *(_QWORD *)(v14 + 8) + v15;
    if ( v16 )
      v17 = -1;
    if ( *(_QWORD *)v14 < v17 && *(int *)(v14 + 16) > 0 )
    {
      v29 = *(unsigned int *)(a1 + 96);
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
      {
        sub_C8D5F0(a1 + 88, (const void *)(a1 + 104), v29 + 1, 4u, v12, v13);
        v29 = *(unsigned int *)(a1 + 96);
      }
      *(_DWORD *)(*(_QWORD *)(a1 + 88) + 4 * v29) = v11;
      ++*(_DWORD *)(a1 + 96);
    }
    v18 = *((_DWORD *)v1 + 16);
    v19 = v11 + 1;
    if ( v18 == v11 + 1 )
      break;
    v20 = v19 >> 6;
    v21 = (unsigned int)(v18 - 1) >> 6;
    if ( v19 >> 6 > v21 )
      break;
    v22 = 64 - (v19 & 0x3F);
    v23 = *v1;
    v24 = 0xFFFFFFFFFFFFFFFFLL >> v22;
    v25 = v20;
    if ( v22 == 64 )
      v24 = 0;
    v26 = ~v24;
    while ( 1 )
    {
      _RAX = *(_QWORD *)(v23 + 8 * v25);
      if ( v20 == (_DWORD)v25 )
        _RAX = v26 & *(_QWORD *)(v23 + 8 * v25);
      if ( v21 == (_DWORD)v25 )
        _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)*((_DWORD *)v1 + 16);
      if ( _RAX )
        break;
      if ( v21 < (unsigned int)++v25 )
        return *(_DWORD *)(a1 + 96) != 0;
    }
    __asm { tzcnt   rax, rax }
    v11 = ((_DWORD)v25 << 6) + _RAX;
  }
  while ( v11 != -1 );
  return *(_DWORD *)(a1 + 96) != 0;
}
