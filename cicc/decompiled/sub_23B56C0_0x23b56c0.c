// Function: sub_23B56C0
// Address: 0x23b56c0
//
__int64 __fastcall sub_23B56C0(__int64 a1, __int64 a2)
{
  int v2; // edx
  unsigned int v4; // r8d
  _QWORD *v7; // rsi
  _QWORD *v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // r10
  __int64 v12; // r9
  int v13; // r11d
  unsigned int v14; // ebx
  __int64 *v15; // rsi
  __int64 v16; // r12
  int v17; // esi
  int v18; // r13d

  v2 = *(_DWORD *)(a1 + 16);
  v4 = 0;
  if ( v2 != *(_DWORD *)(a2 + 16) )
    return v4;
  if ( !v2 )
    return 1;
  v7 = *(_QWORD **)(a1 + 8);
  v8 = &v7[2 * *(unsigned int *)(a1 + 24)];
  if ( v7 == v8 )
    return 1;
  while ( 1 )
  {
    v9 = *v7;
    v10 = v7;
    LOBYTE(v4) = *v7 == -4096 || *v7 == -8192;
    if ( !(_BYTE)v4 )
      break;
    v7 += 2;
    if ( v8 == v7 )
      return 1;
  }
  if ( v8 == v7 )
    return 1;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = *(unsigned int *)(a2 + 24);
  v13 = v12 - 1;
  if ( !(_DWORD)v12 )
    return v4;
  while ( 1 )
  {
    v14 = v13 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( *v15 != v9 )
      break;
LABEL_12:
    if ( (__int64 *)(v11 + 16 * v12) == v15 || *((_DWORD *)v15 + 2) != *((_DWORD *)v10 + 2) )
      return v4;
    v10 += 2;
    if ( v10 != v8 )
    {
      while ( 1 )
      {
        v9 = *v10;
        if ( *v10 != -4096 && v9 != -8192 )
          break;
        v10 += 2;
        if ( v8 == v10 )
          return 1;
      }
      if ( v8 != v10 )
        continue;
    }
    return 1;
  }
  v17 = 1;
  while ( v16 != -4096 )
  {
    v18 = v17 + 1;
    v14 = v13 & (v17 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( *v15 == v9 )
      goto LABEL_12;
    v17 = v18;
  }
  return v4;
}
