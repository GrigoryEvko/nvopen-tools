// Function: sub_386B3C0
// Address: 0x386b3c0
//
_QWORD *__fastcall sub_386B3C0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  __int64 v4; // r9
  __int64 v5; // rcx
  unsigned int v6; // r11d
  __int64 *v7; // rdx
  __int64 v8; // r10
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // edi
  __int64 *v16; // rdx
  __int64 v17; // r8
  _QWORD *v18; // rax
  int v19; // edx
  int v20; // r12d
  int v21; // edx
  int v22; // r11d

  v2 = *a1;
  v3 = *(unsigned int *)(*a1 + 112);
  if ( !(_DWORD)v3 )
    return 0;
  v4 = *(_QWORD *)(a2 + 64);
  v5 = *(_QWORD *)(v2 + 96);
  v6 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( v4 != *v7 )
  {
    v19 = 1;
    while ( v8 != -8 )
    {
      v20 = v19 + 1;
      v6 = (v3 - 1) & (v19 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( v4 == *v7 )
        goto LABEL_3;
      v19 = v20;
    }
    return 0;
  }
LABEL_3:
  if ( v7 == (__int64 *)(v5 + 16 * v3) )
    return 0;
  v9 = v7[1];
  if ( !v9 )
    return 0;
  if ( *(_BYTE *)(a2 + 16) != 21 )
  {
    v10 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 != v10 && v10 )
      return (_QWORD *)(v10 - 48);
    return 0;
  }
  v12 = *(unsigned int *)(v2 + 80);
  v13 = 0;
  if ( (_DWORD)v12 )
  {
    v14 = *(_QWORD *)(v2 + 64);
    v15 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v16 = (__int64 *)(v14 + 16LL * v15);
    v17 = *v16;
    if ( v4 == *v16 )
    {
LABEL_12:
      if ( v16 != (__int64 *)(v14 + 16 * v12) )
      {
        v13 = (_QWORD *)v16[1];
        goto LABEL_14;
      }
    }
    else
    {
      v21 = 1;
      while ( v17 != -8 )
      {
        v22 = v21 + 1;
        v15 = (v12 - 1) & (v21 + v15);
        v16 = (__int64 *)(v14 + 16LL * v15);
        v17 = *v16;
        if ( v4 == *v16 )
          goto LABEL_12;
        v21 = v22;
      }
    }
    v13 = 0;
  }
LABEL_14:
  v18 = (_QWORD *)(*(_QWORD *)(a2 + 32) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v13 == v18 )
    return 0;
  while ( 1 )
  {
    if ( !v18 )
      BUG();
    if ( *((_BYTE *)v18 - 16) != 21 )
      break;
    v18 = (_QWORD *)(*v18 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v13 == v18 )
      return 0;
  }
  return v18 - 4;
}
