// Function: sub_2AB3010
// Address: 0x2ab3010
//
__int64 __fastcall sub_2AB3010(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  int v3; // ecx
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r10
  unsigned int v7; // ebx
  __int64 *v8; // r8
  __int64 v9; // r11
  __int64 v11; // r10
  __int64 v12; // r11
  int v13; // r13d
  unsigned int i; // edx
  __int64 v15; // r8
  unsigned int v16; // edx
  int v17; // eax
  __int64 v18; // rcx
  int v19; // eax
  unsigned int v20; // r9d
  __int64 v21; // rdx
  int v22; // r8d
  int v23; // r8d
  int v24; // r13d

  v3 = a3;
  v4 = HIDWORD(a3);
  if ( BYTE4(a3) )
  {
    if ( !(_DWORD)a3 )
      return 0;
  }
  else if ( (unsigned int)a3 <= 1 )
  {
    return 0;
  }
  v5 = *(unsigned int *)(a1 + 40);
  v6 = *(_QWORD *)(a1 + 24);
  if ( !(_DWORD)v5 )
    return 0;
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v23 = 1;
    while ( v9 != -4096 )
    {
      v24 = v23 + 1;
      v7 = (v5 - 1) & (v23 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_5;
      v23 = v24;
    }
    return 0;
  }
LABEL_5:
  if ( v8 == (__int64 *)(v6 + 16 * v5) )
    return 0;
  v11 = *(unsigned int *)(a1 + 152);
  v12 = *(_QWORD *)(a1 + 136);
  if ( (_DWORD)v11 )
  {
    v13 = 1;
    for ( i = (v11 - 1) & (((_BYTE)v4 == 0) + 37 * v3 - 1); ; i = (v11 - 1) & v16 )
    {
      v15 = v12 + 40LL * i;
      if ( v3 == *(_DWORD *)v15 && (_BYTE)v4 == *(_BYTE *)(v15 + 4) )
        break;
      if ( *(_DWORD *)v15 == -1 && *(_BYTE *)(v15 + 4) )
        goto LABEL_15;
      v16 = v13 + i;
      ++v13;
    }
  }
  else
  {
LABEL_15:
    v15 = v12 + 40 * v11;
  }
  v17 = *(_DWORD *)(v15 + 32);
  v18 = *(_QWORD *)(v15 + 16);
  if ( v17 )
  {
    v19 = v17 - 1;
    v20 = v19 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v21 = *(_QWORD *)(v18 + 24LL * v20);
    if ( v21 == a2 )
      return 0;
    v22 = 1;
    while ( v21 != -4096 )
    {
      v20 = v19 & (v22 + v20);
      v21 = *(_QWORD *)(v18 + 24LL * v20);
      if ( v21 == a2 )
        return 0;
      ++v22;
    }
  }
  return (unsigned int)sub_2AB2DA0(a1, a2, a3) ^ 1;
}
