// Function: sub_13C40F0
// Address: 0x13c40f0
//
__int64 __fastcall sub_13C40F0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned __int64 v5; // rax
  __int64 v6; // r14
  __int64 *v7; // r14
  char v8; // al
  char v9; // dl
  unsigned __int64 v10; // rcx
  char v11; // al
  char v13; // r8
  unsigned __int64 v14; // rdi
  int v15; // r10d
  unsigned int v16; // r9d
  unsigned __int64 v17; // rsi
  __int64 v18; // r11
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // rsi
  int v22; // esi
  int v23; // r12d

  v3 = sub_14AD280(*a3, *(_QWORD *)(a1 + 8), 6);
  if ( *(_BYTE *)(v3 + 16) > 3u )
    return 7;
  v4 = v3;
  if ( (*(_BYTE *)(v3 + 32) & 0xFu) - 7 > 1 )
    return 7;
  v5 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  if ( (a2 & 4) != 0 )
    v5 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v6 = *(_QWORD *)v5;
  if ( *(_BYTE *)(*(_QWORD *)v5 + 16LL) )
    return 7;
  if ( !sub_13C3050(a1 + 24, v4) )
    return 7;
  v7 = sub_13C1210(a1, v6);
  if ( !v7 )
    return 7;
  v8 = sub_13C34D0(a1, a2, v4);
  v9 = ((*v7 & 4) != 0) + 4;
  v10 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 )
  {
    v13 = *(_BYTE *)(v10 + 8) & 1;
    if ( v13 )
    {
      v14 = v10 + 16;
      v15 = 15;
    }
    else
    {
      v20 = *(unsigned int *)(v10 + 24);
      v14 = *(_QWORD *)(v10 + 16);
      if ( !(_DWORD)v20 )
        goto LABEL_22;
      v15 = v20 - 1;
    }
    v16 = v15 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v17 = v14 + 16LL * v16;
    v18 = *(_QWORD *)v17;
    if ( v4 == *(_QWORD *)v17 )
    {
LABEL_15:
      v19 = 256;
      if ( !v13 )
        v19 = 16LL * *(unsigned int *)(v10 + 24);
      if ( v17 != v19 + v14 )
        v9 |= *(_BYTE *)(v17 + 8);
      goto LABEL_9;
    }
    v22 = 1;
    while ( v18 != -8 )
    {
      v23 = v22 + 1;
      v16 = v15 & (v22 + v16);
      v17 = v14 + 16LL * v16;
      v18 = *(_QWORD *)v17;
      if ( v4 == *(_QWORD *)v17 )
        goto LABEL_15;
      v22 = v23;
    }
    if ( v13 )
    {
      v21 = 256;
      goto LABEL_23;
    }
    v20 = *(unsigned int *)(v10 + 24);
LABEL_22:
    v21 = 16 * v20;
LABEL_23:
    v17 = v14 + v21;
    goto LABEL_15;
  }
LABEL_9:
  v11 = v9 | v8;
  if ( (v11 & 3) != 0 )
    return v11 & 7;
  else
    return 4;
}
