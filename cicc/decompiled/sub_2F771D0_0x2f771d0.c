// Function: sub_2F771D0
// Address: 0x2f771d0
//
char __fastcall sub_2F771D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int16 v9; // dx
  __int64 v10; // rdx
  unsigned __int64 v11; // r12
  bool v12; // zf
  __int16 v13; // dx
  __int64 v14; // r8
  unsigned __int64 i; // rdx
  __int64 j; // rsi
  __int16 v17; // ax
  unsigned int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r9
  int v23; // eax
  int v24; // r10d

  if ( sub_2F753D0(a1) )
  {
    if ( *(_BYTE *)(a1 + 56) )
      goto LABEL_3;
  }
  else
  {
    sub_2F75730(a1, a2, v3, v4, v5, v6);
    if ( *(_BYTE *)(a1 + 56) )
      goto LABEL_3;
  }
  if ( sub_2F753A0(a1) )
    sub_2F751B0(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 64));
LABEL_3:
  v7 = **(_QWORD **)(a1 + 64) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v7 )
LABEL_43:
    BUG();
  v8 = *(_QWORD *)v7;
  if ( (*(_QWORD *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v7 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
        break;
      v8 = *(_QWORD *)v7;
    }
  }
  while ( *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL) != v7 )
  {
    v9 = *(_WORD *)(v7 + 68);
    if ( (unsigned __int16)(v9 - 14) > 4u && v9 != 24 )
      break;
    v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v7 )
      goto LABEL_43;
    v10 = *(_QWORD *)v7;
    if ( (*(_QWORD *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v7 = v10 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
          break;
        v10 = *(_QWORD *)v7;
      }
    }
  }
  v11 = 0;
  v12 = *(_BYTE *)(a1 + 56) == 0;
  *(_QWORD *)(a1 + 64) = v7;
  if ( !v12 )
  {
    v13 = *(_WORD *)(v7 + 68);
    if ( (unsigned __int16)(v13 - 14) <= 4u || v13 == 24 )
    {
LABEL_16:
      LOBYTE(v7) = sub_2F753A0(a1);
      if ( (_BYTE)v7 )
        LOBYTE(v7) = sub_2F75170(*(_QWORD *)(a1 + 48), v11);
      return v7;
    }
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
    for ( i = v7; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    for ( ; (*(_BYTE *)(v7 + 44) & 8) != 0; v7 = *(_QWORD *)(v7 + 8) )
      ;
    for ( j = *(_QWORD *)(v7 + 8); j != i; i = *(_QWORD *)(i + 8) )
    {
      v17 = *(_WORD *)(i + 68);
      if ( (unsigned __int16)(v17 - 14) > 4u && v17 != 24 )
        break;
    }
    v18 = *(_DWORD *)(v14 + 144);
    v19 = *(_QWORD *)(v14 + 128);
    if ( v18 )
    {
      v20 = (v18 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( *v21 == i )
      {
LABEL_37:
        v11 = v21[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_16;
      }
      v23 = 1;
      while ( v22 != -4096 )
      {
        v24 = v23 + 1;
        v20 = (v18 - 1) & (v23 + v20);
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == i )
          goto LABEL_37;
        v23 = v24;
      }
    }
    v21 = (__int64 *)(v19 + 16LL * v18);
    goto LABEL_37;
  }
  return v7;
}
