// Function: sub_2AB7850
// Address: 0x2ab7850
//
__int64 __fastcall sub_2AB7850(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // r13
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r14
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi

  if ( *(_BYTE *)(a1 + 860) )
  {
    if ( *(_BYTE *)(a1 + 700) )
      goto LABEL_3;
LABEL_69:
    _libc_free(*(_QWORD *)(a1 + 680));
    if ( *(_BYTE *)(a1 + 540) )
      goto LABEL_4;
LABEL_70:
    _libc_free(*(_QWORD *)(a1 + 520));
    goto LABEL_4;
  }
  _libc_free(*(_QWORD *)(a1 + 840));
  if ( !*(_BYTE *)(a1 + 700) )
    goto LABEL_69;
LABEL_3:
  if ( !*(_BYTE *)(a1 + 540) )
    goto LABEL_70;
LABEL_4:
  sub_C7D6A0(*(_QWORD *)(a1 + 392), (unsigned __int64)*(unsigned int *)(a1 + 408) << 6, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 360), 40LL * *(unsigned int *)(a1 + 376), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 328), 16LL * *(unsigned int *)(a1 + 344), 8);
  if ( !*(_BYTE *)(a1 + 284) )
    _libc_free(*(_QWORD *)(a1 + 264));
  v2 = *(unsigned int *)(a1 + 248);
  if ( (_DWORD)v2 )
  {
    v3 = *(_QWORD *)(a1 + 232);
    v4 = v3 + 72 * v2;
    while ( 1 )
    {
      while ( *(_DWORD *)v3 != -1 )
      {
        if ( (*(_DWORD *)v3 != -2 || *(_BYTE *)(v3 + 4)) && !*(_BYTE *)(v3 + 36) )
          goto LABEL_14;
LABEL_10:
        v3 += 72;
        if ( v4 == v3 )
          goto LABEL_15;
      }
      if ( *(_BYTE *)(v3 + 4) || *(_BYTE *)(v3 + 36) )
        goto LABEL_10;
LABEL_14:
      v5 = *(_QWORD *)(v3 + 16);
      v3 += 72;
      _libc_free(v5);
      if ( v4 == v3 )
      {
LABEL_15:
        v2 = *(unsigned int *)(a1 + 248);
        break;
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 72 * v2, 8);
  v6 = *(unsigned int *)(a1 + 216);
  if ( !(_DWORD)v6 )
    goto LABEL_26;
  v7 = *(_QWORD *)(a1 + 200);
  v8 = v7 + 72 * v6;
  do
  {
    while ( *(_DWORD *)v7 == -1 )
    {
      if ( !*(_BYTE *)(v7 + 4) && !*(_BYTE *)(v7 + 36) )
        goto LABEL_24;
LABEL_20:
      v7 += 72;
      if ( v8 == v7 )
        goto LABEL_25;
    }
    if ( *(_DWORD *)v7 == -2 && !*(_BYTE *)(v7 + 4) || *(_BYTE *)(v7 + 36) )
      goto LABEL_20;
LABEL_24:
    v9 = *(_QWORD *)(v7 + 16);
    v7 += 72;
    _libc_free(v9);
  }
  while ( v8 != v7 );
LABEL_25:
  v6 = *(unsigned int *)(a1 + 216);
LABEL_26:
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 72 * v6, 8);
  v10 = *(unsigned int *)(a1 + 184);
  if ( !(_DWORD)v10 )
    goto LABEL_36;
  v11 = *(_QWORD *)(a1 + 168);
  v12 = v11 + 72 * v10;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v11 == -1 )
      {
        if ( !*(_BYTE *)(v11 + 4) && !*(_BYTE *)(v11 + 36) )
          break;
        goto LABEL_30;
      }
      if ( *(_DWORD *)v11 == -2 && !*(_BYTE *)(v11 + 4) || *(_BYTE *)(v11 + 36) )
      {
LABEL_30:
        v11 += 72;
        if ( v12 == v11 )
          goto LABEL_35;
        continue;
      }
      break;
    }
    v13 = *(_QWORD *)(v11 + 16);
    v11 += 72;
    _libc_free(v13);
    if ( v12 != v11 )
      continue;
    break;
  }
LABEL_35:
  v10 = *(unsigned int *)(a1 + 184);
LABEL_36:
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 72 * v10, 8);
  v14 = *(unsigned int *)(a1 + 152);
  if ( !(_DWORD)v14 )
    goto LABEL_45;
  v15 = *(_QWORD *)(a1 + 136);
  v16 = v15 + 40 * v14;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v15 != -1 )
      {
        if ( *(_DWORD *)v15 == -2 && !*(_BYTE *)(v15 + 4) )
        {
LABEL_40:
          v15 += 40;
          if ( v16 == v15 )
            goto LABEL_44;
          continue;
        }
LABEL_39:
        sub_C7D6A0(*(_QWORD *)(v15 + 16), 24LL * *(unsigned int *)(v15 + 32), 8);
        goto LABEL_40;
      }
      break;
    }
    if ( !*(_BYTE *)(v15 + 4) )
      goto LABEL_39;
    v15 += 40;
    if ( v16 != v15 )
      continue;
    break;
  }
LABEL_44:
  v14 = *(unsigned int *)(a1 + 152);
LABEL_45:
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 40 * v14, 8);
  v17 = *(unsigned int *)(a1 + 88);
  if ( !(_DWORD)v17 )
    goto LABEL_55;
  v18 = *(_QWORD *)(a1 + 72);
  v19 = v18 + 72 * v17;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v18 == -1 )
      {
        if ( !*(_BYTE *)(v18 + 4) && !*(_BYTE *)(v18 + 36) )
          break;
        goto LABEL_49;
      }
      if ( *(_DWORD *)v18 == -2 && !*(_BYTE *)(v18 + 4) || *(_BYTE *)(v18 + 36) )
      {
LABEL_49:
        v18 += 72;
        if ( v19 == v18 )
          goto LABEL_54;
        continue;
      }
      break;
    }
    v20 = *(_QWORD *)(v18 + 16);
    v18 += 72;
    _libc_free(v20);
    if ( v19 != v18 )
      continue;
    break;
  }
LABEL_54:
  v17 = *(unsigned int *)(a1 + 88);
LABEL_55:
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 72 * v17, 8);
  v21 = *(_QWORD *)(a1 + 48);
  if ( a1 + 64 != v21 )
    _libc_free(v21);
  return sub_C7D6A0(*(_QWORD *)(a1 + 24), 16LL * *(unsigned int *)(a1 + 40), 8);
}
