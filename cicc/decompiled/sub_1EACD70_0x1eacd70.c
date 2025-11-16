// Function: sub_1EACD70
// Address: 0x1eacd70
//
__int64 __fastcall sub_1EACD70(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  char v8; // dl
  __int64 v9; // rdi
  int v10; // esi
  unsigned int v11; // eax
  unsigned int v13; // esi
  unsigned int v14; // eax
  _QWORD *v15; // r10
  int v16; // ecx
  unsigned int v17; // edi
  __int64 v18; // rax
  int v19; // r11d
  __int64 v20; // rsi
  int v21; // edx
  unsigned int v22; // eax
  __int64 v23; // rdi
  __int64 v24; // rsi
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rdi
  int v28; // edx
  int v29; // edx

  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( v8 )
  {
    v9 = a1 + 16;
    v10 = 15;
  }
  else
  {
    v13 = *(_DWORD *)(a1 + 24);
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v13 )
    {
      v14 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v15 = 0;
      v16 = (v14 >> 1) + 1;
LABEL_8:
      v17 = 3 * v13;
      goto LABEL_9;
    }
    v10 = v13 - 1;
  }
  v11 = v10 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  a6 = (_QWORD *)(v9 + 8LL * v11);
  a5 = (_QWORD *)*a6;
  if ( *a2 == *a6 )
    return 0;
  v19 = 1;
  v15 = 0;
  while ( a5 != (_QWORD *)-8LL )
  {
    if ( v15 || a5 != (_QWORD *)-16LL )
      a6 = v15;
    v11 = v10 & (v19 + v11);
    a5 = *(_QWORD **)(v9 + 8LL * v11);
    if ( (_QWORD *)*a2 == a5 )
      return 0;
    ++v19;
    v15 = a6;
    a6 = (_QWORD *)(v9 + 8LL * v11);
  }
  v14 = *(_DWORD *)(a1 + 8);
  if ( !v15 )
    v15 = a6;
  ++*(_QWORD *)a1;
  v16 = (v14 >> 1) + 1;
  if ( !v8 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v17 = 48;
  v13 = 16;
LABEL_9:
  if ( 4 * v16 >= v17 )
  {
    sub_1EAC9B0(a1, 2 * v13);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v20 = a1 + 16;
      v21 = 15;
    }
    else
    {
      v28 = *(_DWORD *)(a1 + 24);
      v20 = *(_QWORD *)(a1 + 16);
      if ( !v28 )
        goto LABEL_55;
      v21 = v28 - 1;
    }
    v22 = v21 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v15 = (_QWORD *)(v20 + 8LL * v22);
    v23 = *v15;
    if ( *v15 != *a2 )
    {
      LODWORD(a6) = 1;
      a5 = 0;
      while ( v23 != -8 )
      {
        if ( v23 == -16 && !a5 )
          a5 = v15;
        v22 = v21 & ((_DWORD)a6 + v22);
        v15 = (_QWORD *)(v20 + 8LL * v22);
        v23 = *v15;
        if ( *a2 == *v15 )
          goto LABEL_25;
        LODWORD(a6) = (_DWORD)a6 + 1;
      }
      goto LABEL_31;
    }
LABEL_25:
    v14 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v13 - *(_DWORD *)(a1 + 12) - v16 <= v13 >> 3 )
  {
    sub_1EAC9B0(a1, v13);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v24 = a1 + 16;
      v25 = 15;
      goto LABEL_28;
    }
    v29 = *(_DWORD *)(a1 + 24);
    v24 = *(_QWORD *)(a1 + 16);
    if ( v29 )
    {
      v25 = v29 - 1;
LABEL_28:
      v26 = v25 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v15 = (_QWORD *)(v24 + 8LL * v26);
      v27 = *v15;
      if ( *v15 != *a2 )
      {
        LODWORD(a6) = 1;
        a5 = 0;
        while ( v27 != -8 )
        {
          if ( v27 == -16 && !a5 )
            a5 = v15;
          v26 = v25 & ((_DWORD)a6 + v26);
          v15 = (_QWORD *)(v24 + 8LL * v26);
          v27 = *v15;
          if ( *a2 == *v15 )
            goto LABEL_25;
          LODWORD(a6) = (_DWORD)a6 + 1;
        }
LABEL_31:
        if ( a5 )
          v15 = a5;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_55:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *v15 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v15 = *a2;
  v18 = *(unsigned int *)(a1 + 152);
  if ( (unsigned int)v18 >= *(_DWORD *)(a1 + 156) )
  {
    sub_16CD150(a1 + 144, (const void *)(a1 + 160), 0, 8, (int)a5, (int)a6);
    v18 = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * v18) = *a2;
  ++*(_DWORD *)(a1 + 152);
  return 1;
}
