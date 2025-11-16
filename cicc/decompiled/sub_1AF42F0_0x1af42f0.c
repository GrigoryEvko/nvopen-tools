// Function: sub_1AF42F0
// Address: 0x1af42f0
//
__int64 __fastcall sub_1AF42F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  char v8; // dl
  __int64 v9; // rdi
  int v10; // esi
  __int64 result; // rax
  unsigned int v12; // esi
  unsigned int v13; // eax
  _QWORD *v14; // r10
  int v15; // ecx
  unsigned int v16; // edi
  int v17; // r11d
  __int64 v18; // rsi
  int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rdi
  __int64 v22; // rsi
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // rdi
  int v26; // edx
  int v27; // edx

  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( v8 )
  {
    v9 = a1 + 16;
    v10 = 15;
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 24);
    v9 = *(_QWORD *)(a1 + 16);
    if ( !v12 )
    {
      v13 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v14 = 0;
      v15 = (v13 >> 1) + 1;
LABEL_8:
      v16 = 3 * v12;
      goto LABEL_9;
    }
    v10 = v12 - 1;
  }
  result = v10 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  a6 = (_QWORD *)(v9 + 8 * result);
  a5 = (_QWORD *)*a6;
  if ( *a2 == *a6 )
    return result;
  v17 = 1;
  v14 = 0;
  while ( a5 != (_QWORD *)-8LL )
  {
    if ( v14 || a5 != (_QWORD *)-16LL )
      a6 = v14;
    result = v10 & (unsigned int)(v17 + result);
    a5 = *(_QWORD **)(v9 + 8LL * (unsigned int)result);
    if ( (_QWORD *)*a2 == a5 )
      return result;
    ++v17;
    v14 = a6;
    a6 = (_QWORD *)(v9 + 8LL * (unsigned int)result);
  }
  v13 = *(_DWORD *)(a1 + 8);
  if ( !v14 )
    v14 = a6;
  ++*(_QWORD *)a1;
  v15 = (v13 >> 1) + 1;
  if ( !v8 )
  {
    v12 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v16 = 48;
  v12 = 16;
LABEL_9:
  if ( 4 * v15 >= v16 )
  {
    sub_18F1E30(a1, 2 * v12);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v18 = a1 + 16;
      v19 = 15;
    }
    else
    {
      v26 = *(_DWORD *)(a1 + 24);
      v18 = *(_QWORD *)(a1 + 16);
      if ( !v26 )
        goto LABEL_55;
      v19 = v26 - 1;
    }
    v20 = v19 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v14 = (_QWORD *)(v18 + 8LL * v20);
    v21 = *v14;
    if ( *v14 != *a2 )
    {
      LODWORD(a6) = 1;
      a5 = 0;
      while ( v21 != -8 )
      {
        if ( v21 == -16 && !a5 )
          a5 = v14;
        v20 = v19 & ((_DWORD)a6 + v20);
        v14 = (_QWORD *)(v18 + 8LL * v20);
        v21 = *v14;
        if ( *a2 == *v14 )
          goto LABEL_25;
        LODWORD(a6) = (_DWORD)a6 + 1;
      }
      goto LABEL_31;
    }
LABEL_25:
    v13 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v12 - *(_DWORD *)(a1 + 12) - v15 <= v12 >> 3 )
  {
    sub_18F1E30(a1, v12);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v22 = a1 + 16;
      v23 = 15;
      goto LABEL_28;
    }
    v27 = *(_DWORD *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    if ( v27 )
    {
      v23 = v27 - 1;
LABEL_28:
      v24 = v23 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v14 = (_QWORD *)(v22 + 8LL * v24);
      v25 = *v14;
      if ( *v14 != *a2 )
      {
        LODWORD(a6) = 1;
        a5 = 0;
        while ( v25 != -8 )
        {
          if ( v25 == -16 && !a5 )
            a5 = v14;
          v24 = v23 & ((_DWORD)a6 + v24);
          v14 = (_QWORD *)(v22 + 8LL * v24);
          v25 = *v14;
          if ( *a2 == *v14 )
            goto LABEL_25;
          LODWORD(a6) = (_DWORD)a6 + 1;
        }
LABEL_31:
        if ( a5 )
          v14 = a5;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_55:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v13 >> 1) + 2) | v13 & 1;
  if ( *v14 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v14 = *a2;
  result = *(unsigned int *)(a1 + 152);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 156) )
  {
    sub_16CD150(a1 + 144, (const void *)(a1 + 160), 0, 8, (int)a5, (int)a6);
    result = *(unsigned int *)(a1 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 144) + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 152);
  return result;
}
