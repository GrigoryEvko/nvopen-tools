// Function: sub_1DD42F0
// Address: 0x1dd42f0
//
__int64 __fastcall sub_1DD42F0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4, _DWORD *a5)
{
  __int64 result; // rax
  __int64 v8; // r9
  int v9; // esi
  unsigned int v10; // ecx
  int v11; // edi
  unsigned int v12; // esi
  unsigned int v13; // edx
  _DWORD *v14; // r10
  int v15; // ecx
  unsigned int v16; // edi
  int v17; // r11d
  __int64 v18; // rsi
  int v19; // eax
  unsigned int v20; // edx
  int v21; // edi
  __int64 v22; // rsi
  int v23; // ecx
  unsigned int v24; // eax
  int v25; // edi
  int v26; // eax
  int v27; // ecx

  result = *(_BYTE *)(a1 + 8) & 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v8 = a1 + 16;
    v9 = 7;
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 24);
    v8 = *(_QWORD *)(a1 + 16);
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
    v9 = v12 - 1;
  }
  v10 = v9 & (37 * *a2);
  a5 = (_DWORD *)(v8 + 4LL * v10);
  v11 = *a5;
  if ( *a2 == *a5 )
    return result;
  v17 = 1;
  v14 = 0;
  while ( v11 != 0x7FFFFFFF )
  {
    if ( v14 || v11 != 0x80000000 )
      a5 = v14;
    v10 = v9 & (v17 + v10);
    v11 = *(_DWORD *)(v8 + 4LL * v10);
    if ( *a2 == v11 )
      return result;
    ++v17;
    v14 = a5;
    a5 = (_DWORD *)(v8 + 4LL * v10);
  }
  v13 = *(_DWORD *)(a1 + 8);
  if ( !v14 )
    v14 = a5;
  ++*(_QWORD *)a1;
  v15 = (v13 >> 1) + 1;
  if ( !(_BYTE)result )
  {
    v12 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v16 = 24;
  v12 = 8;
LABEL_9:
  if ( 4 * v15 >= v16 )
  {
    sub_1DD3F40(a1, 2 * v12);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v18 = a1 + 16;
      v19 = 7;
    }
    else
    {
      v26 = *(_DWORD *)(a1 + 24);
      v18 = *(_QWORD *)(a1 + 16);
      if ( !v26 )
        goto LABEL_55;
      v19 = v26 - 1;
    }
    v20 = v19 & (37 * *a2);
    v14 = (_DWORD *)(v18 + 4LL * v20);
    v21 = *v14;
    if ( *v14 != *a2 )
    {
      LODWORD(v8) = 1;
      a5 = 0;
      while ( v21 != 0x7FFFFFFF )
      {
        if ( v21 == 0x80000000 && !a5 )
          a5 = v14;
        v20 = v19 & (v8 + v20);
        v14 = (_DWORD *)(v18 + 4LL * v20);
        v21 = *v14;
        if ( *a2 == *v14 )
          goto LABEL_25;
        LODWORD(v8) = v8 + 1;
      }
      goto LABEL_31;
    }
LABEL_25:
    v13 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v12 - *(_DWORD *)(a1 + 12) - v15 <= v12 >> 3 )
  {
    sub_1DD3F40(a1, v12);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v22 = a1 + 16;
      v23 = 7;
      goto LABEL_28;
    }
    v27 = *(_DWORD *)(a1 + 24);
    v22 = *(_QWORD *)(a1 + 16);
    if ( v27 )
    {
      v23 = v27 - 1;
LABEL_28:
      v24 = v23 & (37 * *a2);
      v14 = (_DWORD *)(v22 + 4LL * v24);
      v25 = *v14;
      if ( *v14 != *a2 )
      {
        LODWORD(v8) = 1;
        a5 = 0;
        while ( v25 != 0x7FFFFFFF )
        {
          if ( v25 == 0x80000000 && !a5 )
            a5 = v14;
          v24 = v23 & (v8 + v24);
          v14 = (_DWORD *)(v22 + 4LL * v24);
          v25 = *v14;
          if ( *a2 == *v14 )
            goto LABEL_25;
          LODWORD(v8) = v8 + 1;
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
  if ( *v14 != 0x7FFFFFFF )
    --*(_DWORD *)(a1 + 12);
  *v14 = *a2;
  result = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 60) )
  {
    sub_16CD150(a1 + 48, (const void *)(a1 + 64), 0, 4, (int)a5, v8);
    result = *(unsigned int *)(a1 + 56);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 48) + 4 * result) = *a2;
  ++*(_DWORD *)(a1 + 56);
  return result;
}
