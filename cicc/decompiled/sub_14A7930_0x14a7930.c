// Function: sub_14A7930
// Address: 0x14a7930
//
__int64 __fastcall sub_14A7930(__int64 a1, _QWORD *a2)
{
  char v4; // dl
  __int64 v5; // rdi
  int v6; // esi
  __int64 result; // rax
  _QWORD *v8; // r9
  __int64 v9; // r8
  unsigned int v10; // esi
  unsigned int v11; // eax
  _QWORD *v12; // r10
  int v13; // ecx
  unsigned int v14; // edi
  int v15; // r11d
  __int64 v16; // rsi
  int v17; // edx
  unsigned int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // rsi
  int v21; // edx
  unsigned int v22; // eax
  __int64 v23; // rdi
  int v24; // r9d
  _QWORD *v25; // r8
  int v26; // edx
  int v27; // edx
  int v28; // r9d

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 3;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v10 )
    {
      v11 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v12 = 0;
      v13 = (v11 >> 1) + 1;
LABEL_8:
      v14 = 3 * v10;
      goto LABEL_9;
    }
    v6 = v10 - 1;
  }
  result = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v5 + 8 * result);
  v9 = *v8;
  if ( *a2 == *v8 )
    return result;
  v15 = 1;
  v12 = 0;
  while ( v9 != -8 )
  {
    if ( v12 || v9 != -16 )
      v8 = v12;
    result = v6 & (unsigned int)(v15 + result);
    v9 = *(_QWORD *)(v5 + 8LL * (unsigned int)result);
    if ( *a2 == v9 )
      return result;
    ++v15;
    v12 = v8;
    v8 = (_QWORD *)(v5 + 8LL * (unsigned int)result);
  }
  v11 = *(_DWORD *)(a1 + 8);
  if ( !v12 )
    v12 = v8;
  ++*(_QWORD *)a1;
  v13 = (v11 >> 1) + 1;
  if ( !v4 )
  {
    v10 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v14 = 12;
  v10 = 4;
LABEL_9:
  if ( 4 * v13 >= v14 )
  {
    sub_14A7580(a1, 2 * v10);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v16 = a1 + 16;
      v17 = 3;
    }
    else
    {
      v26 = *(_DWORD *)(a1 + 24);
      v16 = *(_QWORD *)(a1 + 16);
      if ( !v26 )
        goto LABEL_55;
      v17 = v26 - 1;
    }
    v18 = v17 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v12 = (_QWORD *)(v16 + 8LL * v18);
    v19 = *v12;
    if ( *v12 != *a2 )
    {
      v28 = 1;
      v25 = 0;
      while ( v19 != -8 )
      {
        if ( v19 == -16 && !v25 )
          v25 = v12;
        v18 = v17 & (v28 + v18);
        v12 = (_QWORD *)(v16 + 8LL * v18);
        v19 = *v12;
        if ( *a2 == *v12 )
          goto LABEL_25;
        ++v28;
      }
      goto LABEL_31;
    }
LABEL_25:
    v11 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v10 - *(_DWORD *)(a1 + 12) - v13 <= v10 >> 3 )
  {
    sub_14A7580(a1, v10);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v20 = a1 + 16;
      v21 = 3;
      goto LABEL_28;
    }
    v27 = *(_DWORD *)(a1 + 24);
    v20 = *(_QWORD *)(a1 + 16);
    if ( v27 )
    {
      v21 = v27 - 1;
LABEL_28:
      v22 = v21 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v12 = (_QWORD *)(v20 + 8LL * v22);
      v23 = *v12;
      if ( *v12 != *a2 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != -8 )
        {
          if ( v23 == -16 && !v25 )
            v25 = v12;
          v22 = v21 & (v24 + v22);
          v12 = (_QWORD *)(v20 + 8LL * v22);
          v23 = *v12;
          if ( *a2 == *v12 )
            goto LABEL_25;
          ++v24;
        }
LABEL_31:
        if ( v25 )
          v12 = v25;
        goto LABEL_25;
      }
      goto LABEL_25;
    }
LABEL_55:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v11 >> 1) + 2) | v11 & 1;
  if ( *v12 != -8 )
    --*(_DWORD *)(a1 + 12);
  *v12 = *a2;
  result = *(unsigned int *)(a1 + 56);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 60) )
  {
    sub_16CD150(a1 + 48, a1 + 64, 0, 8);
    result = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * result) = *a2;
  ++*(_DWORD *)(a1 + 56);
  return result;
}
