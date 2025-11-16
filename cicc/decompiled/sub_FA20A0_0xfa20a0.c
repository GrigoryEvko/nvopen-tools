// Function: sub_FA20A0
// Address: 0xfa20a0
//
_QWORD *__fastcall sub_FA20A0(__int64 a1, __int64 *a2)
{
  char v4; // cl
  __int64 v5; // rdi
  int v6; // esi
  unsigned int v7; // r9d
  _QWORD *v8; // rax
  __int64 v9; // r8
  _QWORD *result; // rax
  unsigned int v11; // esi
  unsigned int v12; // edx
  int v13; // eax
  unsigned int v14; // edi
  _QWORD *v15; // rax
  __int64 v16; // rdx
  int v17; // r11d
  _QWORD *v18; // r10
  __int64 v19; // rsi
  int v20; // edi
  unsigned int v21; // edx
  __int64 v22; // r8
  int v23; // eax
  int v24; // r10d
  _QWORD *v25; // r9
  _QWORD *v26; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( v4 )
  {
    v5 = a1 + 16;
    v6 = 7;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v11 )
    {
      v12 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v26 = 0;
      v13 = (v12 >> 1) + 1;
LABEL_8:
      v14 = 3 * v11;
      goto LABEL_9;
    }
    v6 = v11 - 1;
  }
  v7 = v6 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v8 = (_QWORD *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( *a2 == *v8 )
    return v8 + 1;
  v17 = 1;
  v18 = 0;
  while ( v9 != -4096 )
  {
    if ( !v18 && v9 == -8192 )
      v18 = v8;
    v7 = v6 & (v7 + v17);
    v8 = (_QWORD *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *a2 == *v8 )
      return v8 + 1;
    ++v17;
  }
  v12 = *(_DWORD *)(a1 + 8);
  v14 = 24;
  v11 = 8;
  if ( !v18 )
    v18 = v8;
  ++*(_QWORD *)a1;
  v26 = v18;
  v13 = (v12 >> 1) + 1;
  if ( !v4 )
  {
    v11 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  if ( 4 * v13 >= v14 )
  {
    sub_F5E3E0(a1, 2 * v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v19 = a1 + 16;
      v20 = 7;
    }
    else
    {
      v23 = *(_DWORD *)(a1 + 24);
      v19 = *(_QWORD *)(a1 + 16);
      v20 = v23 - 1;
      if ( !v23 )
      {
        v26 = 0;
        v12 = *(_DWORD *)(a1 + 8);
        v15 = 0;
        goto LABEL_12;
      }
    }
    v21 = v20 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v15 = (_QWORD *)(v19 + 16LL * v21);
    v22 = *v15;
    if ( *v15 == *a2 )
    {
LABEL_24:
      v26 = v15;
      v12 = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v24 = 1;
      v25 = 0;
      while ( v22 != -4096 )
      {
        if ( !v25 && v22 == -8192 )
          v25 = v15;
        v21 = v20 & (v24 + v21);
        v15 = (_QWORD *)(v19 + 16LL * v21);
        v22 = *v15;
        if ( *a2 == *v15 )
          goto LABEL_24;
        ++v24;
      }
      v12 = *(_DWORD *)(a1 + 8);
      if ( !v25 )
        v25 = v15;
      v26 = v25;
      v15 = v25;
    }
    goto LABEL_12;
  }
  if ( v11 - *(_DWORD *)(a1 + 12) - v13 <= v11 >> 3 )
  {
    sub_F5E3E0(a1, v11);
    sub_F9BF10(a1, a2, &v26);
    v15 = v26;
    v12 = *(_DWORD *)(a1 + 8);
  }
  else
  {
    v15 = v26;
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v12 >> 1) + 2) | v12 & 1;
  if ( *v15 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v16 = *a2;
  *((_DWORD *)v15 + 2) = 0;
  result = v15 + 1;
  *(result - 1) = v16;
  return result;
}
