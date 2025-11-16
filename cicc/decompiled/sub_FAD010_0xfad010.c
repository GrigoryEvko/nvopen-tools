// Function: sub_FAD010
// Address: 0xfad010
//
__int64 *__fastcall sub_FAD010(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v8; // cl
  __int64 v9; // r8
  int v10; // esi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 *result; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // edi
  __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 *v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rdx
  int v23; // r11d
  __int64 *v24; // r10
  __int64 v25; // rdi
  int v26; // esi
  unsigned int v27; // edx
  __int64 v28; // r8
  int v29; // eax
  int v30; // r10d
  __int64 *v31; // r9
  __int64 *v32; // [rsp+8h] [rbp-28h] BYREF

  v8 = *(_BYTE *)(a1 + 8) & 1;
  if ( v8 )
  {
    v9 = a1 + 16;
    v10 = 3;
  }
  else
  {
    v14 = *(unsigned int *)(a1 + 24);
    v9 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v14 )
    {
      v15 = *(unsigned int *)(a1 + 8);
      ++*(_QWORD *)a1;
      v32 = 0;
      v16 = ((unsigned int)v15 >> 1) + 1;
LABEL_8:
      v17 = 3 * v14;
      goto LABEL_9;
    }
    v10 = v14 - 1;
  }
  v11 = v10 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
  v12 = (__int64 *)(v9 + 88LL * v11);
  a6 = *v12;
  if ( *a2 == *v12 )
    return v12 + 1;
  v23 = 1;
  v24 = 0;
  while ( a6 != -4096 )
  {
    if ( !v24 && a6 == -8192 )
      v24 = v12;
    v11 = v10 & (v23 + v11);
    v12 = (__int64 *)(v9 + 88LL * v11);
    a6 = *v12;
    if ( *a2 == *v12 )
      return v12 + 1;
    ++v23;
  }
  v15 = *(unsigned int *)(a1 + 8);
  v17 = 12;
  v14 = 4;
  if ( !v24 )
    v24 = v12;
  ++*(_QWORD *)a1;
  v32 = v24;
  v16 = ((unsigned int)v15 >> 1) + 1;
  if ( !v8 )
  {
    v14 = *(unsigned int *)(a1 + 24);
    goto LABEL_8;
  }
LABEL_9:
  v18 = 4 * v16;
  if ( (unsigned int)v18 >= v17 )
  {
    sub_FACA90(a1, (_QWORD *)(unsigned int)(2 * v14), v15, v18, v9, a6);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v25 = a1 + 16;
      v26 = 3;
    }
    else
    {
      v29 = *(_DWORD *)(a1 + 24);
      v25 = *(_QWORD *)(a1 + 16);
      v26 = v29 - 1;
      if ( !v29 )
      {
        v32 = 0;
        LODWORD(v15) = *(_DWORD *)(a1 + 8);
        v20 = 0;
        goto LABEL_12;
      }
    }
    v27 = v26 & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
    v20 = (__int64 *)(v25 + 88LL * v27);
    v28 = *v20;
    if ( *v20 == *a2 )
    {
LABEL_24:
      v32 = v20;
      LODWORD(v15) = *(_DWORD *)(a1 + 8);
    }
    else
    {
      v30 = 1;
      v31 = 0;
      while ( v28 != -4096 )
      {
        if ( !v31 && v28 == -8192 )
          v31 = v20;
        v27 = v26 & (v30 + v27);
        v20 = (__int64 *)(v25 + 88LL * v27);
        v28 = *v20;
        if ( *a2 == *v20 )
          goto LABEL_24;
        ++v30;
      }
      LODWORD(v15) = *(_DWORD *)(a1 + 8);
      if ( !v31 )
        v31 = v20;
      v32 = v31;
      v20 = v31;
    }
    goto LABEL_12;
  }
  v19 = (_DWORD)v14 - *(_DWORD *)(a1 + 12) - v16;
  if ( (unsigned int)v19 <= (unsigned int)v14 >> 3 )
  {
    sub_FACA90(a1, (_QWORD *)v14, v15, v19, v9, a6);
    sub_F9DA50(a1, a2, &v32);
    v20 = v32;
    LODWORD(v15) = *(_DWORD *)(a1 + 8);
  }
  else
  {
    v20 = v32;
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * ((unsigned int)v15 >> 1) + 2) | v15 & 1;
  if ( *v20 != -4096 )
    --*(_DWORD *)(a1 + 12);
  v21 = *a2;
  v20[2] = 0x400000000LL;
  *v20 = v21;
  v22 = v20 + 3;
  result = v20 + 1;
  *result = (__int64)v22;
  return result;
}
