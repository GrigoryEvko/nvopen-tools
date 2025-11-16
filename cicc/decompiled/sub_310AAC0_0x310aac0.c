// Function: sub_310AAC0
// Address: 0x310aac0
//
__int64 __fastcall sub_310AAC0(__int64 a1, __int64 a2)
{
  int v4; // edi
  __int64 v5; // rsi
  int v6; // r8d
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 result; // rax
  __int64 v12; // rax
  char v13; // di
  __int64 v14; // r8
  int v15; // esi
  unsigned int v16; // ecx
  _QWORD *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rax
  unsigned int v20; // esi
  int v21; // eax
  unsigned int v22; // edx
  _QWORD *v23; // r9
  int v24; // ecx
  unsigned int v25; // edi
  int v26; // r11d
  int v27; // r9d
  __int64 v28; // rdi
  int v29; // ecx
  unsigned int v30; // edx
  __int64 v31; // rsi
  __int64 v32; // rdi
  int v33; // ecx
  unsigned int v34; // edx
  __int64 v35; // rsi
  int v36; // r10d
  _QWORD *v37; // r8
  int v38; // ecx
  int v39; // ecx
  int v40; // r10d
  __int64 v41; // [rsp+8h] [rbp-28h]
  __int64 v42; // [rsp+8h] [rbp-28h]

  v4 = *(_BYTE *)(a1 + 16) & 1;
  if ( v4 )
  {
    v5 = a1 + 24;
    v6 = 3;
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 32);
    v5 = *(_QWORD *)(a1 + 24);
    if ( !(_DWORD)v12 )
      goto LABEL_15;
    v6 = v12 - 1;
  }
  v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
    goto LABEL_4;
  v21 = 1;
  while ( v9 != -4096 )
  {
    v27 = v21 + 1;
    v7 = v6 & (v21 + v7);
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
      goto LABEL_4;
    v21 = v27;
  }
  if ( (_BYTE)v4 )
  {
    v19 = 64;
    goto LABEL_16;
  }
  v12 = *(unsigned int *)(a1 + 32);
LABEL_15:
  v19 = 16 * v12;
LABEL_16:
  v8 = (__int64 *)(v5 + v19);
LABEL_4:
  v10 = 64;
  if ( !(_BYTE)v4 )
    v10 = 16LL * *(unsigned int *)(a1 + 32);
  if ( v8 != (__int64 *)(v5 + v10) )
    return v8[1];
  result = sub_310AE90(a1, a2);
  v13 = *(_BYTE *)(a1 + 16) & 1;
  if ( v13 )
  {
    v14 = a1 + 24;
    v15 = 3;
  }
  else
  {
    v20 = *(_DWORD *)(a1 + 32);
    v14 = *(_QWORD *)(a1 + 24);
    if ( !v20 )
    {
      v22 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)(a1 + 8);
      v23 = 0;
      v24 = (v22 >> 1) + 1;
LABEL_24:
      v25 = 3 * v20;
      goto LABEL_25;
    }
    v15 = v20 - 1;
  }
  v16 = v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v17 = (_QWORD *)(v14 + 16LL * v16);
  v18 = *v17;
  if ( *v17 == a2 )
    return v17[1];
  v26 = 1;
  v23 = 0;
  while ( v18 != -4096 )
  {
    if ( v18 == -8192 && !v23 )
      v23 = v17;
    v16 = v15 & (v26 + v16);
    v17 = (_QWORD *)(v14 + 16LL * v16);
    v18 = *v17;
    if ( *v17 == a2 )
      return v17[1];
    ++v26;
  }
  if ( !v23 )
    v23 = v17;
  v22 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)(a1 + 8);
  v24 = (v22 >> 1) + 1;
  if ( !v13 )
  {
    v20 = *(_DWORD *)(a1 + 32);
    goto LABEL_24;
  }
  v25 = 12;
  v20 = 4;
LABEL_25:
  if ( 4 * v24 >= v25 )
  {
    v41 = result;
    sub_DB0DD0(a1 + 8, 2 * v20);
    result = v41;
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    {
      v28 = a1 + 24;
      v29 = 3;
    }
    else
    {
      v38 = *(_DWORD *)(a1 + 32);
      v28 = *(_QWORD *)(a1 + 24);
      if ( !v38 )
        goto LABEL_71;
      v29 = v38 - 1;
    }
    v30 = v29 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v23 = (_QWORD *)(v28 + 16LL * v30);
    v31 = *v23;
    if ( *v23 != a2 )
    {
      v40 = 1;
      v37 = 0;
      while ( v31 != -4096 )
      {
        if ( v31 == -8192 && !v37 )
          v37 = v23;
        v30 = v29 & (v40 + v30);
        v23 = (_QWORD *)(v28 + 16LL * v30);
        v31 = *v23;
        if ( *v23 == a2 )
          goto LABEL_41;
        ++v40;
      }
      goto LABEL_47;
    }
LABEL_41:
    v22 = *(_DWORD *)(a1 + 16);
    goto LABEL_27;
  }
  if ( v20 - *(_DWORD *)(a1 + 20) - v24 <= v20 >> 3 )
  {
    v42 = result;
    sub_DB0DD0(a1 + 8, v20);
    result = v42;
    if ( (*(_BYTE *)(a1 + 16) & 1) != 0 )
    {
      v32 = a1 + 24;
      v33 = 3;
      goto LABEL_44;
    }
    v39 = *(_DWORD *)(a1 + 32);
    v32 = *(_QWORD *)(a1 + 24);
    if ( v39 )
    {
      v33 = v39 - 1;
LABEL_44:
      v34 = v33 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v23 = (_QWORD *)(v32 + 16LL * v34);
      v35 = *v23;
      if ( *v23 != a2 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( v35 == -8192 && !v37 )
            v37 = v23;
          v34 = v33 & (v36 + v34);
          v23 = (_QWORD *)(v32 + 16LL * v34);
          v35 = *v23;
          if ( *v23 == a2 )
            goto LABEL_41;
          ++v36;
        }
LABEL_47:
        if ( v37 )
          v23 = v37;
        goto LABEL_41;
      }
      goto LABEL_41;
    }
LABEL_71:
    *(_DWORD *)(a1 + 16) = (2 * (*(_DWORD *)(a1 + 16) >> 1) + 2) | *(_DWORD *)(a1 + 16) & 1;
    BUG();
  }
LABEL_27:
  *(_DWORD *)(a1 + 16) = (2 * (v22 >> 1) + 2) | v22 & 1;
  if ( *v23 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v23 = a2;
  v23[1] = result;
  return result;
}
