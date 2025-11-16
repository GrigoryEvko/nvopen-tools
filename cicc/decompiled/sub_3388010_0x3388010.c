// Function: sub_3388010
// Address: 0x3388010
//
__int64 __fastcall sub_3388010(__int64 a1, __int64 *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r14
  char v10; // cl
  __int64 v11; // rdi
  int v12; // esi
  unsigned int v13; // edx
  __int64 v14; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  __int64 v18; // r8
  int v19; // edx
  unsigned int v20; // edi
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // r11d
  __int64 v28; // r14
  __int64 v29; // r12
  _QWORD *v30; // rdx
  __int64 v31; // rcx
  int v32; // edx
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // rcx
  int v36; // edx
  unsigned int v37; // eax
  __int64 v38; // rsi
  __int64 v39; // rdi
  int v40; // edx
  int v41; // edx

  v9 = *a2;
  v10 = *(_BYTE *)(a1 + 8) & 1;
  if ( v10 )
  {
    v11 = a1 + 16;
    v12 = 7;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_8:
      v20 = 3 * v16;
      goto LABEL_9;
    }
    v12 = v16 - 1;
  }
  v13 = v12 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v14 = v11 + 16LL * v13;
  a6 = *(_QWORD *)v14;
  if ( v9 == *(_QWORD *)v14 )
    return *(_QWORD *)(a1 + 144) + 16LL * *(unsigned int *)(v14 + 8);
  v27 = 1;
  v18 = 0;
  while ( a6 != -4096 )
  {
    if ( !v18 && a6 == -8192 )
      v18 = v14;
    v13 = v12 & (v27 + v13);
    v14 = v11 + 16LL * v13;
    a6 = *(_QWORD *)v14;
    if ( v9 == *(_QWORD *)v14 )
      return *(_QWORD *)(a1 + 144) + 16LL * *(unsigned int *)(v14 + 8);
    ++v27;
  }
  if ( !v18 )
    v18 = v14;
  v17 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v19 = (v17 >> 1) + 1;
  if ( !v10 )
  {
    v16 = *(_DWORD *)(a1 + 24);
    goto LABEL_8;
  }
  v20 = 24;
  v16 = 8;
LABEL_9:
  if ( v20 <= 4 * v19 )
  {
    sub_3387BD0(a1, 2 * v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v31 = a1 + 16;
      v32 = 7;
    }
    else
    {
      v40 = *(_DWORD *)(a1 + 24);
      v31 = *(_QWORD *)(a1 + 16);
      if ( !v40 )
        goto LABEL_60;
      v32 = v40 - 1;
    }
    v33 = v32 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v18 = v31 + 16LL * v33;
    v34 = *(_QWORD *)v18;
    if ( v9 != *(_QWORD *)v18 )
    {
      a6 = 1;
      v39 = 0;
      while ( v34 != -4096 )
      {
        if ( !v39 && v34 == -8192 )
          v39 = v18;
        v33 = v32 & (a6 + v33);
        v18 = v31 + 16LL * v33;
        v34 = *(_QWORD *)v18;
        if ( v9 == *(_QWORD *)v18 )
          goto LABEL_30;
        a6 = (unsigned int)(a6 + 1);
      }
      goto LABEL_36;
    }
LABEL_30:
    v17 = *(_DWORD *)(a1 + 8);
    goto LABEL_11;
  }
  if ( v16 - *(_DWORD *)(a1 + 12) - v19 <= v16 >> 3 )
  {
    sub_3387BD0(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v35 = a1 + 16;
      v36 = 7;
      goto LABEL_33;
    }
    v41 = *(_DWORD *)(a1 + 24);
    v35 = *(_QWORD *)(a1 + 16);
    if ( v41 )
    {
      v36 = v41 - 1;
LABEL_33:
      v37 = v36 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v18 = v35 + 16LL * v37;
      v38 = *(_QWORD *)v18;
      if ( v9 != *(_QWORD *)v18 )
      {
        a6 = 1;
        v39 = 0;
        while ( v38 != -4096 )
        {
          if ( !v39 && v38 == -8192 )
            v39 = v18;
          v37 = v36 & (a6 + v37);
          v18 = v35 + 16LL * v37;
          v38 = *(_QWORD *)v18;
          if ( v9 == *(_QWORD *)v18 )
            goto LABEL_30;
          a6 = (unsigned int)(a6 + 1);
        }
LABEL_36:
        if ( v39 )
          v18 = v39;
        goto LABEL_30;
      }
      goto LABEL_30;
    }
LABEL_60:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *(_QWORD *)v18 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_DWORD *)(v18 + 8) = 0;
  *(_QWORD *)v18 = v9;
  *(_DWORD *)(v18 + 8) = *(_DWORD *)(a1 + 152);
  v21 = *(unsigned int *)(a1 + 152);
  v22 = *(unsigned int *)(a1 + 156);
  v23 = *(_DWORD *)(a1 + 152);
  if ( v21 >= v22 )
  {
    v28 = *a2;
    v29 = (unsigned __int8)*a3;
    if ( v22 < v21 + 1 )
    {
      sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), v21 + 1, 0x10u, v21 + 1, a6);
      v21 = *(unsigned int *)(a1 + 152);
    }
    v30 = (_QWORD *)(*(_QWORD *)(a1 + 144) + 16 * v21);
    *v30 = v28;
    v30[1] = v29;
    v24 = *(_QWORD *)(a1 + 144);
    v26 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
    *(_DWORD *)(a1 + 152) = v26;
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 144);
    v25 = v24 + 16 * v21;
    if ( v25 )
    {
      *(_QWORD *)v25 = *a2;
      *(_BYTE *)(v25 + 8) = *a3;
      v23 = *(_DWORD *)(a1 + 152);
      v24 = *(_QWORD *)(a1 + 144);
    }
    v26 = (unsigned int)(v23 + 1);
    *(_DWORD *)(a1 + 152) = v26;
  }
  return v24 + 16 * v26 - 16;
}
