// Function: sub_36F0A50
// Address: 0x36f0a50
//
__int64 __fastcall sub_36F0A50(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v8; // r13
  char v9; // dl
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // eax
  unsigned __int8 *v13; // rcx
  __int64 v14; // rax
  unsigned int v16; // esi
  unsigned int v17; // eax
  unsigned __int8 *v18; // r14
  int v19; // ecx
  unsigned int v20; // edi
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // rcx
  int v24; // eax
  unsigned int v25; // edx
  unsigned __int8 v26; // si
  __int64 v27; // rcx
  int v28; // edx
  unsigned int v29; // eax
  unsigned __int8 v30; // si
  unsigned __int8 *v31; // rdi
  int v32; // eax
  int v33; // edx

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8) & 1;
  if ( v9 )
  {
    v10 = a1 + 16;
    v11 = 7;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 24);
    v10 = *(_QWORD *)(a1 + 16);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_9:
      v20 = 3 * v16;
      goto LABEL_10;
    }
    v11 = v16 - 1;
  }
  v12 = v11 & (37 * v8);
  v13 = (unsigned __int8 *)(v10 + 8LL * v12);
  a5 = *v13;
  if ( v8 == (_BYTE)a5 )
  {
LABEL_4:
    v14 = *((unsigned int *)v13 + 1);
    return *(_QWORD *)(a1 + 80) + 8 * v14 + 4;
  }
  a6 = 1;
  v18 = 0;
  while ( (_BYTE)a5 != 0xFF )
  {
    if ( !v18 && (_BYTE)a5 == 0xFE )
      v18 = v13;
    v12 = v11 & (a6 + v12);
    v13 = (unsigned __int8 *)(v10 + 8LL * v12);
    a5 = *v13;
    if ( v8 == (_BYTE)a5 )
      goto LABEL_4;
    a6 = (unsigned int)(a6 + 1);
  }
  v17 = *(_DWORD *)(a1 + 8);
  if ( !v18 )
    v18 = v13;
  ++*(_QWORD *)a1;
  v19 = (v17 >> 1) + 1;
  if ( !v9 )
  {
    v16 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 24;
  v16 = 8;
LABEL_10:
  if ( v20 <= 4 * v19 )
  {
    sub_36F0660(a1, 2 * v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v23 = a1 + 16;
      v24 = 7;
    }
    else
    {
      v32 = *(_DWORD *)(a1 + 24);
      v23 = *(_QWORD *)(a1 + 16);
      if ( !v32 )
        goto LABEL_56;
      v24 = v32 - 1;
    }
    v25 = v24 & (37 * v8);
    v18 = (unsigned __int8 *)(v23 + 8LL * v25);
    v26 = *v18;
    if ( v8 != *v18 )
    {
      a5 = 1;
      v31 = 0;
      while ( v26 != 0xFF )
      {
        if ( !v31 && v26 == 0xFE )
          v31 = v18;
        a6 = (unsigned int)(a5 + 1);
        v25 = v24 & (a5 + v25);
        v18 = (unsigned __int8 *)(v23 + 8LL * v25);
        v26 = *v18;
        if ( v8 == *v18 )
          goto LABEL_26;
        a5 = (unsigned int)a6;
      }
      goto LABEL_32;
    }
LABEL_26:
    v17 = *(_DWORD *)(a1 + 8);
    goto LABEL_12;
  }
  if ( v16 - *(_DWORD *)(a1 + 12) - v19 <= v16 >> 3 )
  {
    sub_36F0660(a1, v16);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v27 = a1 + 16;
      v28 = 7;
      goto LABEL_29;
    }
    v33 = *(_DWORD *)(a1 + 24);
    v27 = *(_QWORD *)(a1 + 16);
    if ( v33 )
    {
      v28 = v33 - 1;
LABEL_29:
      v29 = v28 & (37 * v8);
      v18 = (unsigned __int8 *)(v27 + 8LL * v29);
      v30 = *v18;
      if ( v8 != *v18 )
      {
        a5 = 1;
        v31 = 0;
        while ( v30 != 0xFF )
        {
          if ( !v31 && v30 == 0xFE )
            v31 = v18;
          a6 = (unsigned int)(a5 + 1);
          v29 = v28 & (a5 + v29);
          v18 = (unsigned __int8 *)(v27 + 8LL * v29);
          v30 = *v18;
          if ( v8 == *v18 )
            goto LABEL_26;
          a5 = (unsigned int)a6;
        }
LABEL_32:
        if ( v31 )
          v18 = v31;
        goto LABEL_26;
      }
      goto LABEL_26;
    }
LABEL_56:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *v18 != 0xFF )
    --*(_DWORD *)(a1 + 12);
  *v18 = v8;
  *((_DWORD *)v18 + 1) = 0;
  v21 = *(unsigned int *)(a1 + 88);
  v22 = *a2;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 92) )
  {
    sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v21 + 1, 8u, a5, a6);
    v21 = *(unsigned int *)(a1 + 88);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v21) = v22;
  v14 = *(unsigned int *)(a1 + 88);
  *(_DWORD *)(a1 + 88) = v14 + 1;
  *((_DWORD *)v18 + 1) = v14;
  return *(_QWORD *)(a1 + 80) + 8 * v14 + 4;
}
