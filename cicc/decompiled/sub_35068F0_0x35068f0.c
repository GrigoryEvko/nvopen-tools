// Function: sub_35068F0
// Address: 0x35068f0
//
bool __fastcall sub_35068F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  unsigned __int8 **v8; // rcx
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // r11d
  __int64 v13; // r9
  __int64 *v14; // rdx
  unsigned int v15; // r8d
  _QWORD *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdi
  unsigned __int64 *v19; // rbx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v23; // rcx
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned __int64 v28; // r15
  int v29; // eax
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // eax
  __int64 v33; // rdi
  int v34; // r10d
  __int64 *v35; // r9
  int v36; // eax
  int v37; // eax
  __int64 v38; // rdi
  __int64 *v39; // r8
  unsigned int v40; // ebx
  int v41; // r9d
  __int64 v42; // rsi

  if ( !a2 )
    return 0;
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
  {
    if ( *(_DWORD *)(a2 - 24) == 2 )
      v7 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    else
      v7 = 0;
    v8 = *(unsigned __int8 ***)(a2 - 32);
  }
  else
  {
    v23 = a2 - 16;
    if ( ((*(_WORD *)(a2 - 16) >> 6) & 0xF) == 2 )
      v7 = *(_QWORD *)(v23 - 8LL * ((v6 >> 2) & 0xF) + 8);
    else
      v7 = 0;
    v8 = (unsigned __int8 **)(v23 - 8LL * ((v6 >> 2) & 0xF));
  }
  v9 = sub_35057B0((_QWORD *)a1, *v8, v7);
  if ( !v9 )
    return 0;
  if ( *(_QWORD *)(a1 + 224) == v9 && *(_QWORD *)a1 == *(_QWORD *)(a3 + 32) )
    return 1;
  v10 = *(_DWORD *)(a1 + 256);
  v11 = a1 + 232;
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 232);
    goto LABEL_48;
  }
  v12 = 1;
  v13 = *(_QWORD *)(a1 + 240);
  v14 = 0;
  v15 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = (_QWORD *)(v13 + 16LL * v15);
  v17 = *v16;
  if ( *v16 != a2 )
  {
    while ( v17 != -4096 )
    {
      if ( v17 == -8192 && !v14 )
        v14 = v16;
      v15 = (v10 - 1) & (v12 + v15);
      v16 = (_QWORD *)(v13 + 16LL * v15);
      v17 = *v16;
      if ( *v16 == a2 )
        goto LABEL_10;
      ++v12;
    }
    if ( !v14 )
      v14 = v16;
    v24 = *(_DWORD *)(a1 + 248);
    ++*(_QWORD *)(a1 + 232);
    v25 = v24 + 1;
    if ( 4 * (v24 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 252) - v25 > v10 >> 3 )
      {
LABEL_35:
        *(_DWORD *)(a1 + 248) = v25;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a1 + 252);
        *v14 = a2;
        v19 = (unsigned __int64 *)(v14 + 1);
        v14[1] = 0;
        goto LABEL_38;
      }
      sub_35066C0(v11, v10);
      v36 = *(_DWORD *)(a1 + 256);
      if ( v36 )
      {
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 240);
        v39 = 0;
        v40 = v37 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v41 = 1;
        v25 = *(_DWORD *)(a1 + 248) + 1;
        v14 = (__int64 *)(v38 + 16LL * v40);
        v42 = *v14;
        if ( *v14 != a2 )
        {
          while ( v42 != -4096 )
          {
            if ( !v39 && v42 == -8192 )
              v39 = v14;
            v40 = v37 & (v41 + v40);
            v14 = (__int64 *)(v38 + 16LL * v40);
            v42 = *v14;
            if ( *v14 == a2 )
              goto LABEL_35;
            ++v41;
          }
          if ( v39 )
            v14 = v39;
        }
        goto LABEL_35;
      }
LABEL_71:
      ++*(_DWORD *)(a1 + 248);
      BUG();
    }
LABEL_48:
    sub_35066C0(v11, 2 * v10);
    v29 = *(_DWORD *)(a1 + 256);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 240);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = *(_DWORD *)(a1 + 248) + 1;
      v14 = (__int64 *)(v31 + 16LL * v32);
      v33 = *v14;
      if ( *v14 != a2 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v14;
          v32 = v30 & (v34 + v32);
          v14 = (__int64 *)(v31 + 16LL * v32);
          v33 = *v14;
          if ( *v14 == a2 )
            goto LABEL_35;
          ++v34;
        }
        if ( v35 )
          v14 = v35;
      }
      goto LABEL_35;
    }
    goto LABEL_71;
  }
LABEL_10:
  v18 = v16[1];
  v19 = v16 + 1;
  if ( !v18 )
  {
LABEL_38:
    v26 = sub_22077B0(0x40u);
    v27 = v26;
    if ( v26 )
    {
      *(_QWORD *)v26 = 0;
      *(_QWORD *)(v26 + 8) = v26 + 32;
      *(_QWORD *)(v26 + 16) = 4;
      *(_DWORD *)(v26 + 24) = 0;
      *(_BYTE *)(v26 + 28) = 1;
    }
    v28 = *v19;
    *v19 = v26;
    if ( v28 )
    {
      if ( !*(_BYTE *)(v28 + 28) )
        _libc_free(*(_QWORD *)(v28 + 8));
      j_j___libc_free_0(v28);
      v27 = *v19;
    }
    sub_35059C0((_QWORD *)a1, a2, v27);
    v18 = *v19;
  }
  if ( *(_BYTE *)(v18 + 28) )
  {
    v20 = *(_QWORD **)(v18 + 8);
    v21 = &v20[*(unsigned int *)(v18 + 20)];
    if ( v20 != v21 )
    {
      while ( a3 != *v20 )
      {
        if ( v21 == ++v20 )
          return 0;
      }
      return 1;
    }
    return 0;
  }
  return sub_C8CA60(v18, a3) != 0;
}
